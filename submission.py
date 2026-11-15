from types import SimpleNamespace
import cvxpy as cp
import numpy as np
import pandas as pd

from score import score, validate

# ---- walk-forward fitting ----
REFIT_EVERY = 63              # days between refits
FIRST_FIT_DAYS = 252          # days of history before the first fit (~1 year: first fit 2018-12-24); flat before it
SPEED_DAYS = 252              # days of signals used once to measure each signal's speed: the year before the first fit

# ---- return model ----
HORIZON = 3                   # the regression predicts the sum of the next 3 days' returns
SPEED_CUTOFF = 5              # fast if a signal's autocorrelation drops below 0.5 within this many days, else slow
LAGS = {'fast': (0,), 'slow': (0, 5)}   # per signal speed: 0 = today's value, L = today − L days ago
RIDGE_LAMBDA = 0.3            # ridge penalty
VOL_LOOKBACK = 252            # days in the vol that puts returns in vol units and scales the forecast back

# ---- risk model ----
COV_LOOKBACK = 252            # days of past returns in the covariance
COV_SHRINK = 0.5              # pull correlations toward 0: 0 = sample covariance, 1 = diagonal only

# ---- optimiser ----
RISK_AVERSION = 1.5           # bigger = smaller positions, less vol
COST_MULT = 1.0               # weight on trading cost in the objective

# ---- README limits, units ----
GROSS_MAX, NET_MAX = 2.5e6, 1.0e6
MM = 1e6                      # inside the optimiser positions are in $mm per bp; positions.csv is in $ per bp
ANN = 252

# Data
def build_data(signals_table, returns_table, products):
    #build return data and signals data into a SimpleNamespace for the rest of the pipeline
    dates = pd.DatetimeIndex(np.sort(signals_table['date'].unique()))
    names = products['product'].tolist()
    signal_names = [column for column in signals_table.columns if column.startswith('S')]

    def wide(table, column):
        return table.pivot(index='date', columns='product', values=column).reindex(index=dates, columns=names).to_numpy()

    ret_close = wide(returns_table, 'ret_close')        # yesterday's close → today's close (bp)
    ret_intra = wide(returns_table, 'ret_intra')        # today's 09:00 → today's close (bp)
    fwd = np.full_like(ret_close, np.nan)               # 09:00 day d → 09:00 day d+1: what pos[d] earns
    fwd[:-1] = ret_intra[:-1] + (ret_close - ret_intra)[1:]
    limits = products.set_index('product').loc[names]
    return SimpleNamespace(
        dates=dates, names=names,
        signals=np.stack([wide(signals_table, s) for s in signal_names], axis=2),
        ret_close=ret_close, fwd=fwd,
        n_known=int(np.isfinite(ret_close).all(axis=1).sum()),       # days with returns
        bucket=limits['bucket'].to_numpy(),                          # B1 (shortest maturity) … B4
        cost=limits['cost'].to_numpy(float),                         # bp per $ traded
        max_pos=limits['max_pos'].to_numpy(float) / MM,
        max_trade=limits['max_trade'].to_numpy(float) / MM,
        min_trade=limits['min_trade'].to_numpy(float) / MM,
    )

# 1. Features
def signal_speeds(signals):
    # classify speed of signal based on its autocorrelation half life: the first lag at which its autocorrelation
    # (averaged over products) drops below 0.5. fast ≤ SPEED_CUTOFF days, slow beyond. The signals form two clusters
    # (half-lives 1-3 days and 8+ days); the cutoff sits between them and equals the lag slow signals get.
    speeds = []
    for k in range(signals.shape[2]):
        series = pd.DataFrame(signals[:, :, k])
        half_life = np.inf
        for lag in range(1, 61):
            if series.corrwith(series.shift(lag)).mean() < 0.5:
                half_life = lag
                break
        speeds.append('fast' if half_life <= SPEED_CUTOFF else 'slow')
    return speeds


def lag_features(signals, speeds):
    # the signals as given (they arrive near unit scale, clipped at ±2), plus for each lag L > 0 of the signal's speed
    # today's value minus the value L days ago (today's value for the first L days, which have no earlier value)
    columns = []
    for k, speed in enumerate(speeds):
        today = signals[:, :, k]
        for lag in LAGS[speed]:
            if lag == 0:
                columns.append(today)
            else:
                days_ago = np.zeros_like(today)
                days_ago[lag:] = today[:-lag]
                columns.append(today - days_ago)
    return np.stack(columns, axis=2)


# 2. Vol and the target

def trailing_vol(ret_close, lookback):
    # sample vol of the last `lookback` days of returns up to day d−1, NaN where that runs past the first row given.
    n_known = int(np.isfinite(ret_close).all(axis=1).sum())
    vol = pd.DataFrame(ret_close[:n_known]).rolling(lookback, min_periods=60).std()
    vol = vol.reindex(range(len(ret_close))).ffill()
    return vol.shift(1).to_numpy()                  # row d: returns up to d−1

def h_day_sum(daily, horizon):
    return pd.DataFrame(daily).rolling(horizon).sum().shift(1 - horizon).to_numpy()


# 3. Walk-forward fitting and the forecast

def fit_days(data, refit_every):
    #Every `refit_every` days from FIRST_FIT_DAYS on
    return [t for t in range(FIRST_FIT_DAYS, len(data.dates), refit_every) if t - 2 < data.n_known]


def fit_betas(features, fwd_in_vols, buckets):
    #Ridge regression (for each bucket). Returns products × features (one row per product, shared per bucket).
    target = h_day_sum(fwd_in_vols, HORIZON)
    n_features = features.shape[2]
    beta = np.zeros((features.shape[1], n_features))
    for bucket in np.unique(buckets):                   # ridge, pooled over the bucket's products and days
        members = buckets == bucket
        X = features[:, members].reshape(-1, n_features)
        y = target[:, members].reshape(-1)
        known = ~np.isnan(y)
        X, y = X[known], y[known]
        beta[members] = np.linalg.solve(X.T @ X + RIDGE_LAMBDA * len(y) * np.eye(n_features), X.T @ y)
    fitted = (features * beta).sum(axis=2)              # calibration
    for bucket in np.unique(buckets):
        members = buckets == bucket
        f, y = fitted[:, members].ravel(), target[:, members].ravel()
        known = np.isfinite(y)
        beta[members] *= max(f[known] @ y[known] / max(f[known] @ f[known], 1e-12), 0.0)
    return beta


def forecast(features, vol, beta):
    # Expected next-HORIZON-day return in bp
    in_vols = (features * beta).sum(axis=2)
    in_vols = in_vols - in_vols.mean(axis=1, keepdims=True)     # demean across products, each day
    return in_vols * vol


# 4. Risk

def sample_covariance(ret_close, n_known):
    #cov[d]: covariance of the last COV_LOOKBACK days of returns up to day d−1 + shrinkage
    n_days, n_products = ret_close.shape
    cov = np.full((n_days, n_products, n_products), np.nan)
    for t in range(n_days):
        stop = min(t, n_known)                           # rows < stop are known at 09:00 on day t
        past = ret_close[max(0, stop - COV_LOOKBACK):stop]
        if len(past) < 60:
            continue
        sample = np.cov(past, rowvar=False)
        cov[t] = (1 - COV_SHRINK) * sample + COV_SHRINK * np.diag(np.diag(sample))
    return cov


# 5. Positions: the daily optimization

class DailyMVO:
    """Each day, positions x ($mm per bp):

        maximise    alpha·x − COST_MULT·Σ cost_i·|x_i − x_prev_i| − ½·RISK_AVERSION·HORIZON·x'Σx
        subject to  |x_i| ≤ max_pos_i, |x_i − x_prev_i| ≤ max_trade_i, Σ|x_i| ≤ 2.5, |Σ x_i| ≤ 1.0

    min_trade (each trade is 0 or ≥ min_trade) isn't convex: trades that come out smaller are frozen at 0 and the
    problem is solved again. The problem is built once; each day only alpha, the covariance and yesterday's
    positions change."""

    def __init__(self, data):
        n = len(data.names)
        self.data = data
        self.alpha = cp.Parameter(n)
        self.x_prev = cp.Parameter(n)
        self.chol = cp.Parameter((n, n))                 # chol'·chol = Σ, so x'Σx = |chol·x|²
        self.trade_cap = cp.Parameter(n, nonneg=True)     # max_trade, or 0 for a product frozen today
        self.x = cp.Variable(n)
        trade = self.x - self.x_prev
        expected_return = self.alpha @ self.x
        trading_cost = COST_MULT * data.cost @ cp.abs(trade)
        risk = 0.5 * RISK_AVERSION * HORIZON * cp.sum_squares(self.chol @ self.x)
        constraints = [
            cp.abs(self.x) <= data.max_pos,
            cp.abs(trade) <= self.trade_cap,
            cp.norm1(self.x) <= GROSS_MAX / MM,
            cp.abs(cp.sum(self.x)) <= NET_MAX / MM,
        ]
        self.problem = cp.Problem(cp.Maximize(expected_return - trading_cost - risk), constraints)

    def solve_once(self):
        try:
            self.problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            return None
        return self.x.value if self.problem.status in ('optimal', 'optimal_inaccurate') else None

    def solve(self, alpha, cov, x_prev):
        d = self.data
        self.alpha.value = alpha
        self.x_prev.value = x_prev
        self.chol.value = np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov))).T
        trade_cap = d.max_trade.copy()
        for _ in range(3):
            self.trade_cap.value = trade_cap
            x = self.solve_once()
            if x is None:
                return x_prev                            # give up for today: no trades
            small = np.abs(x - x_prev) < d.min_trade
            if not (small & (trade_cap > 0)).any():
                break
            trade_cap = np.where(small, 0.0, trade_cap)
        x = np.where(np.abs(x - x_prev) < d.min_trade, x_prev, x)    # clean up solver round-off
        x = x_prev + np.clip(x - x_prev, -d.max_trade, d.max_trade)
        x = np.clip(x, -d.max_pos, d.max_pos)
        if np.abs(x).sum() > GROSS_MAX / MM + 1e-6 or abs(x.sum()) > NET_MAX / MM + 1e-6:
            return x_prev
        return x


def trade(alpha, cov, data):
    # Positions ($ per bp, days × products)
    optimiser = DailyMVO(data)
    positions = np.zeros((len(data.dates), len(data.names)))
    x = np.zeros(len(data.names))
    for t in range(len(data.dates)):
        if not np.isnan(alpha[t]).any() and not np.isnan(cov[t]).any():
            x = optimiser.solve(alpha[t], cov[t], x)
        positions[t] = x
    return pd.DataFrame(positions * MM, index=data.dates, columns=data.names)


# The whole pipeline
def make_positions(data, refit_every=REFIT_EVERY):
    # entire pipeline caller 
    features = lag_features(data.signals, signal_speeds(data.signals[:SPEED_DAYS]))
    vol = trailing_vol(data.ret_close, VOL_LOOKBACK)
    fwd_in_vols = data.fwd / vol
    days = fit_days(data, refit_every)
    alpha = np.full(vol.shape, np.nan)
    for k, t in enumerate(days):
        until = days[k + 1] if k + 1 < len(days) else len(data.dates)
        beta = fit_betas(features[:t - 1], fwd_in_vols[:t - 1], data.bucket)     # rows 0 … t−2 only
        alpha[t:until] = forecast(features[t:until], vol[t:until], beta)
    cov = sample_covariance(data.ret_close, data.n_known)
    return trade(alpha, cov, data), days


def to_long(positions):
    return positions.rename_axis(index='date', columns='product').reset_index().melt(
        id_vars='date', var_name='product', value_name='pos')


def main():
    # run from the folder that holds the data files: python submission.py
    signals_table = pd.read_parquet('signals.parquet')
    returns_table = pd.read_parquet('returns.parquet')
    products = pd.read_csv('products.csv')
    signals_table['date'] = pd.to_datetime(signals_table['date'])
    returns_table['date'] = pd.to_datetime(returns_table['date'])

    positions, days = make_positions(build_data(signals_table, returns_table, products))
    table = to_long(positions)
    breaches = validate(table, signals_table, returns_table, products)
    if not breaches.empty:
        print(breaches.to_string())
        raise SystemExit('positions.csv not written: the positions break a README rule')
    table.to_csv('positions.csv', index=False)
    print(f'wrote positions.csv: {len(table):,} rows, {table["date"].nunique()} days × '
          f'{table["product"].nunique()} products; flat until the first fit on {positions.index[days[0]].date()}')
    print(score(table, returns_table, products).round(2).to_string())


if __name__ == '__main__':
    main()
