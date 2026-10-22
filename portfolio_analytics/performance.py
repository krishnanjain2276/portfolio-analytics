"""Return statistics, and the harder question: could this track record just be luck?

The usual tearsheet numbers (CAGR, Sharpe, Sortino, drawdowns) are here, plus three tools
from Bailey and Lopez de Prado that take the sampling noise in a Sharpe ratio seriously:

* Probabilistic Sharpe Ratio (PSR): how likely the true Sharpe is above zero, allowing for
  skew and fat tails.
* Deflated Sharpe Ratio (DSR): the same, but against the best Sharpe you'd expect from N
  strategies with no skill at all. Trying lots of variants and keeping the best one is
  exactly how people fool themselves.
* Minimum Track Record Length: how many years of this you'd need before PSR reaches 95%.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from . import config

EULER_GAMMA = 0.5772156649015329
TD = config.TRADING_DAYS


def daily_rate(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1 / TD) - 1


# The standard numbers


def annualised_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    return float((1 + returns).prod() ** (TD / len(returns)) - 1)


def annualised_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * math.sqrt(TD))


def sharpe(returns: pd.Series, rf: float = config.RISK_FREE) -> float:
    excess = returns - daily_rate(rf)
    sd = excess.std(ddof=1)
    return float(excess.mean() / sd * math.sqrt(TD)) if sd > 0 else float("nan")


def sortino(returns: pd.Series, rf: float = config.RISK_FREE) -> float:
    """Like Sharpe, but only downside volatility counts as risk."""
    excess = returns - daily_rate(rf)
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return float(excess.mean() / downside * math.sqrt(TD)) if downside > 0 else float("nan")


def wealth(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    w = wealth(returns)
    return (w / w.cummax() - 1).rename("drawdown")


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min()) if len(returns) else float("nan")


def calmar(returns: pd.Series) -> float:
    worst = max_drawdown(returns)
    return annualised_return(returns) / abs(worst) if worst < 0 else float("nan")


def omega(returns: pd.Series, threshold_annual: float = 0.0) -> float:
    threshold = daily_rate(threshold_annual)
    gains = (returns - threshold).clip(lower=0).sum()
    losses = (threshold - returns).clip(lower=0).sum()
    return float(gains / losses) if losses > 0 else float("inf")


def drawdown_episodes(returns: pd.Series, top: int = 10) -> pd.DataFrame:
    dd = drawdown_series(returns)
    days = list(dd.index)
    episodes = []
    peak = None
    previous_day = days[0] if days else None
    for day, depth in dd.items():
        if peak is None and depth < 0:
            peak = previous_day
        elif peak is not None and depth == 0:
            episodes.append((peak, dd.loc[peak:day].idxmin(), day))
            peak = None
        previous_day = day
    if peak is not None:  # still under water at the end
        episodes.append((peak, dd.loc[peak:].idxmin(), None))

    rows = []
    for peak, trough, recovered in episodes:
        end = days.index(recovered) if recovered is not None else len(days) - 1
        rows.append({
            "peak": peak.date(),
            "trough": trough.date(),
            "recovered": recovered.date() if recovered is not None else "not yet",
            "depth": dd[trough],
            "days_to_trough": days.index(trough) - days.index(peak),
            "days_to_recover": days.index(recovered) - days.index(trough) if recovered is not None else np.nan,
            "total_days": end - days.index(peak),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("depth").head(top).reset_index(drop=True)


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    monthly = (1 + returns).resample("ME").prod() - 1
    grid = (pd.DataFrame({"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values})
            .pivot(index="year", columns="month", values="ret"))
    grid.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in grid.columns]
    grid["Year"] = (1 + returns).groupby(returns.index.year).prod() - 1
    return grid


# Could it be luck?


def _moments(returns: pd.Series) -> tuple[float, float, int]:
    """Skew, (non-excess) kurtosis and sample size."""
    x = returns.dropna().to_numpy()
    return stats.skew(x, bias=False), stats.kurtosis(x, fisher=False, bias=False), len(x)


def per_period_sharpe(returns: pd.Series, rf: float = config.RISK_FREE) -> float:
    """Daily, not annualised. The PSR formulas are written in per-period terms."""
    excess = returns - daily_rate(rf)
    return float(excess.mean() / excess.std(ddof=1))


def sharpe_std_error(sr: float, skew: float, kurt: float, n: int) -> float:
    """Standard error of a Sharpe estimate. Negative skew and fat tails make it wider."""
    return math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12) / (n - 1))


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0,
                         rf: float = config.RISK_FREE) -> float:
    """Probability that the true (per-period) Sharpe beats ``sr_benchmark``."""
    sr = per_period_sharpe(returns, rf)
    skew, kurt, n = _moments(returns)
    return float(stats.norm.cdf((sr - sr_benchmark) / sharpe_std_error(sr, skew, kurt, n)))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected best per-period Sharpe among N strategies that have no real edge."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: pd.Series, n_trials: int = 1, trial_sharpes=None,
                    rf: float = config.RISK_FREE) -> dict:
    """PSR measured against the luckiest of ``n_trials`` skill-less strategies.

    If you don't have the Sharpe ratios of the variants you tried, their spread is taken to
    be the sampling error of this Sharpe estimate, which is a reasonable neutral default.
    """
    sr = per_period_sharpe(returns, rf)
    skew, kurt, n = _moments(returns)
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var_sr = float(np.var(trial_sharpes, ddof=1))
    else:
        var_sr = sharpe_std_error(sr, skew, kurt, n) ** 2
    hurdle = expected_max_sharpe(n_trials, var_sr)
    return {
        "dsr": probabilistic_sharpe(returns, hurdle, rf),
        "sr0_annual": hurdle * math.sqrt(TD),
        "n_trials": n_trials,
    }


def min_track_record(returns: pd.Series, sr_benchmark: float = 0.0, prob: float = 0.95,
                     rf: float = config.RISK_FREE) -> float:
    """Years of history needed for PSR to reach ``prob``. Infinite if the Sharpe is too low."""
    sr = per_period_sharpe(returns, rf)
    if sr <= sr_benchmark:
        return float("inf")
    skew, kurt, _ = _moments(returns)
    z = stats.norm.ppf(prob)
    n = 1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - sr_benchmark)) ** 2
    return float(n / TD)


def stationary_bootstrap_indices(n: int, n_samples: int, length: int | None = None,
                                 block: float = config.BOOTSTRAP_BLOCK, rng=None) -> np.ndarray:
    """Resampling indices for the stationary bootstrap (Politis and Romano, 1994).

    Instead of shuffling single days, which would destroy volatility clustering, it copies
    runs of consecutive days. Run lengths are geometric with mean ``block``, and the sample
    wraps around at the end.
    """
    rng = np.random.default_rng(config.SEED) if rng is None else rng
    length = n if length is None else length
    idx = np.empty((n_samples, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_samples)
    start_new_block = rng.random((n_samples, length)) < 1.0 / block
    fresh_starts = rng.integers(0, n, (n_samples, length))
    for t in range(1, length):
        idx[:, t] = np.where(start_new_block[:, t], fresh_starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_sharpe_ci(returns: pd.Series, rf: float = config.RISK_FREE, level: float = 0.95,
                        n_samples: int = config.BOOTSTRAP_SAMPLES) -> tuple[float, float]:
    excess = (returns - daily_rate(rf)).to_numpy()
    samples = excess[stationary_bootstrap_indices(len(excess), n_samples)]
    sharpes = samples.mean(axis=1) / samples.std(axis=1, ddof=1) * math.sqrt(TD)
    tail = (1 - level) / 2
    return float(np.quantile(sharpes, tail)), float(np.quantile(sharpes, 1 - tail))


# Money-weighted return


def xirr(dates, amounts) -> float:
    """Annual IRR of irregular cash flows, from the investor's side (deposits negative)."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    amounts = np.asarray(amounts, dtype=float)
    years = (dates - dates.iloc[0]).dt.days.to_numpy() / 365.0

    def npv(rate):
        return np.sum(amounts / (1 + rate) ** years)

    try:
        return float(brentq(npv, -0.9999, 100.0, maxiter=500))
    except ValueError:
        return float("nan")


def portfolio_xirr(nav: pd.Series, flows: pd.Series) -> float:
    real_flows = flows[flows != 0]
    dates = [*real_flows.index, nav.index[-1]]
    amounts = [*(-real_flows.values), nav.iloc[-1]]
    return xirr(dates, amounts)


def summary(returns: pd.Series, bench: pd.Series | None = None, rf: float = config.RISK_FREE,
            n_trials: int = 1) -> dict:
    skew, kurt, n = _moments(returns)
    deflated = deflated_sharpe(returns, n_trials, rf=rf)
    ci_low, ci_high = bootstrap_sharpe_ci(returns, rf)
    out = {
        "start": returns.index[0].date(),
        "end": returns.index[-1].date(),
        "days": n,
        "years": n / TD,
        "total_return": float((1 + returns).prod() - 1),
        "cagr": annualised_return(returns),
        "vol": annualised_vol(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "calmar": calmar(returns),
        "omega": omega(returns),
        "max_drawdown": max_drawdown(returns),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "hit_rate": float((returns > 0).mean()),
        "psr": probabilistic_sharpe(returns, 0.0, rf),
        "dsr": deflated["dsr"],
        "dsr_benchmark_sharpe": float(deflated["sr0_annual"]),
        "n_trials": n_trials,
        "min_track_record_years": min_track_record(returns, 0.0, 0.95, rf),
        "sharpe_ci_low": ci_low,
        "sharpe_ci_high": ci_high,
    }
    if bench is not None:
        b = bench.reindex(returns.index).dropna()
        out.update({
            "bench_cagr": annualised_return(b),
            "bench_vol": annualised_vol(b),
            "bench_sharpe": sharpe(b, rf),
            "bench_max_drawdown": max_drawdown(b),
        })
    return out
