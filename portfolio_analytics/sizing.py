"""How big should positions be? Four standard answers.

* Half-Kelly: the growth-optimal leverage, halved because the inputs are estimates.
* Volatility targeting: scale everything so forecast volatility hits a target.
* Fixed-fractional: lots such that a stop-out costs a fixed share of NAV.
* Risk parity: stock weights where each stock contributes the same amount of risk.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .risk import ewma_vol

TD = config.TRADING_DAYS


def kelly_leverage(returns: pd.Series, rf: float = config.RISK_FREE,
                   fraction: float = config.KELLY_FRACTION) -> float:
    """For a continuous strategy, full Kelly leverage is (mu - rf) / sigma^2."""
    mu = returns.mean() * TD
    variance = returns.var(ddof=1) * TD
    return float(fraction * (mu - rf) / variance) if variance > 0 else float("nan")


def vol_target_leverage(returns: pd.Series, target: float = config.VOL_TARGET) -> float:
    """Multiply every position by this to bring forecast volatility to ``target``."""
    forecast = ewma_vol(returns).iloc[-1] * math.sqrt(TD)
    return float(target / forecast) if forecast > 0 else float("nan")


def fixed_fractional_lots(nav: float, stop_distance: float, lot_size: float,
                          risk_fraction: float = config.RISK_PER_TRADE) -> int:
    """Lots such that a stop ``stop_distance`` points away loses at most risk_fraction x NAV.

    Always rounds down: you can't trade a fraction of a lot, and rounding up would break
    the risk limit.
    """
    if stop_distance <= 0 or lot_size <= 0:
        return 0
    return int(math.floor(nav * risk_fraction / (stop_distance * lot_size)))


def risk_parity_weights(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 10_000) -> np.ndarray:
    """Long-only equal-risk-contribution weights, by cyclical coordinate descent.

    Each pass solves, for one weight at a time, the quadratic
        cov_ii w_i^2 + s_i w_i - b_i sigma = 0     with s_i = sum_{j != i} cov_ij w_j
    and keeps the positive root. It converges in a handful of passes for a few dozen stocks.
    """
    cov = np.asarray(cov, float)
    n = cov.shape[0]
    budget = np.ones(n) / n
    weights = 1 / np.sqrt(np.diag(cov))  # inverse-vol is a good starting point
    weights /= weights.sum()
    for _ in range(max_iter):
        previous = weights.copy()
        for i in range(n):
            others = cov[i] @ weights - cov[i, i] * weights[i]
            sigma = math.sqrt(weights @ cov @ weights)
            weights[i] = (-others + math.sqrt(others**2 + 4 * cov[i, i] * budget[i] * sigma)) / (2 * cov[i, i])
        if np.max(np.abs(weights - previous)) < tol:
            break
    return weights / weights.sum()


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sigma = math.sqrt(weights @ cov @ weights)
    return weights * (cov @ weights) / sigma


def sizing_table(returns: pd.Series, nav: float, current_gross_leverage: float,
                 prices: pd.DataFrame, equities: list[str], rf: float = config.RISK_FREE) -> dict:
    out = {
        "half_kelly_leverage": kelly_leverage(returns, rf),
        "vol_target": config.VOL_TARGET,
        "ewma_vol_forecast": ewma_vol(returns).iloc[-1] * math.sqrt(TD),
        "realised_vol_3m": returns.tail(63).std(ddof=1) * math.sqrt(TD),
        "vol_target_scale": vol_target_leverage(returns),
        "current_gross_leverage": current_gross_leverage,
    }
    if len(equities) >= 2:
        rets = prices[equities].pct_change(fill_method=None).tail(config.VAR_WINDOW).dropna()
        if len(rets) > 30:
            cov = rets.cov().to_numpy()
            weights = risk_parity_weights(cov)
            share = risk_contributions(weights, cov) / math.sqrt(weights @ cov @ weights)
            out["risk_parity"] = pd.DataFrame({"weight": weights, "risk_contribution": share},
                                              index=equities)
    return out
