"""What might the next year look like? Simulated by resampling the past.

Rather than assume normal returns, each simulated year is stitched together from blocks of
real history (the stationary bootstrap), so fat tails and volatile spells come along for
free. The catch is equally obvious: it can only replay the kinds of days you've already had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .performance import stationary_bootstrap_indices

PERCENTILES = [5, 25, 50, 75, 95]


def simulate(returns: pd.Series, n_paths: int = config.MC_PATHS, horizon: int = config.MC_HORIZON,
             block: float = config.BOOTSTRAP_BLOCK, seed: int = config.SEED) -> dict:
    history = returns.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(history), n_paths, horizon, block, rng)

    paths = np.cumprod(1 + history[idx], axis=1)  # growth of 1 along each path
    final_return = paths[:, -1] - 1
    running_peak = np.maximum.accumulate(np.maximum(paths, 1.0), axis=1)
    worst_drawdown = (paths / running_peak - 1).min(axis=1)

    fan = pd.DataFrame(np.percentile(paths, PERCENTILES, axis=0).T,
                       columns=[f"p{p}" for p in PERCENTILES])
    return {
        "terminal": final_return,
        "max_drawdown": worst_drawdown,
        "fan": fan,
        "prob_loss": float((final_return < 0).mean()),
        "prob_loss_10": float((final_return < -0.10).mean()),
        "prob_dd_20": float((worst_drawdown < -0.20).mean()),
        "terminal_pct": {p: float(np.percentile(final_return, p)) for p in PERCENTILES},
        "mdd_pct": {p: float(np.percentile(worst_drawdown, p)) for p in PERCENTILES},
        "expected_return": float(final_return.mean()),
        "n_paths": n_paths,
        "horizon": horizon,
    }
