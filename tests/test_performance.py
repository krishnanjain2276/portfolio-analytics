import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from portfolio_analytics import performance as perf


def daily_returns(mean=0.0008, sd=0.01, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n), index=pd.bdate_range("2022-01-03", periods=n))


def test_psr_rises_with_sharpe():
    weaker, stronger = daily_returns(0.0008), daily_returns(0.0016)
    assert perf.probabilistic_sharpe(stronger, rf=0) > perf.probabilistic_sharpe(weaker, rf=0) > 0.5


def test_deflated_sharpe_penalises_trying_more_strategies():
    returns = daily_returns()
    one_try = perf.deflated_sharpe(returns, 1, rf=0)["dsr"]
    fifty_tries = perf.deflated_sharpe(returns, 50, rf=0)["dsr"]
    assert one_try == pytest.approx(perf.probabilistic_sharpe(returns, 0, rf=0))
    assert fifty_tries < one_try


def test_min_track_record_is_exactly_long_enough():
    returns = daily_returns(0.0008, 0.01, 1500)
    years = perf.min_track_record(returns, 0.0, 0.95, rf=0)
    sr = perf.per_period_sharpe(returns, rf=0)
    skew, kurt, _ = perf._moments(returns)
    n = years * 252
    z = sr * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    assert stats.norm.cdf(z) == pytest.approx(0.95, abs=1e-6)


def test_xirr_of_a_simple_ten_percent_year():
    assert perf.xirr(["2025-01-01", "2026-01-01"], [-100, 110]) == pytest.approx(0.10, abs=1e-6)


def test_max_drawdown_on_a_known_path():
    returns = pd.Series([0.10, -0.20, 0.05, 0.30], index=pd.bdate_range("2026-01-01", periods=4))
    assert perf.max_drawdown(returns) == pytest.approx(-0.20)


def test_drawdown_episode_dates():
    returns = pd.Series([0.0, 0.10, -0.10, -0.05, 0.20, 0.0], index=pd.bdate_range("2026-01-05", periods=6))
    episode = perf.drawdown_episodes(returns).iloc[0]
    assert str(episode.peak) == "2026-01-06" and str(episode.trough) == "2026-01-08"
    assert str(episode.recovered) == "2026-01-09"


def test_stationary_bootstrap_has_the_right_block_length():
    idx = perf.stationary_bootstrap_indices(500, 200, 400, block=10, rng=np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 500
    share_of_new_blocks = (np.diff(idx, axis=1) % 500 != 1).mean()
    assert 1 / share_of_new_blocks == pytest.approx(10, rel=0.15)
