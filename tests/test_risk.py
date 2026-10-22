import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import risk, sizing


def test_kupiec_passes_a_correct_model_and_fails_a_wrong_one():
    rng = np.random.default_rng(3)
    calm = pd.Series(rng.standard_normal(3000) * 0.01)
    assert risk.kupiec_test(calm, 0.99)["result"] == "PASS"
    # a quiet first year, then frequent crashes the trailing VaR hasn't seen
    crashes = pd.Series(np.where(rng.random(3000) < 0.05, -0.05, 0.0) + rng.standard_normal(3000) * 0.002)
    regime_change = pd.concat([pd.Series(rng.standard_normal(1000) * 0.002), crashes], ignore_index=True)
    assert risk.kupiec_test(regime_change, 0.99)["result"] == "FAIL"


def test_cornish_fisher_matches_gaussian_on_normal_data():
    returns = pd.Series(np.random.default_rng(5).standard_normal(200_000) * 0.01)
    assert risk.var_cornish_fisher(returns, 0.99) == pytest.approx(risk.var_gaussian(returns, 0.99), rel=0.02)


def test_capm_recovers_beta_and_newey_west_with_no_lags_is_white():
    rng = np.random.default_rng(7)
    market = pd.Series(rng.normal(0.0004, 0.01, 2000))
    portfolio = 0.0001 + 1.3 * market + rng.normal(0, 0.004, 2000)
    assert risk.capm(portfolio, market, rf=0)["beta"] == pytest.approx(1.3, abs=0.03)

    X = np.column_stack([np.ones(2000), market])
    residuals = portfolio.to_numpy() - X @ np.linalg.lstsq(X, portfolio, rcond=None)[0]
    bread = np.linalg.inv(X.T @ X)
    white = bread @ (X * residuals[:, None] ** 2).T @ X @ bread
    assert np.allclose(risk.newey_west_cov(X, residuals, lag=0), white)


def test_euler_contributions_add_up_to_total_volatility(demo_results):
    contributions = demo_results["euler"]["vol_contribution_daily"]
    assert contributions.sum() == pytest.approx(demo_results["euler_sigma_daily"], rel=1e-9)


def test_no_shock_means_no_stress_pnl(demo_results):
    assert demo_results["stress_grid"].loc["+0%", "IV +0%"] == pytest.approx(0.0, abs=1e-12)


def test_a_long_put_gains_in_a_crash():
    book = pd.DataFrame([{"symbol": "P", "asset_class": "option", "units": 65, "price": 100.0, "beta": 1.0,
                          "spot": 25000.0, "strike": 24000.0, "option_type": "P", "T": 30 / 365, "iv": 0.15}])
    assert risk.revalue(book, -0.10, 1.0).iloc[0] > 0
    assert risk.revalue(book, +0.05, -0.2).iloc[0] < 0


def test_risk_parity_gives_equal_contributions():
    rng = np.random.default_rng(11)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T / 5 + np.diag(rng.uniform(0.01, 0.05, 5))
    weights = sizing.risk_parity_weights(cov)
    contributions = sizing.risk_contributions(weights, cov)
    assert weights.sum() == pytest.approx(1.0)
    assert np.allclose(contributions / contributions.sum(), 0.2, atol=1e-6)


def test_fixed_fractional_always_rounds_down():
    # 1% of 25 lakh = 25,000; a 150-point stop on a 65 lot = 9,750 a lot, so 2 lots, not 2.56
    assert sizing.fixed_fractional_lots(2_500_000, 150, 65, 0.01) == 2


def test_half_kelly_formula():
    returns = pd.Series(np.random.default_rng(1).normal(0.0006, 0.01, 5000))
    mu, variance = returns.mean() * 252, returns.var(ddof=1) * 252
    assert sizing.kelly_leverage(returns, rf=0.0, fraction=0.5) == pytest.approx(0.5 * mu / variance)
