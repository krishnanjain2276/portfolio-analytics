"""Runs every analysis in order and collects the results in one dictionary."""

from __future__ import annotations

import warnings

import pandas as pd

from . import attribution, config, montecarlo, performance, risk, signals, sizing, verdicts
from .data import PortfolioData
from .portfolio import open_positions, run_ledger, time_weighted_returns


def run_all(data: PortfolioData, rf: float = config.RISK_FREE, n_trials: int = 1,
            benchmark: str = config.BENCHMARK, mc_paths: int = config.MC_PATHS) -> dict:
    """Ledger -> returns -> performance and risk -> live book -> verdicts."""
    ledger = run_ledger(data.instruments, data.prices, data.trades, data.cashflows)
    nav = ledger.nav
    returns = time_weighted_returns(nav, ledger.flows)
    if len(returns) < 30:
        raise ValueError("Need at least 30 days of history to say anything useful.")

    benchmark_closes = data.prices[benchmark].reindex(nav.index).ffill()
    benchmark_returns = benchmark_closes.pct_change(fill_method=None).reindex(returns.index).fillna(0.0)
    today = nav.index[-1]
    nav_today = float(nav.iloc[-1])

    results: dict = {
        "rf": rf, "benchmark": benchmark, "as_of": today.date(), "nav_last": nav_today,
        "notes": [*data.notes, *ledger.notes], "modelled": data.modelled,
        "ledger": ledger, "nav": nav, "returns": returns, "bench_returns": benchmark_returns,
    }

    # How did it do?
    summary = performance.summary(returns, benchmark_returns, rf, n_trials)
    summary["xirr"] = performance.portfolio_xirr(nav, ledger.flows)
    summary["net_capital"] = float(ledger.flows.sum())
    summary["total_pnl"] = nav_today - summary["net_capital"]
    results.update(
        summary=summary,
        drawdowns=performance.drawdown_episodes(returns),
        monthly=performance.monthly_table(returns),
        capm=risk.capm(returns, benchmark_returns, rf),
        rolling_beta=risk.rolling_beta(returns, benchmark_returns),
        var=risk.var_table(returns, nav_today),
        kupiec={level: risk.kupiec_test(returns, level) for level in config.VAR_LEVELS},
    )

    # What is held now, and what could hurt it?
    positions = open_positions(ledger, data.instruments, data.prices)
    book = risk.live_book(ledger.units.iloc[-1], data.instruments, data.prices, today, rf, bench=benchmark)
    euler, sigma = risk.euler_decomposition(book, data.prices, benchmark)
    euler_sleeves = pd.DataFrame()
    if len(euler):
        euler_sleeves = (euler.groupby("sleeve")[["exposure", "vol_contribution_daily", "share_of_risk"]]
                         .sum().sort_values("vol_contribution_daily", ascending=False))
    results.update(
        positions=positions, book=book, greeks=risk.book_greeks_summary(book, nav_today),
        euler=euler, euler_sigma_daily=sigma, euler_sleeves=euler_sleeves,
        stress_grid=risk.stress_grid(book, nav_today, rf=rf),
        scenarios=risk.scenario_table(book, nav_today, rf=rf),
        montecarlo=montecarlo.simulate(returns, n_paths=mc_paths),
    )

    gross_exposure = float(book["own_exposure"].abs().sum()) if len(book) else 0.0
    held_stocks = [s for s in book.symbol if data.instruments[s].asset_class == "equity"] if len(book) else []
    results["sizing"] = sizing.sizing_table(returns, nav_today, gross_exposure / nav_today, data.prices,
                                            held_stocks, rf)

    # Where did the money come from?
    results.update(
        attr_sleeve=attribution.pnl_by_sleeve(ledger.pnl, data.instruments),
        attr_instrument=attribution.pnl_by_instrument(ledger.pnl, data.instruments),
        attr_monthly=attribution.monthly_pnl_by_class(ledger.pnl, data.instruments),
        trading=attribution.trading_stats(ledger, nav),
    )

    # And what should be done about it?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # short price histories give harmless NaN warnings
        results["index_view"] = signals.index_view(data, benchmark, rf, nav_today)
        results["signals"] = signals.stock_signals(data, positions, nav_today, benchmark)
        results["position_verdicts"] = verdicts.position_verdicts(results, data)
        results["section_verdicts"] = verdicts.section_verdicts(results, data)
        results["overall"] = verdicts.overall_verdict(results["section_verdicts"])
    return results
