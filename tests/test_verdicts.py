import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import signals, verdicts
from portfolio_analytics.instruments import parse_symbol


def trending_prices(daily_drift, n=300, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(daily_drift, 0.008, n)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=pd.bdate_range("2025-01-01", periods=n))


def test_uptrend_scores_positive_and_downtrend_negative():
    up, _ = signals.score(signals.technicals(trending_prices(0.002)))
    down, _ = signals.score(signals.technicals(trending_prices(-0.002)))
    assert -100 <= down < 0 < up <= 100
    assert signals.rating(up) in ("BUY", "STRONG BUY")
    assert signals.rating(down) in ("SELL", "STRONG SELL")


@pytest.mark.parametrize("view, regime, expected", [
    ("BULLISH", "CHEAP", "BUY CALL"), ("BULLISH", "RICH", "BULL PUT SPREAD"),
    ("BEARISH", "FAIR", "BUY PUT"), ("BEARISH", "RICH", "BEAR CALL SPREAD"),
    ("NEUTRAL", "RICH", "IRON CONDOR"), ("NEUTRAL", "CHEAP", "STAY FLAT"),
])
def test_view_and_vol_regime_pick_the_strategy(view, regime, expected):
    idea = signals.build_strategy(view, False, regime, 25000, 0.15, pd.Timestamp("2026-10-20"),
                                  0.065, 0.012, nav=5_000_000, avg_move_pts=150)
    assert idea["strategy"].startswith(expected)
    if len(idea["legs"]) == 2:  # credit spread: you can lose the width minus what you collected
        width = abs(idea["legs"][0]["strike"] - idea["legs"][1]["strike"])
        assert idea["net"] > 0
        assert idea["max_loss_pts"] == pytest.approx(width - idea["net"])
    if idea["legs"]:
        assert idea["lots"] * idea["max_loss_per_lot"] <= 5_000_000 * 0.01


class FakeData:
    def __init__(self, symbols):
        self.instruments = {s: parse_symbol(s) for s in symbols}
        self.prices = pd.DataFrame({"NIFTY50": [25000.0]}, index=[pd.Timestamp("2026-10-20")])


def results_with(positions, view="BULLISH", as_of="2026-10-20"):
    return {"positions": pd.DataFrame(positions), "book": pd.DataFrame(), "signals": pd.DataFrame(),
            "index_view": {"view": view, "technicals": {"avg_move_pts": 100}},
            "as_of": pd.Timestamp(as_of).date()}


def option_position(symbol, lots, pnl_pct, last_price=100.0):
    return {"symbol": symbol, "asset_class": "option", "lots": lots, "units": lots * 65,
            "unrealised_pct": pnl_pct, "unrealised_pnl": 0.0, "last_price": last_price, "underlying": "NIFTY50"}


def verdict_for(position, view="BULLISH", as_of="2026-10-20"):
    table = verdicts.position_verdicts(results_with([position], view, as_of), FakeData([position["symbol"]]))
    return table.verdict.iloc[0]


def test_long_call_up_80_percent_books_profit():
    assert verdict_for(option_position("NIFTY26NOV25500CE", 1, 0.8)) == "BOOK PROFIT"


def test_short_put_with_most_premium_captured_is_bought_back():
    assert verdict_for(option_position("NIFTY26NOV24000PE", -1, 0.75)).startswith("BUY BACK")


def test_long_call_against_a_bearish_view_is_closed():
    assert verdict_for(option_position("NIFTY26NOV25500CE", 1, 0.1), view="BEARISH") == "CLOSE"


def test_future_two_days_from_expiry_with_the_trend_is_rolled():
    future = {"symbol": "NIFTY26OCTFUT", "asset_class": "future", "lots": 1, "units": 65,
              "unrealised_pct": 0.01, "unrealised_pnl": 1000.0, "last_price": 25000, "underlying": "NIFTY50"}
    assert verdict_for(future, as_of="2026-10-25") == "ROLL"


def test_every_analysis_and_position_gets_a_verdict(demo_results):
    areas = {v.area for v in demo_results["section_verdicts"]}
    assert {"Returns", "Statistical validity", "Value at Risk", "Stress test", "Position sizing",
            "NIFTY view & strategy"} <= areas
    assert all(v.tone in ("good", "warn", "bad", "neutral") and v.action for v in demo_results["section_verdicts"])
    position_table = demo_results["position_verdicts"]
    assert len(position_table) == len(demo_results["positions"])
    assert position_table.verdict.notna().all()
    assert demo_results["overall"]["headline"]


def test_reports_are_written(demo_results, tmp_path):
    from portfolio_analytics.excel import to_excel
    from portfolio_analytics.report import to_html

    page = to_html(demo_results, tmp_path / "tearsheet.html")
    workbook = to_excel(demo_results, tmp_path / "tearsheet.xlsx")
    assert "Verdict" in page.read_text()
    assert workbook.stat().st_size > 10_000
    with pd.ExcelFile(workbook) as book:
        assert "Verdicts" in book.sheet_names
