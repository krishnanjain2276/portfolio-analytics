import pandas as pd
import pytest

from portfolio_analytics import inputs
from portfolio_analytics.data import clean_trades, fill_derivative_prices
from portfolio_analytics.tradebook import import_tradebook, parse_dates


def test_template_then_add_then_import(tmp_path):
    inputs.init_folder(tmp_path)
    inputs.add_cash(tmp_path, "2025-01-01", 1_000_000)
    inputs.add_trade(tmp_path, "2025-01-02", "RELIANCE", "BUY", 10, 1250)

    broker_file = tmp_path / "tradebook.csv"
    pd.DataFrame({
        "symbol": ["RELIANCE", "NIFTY26NOV25000CE", "NIFTY26NOV25000CE"],
        "trade_date": ["2025-01-02", "2026-10-12", "2026-10-12"],
        "trade_type": ["buy", "sell", "sell"],
        "quantity": [10, 65, 65],
        "price": [1250, 180, 182],
        "order_id": ["a", "b", "b"],  # one order filled in two pieces
    }).to_csv(broker_file, index=False)

    result = import_tradebook(tmp_path, broker_file)
    assert result["added"] == 1
    assert result["duplicates_skipped"] == 1  # the RELIANCE trade was already there

    trades = pd.read_excel(tmp_path / "input.xlsx", sheet_name="Trades")
    option = trades[trades.Symbol == "NIFTY26NOV25000CE"].iloc[0]
    assert option.Quantity == pytest.approx(2.0)  # 130 units is 2 lots of 65
    assert option.Price == pytest.approx(181.0)  # volume-weighted across the two fills
    assert pd.Timestamp(option.Date) == pd.Timestamp("2026-10-12")


def test_iso_and_indian_dates_both_work():
    dates = pd.Series(["2026-02-10", "10-02-2026", "10/02/2026 09:15"])
    assert list(parse_dates(dates)) == [pd.Timestamp("2026-02-10")] * 3


def test_side_column_and_column_name_variants():
    raw = pd.DataFrame({"Trade Date": ["2026-01-05"], "Symbol": ["infy.ns"], "Type": ["sell"],
                        "Qty": [5], "Price": [1500]})
    trades = clean_trades(raw)
    assert trades.loc[0, "symbol"] == "INFY"
    assert trades.loc[0, "quantity"] == -5
    assert trades.loc[0, "fees_estimated"]


def test_missing_option_closes_are_modelled_from_the_trade(tiny):
    instruments, prices, _, days = tiny
    prices = prices.drop(columns=["NIFTY26JAN104CE"])
    trades = pd.DataFrame({"date": [days[2]], "symbol": ["NIFTY26JAN104CE"], "quantity": [1],
                           "price": [1.5], "fees": [0.0]})
    filled, modelled = fill_derivative_prices(prices, instruments, trades)
    assert modelled == ["NIFTY26JAN104CE"]
    assert filled.loc[days[2]:days[6], "NIFTY26JAN104CE"].notna().all()
    assert filled.loc[days[2], "NIFTY26JAN104CE"] == pytest.approx(1.5, abs=0.05)  # reprices the trade


def test_validate_spots_a_missing_buy(tmp_path, demo):
    inputs.init_folder(tmp_path)
    inputs.add_cash(tmp_path, "2023-01-02", 100_000)
    inputs.add_trade(tmp_path, "2023-02-01", "ITC", "SELL", 10, 300)
    demo.prices[["NIFTY50", "ITC"]].to_csv(tmp_path / "prices.csv")
    messages = inputs.validate(tmp_path)
    assert any(level == "WARN" and "Short stock" in text for level, text in messages)
