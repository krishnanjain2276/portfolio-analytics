import datetime as dt

from portfolio_analytics.charges import estimate_fees
from portfolio_analytics.instruments import infer_instrument, lot_size, monthly_expiry, option_symbol, parse_symbol


def test_monthly_future_symbol():
    future = parse_symbol("NIFTY26OCTFUT")
    assert future.asset_class == "future"
    assert future.underlying == "NIFTY50"
    assert future.expiry.date() == dt.date(2026, 10, 27)
    assert future.multiplier == 65


def test_monthly_and_weekly_option_symbols():
    monthly = parse_symbol("NIFTY26NOV25000CE")
    assert (monthly.option_type, monthly.strike, monthly.expiry.date()) == ("C", 25000.0, dt.date(2026, 11, 24))
    weekly = parse_symbol("NIFTY26O2025000PE")
    assert (weekly.option_type, weekly.strike, weekly.expiry.date()) == ("P", 25000.0, dt.date(2026, 10, 20))


def test_a_stock_is_not_a_derivative_symbol():
    assert parse_symbol("RELIANCE") is None


def test_symbols_round_trip():
    assert option_symbol("NIFTY50", dt.date(2026, 11, 24), 25000, "C") == "NIFTY26NOV25000CE"


def test_expiry_moved_from_thursday_to_tuesday():
    assert monthly_expiry(2025, 6) == dt.date(2025, 6, 26)  # last Thursday
    assert monthly_expiry(2025, 9) == dt.date(2025, 9, 30)  # last Tuesday


def test_lot_size_follows_the_nse_changes():
    assert lot_size("NIFTY50", dt.date(2024, 6, 27)) == 25
    assert lot_size("NIFTY50", dt.date(2025, 6, 26)) == 75
    assert lot_size("NIFTY50", dt.date(2026, 3, 31)) == 65


def test_messy_symbols_are_cleaned_and_get_a_sector():
    stock = infer_instrument("hdfcbank.ns")
    assert (stock.symbol, stock.asset_class, stock.sector) == ("HDFCBANK", "equity", "Financials")
    assert infer_instrument("Nifty 50").symbol == "NIFTY50"


def test_sell_side_stt_went_up_in_april_2026():
    call = parse_symbol("NIFTY26NOV25000CE")
    sell_before = estimate_fees(call, -1, 100, "2026-03-15")
    sell_after = estimate_fees(call, -1, 100, "2026-04-15")
    buy_after = estimate_fees(call, 1, 100, "2026-04-15")
    assert sell_after > sell_before > 0
    assert sell_after > buy_after  # STT is sell-side only on options
