import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.instruments import Instrument
from portfolio_analytics.portfolio import run_ledger, time_weighted_returns


def make_trades(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "quantity", "price", "fees"])


def test_nav_reconciles_on_every_day(demo):
    ledger = run_ledger(demo.instruments, demo.prices, demo.trades, demo.cashflows)
    money_in = ledger.flows.cumsum()
    assert np.allclose(ledger.nav, money_in + ledger.pnl.sum(axis=1), atol=1e-6)
    assert np.allclose(ledger.nav, ledger.cash + ledger.market_value.sum(axis=1), atol=1e-6)


def test_futures_are_marked_to_market(tiny):
    instruments, prices, cash, days = tiny
    trades = make_trades([(days[1], "NIFTY26JANFUT", 1, 101.0, 5.0),
                          (days[3], "NIFTY26JANFUT", -1, 103.8, 5.0)])
    ledger = run_ledger(instruments, prices, trades, cash)
    # 10 units x (103.8 - 101) less 10 of fees, and a future never carries market value
    assert ledger.pnl["NIFTY26JANFUT"].iloc[-1] == pytest.approx(10 * 2.8 - 10.0)
    assert (ledger.market_value["NIFTY26JANFUT"] == 0).all()
    assert ledger.nav.iloc[-1] == pytest.approx(10_000 + 18.0)


def test_options_cash_settle_at_expiry(tiny):
    instruments, prices, cash, days = tiny
    ledger = run_ledger(instruments, prices, make_trades([(days[2], "NIFTY26JAN104CE", 1, 1.5, 0.0)]), cash)
    spot_at_expiry = prices.loc[days[7], "NIFTY50"]
    assert ledger.units["NIFTY26JAN104CE"].iloc[-1] == 0
    assert ledger.pnl["NIFTY26JAN104CE"].iloc[-1] == pytest.approx(10 * (max(spot_at_expiry - 104, 0) - 1.5))


def test_stock_round_trip(tiny):
    instruments, prices, cash, days = tiny
    trades = make_trades([(days[0], "ABC", 10, 50.0, 1.0), (days[4], "ABC", -10, 54.0, 1.0)])
    ledger = run_ledger(instruments, prices, trades, cash)
    assert ledger.pnl["ABC"].iloc[-1] == pytest.approx(10 * 4 - 2)
    assert ledger.units["ABC"].iloc[-1] == 0


def test_average_cost_survives_a_partial_sale(tiny):
    instruments, prices, cash, days = tiny
    trades = make_trades([(days[0], "ABC", 10, 50.0, 0.0), (days[1], "ABC", 10, 52.0, 0.0),
                          (days[2], "ABC", -5, 53.0, 0.0)])
    ledger = run_ledger(instruments, prices, trades, cash)
    assert ledger.avg_cost["ABC"] == pytest.approx(51.0)


def test_idle_cash_earns_nothing_whatever_the_flows(tiny):
    instruments, prices, _, days = tiny
    cash = pd.DataFrame({"date": [days[0], days[3], days[6]], "amount": [10_000.0, 5_000.0, -2_000.0]})
    ledger = run_ledger(instruments, prices, make_trades([]), cash)
    assert np.allclose(time_weighted_returns(ledger.nav, ledger.flows), 0.0)


def test_returns_do_not_depend_on_portfolio_size(demo):
    small = run_ledger(demo.instruments, demo.prices, demo.trades, demo.cashflows)
    trades, cashflows = demo.trades.copy(), demo.cashflows.copy()
    trades[["quantity", "fees"]] *= 2
    cashflows["amount"] *= 2
    double = run_ledger(demo.instruments, demo.prices, trades, cashflows)
    assert np.allclose(time_weighted_returns(small.nav, small.flows),
                       time_weighted_returns(double.nav, double.flows), atol=1e-12)


def test_contract_expiring_after_the_data_stays_open(tiny):
    instruments, prices, cash, days = tiny
    march = Instrument("NIFTY26MARFUT", "future", 10, "NIFTY50", expiry=pd.Timestamp("2026-03-31"))
    instruments = {**instruments, march.symbol: march}
    prices = prices.assign(NIFTY26MARFUT=prices["NIFTY50"] + 1)
    ledger = run_ledger(instruments, prices, make_trades([(days[2], "NIFTY26MARFUT", 1, 103.0, 0.0)]), cash)
    assert ledger.units["NIFTY26MARFUT"].iloc[-1] == 10


def test_weekend_trade_is_booked_on_monday(tiny):
    instruments, prices, cash, days = tiny
    saturday = pd.Timestamp("2026-01-10")
    ledger = run_ledger(instruments, prices, make_trades([(saturday, "ABC", 1, 55.0, 0.0)]), cash)
    assert ledger.trades["date"].iloc[0] == pd.Timestamp("2026-01-12")
    assert any("non-trading" in note for note in ledger.notes)


def test_trade_after_expiry_is_rejected(tiny):
    instruments, prices, cash, days = tiny
    with pytest.raises(ValueError, match="after its expiry"):
        run_ledger(instruments, prices, make_trades([(days[9], "NIFTY26JAN104CE", 1, 1.0, 0.0)]), cash)
