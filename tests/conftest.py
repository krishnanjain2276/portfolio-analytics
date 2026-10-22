import numpy as np
import pandas as pd
import pytest

from portfolio_analytics.data import load_folder
from portfolio_analytics.demo import generate_demo
from portfolio_analytics.instruments import Instrument


@pytest.fixture(scope="session")
def demo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo")
    generate_demo(folder)
    return load_folder(folder)


@pytest.fixture(scope="session")
def demo_results(demo):
    from portfolio_analytics.analysis import run_all

    return run_all(demo, n_trials=3, mc_paths=500)


@pytest.fixture
def tiny():
    """Ten business days: the index rising 1 point a day, one stock, one future, one call.

    Small enough to check every number by hand. Both derivatives expire on day 8 (index 7).
    """
    dates = pd.bdate_range("2026-01-05", periods=10)
    index_level = pd.Series(np.linspace(100, 109, 10), dates)
    prices = pd.DataFrame({
        "NIFTY50": index_level,
        "ABC": np.linspace(50, 59, 10),
        "NIFTY26JANFUT": index_level + 0.5,
        "NIFTY26JAN104CE": np.maximum(index_level - 104, 0) + 1.0,
    }, index=dates)
    expiry = dates[7]
    instruments = {
        "NIFTY50": Instrument("NIFTY50", "index"),
        "ABC": Instrument("ABC", "equity", 1, sector="Test"),
        "NIFTY26JANFUT": Instrument("NIFTY26JANFUT", "future", 10, "NIFTY50", expiry=expiry),
        "NIFTY26JAN104CE": Instrument("NIFTY26JAN104CE", "option", 10, "NIFTY50", 104.0, expiry, "C"),
    }
    prices.loc[expiry:, ["NIFTY26JANFUT", "NIFTY26JAN104CE"]] = np.nan
    prices.loc[expiry, "NIFTY26JANFUT"] = index_level[expiry]  # a future converges to spot at expiry
    cashflows = pd.DataFrame({"date": [dates[0]], "amount": [10_000.0]})
    return instruments, prices, cashflows, dates
