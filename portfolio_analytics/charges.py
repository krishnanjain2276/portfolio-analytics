"""Estimated Indian transaction charges.

Used only when a trade's fee is left blank. It adds up brokerage, STT, exchange charges, the
SEBI fee, stamp duty and GST at the rates in ``config.CHARGES``. A discount broker's contract
note should come out within a few rupees of this, but the contract note is always the truth.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .instruments import Instrument


def _fno_stt_rates(trade_date) -> tuple[float, float]:
    day = pd.Timestamp(trade_date).date()
    in_force = [row for row in config.CHARGES["stt_fno_schedule"] if day >= row[0]]
    _, futures_rate, options_rate = in_force[-1]
    return futures_rate, options_rate


def estimate_fees(instrument: Instrument, quantity: float, price: float, trade_date,
                  intraday: bool = False) -> float:
    """Total charges in rupees for one trade.

    ``quantity`` is signed: shares for stocks, lots for F&O, positive for a buy.
    """
    c = config.CHARGES
    is_buy = quantity > 0
    turnover = abs(quantity) * instrument.multiplier * price  # premium turnover for options
    if turnover == 0:
        return 0.0

    if instrument.asset_class in ("equity", "index"):
        if intraday:
            brokerage = min(c["brokerage_flat_per_order"], c["brokerage_pct_cap"] * turnover)
            stt = 0.0 if is_buy else 0.00025 * turnover
            stamp = 0.00003 * turnover if is_buy else 0.0
        else:
            brokerage = c["brokerage_delivery"]
            stt = c["stt_equity_delivery"] * turnover
            stamp = c["stamp_equity_delivery"] * turnover if is_buy else 0.0
        exchange = c["exchange_equity"] * turnover

    elif instrument.asset_class == "future":
        futures_stt, _ = _fno_stt_rates(trade_date)
        brokerage = min(c["brokerage_flat_per_order"], c["brokerage_pct_cap"] * turnover)
        stt = 0.0 if is_buy else futures_stt * turnover
        exchange = c["exchange_futures"] * turnover
        stamp = c["stamp_futures"] * turnover if is_buy else 0.0

    else:  # option
        _, options_stt = _fno_stt_rates(trade_date)
        brokerage = c["brokerage_flat_per_order"]
        stt = 0.0 if is_buy else options_stt * turnover
        exchange = c["exchange_options"] * turnover
        stamp = c["stamp_options"] * turnover if is_buy else 0.0

    sebi = c["sebi_fee"] * turnover
    gst = c["gst"] * (brokerage + exchange + sebi)
    return round(brokerage + stt + exchange + sebi + stamp + gst, 2)
