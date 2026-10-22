"""The ledger: replays every trade and cash flow and rebuilds the portfolio day by day.

Everything else in the package reads from what this produces, so the accounting rules are
spelled out here:

* Deposits and withdrawals hit cash at the start of the day they are dated.
* Stocks and options: buying pays ``units x price + fees`` out of cash, and the position is
  then carried at ``units x close``.
* Futures: no premium changes hands. Every day the change in value,
  ``units x (close - previous mark)``, is swept into cash, just like exchange daily settlement.
  So a future always has zero market value and shows up only through its P&L. On the day you
  trade, the P&L is measured from your trade price rather than yesterday's close.
* Options and futures cash-settle on their last trading day on or before expiry, against the
  close of the underlying.

Each instrument's running P&L is its market value plus every rupee of cash it has generated.
That gives an identity the tests check on every single day:

    NAV = money put in - money taken out + sum of every instrument's P&L
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .instruments import Instrument


@dataclass
class LedgerResult:
    nav: pd.Series
    cash: pd.Series
    flows: pd.Series  # external flows: deposits +, withdrawals -
    units: pd.DataFrame  # units held (shares, or lots x lot size)
    market_value: pd.DataFrame
    pnl: pd.DataFrame  # cumulative P&L per instrument
    fees: pd.Series  # fees paid each day
    trades: pd.DataFrame  # the blotter with a few useful columns added
    avg_cost: dict = field(default_factory=dict)
    opened: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def capital(self) -> pd.Series:
        return self.flows.cumsum()


@dataclass
class _Position:
    """Running state for one instrument while the ledger is being replayed."""

    instrument: Instrument
    units: float = 0.0
    cash: float = 0.0  # every rupee this instrument has paid or received so far
    last_mark: float = float("nan")  # futures only: the price P&L was last settled at
    avg_cost: float = 0.0
    opened: pd.Timestamp | None = None

    def record_fill(self, units: float, price: float, day: pd.Timestamp) -> None:
        """Keep the average entry price up to date.

        Adding to a position blends the price in. Trimming it leaves the average alone.
        Flipping from long to short (or back) starts afresh at the trade price.
        """
        before, after = self.units, self.units + units
        if before == 0 or np.sign(before) == np.sign(units):
            self.avg_cost = (self.avg_cost * abs(before) + price * abs(units)) / abs(after) if after else 0.0
            if before == 0:
                self.opened = day
        elif after == 0:
            self.avg_cost, self.opened = 0.0, None
        elif np.sign(after) != np.sign(before):
            self.avg_cost, self.opened = price, day
        self.units = after

    def close_out(self) -> None:
        self.units, self.avg_cost, self.opened = 0.0, 0.0, None


def _move_to_trading_days(dates: pd.Series, calendar: pd.DatetimeIndex, what: str,
                          notes: list) -> pd.Series:
    """A trade dated on a Saturday is booked on Monday (the next day with prices)."""
    positions = calendar.searchsorted(dates.values, side="left")
    too_late = positions >= len(calendar)
    if too_late.any():
        first_bad = dates[too_late].iloc[0]
        raise ValueError(f"{what} dated {first_bad.date()} is after the last price date "
                         f"{calendar[-1].date()}. Add prices up to that date.")
    moved_dates = pd.Series(calendar[positions], index=dates.index)
    moved = moved_dates != dates
    if moved.any():
        notes.append(f"{int(moved.sum())} {what} dated on non-trading days were moved to the "
                     f"next trading day (first: {dates[moved].iloc[0].date()}).")
    return moved_dates


def _settlement_day(instrument: Instrument, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Last trading day on or before expiry, or None if it is still live at the end."""
    if instrument.expiry > calendar[-1]:
        return None
    position = calendar.searchsorted(instrument.expiry, side="right") - 1
    return calendar[position] if position >= 0 else None


def run_ledger(instruments: dict[str, Instrument], prices: pd.DataFrame, trades: pd.DataFrame,
               cashflows: pd.DataFrame) -> LedgerResult:
    if cashflows.empty:
        raise ValueError("cashflows is empty: the first row must be your starting capital.")

    notes: list[str] = []
    prices = prices.sort_index()
    start = cashflows["date"].min()
    if len(trades):
        start = min(start, trades["date"].min())
    calendar = prices.index[prices.index.searchsorted(start):]
    if len(calendar) == 0:
        raise ValueError("There are no prices on or after your first cash flow or trade.")
    closes = prices.loc[calendar].ffill()

    trades, cashflows = trades.copy(), cashflows.copy()
    if len(trades):
        trades["date"] = _move_to_trading_days(trades["date"], calendar, "trades", notes)
    cashflows["date"] = _move_to_trading_days(cashflows["date"], calendar, "cash flows", notes)

    symbols = sorted(set(trades["symbol"]))
    for symbol in symbols:
        if symbol not in instruments:
            raise ValueError(f"There is a trade in {symbol} but it is not in instruments.")
        if symbol not in closes.columns:
            raise ValueError(f"There is a trade in {symbol} but no price column for it.")

    settles_on = {}
    for symbol in symbols:
        instrument = instruments[symbol]
        if not instrument.is_derivative:
            continue
        settles_on[symbol] = _settlement_day(instrument, calendar)
        late = trades[(trades.symbol == symbol) & (trades.date > instrument.expiry)]
        if len(late):
            raise ValueError(f"The trade in {symbol} on {late.date.iloc[0].date()} is after its "
                             f"expiry ({instrument.expiry.date()}).")

    flows = cashflows.groupby("date")["amount"].sum().reindex(calendar, fill_value=0.0)
    trades_by_day = dict(tuple(trades.groupby("date"))) if len(trades) else {}
    book = {symbol: _Position(instruments[symbol]) for symbol in symbols}
    external_cash = 0.0

    history = {"units": [], "market_value": [], "pnl": [], "cash": [], "fees": []}
    blotter = []

    def price_on(day, symbol, why):
        value = closes.at[day, symbol]
        if np.isnan(value):
            raise ValueError(f"No price for {symbol} on {day.date()} ({why}).")
        return value

    for day in calendar:
        external_cash += flows[day]
        fees_today = 0.0

        # 1. Settle yesterday's futures positions to today's close.
        for symbol, pos in book.items():
            if pos.instrument.asset_class == "future" and pos.units:
                close = price_on(day, symbol, "a position is open")
                pos.cash += pos.units * (close - pos.last_mark)
                pos.last_mark = close

        # 2. Book today's trades.
        if day in trades_by_day:
            for trade in trades_by_day[day].to_dict("records"):
                pos = book[trade["symbol"]]
                instrument = pos.instrument
                units = float(trade["quantity"]) * instrument.multiplier
                price, fee = float(trade["price"]), float(trade["fees"])
                fees_today += fee

                if instrument.asset_class == "future":
                    close = price_on(day, trade["symbol"], "trade day")
                    cash_impact = units * (close - price) - fee
                    pos.last_mark = close
                else:
                    cash_impact = -units * price - fee
                pos.cash += cash_impact
                pos.record_fill(units, price, day)

                has_underlying = instrument.underlying in closes.columns
                underlying_price = closes.at[day, instrument.underlying] if has_underlying else price
                blotter.append({
                    **trade, "date": day, "units": units, "asset_class": instrument.asset_class,
                    "sleeve": instrument.sleeve, "notional": abs(units * price),
                    "underlying_notional": abs(units * underlying_price), "cash_impact": cash_impact,
                })

        # 3. Cash-settle anything expiring today against the underlying's close.
        for symbol, pos in book.items():
            if pos.units and settles_on.get(symbol) == day:
                instrument = pos.instrument
                spot = price_on(day, instrument.underlying, f"needed to settle {symbol}")
                if instrument.asset_class == "option":
                    if instrument.option_type == "C":
                        intrinsic = max(spot - instrument.strike, 0)
                    else:
                        intrinsic = max(instrument.strike - spot, 0)
                    pos.cash += pos.units * intrinsic
                else:
                    pos.cash += pos.units * (spot - pos.last_mark)
                pos.close_out()

        # 4. Value everything at today's close.
        market_value = {}
        for symbol, pos in book.items():
            if pos.units == 0 or pos.instrument.asset_class == "future":
                market_value[symbol] = 0.0
            else:
                market_value[symbol] = pos.units * price_on(day, symbol, "a position is open")

        history["units"].append({s: p.units for s, p in book.items()})
        history["market_value"].append(market_value)
        history["pnl"].append({s: market_value[s] + p.cash for s, p in book.items()})
        history["cash"].append(external_cash + sum(p.cash for p in book.values()))
        history["fees"].append(fees_today)

    def frame(name):
        return pd.DataFrame(history[name], index=calendar, columns=symbols, dtype=float)

    market_value = frame("market_value")
    cash = pd.Series(history["cash"], index=calendar, name="cash")
    nav = (cash + market_value.sum(axis=1)).rename("nav")
    if (nav <= 0).any():
        warnings.warn("NAV goes to zero or below on some days; returns there are undefined.",
                      stacklevel=2)

    return LedgerResult(
        nav=nav, cash=cash, flows=flows.rename("flows"), units=frame("units"),
        market_value=market_value, pnl=frame("pnl"),
        fees=pd.Series(history["fees"], index=calendar, name="fees"),
        trades=pd.DataFrame(blotter),
        avg_cost={s: p.avg_cost for s, p in book.items()},
        opened={s: p.opened for s, p in book.items()},
        notes=notes,
    )


def time_weighted_returns(nav: pd.Series, flows: pd.Series) -> pd.Series:
    """Daily returns with deposits and withdrawals taken out.

    r_t = NAV_t / (NAV_{t-1} + F_t) - 1, treating a flow as arriving at the start of the day.
    Without this, depositing money would look like a great day.
    """
    starting_value = nav.shift(1).fillna(0.0) + flows.reindex(nav.index, fill_value=0.0)
    returns = nav / starting_value - 1.0
    returns[starting_value <= 0] = np.nan
    return returns.dropna().rename("twr")


def open_positions(ledger: LedgerResult, instruments: dict, prices: pd.DataFrame) -> pd.DataFrame:
    last_day = ledger.units.index[-1]
    rows = []
    for symbol, units in ledger.units.loc[last_day].items():
        if units == 0:
            continue
        instrument = instruments[symbol]
        last_price = prices[symbol].loc[:last_day].ffill().iloc[-1]
        entry = ledger.avg_cost.get(symbol, np.nan)
        rows.append({
            "symbol": symbol,
            "asset_class": instrument.asset_class,
            "sleeve": instrument.sleeve,
            "units": units,
            "lots": units / instrument.multiplier if instrument.is_derivative else np.nan,
            "avg_cost": entry,
            "last_price": last_price,
            "market_value": ledger.market_value.loc[last_day, symbol],
            "notional": units * last_price,
            "unrealised_pnl": units * (last_price - entry),
            # positive means the position is winning, for longs and shorts alike
            "unrealised_pct": (last_price / entry - 1) * np.sign(units) if entry else np.nan,
            "total_pnl": ledger.pnl.loc[last_day, symbol],
            "opened": ledger.opened.get(symbol),
            "expiry": instrument.expiry,
            "strike": instrument.strike,
            "option_type": instrument.option_type,
            "underlying": instrument.underlying,
        })
    return pd.DataFrame(rows)
