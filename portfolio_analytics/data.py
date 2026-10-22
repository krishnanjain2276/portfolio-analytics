"""Reading your data in, cleaning it up, and filling gaps sensibly.

A data folder can hold either one Excel workbook (``input.xlsx``, made by the ``init``
command) or four CSVs, or a mix of the two (the workbook wins). The cleaning functions are
forgiving about the things people actually do: BUY/SELL columns instead of signed
quantities, "Qty" instead of "quantity", ".NS" on the end of symbols, blank fees.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .charges import estimate_fees
from .instruments import Instrument, load_instruments, normalise_symbol
from .options import bs_price, forward_price, implied_vol

WORKBOOK = "input.xlsx"


@dataclass
class PortfolioData:
    instruments: dict[str, Instrument]
    prices: pd.DataFrame
    trades: pd.DataFrame
    cashflows: pd.DataFrame
    notes: list = field(default_factory=list)
    modelled: list = field(default_factory=list)  # contracts whose closes were modelled


# --- Reading --------------------------------------------------------------------------


def read_table(folder: Path, name: str) -> pd.DataFrame | None:
    """A sheet called ``name`` in the workbook if it has rows, else ``name``.csv/.xlsx."""
    workbook = folder / WORKBOOK
    if workbook.exists():
        with pd.ExcelFile(workbook) as book:
            sheets = [s for s in book.sheet_names if s.strip().lower() == name]
            table = book.parse(sheets[0]).dropna(how="all") if sheets else None
        if table is not None and len(table):
            return table
    for extension in (".csv", ".xlsx"):
        path = folder / f"{name}{extension}"
        if path.exists():
            table = pd.read_csv(path) if extension == ".csv" else pd.read_excel(path)
            return table.dropna(how="all")
    return None


def tidy_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table.columns = [str(c).strip().lower().replace(" ", "_") for c in table.columns]
    return table


_SIDE_SIGN = {"BUY": 1, "B": 1, "LONG": 1, "SELL": -1, "S": -1, "SHORT": -1}
_TRADE_COLUMN_ALIASES = {
    "qty": "quantity", "lots": "quantity", "trade_date": "date", "type": "side",
    "trade_type": "side", "fee": "fees", "charges": "fees",
}


def clean_trades(raw: pd.DataFrame | None, instruments: dict | None = None) -> pd.DataFrame:
    """Standard columns: date, symbol, quantity (signed), price, fees, fees_estimated.

    Quantity can be signed already, or unsigned with a BUY/SELL ``side`` column. Blank fees
    are estimated if ``instruments`` is given, and ``fees_estimated`` says which ones.
    """
    columns = ["date", "symbol", "quantity", "price", "fees", "fees_estimated"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=columns)

    raw = tidy_columns(raw).rename(columns=_TRADE_COLUMN_ALIASES)
    missing = {"date", "symbol", "quantity", "price"} - set(raw.columns)
    if missing:
        raise ValueError(f"trades is missing column(s): {', '.join(sorted(missing))}")
    raw = raw.dropna(subset=["date", "symbol"])

    trades = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]),
        "symbol": raw["symbol"].map(normalise_symbol),
        "quantity": pd.to_numeric(raw["quantity"], errors="coerce"),
        "price": pd.to_numeric(raw["price"], errors="coerce"),
    })

    if "side" in raw.columns:
        side_text = raw["side"].astype(str).str.strip()
        sign = side_text.str.upper().map(_SIDE_SIGN)
        unknown = sign.isna() & raw["side"].notna() & (side_text != "")
        if unknown.any():
            raise ValueError(f"Unknown side {raw['side'][unknown].iloc[0]!r}: use BUY or SELL.")
        trades["quantity"] = np.where(sign.notna(), trades["quantity"].abs() * sign.fillna(1),
                                      trades["quantity"])

    bad_rows = trades[["quantity", "price"]].isna().any(axis=1)
    if bad_rows.any():
        # +2: one for the header row, one because spreadsheets count from 1
        raise ValueError(f"trades row {bad_rows.idxmax() + 2}: quantity and price must be numbers.")

    fees = pd.to_numeric(raw["fees"], errors="coerce") if "fees" in raw.columns else pd.Series(np.nan, index=raw.index)
    trades["fees"] = fees.to_numpy()
    trades["fees_estimated"] = fees.isna().to_numpy()
    if instruments is not None:
        for i in trades.index[trades["fees_estimated"]]:
            row = trades.loc[i]
            trades.at[i, "fees"] = estimate_fees(instruments[row.symbol], row.quantity, row.price, row.date)
    return trades.sort_values("date", kind="stable").reset_index(drop=True)


def clean_cashflows(raw: pd.DataFrame | None) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("cashflows is empty: add at least your starting capital.")
    raw = tidy_columns(raw)
    if "amount" not in raw.columns:
        if {"deposit", "withdrawal"} <= set(raw.columns):
            deposits = pd.to_numeric(raw["deposit"], errors="coerce").fillna(0)
            withdrawals = pd.to_numeric(raw["withdrawal"], errors="coerce").fillna(0)
            raw["amount"] = deposits - withdrawals
        else:
            raise ValueError("cashflows needs a date column and an amount column.")
    flows = pd.DataFrame({"date": pd.to_datetime(raw["date"]),
                          "amount": pd.to_numeric(raw["amount"], errors="coerce")}).dropna()
    return flows.sort_values("date").reset_index(drop=True)


def clean_prices(raw: pd.DataFrame | None) -> pd.DataFrame:
    """First column is the date; every other column is one symbol's closes."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    prices = raw.copy()
    date_column = prices.columns[0]
    prices[date_column] = pd.to_datetime(prices[date_column])
    prices = prices.set_index(date_column).sort_index()
    prices.index.name = "date"
    prices.columns = [normalise_symbol(c) for c in prices.columns]
    prices = prices.T.groupby(level=0).first().T  # 'RELIANCE' and 'reliance.ns' become one column
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices[~prices.index.duplicated(keep="last")]


# --- Filling in missing derivative closes ---------------------------------------------


def _implied_vols_from_trades(symbol, instrument, trades, spot, calendar, rf, q) -> pd.Series:
    """The implied vol at each of your own trades in this option, where it can be solved."""
    vols = {}
    for trade in trades[trades.symbol == symbol].itertuples():
        if trade.date > calendar[-1]:
            continue
        day = calendar[calendar.searchsorted(trade.date)]
        T = max((instrument.expiry - day).days, 0) / 365
        iv = implied_vol(trade.price, spot.loc[day], instrument.strike, T, rf, q, instrument.option_type)
        if not np.isnan(iv):
            vols[day] = iv
    return pd.Series(vols, dtype=float).sort_index()


def fill_derivative_prices(prices: pd.DataFrame, instruments: dict, trades: pd.DataFrame,
                           rf: float = config.RISK_FREE, q: float = config.INDEX_DIVIDEND_YIELD
                           ) -> tuple[pd.DataFrame, list[str]]:
    """Fill gaps in the closes of the futures and options you traded.

    Derivative closes are the hardest prices to collect, so you don't have to. Futures are
    filled at fair value, S x exp((r - q) T). Options are priced with Black-Scholes at the
    implied vol of your own most recent trade in that contract (or, if there isn't a usable
    one, recent realised vol plus a typical premium). Real closes are never overwritten.
    Returns the filled prices and the list of contracts that needed it.
    """
    prices = prices.copy()
    modelled = []
    for symbol in sorted(set(trades["symbol"])):
        instrument = instruments[symbol]
        if not instrument.is_derivative:
            continue
        if instrument.underlying not in prices.columns:
            raise ValueError(f"{symbol} needs prices for its underlying {instrument.underlying}.")
        if symbol not in prices.columns:
            prices[symbol] = np.nan

        first_trade = trades.loc[trades.symbol == symbol, "date"].min()
        live_days = prices.index[(prices.index >= first_trade - pd.Timedelta(days=5))
                                 & (prices.index <= instrument.expiry)]
        gaps = live_days[prices.loc[live_days, symbol].isna()]
        if len(gaps) == 0:
            continue
        modelled.append(symbol)

        spot = prices[instrument.underlying].ffill()
        S = spot.loc[gaps].to_numpy()
        T = np.array([max((instrument.expiry - day).days, 0) / 365 for day in gaps])

        if instrument.asset_class == "future":
            prices.loc[gaps, symbol] = np.round(forward_price(S, T, rf, q), 2)
            continue

        trade_vols = _implied_vols_from_trades(symbol, instrument, trades, spot, prices.index, rf, q)
        daily_vol = spot.pct_change(fill_method=None).ewm(alpha=1 - config.EWMA_LAMBDA).std()
        realised = daily_vol * math.sqrt(config.TRADING_DAYS)
        fallback = (realised * config.VERDICT["iv_premium_assumed"]).reindex(gaps).bfill().ffill().fillna(0.15)
        if len(trade_vols):
            # most recent trade's IV; before the first trade, use the first one
            iv = trade_vols.reindex(gaps, method="ffill").fillna(trade_vols.iloc[0])
        else:
            iv = fallback
        modelled_prices = bs_price(S, instrument.strike, T, rf, iv.to_numpy(), q, instrument.option_type)
        prices.loc[gaps, symbol] = np.maximum(np.round(np.asarray(modelled_prices) / 0.05) * 0.05, 0.05)
    return prices, modelled


# --- Loading a folder -----------------------------------------------------------------


def load_folder(folder: str | os.PathLike, rf: float = config.RISK_FREE,
                benchmark: str = config.BENCHMARK, model_missing: bool = True) -> PortfolioData:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"{folder} does not exist. Run `init --data {folder}` first.")
    notes = []

    raw_trades = read_table(folder, "trades")
    raw_instruments = read_table(folder, "instruments")
    cashflows = clean_cashflows(read_table(folder, "cashflows"))

    prices = clean_prices(read_table(folder, "prices"))
    downloaded = folder / "prices.csv"  # fetch-prices writes here even when a workbook exists
    if (folder / WORKBOOK).exists() and downloaded.exists() and not prices.empty:
        prices = prices.combine_first(clean_prices(pd.read_csv(downloaded)))
    if prices.empty:
        raise ValueError("No prices found. Add a Prices sheet or prices.csv, or run "
                         f"`fetch-prices --data {folder}` (needs yfinance).")

    traded = []
    if raw_trades is not None and len(raw_trades):
        traded = sorted(set(tidy_columns(raw_trades)["symbol"].dropna().map(normalise_symbol)))
    instruments = load_instruments(raw_instruments, traded + list(prices.columns), benchmark)
    trades = clean_trades(raw_trades, instruments)
    if trades["fees_estimated"].any():
        notes.append(f"{int(trades['fees_estimated'].sum())} trades had no fee, so charges were "
                     f"estimated (brokerage, STT, exchange, SEBI, stamp duty, GST).")

    needs_prices = [s for s in traded if not instruments[s].is_derivative] + [benchmark]
    missing = [s for s in needs_prices if s not in prices.columns]
    if missing:
        raise ValueError(f"No price column for: {', '.join(missing)}. Add them to prices or run "
                         f"`fetch-prices --data {folder}`.")

    modelled = []
    if model_missing and len(trades):
        prices, modelled = fill_derivative_prices(prices, instruments, trades, rf)
        if modelled:
            notes.append(f"{len(modelled)} derivative contract(s) had missing closes; those were "
                         f"modelled from the underlying (see the README's methodology notes).")
    return PortfolioData(instruments, prices, trades, cashflows, notes, modelled)


# --- Downloading prices ---------------------------------------------------------------


def yahoo_ticker(symbol: str) -> str:
    return {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK"}.get(symbol, f"{symbol}.NS")


def fetch_prices_yfinance(symbols, start, end=None) -> pd.DataFrame:
    """Daily closes for NSE stocks and indices from Yahoo Finance (``pip install yfinance``)."""
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError("Run `pip install yfinance` to download prices.") from error
    tickers = {yahoo_ticker(s): s for s in symbols}
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:  # a single ticker comes back without the extra column level
        closes = raw[["Close"]]
        closes.columns = list(tickers)
    closes = closes.rename(columns=tickers)
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes.dropna(how="all")
