"""Importing a broker's tradebook export.

Every broker names its columns differently, so columns are matched against a list of common
names (Zerodha Console's tradebook works as is). Three things broker files do that the
engine doesn't want are fixed on the way in:

* F&O quantity is in units (lots x lot size); the engine wants lots.
* One order filled in several pieces shows up as several rows; they're merged into one
  trade at the volume-weighted price.
* Dates can be ISO (2026-10-01) or Indian day-first (01-10-2026).

Rows already in your trades sheet are skipped, so importing the same file twice is harmless.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import read_table
from .inputs import TRADE_COLUMNS, append_rows
from .instruments import normalise_symbol, parse_symbol

COLUMN_NAMES = {
    "date": ["trade_date", "date", "trade date", "order date", "order_execution_time",
             "execution date", "trade time"],
    "symbol": ["tradingsymbol", "trading symbol", "symbol", "scrip", "scrip name", "instrument",
               "stock", "stock name", "security"],
    "side": ["trade_type", "trade type", "transaction type", "buy/sell", "side", "type", "action", "b/s"],
    "quantity": ["quantity", "qty", "filled qty", "traded qty", "shares"],
    "price": ["price", "trade price", "rate", "avg price", "average price", "traded price", "net rate"],
    "order": ["order_id", "order id", "order no", "order number"],
    "fees": ["charges", "total charges", "fees", "brokerage"],
}
REQUIRED = ("date", "symbol", "side", "quantity", "price")


def _find_column(columns, field_name: str) -> str | None:
    by_lowercase = {c.strip().lower(): c for c in columns}
    for candidate in COLUMN_NAMES[field_name]:
        if candidate in by_lowercase:
            return by_lowercase[candidate]
    return None


def parse_dates(column: pd.Series) -> pd.Series:
    """ISO dates are read as is; anything else is read day-first, the Indian way."""
    if pd.api.types.is_datetime64_any_dtype(column):
        return pd.to_datetime(column).dt.normalize()
    text = column.astype(str).str.strip()
    is_iso = text.str.match(r"^\d{4}-\d{2}-\d{2}")
    dates = pd.Series(pd.NaT, index=column.index, dtype="datetime64[ns]")
    if is_iso.any():
        dates[is_iso] = pd.to_datetime(text[is_iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~is_iso).any():
        dates[~is_iso] = pd.to_datetime(text[~is_iso], dayfirst=True, errors="coerce", format="mixed")
    return dates.dt.normalize()


def read_tradebook(file, column_map: dict | None = None, fo_quantity_in_units: bool = True) -> pd.DataFrame:
    file = Path(file)
    raw = pd.read_excel(file) if file.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(file)
    column_map = column_map or {}
    columns = {name: column_map.get(name) or _find_column(raw.columns, name) for name in COLUMN_NAMES}
    missing = [name for name in REQUIRED if not columns[name]]
    if missing:
        raise ValueError(f"Couldn't find the {', '.join(missing)} column(s) in {file.name}. Its columns "
                         f"are {list(raw.columns)}. Tell me which is which with --map, e.g. "
                         f"--map date='Trade Date' symbol=Scrip")

    fills = pd.DataFrame({
        "date": parse_dates(raw[columns["date"]]),
        "symbol": raw[columns["symbol"]].map(normalise_symbol),
        "side": raw[columns["side"]].astype(str).str.strip().str.upper().str[0].map({"B": "BUY", "S": "SELL"}),
        "qty": pd.to_numeric(raw[columns["quantity"]], errors="coerce").abs(),
        "price": pd.to_numeric(raw[columns["price"]], errors="coerce"),
        "order": raw[columns["order"]].astype(str) if columns["order"] else raw.index.astype(str),
        "fees": pd.to_numeric(raw[columns["fees"]], errors="coerce") if columns["fees"] else np.nan,
    }).dropna(subset=["date", "symbol", "side", "qty", "price"])
    fills["notional"] = fills.qty * fills.price

    orders = (fills.groupby(["date", "symbol", "side", "order"], sort=False)
              .agg(qty=("qty", "sum"), notional=("notional", "sum"), fees=("fees", "sum"))
              .reset_index())
    orders["price"] = (orders.notional / orders.qty).round(4)

    rows, warnings = [], []
    for order in orders.itertuples():
        quantity = order.qty
        contract = parse_symbol(order.symbol)
        if contract is not None and fo_quantity_in_units:
            lots = quantity / contract.multiplier
            if not math.isclose(lots, round(lots), abs_tol=1e-6):
                warnings.append(f"{order.symbol}: {quantity:g} units isn't a whole number of lots of "
                                f"{contract.multiplier:g}. Check the lot size (you can override it in "
                                f"the Instruments sheet).")
            quantity = lots
        has_fee = order.fees and not np.isnan(order.fees) and order.fees > 0
        rows.append({"Date": order.date, "Symbol": order.symbol, "Side": order.side,
                     "Quantity": quantity, "Price": order.price,
                     "Fees": order.fees if has_fee else None,
                     "Notes": f"imported from {file.name}"})

    trades = pd.DataFrame(rows)
    trades.attrs["warnings"] = warnings
    return trades


def _trade_key(date, symbol, quantity, price):
    return pd.Timestamp(date).date(), normalise_symbol(symbol), round(abs(float(quantity)), 6), round(float(price), 2)


def import_tradebook(folder, file, column_map=None, fo_quantity_in_units=True) -> dict:
    folder = Path(folder)
    new = read_tradebook(file, column_map, fo_quantity_in_units)
    warnings = new.attrs.get("warnings", [])

    existing = read_table(folder, "trades")
    duplicates = 0
    if existing is not None and len(existing) and len(new):
        existing.columns = [str(c).strip().lower() for c in existing.columns]
        already_there = {_trade_key(r.date, r.symbol, r.quantity, r.price) for r in existing.itertuples()}
        is_new = [_trade_key(r.Date, r.Symbol, r.Quantity, r.Price) not in already_there for r in new.itertuples()]
        duplicates = len(new) - sum(is_new)
        new = new[is_new]

    written_to = append_rows(folder, "Trades", new.to_dict("records"), TRADE_COLUMNS) if len(new) else "(nothing new)"
    return {"added": len(new), "duplicates_skipped": duplicates, "written_to": written_to, "warnings": warnings}
