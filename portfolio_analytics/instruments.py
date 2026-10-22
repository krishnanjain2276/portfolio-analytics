"""What can be traded: stocks, the index, NIFTY futures and NIFTY options.

Most of this module is about not making you type things the engine can work out for itself.
An NSE derivative symbol already contains the underlying, expiry, strike and option type,
so ``parse_symbol`` decodes it:

    NIFTY26OCTFUT        October 2026 monthly future
    NIFTY26OCT25000CE    October 2026 monthly 25000 call
    NIFTY26O2025000PE    weekly 25000 put expiring 20 Oct 2026
                         (YY, then month as 1-9 / O / N / D, then DD)

The lot size is then looked up from the expiry date, because NSE has changed it several times.
"""

from __future__ import annotations

import calendar
import datetime as dt
import math
import re
from dataclasses import dataclass

import pandas as pd

from . import config

ASSET_CLASSES = ("equity", "future", "option", "index")

# Sectors for the large caps people actually trade, so a plain stock symbol needs no setup.
# Anything missing shows up as "Unclassified"; add it in the Instruments sheet.
SECTOR_MAP = {
    "RELIANCE": "Energy", "ONGC": "Energy", "COALINDIA": "Energy",
    "NTPC": "Utilities", "POWERGRID": "Utilities",
    "HDFCBANK": "Financials", "ICICIBANK": "Financials", "SBIN": "Financials",
    "KOTAKBANK": "Financials", "AXISBANK": "Financials", "BAJFINANCE": "Financials",
    "INDUSINDBK": "Financials", "BAJAJFINSV": "Financials", "HDFCLIFE": "Financials",
    "SBILIFE": "Financials",
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT", "LTIM": "IT",
    "HINDUNILVR": "Consumer", "ITC": "Consumer", "NESTLEIND": "Consumer",
    "BRITANNIA": "Consumer", "TATACONSUM": "Consumer", "ASIANPAINT": "Consumer",
    "TITAN": "Consumer", "TRENT": "Consumer",
    "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto", "BAJAJ-AUTO": "Auto",
    "EICHERMOT": "Auto", "HEROMOTOCO": "Auto",
    "SUNPHARMA": "Healthcare", "DRREDDY": "Healthcare", "CIPLA": "Healthcare",
    "APOLLOHOSP": "Healthcare",
    "LT": "Industrials", "ADANIPORTS": "Industrials", "ADANIENT": "Industrials",
    "BEL": "Industrials",
    "ULTRACEMCO": "Materials", "GRASIM": "Materials", "TATASTEEL": "Materials",
    "JSWSTEEL": "Materials", "HINDALCO": "Materials",
    "BHARTIARTL": "Telecom",
}


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: str  # equity | future | option | index
    multiplier: float = 1.0  # lot size for F&O, 1 for stocks
    underlying: str | None = None
    strike: float | None = None
    expiry: pd.Timestamp | None = None
    option_type: str | None = None  # "C" or "P"
    sector: str | None = None

    def __post_init__(self):
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"{self.symbol}: unknown asset_class {self.asset_class!r}")
        if self.asset_class == "option":
            if self.option_type not in ("C", "P"):
                raise ValueError(f"{self.symbol}: option_type must be C or P")
            if self.strike is None or self.expiry is None or self.underlying is None:
                raise ValueError(f"{self.symbol}: an option needs an underlying, strike and expiry")
        if self.asset_class == "future" and (self.expiry is None or self.underlying is None):
            raise ValueError(f"{self.symbol}: a future needs an underlying and expiry")

    @property
    def is_derivative(self) -> bool:
        return self.asset_class in ("future", "option")

    @property
    def sleeve(self) -> str:
        if self.asset_class == "equity":
            return self.sector or "Unclassified"
        if self.asset_class == "future":
            return "Futures"
        if self.asset_class == "option":
            return "Calls" if self.option_type == "C" else "Puts"
        return "Index"


# Expiries and lot sizes


def _last_weekday_of_month(year: int, month: int, weekday: int) -> dt.date:
    last_day = dt.date(year, month, calendar.monthrange(year, month)[1])
    return last_day - dt.timedelta(days=(last_day.weekday() - weekday) % 7)


def monthly_expiry(year: int, month: int) -> dt.date:
    """Last Thursday of the month before Sept 2025, last Tuesday from then on.

    When expiry day is an exchange holiday NSE moves it a day earlier. I don't model
    holidays, so if that matters for a contract you traded, put the real date in the
    Instruments sheet.
    """
    first_of_month = dt.date(year, month, 1)
    weekday = [w for start, w in config.EXPIRY_WEEKDAY_SCHEDULE if first_of_month >= start][-1]
    return _last_weekday_of_month(year, month, weekday)


def lot_size(underlying: str, expiry: dt.date | pd.Timestamp | None) -> int:
    schedule = config.LOT_SIZE_SCHEDULE.get(underlying)
    if not schedule:
        return 1
    if expiry is None:
        return schedule[-1][1]
    expiry_date = pd.Timestamp(expiry).date()
    return [size for start, size in schedule if expiry_date >= start][-1]


def next_monthly_expiry(after: dt.date | pd.Timestamp, min_days: int = 7) -> dt.date:
    """The first monthly expiry at least ``min_days`` after ``after``."""
    today = pd.Timestamp(after).date()
    year, month = today.year, today.month
    for _ in range(3):
        expiry = monthly_expiry(year, month)
        if (expiry - today).days >= min_days:
            return expiry
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return expiry


# Symbols

_MONTH_NUMBER = {name.upper(): i for i, name in enumerate(calendar.month_abbr) if name}
_WEEKLY_MONTH_CODE = {**{str(i): i for i in range(1, 10)}, "O": 10, "N": 11, "D": 12}

_MONTHLY_PATTERN = re.compile(
    r"^(?P<root>[A-Z&\-]+?)(?P<yy>\d{2})"
    r"(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"(?:(?P<fut>FUT)|(?P<strike>\d+(?:\.\d+)?)(?P<cp>CE|PE))$"
)
_WEEKLY_PATTERN = re.compile(
    r"^(?P<root>[A-Z&\-]+?)(?P<yy>\d{2})(?P<m>[1-9OND])(?P<dd>\d{2})"
    r"(?P<strike>\d+(?:\.\d+)?)(?P<cp>CE|PE)$"
)


def normalise_symbol(raw: str) -> str:
    """'reliance.ns ' -> 'RELIANCE', 'Nifty 50' -> 'NIFTY50'."""
    symbol = str(raw).strip().upper().replace(" ", "")
    for suffix in (".NS", ".BO", "-EQ"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
    return config.INDEX_ALIASES.get(symbol, symbol)


def parse_symbol(symbol: str) -> Instrument | None:
    symbol = normalise_symbol(symbol)

    match = _MONTHLY_PATTERN.match(symbol)
    if match:
        underlying = config.INDEX_ALIASES.get(match["root"], match["root"])
        expiry = pd.Timestamp(monthly_expiry(2000 + int(match["yy"]), _MONTH_NUMBER[match["mon"]]))
        lot = lot_size(underlying, expiry)
        if match["fut"]:
            return Instrument(symbol, "future", lot, underlying, expiry=expiry)
        option_type = "C" if match["cp"] == "CE" else "P"
        return Instrument(symbol, "option", lot, underlying, float(match["strike"]), expiry, option_type)

    match = _WEEKLY_PATTERN.match(symbol)
    if match:
        underlying = config.INDEX_ALIASES.get(match["root"], match["root"])
        try:
            month = _WEEKLY_MONTH_CODE[match["m"]]
            expiry = pd.Timestamp(dt.date(2000 + int(match["yy"]), month, int(match["dd"])))
        except ValueError:  # e.g. day 31 in a 30-day month: not a real weekly symbol
            return None
        option_type = "C" if match["cp"] == "CE" else "P"
        return Instrument(symbol, "option", lot_size(underlying, expiry), underlying,
                          float(match["strike"]), expiry, option_type)
    return None


def _symbol_root(underlying: str) -> str:
    return "NIFTY" if underlying == "NIFTY50" else underlying


def option_symbol(underlying: str, expiry: dt.date, strike: float, option_type: str) -> str:
    """Build a monthly option symbol, e.g. ('NIFTY50', 2026-11-24, 25000, 'C') -> NIFTY26NOV25000CE."""
    expiry = pd.Timestamp(expiry)
    suffix = "CE" if option_type == "C" else "PE"
    return f"{_symbol_root(underlying)}{expiry:%y%b}".upper() + f"{int(strike)}{suffix}"


def future_symbol(underlying: str, expiry: dt.date) -> str:
    return f"{_symbol_root(underlying)}{pd.Timestamp(expiry):%y%b}".upper() + "FUT"


def infer_instrument(symbol: str, benchmark: str = config.BENCHMARK) -> Instrument:
    symbol = normalise_symbol(symbol)
    parsed = parse_symbol(symbol)
    if parsed is not None:
        return parsed
    if symbol == benchmark or symbol in config.INDEX_ALIASES.values():
        return Instrument(symbol, "index")
    return Instrument(symbol, "equity", 1.0, sector=SECTOR_MAP.get(symbol, "Unclassified"))


# Loading the Instruments sheet


def _is_blank(value) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    return str(value).strip() in ("", "nan", "None", "NaT")


def instrument_from_row(row: dict) -> Instrument:
    """An Instruments row. Blank cells fall back to whatever the symbol itself implies."""
    symbol = normalise_symbol(row["symbol"])
    guess = infer_instrument(symbol)

    def cell(key, convert, default):
        return default if _is_blank(row.get(key)) else convert(row[key])

    asset_class = cell("asset_class", lambda v: str(v).strip().lower(), guess.asset_class)
    underlying = cell("underlying", normalise_symbol, guess.underlying)
    expiry = cell("expiry", pd.Timestamp, guess.expiry)
    strike = cell("strike", float, guess.strike)
    option_type = cell("option_type", lambda v: str(v).strip().upper()[:1], guess.option_type)

    if not _is_blank(row.get("multiplier")):
        multiplier = float(row["multiplier"])
    elif asset_class in ("future", "option"):
        multiplier = guess.multiplier if guess.is_derivative else float(lot_size(underlying, expiry))
    else:
        multiplier = 1.0

    sector = cell("sector", lambda v: str(v).strip(), guess.sector if asset_class == "equity" else None)
    if asset_class == "equity" and not sector:
        sector = "Unclassified"
    return Instrument(symbol, asset_class, multiplier, underlying, strike, expiry, option_type, sector)


def load_instruments(
    table: pd.DataFrame | None, symbols_needed=(), benchmark: str = config.BENCHMARK
) -> dict[str, Instrument]:
    """Everything the engine will touch, keyed by symbol.

    Rows you wrote in the Instruments sheet win. Every other symbol (from trades, prices or
    the watch-list) is inferred, and so is the underlying of every derivative.
    """
    instruments: dict[str, Instrument] = {}
    if table is not None and len(table):
        for row in table.to_dict("records"):
            if not _is_blank(row.get("symbol")):
                instrument = instrument_from_row(row)
                instruments[instrument.symbol] = instrument

    for symbol in [*symbols_needed, benchmark]:
        symbol = normalise_symbol(symbol)
        if symbol not in instruments:
            instruments[symbol] = infer_instrument(symbol, benchmark)

    for instrument in list(instruments.values()):
        if instrument.underlying and instrument.underlying not in instruments:
            instruments[instrument.underlying] = infer_instrument(instrument.underlying, benchmark)
    return instruments


def instruments_frame(instruments: dict[str, Instrument]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "symbol": i.symbol,
            "asset_class": i.asset_class,
            "multiplier": i.multiplier,
            "underlying": i.underlying or "",
            "strike": i.strike,
            "expiry": i.expiry.date() if i.expiry is not None else None,
            "option_type": i.option_type or "",
            "sector": i.sector or "",
        }
        for i in instruments.values()
    ])
