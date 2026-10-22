"""A synthetic dataset, so the engine can be tried without anyone's real trades.

Prices are simulated:
* NIFTY follows a GARCH(1,1) process with Student-t shocks, so volatility clusters and
  the tails are fat, as in real index data.
* Each stock is beta x NIFTY plus its own fat-tailed noise (a one-factor model).
* Futures are priced at fair value and options with Black-Scholes, using the GARCH vol
  with a simple skew so out-of-the-money puts are dearer than calls.

The trades come from a deliberately simple rules-based strategy (nothing clever, the point
is to exercise every part of the ledger):
* stocks: each month, hold the four with the best 6-month momentum, 55% of NAV
* futures: one lot long above the 100-day average, short well below it, rolled before expiry
* options: always hold a protective put; buy a call in a strong uptrend; sometimes sell an
  out-of-the-money put mid-month when implied vol is rich

None of this is anyone's track record.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .charges import estimate_fees
from .instruments import Instrument, future_symbol, instruments_frame, lot_size, next_monthly_expiry, option_symbol
from .options import bs_price, forward_price

TD = config.TRADING_DAYS
INDEX = "NIFTY50"

# symbol: (sector, beta, idiosyncratic vol, starting price)
DEMO_STOCKS = {
    "RELIANCE": ("Energy", 0.95, 0.20, 1250.0),
    "HDFCBANK": ("Financials", 1.00, 0.18, 800.0),
    "ICICIBANK": ("Financials", 1.10, 0.20, 880.0),
    "INFY": ("IT", 0.80, 0.24, 1500.0),
    "TCS": ("IT", 0.70, 0.20, 3300.0),
    "LT": ("Industrials", 1.20, 0.22, 2100.0),
    "ITC": ("Consumer", 0.60, 0.18, 330.0),
    "MARUTI": ("Auto", 0.90, 0.24, 8600.0),
    "SUNPHARMA": ("Healthcare", 0.60, 0.22, 1000.0),
    "BHARTIARTL": ("Telecom", 0.80, 0.22, 780.0),
}

CASH_FLOWS = [("2023-01-02", 3_000_000.0), ("2024-06-03", 500_000.0), ("2025-11-03", -200_000.0)]



def _student_t(rng, dof, size):
    """Student-t draws scaled to unit variance."""
    return rng.standard_t(dof, size) / math.sqrt(dof / (dof - 2))


def simulate_market(dates: pd.DatetimeIndex, rng) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Returns (NIFTY closes, stock closes, NIFTY's annualised GARCH vol)."""
    n = len(dates)
    long_run_var = 0.14**2 / TD  # 14% a year on average
    arch, garch = 0.08, 0.90
    omega = long_run_var * (1 - arch - garch)
    drift = 0.11 / TD
    shocks = _student_t(rng, 5, n)

    variance, index_returns = np.empty(n), np.empty(n)
    variance[0] = long_run_var
    for t in range(n):
        if t:
            variance[t] = omega + arch * index_returns[t - 1] ** 2 + garch * variance[t - 1]
        index_returns[t] = drift - 0.5 * variance[t] + math.sqrt(variance[t]) * shocks[t]

    nifty = pd.Series(np.round(18100 * np.exp(np.cumsum(index_returns)), 2), dates)
    vol = pd.Series(np.sqrt(variance * TD), dates)

    stocks = {}
    for symbol, (_, beta, idio_vol, start_price) in DEMO_STOCKS.items():
        alpha = rng.normal(0.02, 0.04) / TD
        noise = _student_t(rng, 5, n) * idio_vol / math.sqrt(TD)
        stocks[symbol] = np.round(start_price * np.exp(np.cumsum(alpha + beta * index_returns + noise)), 2)
    return nifty, pd.DataFrame(stocks, index=dates), vol


def _skewed_iv(atm_vol: float, spot: float, strike: float) -> float:
    skew = 1 - 1.2 * math.log(strike / spot)
    return atm_vol * min(max(skew, 0.7), 1.8)



class _DemoStrategy:
    def __init__(self, dates, rng):
        self.dates = dates
        self.rng = rng
        self.nifty, self.stocks, self.garch_vol = simulate_market(dates, rng)
        self.sma20, self.sma50, self.sma100 = (self.nifty.rolling(n).mean() for n in (20, 50, 100))
        self.realised_vol_20d = self.nifty.pct_change().rolling(20).std() * math.sqrt(TD)

        self.contracts: dict[str, Instrument] = {}
        self._price_cache: dict = {}
        self.trades: list[dict] = []
        self.cash = 0.0
        self.shares = dict.fromkeys(self.stocks, 0)
        self.future: tuple[str, int] | None = None  # (symbol, lots)
        self.future_mark = 0.0
        self.options: dict[str, float] = {}  # symbol -> signed lots
        self.option_entry: dict[str, float] = {}

    # contracts and their prices

    def _future_contract(self, expiry) -> str:
        symbol = future_symbol(INDEX, expiry)
        self.contracts.setdefault(symbol, Instrument(symbol, "future", lot_size(INDEX, expiry), INDEX,
                                                     expiry=pd.Timestamp(expiry)))
        return symbol

    def _option_contract(self, expiry, strike, option_type) -> str:
        symbol = option_symbol(INDEX, expiry, strike, option_type)
        self.contracts.setdefault(symbol, Instrument(symbol, "option", lot_size(INDEX, expiry), INDEX,
                                                     float(strike), pd.Timestamp(expiry), option_type))
        return symbol

    def price(self, symbol, day) -> float:
        """Cached, so a trade price and the close written to prices.csv always agree."""
        key = (symbol, day)
        if key not in self._price_cache:
            contract = self.contracts[symbol]
            spot = self.nifty[day]
            T = max((contract.expiry - day).days, 0) / 365
            r, q = config.RISK_FREE, config.INDEX_DIVIDEND_YIELD
            if contract.asset_class == "future":
                fair = float(forward_price(spot, T, r, q))
                value = round(fair * (1 + self.rng.normal(0, 0.0003)), 1)  # a little basis noise
            else:
                iv = _skewed_iv(self.garch_vol[day] * 1.08, spot, contract.strike)
                value = float(bs_price(spot, contract.strike, T, r, iv, q, contract.option_type))
                value = round(max(round(value / 0.05) * 0.05, 0.05), 2)  # NSE tick is 5 paise
            self._price_cache[key] = value
        return self._price_cache[key]

    def trade(self, day, symbol, quantity, price):
        sector = DEMO_STOCKS.get(symbol, ("",))[0]
        instrument = self.contracts.get(symbol) or Instrument(symbol, "equity", 1, sector=sector)
        fee = estimate_fees(instrument, quantity, price, day)
        self.trades.append({"date": day.date(), "symbol": symbol, "quantity": quantity,
                            "price": price, "fees": fee})
        if instrument.asset_class == "future":
            self.cash -= fee
        else:
            self.cash -= quantity * instrument.multiplier * price + fee

    def _buy_option(self, day, symbol, lots):
        price = self.price(symbol, day)
        self.trade(day, symbol, lots, price)
        self.options[symbol] = self.options.get(symbol, 0) + lots
        self.option_entry[symbol] = price

    # the daily loop

    def run(self, cash_flows: dict):
        months_seen = set()
        step = config.STRIKE_STEP[INDEX]
        for i, day in enumerate(self.dates):
            self.cash += cash_flows.get(day, 0.0)
            spot = self.nifty[day]
            self._settle_future(day)
            nav = self._nav(day)
            first_day_of_month = (day.year, day.month) not in months_seen
            months_seen.add((day.year, day.month))

            if first_day_of_month and i > 0:
                self._rebalance_stocks(day, i, nav)
            if i >= 100:
                self._trade_futures_trend(day, spot)
            if i >= 50:
                self._keep_a_protective_put(day, spot, step)
                if first_day_of_month and spot > self.sma20[day] > self.sma50[day]:
                    expiry = next_monthly_expiry(day, 10)
                    self._buy_option(day, self._option_contract(expiry, round(spot * 1.02 / step) * step, "C"), 1)
                self._maybe_sell_a_put(day, spot, step)
            self._manage_options(day, spot)
            if self.future and self.contracts[self.future[0]].expiry.date() <= day.date():
                self.future = None

    def _settle_future(self, day):
        if self.future:
            symbol, lots = self.future
            price = self.price(symbol, day)
            self.cash += lots * self.contracts[symbol].multiplier * (price - self.future_mark)
            self.future_mark = price

    def _nav(self, day) -> float:
        stock_value = sum(units * self.stocks.at[day, s] for s, units in self.shares.items())
        option_value = sum(lots * self.contracts[s].multiplier * self.price(s, day)
                           for s, lots in self.options.items())
        return self.cash + stock_value + option_value

    def _rebalance_stocks(self, day, i, nav):
        if i >= 126:
            momentum = self.stocks.iloc[i] / self.stocks.iloc[i - 126] - 1
            winners = list(momentum.sort_values(ascending=False).index[:4])
        else:
            winners = ["RELIANCE", "HDFCBANK", "INFY", "LT"]
        target_value = 0.55 * nav / 4
        for symbol in self.stocks:
            price = float(self.stocks.at[day, symbol])
            target = int(target_value // price) if symbol in winners else 0
            change = target - self.shares[symbol]
            # skip tiny top-ups; they cost more in charges than they're worth
            if change != 0 and (target == 0 or abs(change) * price > 0.15 * target_value):
                self.trade(day, symbol, change, price)
                self.shares[symbol] += change

    def _trade_futures_trend(self, day, spot):
        sma = self.sma100[day]
        held = self.future[1] if self.future else 0
        # A band around the average stops the position flipping every other day
        if held > 0:
            wanted = 1 if spot > sma * 0.98 else 0
        elif held < 0:
            wanted = -1 if spot < sma else 0
        else:
            wanted = 1 if spot > sma * 1.01 else (-1 if spot < sma * 0.97 else 0)

        if self.future:
            symbol, lots = self.future
            days_left = (self.contracts[symbol].expiry - day).days
            if days_left <= 3 or np.sign(lots) != wanted:
                self.trade(day, symbol, -lots, self.future_mark)
                self.future = None
        if not self.future and wanted != 0:
            symbol = self._future_contract(next_monthly_expiry(day, 6))
            price = self.price(symbol, day)
            self.trade(day, symbol, wanted, price)
            self.future, self.future_mark = (symbol, wanted), price

    def _keep_a_protective_put(self, day, spot, step):
        has_put = any(lots > 0 and self.contracts[s].option_type == "P"
                      and (self.contracts[s].expiry - day).days > 3
                      for s, lots in self.options.items())
        if has_put:
            return
        moneyness = 0.98 if spot < self.sma50[day] else 0.95
        expiry = next_monthly_expiry(day, 10)
        symbol = self._option_contract(expiry, round(spot * moneyness / step) * step, "P")
        if self.options.get(symbol, 0) >= 0:
            self._buy_option(day, symbol, 1)

    def _maybe_sell_a_put(self, day, spot, step):
        mid_month_wednesday = 12 <= day.day <= 14 and day.weekday() == 2
        iv_is_rich = self.garch_vol[day] * 1.08 > self.realised_vol_20d[day] * 1.1
        if not (mid_month_wednesday and spot > self.sma50[day] and iv_is_rich):
            return
        expiry = next_monthly_expiry(day, 10)
        symbol = self._option_contract(expiry, round(spot * 0.95 / step) * step, "P")
        if symbol not in self.options:
            price = self.price(symbol, day)
            self.trade(day, symbol, -1, price)
            self.options[symbol] = -1
            self.option_entry[symbol] = price

    def _manage_options(self, day, spot):
        for symbol in list(self.options):
            contract = self.contracts[symbol]
            lots = self.options[symbol]
            if day.date() >= contract.expiry.date():
                if contract.option_type == "C":
                    intrinsic = max(spot - contract.strike, 0)
                else:
                    intrinsic = max(contract.strike - spot, 0)
                self.cash += lots * contract.multiplier * intrinsic
                del self.options[symbol]
                continue
            price = self.price(symbol, day)
            gain = (price / self.option_entry[symbol] - 1) * np.sign(lots)
            # longs: take a double; shorts: buy back once 70% of the premium is in
            if (lots > 0 and gain >= 1.0) or (lots < 0 and gain >= 0.7):
                self.trade(day, symbol, -lots, price)
                del self.options[symbol]



def generate_demo(folder: str | os.PathLike, start="2023-01-02", end="2026-10-20",
                  seed=config.DEMO_SEED) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, end)
    flows = [(pd.Timestamp(d), amount) for d, amount in CASH_FLOWS]
    flows[0] = (dates[0], flows[0][1])

    sim = _DemoStrategy(dates, np.random.default_rng(seed))
    sim.run(dict(flows))

    instruments = {INDEX: Instrument(INDEX, "index")}
    for symbol, (sector, *_) in DEMO_STOCKS.items():
        instruments[symbol] = Instrument(symbol, "equity", 1, sector=sector)
    instruments.update(sim.contracts)
    instruments_frame(instruments).to_csv(folder / "instruments.csv", index=False)

    # Each contract gets closes for the 75 days before it expires, like a real price history
    contract_closes = {}
    for symbol, contract in sim.contracts.items():
        live = dates[(dates >= contract.expiry - pd.Timedelta(days=75)) & (dates <= contract.expiry)]
        contract_closes[symbol] = pd.Series([sim.price(symbol, d) for d in live], index=live)

    prices = pd.DataFrame({INDEX: sim.nifty}).join(sim.stocks).join(pd.DataFrame(contract_closes))
    prices.index.name = "date"
    prices.to_csv(folder / "prices.csv", float_format="%.2f")
    pd.DataFrame(sim.trades).to_csv(folder / "trades.csv", index=False)
    pd.DataFrame([{"date": d.date(), "amount": a} for d, a in flows]).to_csv(folder / "cashflows.csv", index=False)
    return folder
