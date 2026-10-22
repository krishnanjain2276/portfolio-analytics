"""Trend signals for stocks and for NIFTY, and the NIFTY options trade idea.

Every stock in the price file gets a score from -100 to +100 built from a few well-known,
simple ingredients: where price sits relative to its 20/50/200-day averages, 12-1 month
momentum, strength relative to NIFTY over three months, RSI for timing, and penalties for
being deep below the 52-week high or extremely volatile. Each ingredient's points are kept,
so the report can show exactly why a stock scored what it did.

The same score on NIFTY gives a market view. Combined with whether options look expensive
(implied vol against realised vol) it picks one of six standard strategies:

                    options cheap or fair      options expensive
    bullish         buy a call                 bull put spread
    bearish         buy a put                  bear call spread
    no trend        stay flat                  iron condor

None of this is a prediction. It's a consistent, checkable way of turning trend and vol into
a starting point.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .instruments import future_symbol, lot_size, next_monthly_expiry, option_symbol
from .options import bs_price, implied_vol

RULES = config.VERDICT
TD = config.TRADING_DAYS
NAN = float("nan")


def pct(x, decimals=1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x * 100:.{decimals}f}%"


def rupees(x) -> str:
    """Indian style: lakh and crore for big numbers."""
    if x is None or not math.isfinite(x):
        return "n/a"
    sign, x = ("-" if x < 0 else ""), abs(x)
    if x >= 1e7:
        return f"{sign}Rs {x / 1e7:.2f} cr"
    if x >= 1e5:
        return f"{sign}Rs {x / 1e5:.2f} lakh"
    return f"{sign}Rs {x:,.0f}"


# Technicals


def rsi(closes: pd.Series, period: int = 14) -> float:
    """Wilder's RSI on the latest day."""
    change = closes.diff().dropna()
    if len(change) < period + 1:
        return NAN
    gains = change.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    losses = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if losses == 0:
        return 100.0
    return float(100 - 100 / (1 + gains / losses))


def technicals(closes: pd.Series, benchmark: pd.Series | None = None) -> dict:
    closes = closes.dropna()
    last = float(closes.iloc[-1])
    n = len(closes)
    daily_moves = closes.pct_change().dropna()

    def average(days):
        return float(closes.tail(days).mean()) if n >= days else NAN

    def change_over(days, skip=0):
        return float(closes.iloc[-1 - skip] / closes.iloc[-1 - days] - 1) if n > days else NAN

    t = {
        "last": last,
        "sma20": average(20),
        "sma50": average(50),
        "sma200": average(200),
        "rsi14": rsi(closes),
        "mom_1m": change_over(21),
        "mom_3m": change_over(63),
        "mom_6m": change_over(126),
        # 12-month return skipping the last month, the standard momentum measure
        "mom_12_1": change_over(252, skip=21),
        "high_52w": float(closes.tail(252).max()),
        "vol_20d": float(daily_moves.tail(20).std() * math.sqrt(TD)),
        # average daily move in price points; there's no high/low data, so this stands in for ATR
        "avg_move_pts": float(daily_moves.tail(14).abs().mean() * last),
        "history_days": n,
    }
    t["off_high"] = last / t["high_52w"] - 1
    if benchmark is not None:
        bench = benchmark.dropna()
        if len(bench) > 63:
            t["rel_3m"] = t["mom_3m"] - float(bench.iloc[-1] / bench.iloc[-64] - 1)
    return t


def score(t: dict) -> tuple[int, list[str]]:
    """Trend score in [-100, 100], with the reason for every point."""
    total = 0.0
    reasons = []

    def add(points, reason):
        nonlocal total
        total += points
        reasons.append(f"{reason} ({round(points):+d})")

    last, sma20, sma50, sma200 = t["last"], t["sma20"], t["sma50"], t["sma200"]

    if math.isfinite(sma200):
        add(15 if last > sma200 else -15, f"price {'above' if last > sma200 else 'below'} 200-day average")
        if math.isfinite(sma50):
            if sma50 > sma200:
                add(10, "50-day above 200-day (golden cross)")
            else:
                add(-10, "50-day below 200-day (death cross)")
    if math.isfinite(sma50):
        add(10 if last > sma50 else -10, f"price {'above' if last > sma50 else 'below'} 50-day average")
    if math.isfinite(sma20):
        add(5 if last > sma20 else -5, f"short-term trend {'up' if last > sma20 else 'down'}")

    has_12m = math.isfinite(t["mom_12_1"])
    momentum = t["mom_12_1"] if has_12m else t["mom_6m"]
    if math.isfinite(momentum):
        add(float(np.clip(momentum * 60, -15, 15)), f"{'12-1' if has_12m else '6-month'} momentum {pct(momentum)}")

    relative = t.get("rel_3m", NAN)
    if math.isfinite(relative):
        add(float(np.clip(relative * 150, -15, 15)), f"3-month return vs NIFTY {pct(relative)}")

    r = t["rsi14"]
    if math.isfinite(r):
        in_uptrend = math.isfinite(sma50) and last > sma50
        if r >= RULES["rsi_overbought"]:
            add(-10, f"RSI {r:.0f}: overbought, a poor entry point")
        elif r <= RULES["rsi_oversold"]:
            add(10 if in_uptrend else 3, f"RSI {r:.0f}: oversold{', a dip inside an uptrend' if in_uptrend else ''}")

    if t["off_high"] < -0.25 and math.isfinite(sma200) and last < sma200:
        add(-10, f"{pct(-t['off_high'], 0)} below its 52-week high and still falling")
    if t["vol_20d"] > 0.45:
        add(-5, f"very high volatility ({pct(t['vol_20d'], 0)} annualised)")

    return int(round(np.clip(total, -100, 100))), reasons


def rating(points: int) -> str:
    if points >= RULES["score_strong"]:
        return "STRONG BUY"
    if points >= RULES["score_act"]:
        return "BUY"
    if points > -RULES["score_act"]:
        return "NEUTRAL"
    if points > -RULES["score_strong"]:
        return "SELL"
    return "STRONG SELL"


# What each rating means depends on whether you already own the stock.
ACTION_IF_HELD = {"STRONG BUY": "ADD", "BUY": "HOLD / ADD ON DIPS", "NEUTRAL": "HOLD",
                  "SELL": "TRIM", "STRONG SELL": "EXIT"}
ACTION_IF_NOT_HELD = {"STRONG BUY": "BUY", "BUY": "BUY (staggered)", "NEUTRAL": "WAIT",
                      "SELL": "AVOID", "STRONG SELL": "SHORT (stock futures) / AVOID"}
ACTION_IF_SHORT = {"STRONG BUY": "COVER SHORT", "BUY": "COVER SHORT", "NEUTRAL": "HOLD SHORT",
                   "SELL": "HOLD SHORT", "STRONG SELL": "HOLD SHORT"}


def stock_signals(data, positions: pd.DataFrame, nav: float, bench: str = config.BENCHMARK) -> pd.DataFrame:
    held = {}
    if len(positions):
        held = positions.loc[positions.asset_class == "equity"].set_index("symbol")["units"].to_dict()
    benchmark = data.prices.get(bench)

    rows = []
    for symbol in data.prices.columns:
        instrument = data.instruments.get(symbol)
        closes = data.prices[symbol].dropna()
        if instrument is None or instrument.asset_class != "equity" or len(closes) < 60:
            continue
        t = technicals(closes, benchmark)
        points, reasons = score(t)
        grade = rating(points)
        units = held.get(symbol, 0)
        if units > 0:
            action = ACTION_IF_HELD[grade]
        elif units < 0:
            action = ACTION_IF_SHORT[grade]
        else:
            action = ACTION_IF_NOT_HELD[grade]

        # Stop 2.5 average daily moves away, target at twice that (a 2:1 reward-to-risk),
        # size so that hitting the stop loses 1% of NAV, but never more than 20% of NAV in one name.
        stop_distance = 2.5 * t["avg_move_pts"]
        quantity = int(math.floor(nav * config.RISK_PER_TRADE / stop_distance)) if stop_distance > 0 else 0
        quantity = min(quantity, int(0.20 * nav // t["last"]))
        direction = 1 if points > 0 else -1

        rows.append({
            "symbol": symbol,
            "sector": instrument.sector,
            "held_units": units,
            "last": t["last"],
            "score": points,
            "rating": grade,
            "action": action,
            "stop": t["last"] - direction * stop_distance,
            "target": t["last"] + direction * 2 * stop_distance,
            "suggested_qty_1pct_risk": quantity,
            "rsi14": t["rsi14"],
            "mom_3m": t["mom_3m"],
            "mom_12_1": t["mom_12_1"],
            "off_52w_high": t["off_high"],
            "vol_20d": t["vol_20d"],
            "why": "; ".join(reasons),
        })
    table = pd.DataFrame(rows)
    return table.sort_values("score", ascending=False).reset_index(drop=True) if len(table) else table


# NIFTY view and the options trade idea


def market_implied_vol(data, bench, rf, q) -> tuple[float | None, str | None]:
    """IV of the nearest-expiry, closest-to-the-money NIFTY option with a real close today.

    Modelled closes don't count: they were priced from an assumed vol, so backing a vol out
    of them would just return the assumption.
    """
    today = data.prices.index[-1]
    spot = float(data.prices[bench].ffill().iloc[-1])
    best = None
    for symbol, inst in data.instruments.items():
        if inst.asset_class != "option" or inst.underlying != bench or symbol not in data.prices:
            continue
        if symbol in getattr(data, "modelled", []) or inst.expiry <= today + pd.Timedelta(days=2):
            continue
        close = data.prices[symbol].get(today, NAN)
        if not math.isfinite(close):
            continue
        closeness = ((inst.expiry - today).days, abs(inst.strike / spot - 1))
        if best is None or closeness < best[0]:
            best = (closeness, symbol, inst, close)

    if best is None or best[0][1] > 0.06:  # nothing within 6% of the money
        return None, None
    _, symbol, inst, close = best
    T = (inst.expiry - today).days / 365
    iv = implied_vol(close, spot, inst.strike, T, rf, q, inst.option_type)
    return (iv, symbol) if math.isfinite(iv) else (None, None)


def build_strategy(view: str, strong: bool, regime: str, spot: float, iv: float, today, rf: float,
                   q: float, nav: float, avg_move_pts: float, underlying: str = config.BENCHMARK) -> dict:
    """Turn a view and a vol regime into concrete option legs, sized to the risk budget."""
    expiry = next_monthly_expiry(today, 15)  # at least two weeks, so theta isn't everything
    T = (pd.Timestamp(expiry) - pd.Timestamp(today)).days / 365
    lot = lot_size(underlying, expiry)
    step = config.STRIKE_STEP.get(underlying, 50)
    risk_budget = nav * config.RISK_PER_TRADE

    def strike(moneyness):
        return round(spot * moneyness / step) * step

    def leg(action, moneyness, option_type):
        k = strike(moneyness)
        premium = float(bs_price(spot, k, T, rf, iv, q, option_type))
        return {"action": action, "symbol": option_symbol(underlying, expiry, k, option_type),
                "strike": k, "type": "CE" if option_type == "C" else "PE",
                "premium_est": round(premium, 1), "lot": lot}

    expensive = regime == "RICH"
    if view == "BULLISH" and expensive:
        name = "BULL PUT SPREAD (sell put spread)"
        legs = [leg("SELL", 0.99, "P"), leg("BUY", 0.97, "P")]
        why = "Bullish view, but options are expensive: collect premium instead of paying it."
    elif view == "BULLISH":
        name = "BUY CALL" + (" / LONG FUTURES" if strong else "")
        legs = [leg("BUY", 1.01, "C")]
        why = "Bullish view and options are not expensive: buy upside with a limited loss."
    elif view == "BEARISH" and expensive:
        name = "BEAR CALL SPREAD (sell call spread)"
        legs = [leg("SELL", 1.01, "C"), leg("BUY", 1.03, "C")]
        why = "Bearish view and options are expensive: sell call premium with capped risk."
    elif view == "BEARISH":
        name = "BUY PUT" + (" / SHORT FUTURES" if strong else "")
        legs = [leg("BUY", 0.99, "P")]
        why = "Bearish view and options are not expensive: buy downside exposure or protection."
    elif expensive:
        name = "IRON CONDOR (sell range)"
        legs = [leg("SELL", 0.97, "P"), leg("BUY", 0.95, "P"), leg("SELL", 1.03, "C"), leg("BUY", 1.05, "C")]
        why = "No clear trend and options are expensive: sell a range with defined risk."
    else:
        name = "STAY FLAT (no index trade)"
        legs = []
        why = ("No clear trend and options aren't expensive, so there's no edge in selling vol and no "
               "direction to buy. A long straddle only makes sense ahead of a known event.")

    idea = {"strategy": name, "rationale": why, "expiry": pd.Timestamp(expiry).date(),
            "lot_size": lot, "legs": legs, "dte": int(round(T * 365))}
    if legs:
        idea.update(_payoff(legs))
        idea["max_loss_per_lot"] = idea["max_loss_pts"] * lot
        idea["lots"] = int(risk_budget // idea["max_loss_per_lot"]) if idea["max_loss_per_lot"] > 0 else 0
        idea["net_per_lot"] = idea["net"] * lot
        max_profit = idea["max_profit_pts"]
        idea["reward_risk"] = max_profit / idea["max_loss_pts"] if math.isfinite(max_profit) else float("inf")

    if strong and view in ("BULLISH", "BEARISH") and not expensive:
        stop = 2.5 * avg_move_pts
        idea["futures_alt"] = {
            "action": "BUY" if view == "BULLISH" else "SELL",
            "symbol": future_symbol(underlying, expiry),
            "stop_pts": round(stop),
            "lots": int(risk_budget // (stop * lot)) if stop > 0 else 0,
        }
    return idea


def _payoff(legs: list[dict]) -> dict:
    """Net premium (positive = credit), max loss and profit in points, and breakevens."""
    credit = sum((1 if leg["action"] == "SELL" else -1) * leg["premium_est"] for leg in legs)

    if len(legs) == 1:  # a long call or put
        leg = legs[0]
        breakeven = leg["strike"] + leg["premium_est"] if leg["type"] == "CE" else leg["strike"] - leg["premium_est"]
        return {"net": -leg["premium_est"], "max_loss_pts": leg["premium_est"],
                "max_profit_pts": float("inf"), "breakevens": [breakeven]}

    if len(legs) == 2:  # a credit spread; the first leg is the short one
        short = legs[0]
        width = abs(legs[0]["strike"] - legs[1]["strike"])
        breakeven = short["strike"] - credit if short["type"] == "PE" else short["strike"] + credit
        return {"net": credit, "max_loss_pts": width - credit, "max_profit_pts": credit,
                "breakevens": [breakeven]}

    # iron condor: short put, long put, short call, long call
    width = max(abs(legs[0]["strike"] - legs[1]["strike"]), abs(legs[2]["strike"] - legs[3]["strike"]))
    return {"net": credit, "max_loss_pts": width - credit, "max_profit_pts": credit,
            "breakevens": [legs[0]["strike"] - credit, legs[2]["strike"] + credit]}


def index_view(data, bench: str = config.BENCHMARK, rf: float = config.RISK_FREE, nav: float = 0.0,
               q: float = config.INDEX_DIVIDEND_YIELD) -> dict:
    closes = data.prices[bench].dropna()
    t = technicals(closes)
    points, reasons = score(t)
    if points >= RULES["score_act"]:
        view = "BULLISH"
    elif points <= -RULES["score_act"]:
        view = "BEARISH"
    else:
        view = "NEUTRAL"
    strong = abs(points) >= RULES["score_strong"]

    realised = t["vol_20d"]
    iv, source = market_implied_vol(data, bench, rf, q)
    if iv is None:
        iv = realised * RULES["iv_premium_assumed"]
        source = (f"assumed ({RULES['iv_premium_assumed']:.2f} x realised vol, because there's no live "
                  f"NIFTY option close in your data)")
    else:
        source = f"market ({source})"
    ratio = iv / realised if realised > 0 else NAN
    if ratio >= RULES["iv_rich"]:
        regime = "RICH"
    elif ratio <= RULES["iv_cheap"]:
        regime = "CHEAP"
    else:
        regime = "FAIR"

    idea = build_strategy(view, strong, regime, t["last"], iv, closes.index[-1], rf, q, nav,
                          t["avg_move_pts"], bench)
    return {
        "symbol": bench, "score": points, "rating": rating(points),
        "view": ("STRONGLY " if strong else "") + view, "why": reasons,
        "spot": t["last"], "iv": iv, "iv_source": source, "rv20": realised, "iv_rv": ratio,
        "vol_regime": regime, "rsi14": t["rsi14"], "technicals": t, **idea,
    }
