"""Turning the numbers into decisions.

Three kinds of verdict, all simple rules on the thresholds in ``config.VERDICT``:

1. One verdict per analysis: returns, risk-adjusted return, luck, alpha, VaR, Greeks,
   concentration, stress, simulation, sizing and costs. Each comes with an action and the
   numbers behind it.
2. One verdict per open position: HOLD, BOOK PROFIT, CUT LOSS, ROLL, CLOSE and so on.
3. An overall headline with the few things to do first.

Stock signals and the NIFTY trade idea are in ``signals.py``. None of this is advice: see
``config.DISCLAIMER``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import config
from .instruments import future_symbol, lot_size, next_monthly_expiry, option_symbol
from .options import bs_price
from .signals import pct, rupees

RULES = config.VERDICT
CRASH = "Crash -12%, vol +100%"


@dataclass
class Verdict:
    area: str
    verdict: str
    tone: str  # good | warn | bad | neutral
    action: str
    why: list = field(default_factory=list)

    def as_row(self) -> dict:
        row = asdict(self)
        row["why"] = " ".join(self.why)
        return row


# --- One verdict per open position ----------------------------------------------------


def _equity_verdict(position, signals) -> dict:
    if position.symbol not in signals.index:
        return {"verdict": "HOLD", "tone": "neutral", "action": "Not enough price history for a signal.", "why": ""}
    s = signals.loc[position.symbol]
    if s["score"] >= RULES["score_act"]:
        tone = "good"
    elif s["score"] <= -RULES["score_act"]:
        tone = "bad"
    else:
        tone = "neutral"
    return {"verdict": s["action"], "tone": tone,
            "action": f"Score {s['score']:+d} ({s['rating']}). Stop {s['stop']:,.1f}.", "why": s["why"]}


def _future_verdict(position, inst, days_left, view, trailing_stop_pts, today) -> dict:
    is_long = position.lots > 0
    with_trend = (is_long and "BULLISH" in view) or (not is_long and "BEARISH" in view)
    against_trend = (is_long and "BEARISH" in view) or (not is_long and "BULLISH" in view)

    if days_left <= RULES["dte_roll"]:
        if with_trend:
            next_contract = future_symbol(inst.underlying, next_monthly_expiry(today, 7))
            verdict, action = "ROLL", f"Expires in {days_left} days. Roll to {next_contract}."
        else:
            verdict, action = "CLOSE", f"Expires in {days_left} days and the trend no longer supports it."
        tone = "warn"
    elif against_trend:
        verdict, tone = "CLOSE / REVERSE", "bad"
        action = f"You're {'long' if is_long else 'short'} but the NIFTY view is {view}."
    elif with_trend:
        verdict, tone = "HOLD", "good"
        action = f"The trend agrees. Trail a stop about {trailing_stop_pts:,.0f} points away."
    else:
        verdict, tone = "HOLD WITH TIGHT STOP", "neutral"
        action = "No clear trend: keep it only with a stop in place."
    return {"verdict": verdict, "tone": tone, "action": action,
            "why": f"Unrealised P&L {rupees(position.unrealised_pnl)}."}


def _long_option_verdict(position, inst, days_left, in_the_money, is_hedge, with_view, view, today):
    gain = position.unrealised_pct
    if days_left <= RULES["dte_close_long_otm"] and not in_the_money:
        if is_hedge:
            next_put = option_symbol(inst.underlying, next_monthly_expiry(today, 7), inst.strike, "P")
            return ("ROLL HEDGE", "warn",
                    f"Expires out of the money in {days_left} days. Let it go and buy {next_put} "
                    f"(or a strike about 5% below spot) to keep the stocks protected.")
        cost_to_sell = 2 * config.CHARGES["brokerage_flat_per_order"] * (1 + config.CHARGES["gst"])
        if position.last_price * inst.multiplier < cost_to_sell:
            return ("LET EXPIRE", "neutral",
                    "Almost worthless: selling it would cost more in charges than it brings back.")
        return "CLOSE", "warn", f"Out of the money with {days_left} days left, when decay is fastest. Sell what's left."
    if math.isfinite(gain) and gain >= RULES["long_take_profit"]:
        return "BOOK PROFIT", "good", f"Up {pct(gain, 0)}: sell at least half and trail the rest."
    if math.isfinite(gain) and gain <= RULES["long_stop"] and not is_hedge:
        return "CUT LOSS", "bad", f"Down {pct(-gain, 0)} of the premium and the move hasn't come. Exit."
    if is_hedge:
        return "HOLD (HEDGE)", "neutral", "Protects the stocks. Its job is to lose a little most months."
    if not with_view:
        kind = "call" if inst.option_type == "C" else "put"
        return "CLOSE", "bad", f"A long {kind} doesn't fit a {view} view."
    return "HOLD", "good", "The view still supports it. Get out if it loses half its value."


def _short_option_verdict(position, inst, days_left, in_the_money, with_view, view, theta_per_day):
    captured = position.unrealised_pct  # for a short option, the share of premium kept so far
    if math.isfinite(captured) and captured >= RULES["short_take_profit"]:
        return ("BUY BACK (BOOK PROFIT)", "good",
                f"{pct(captured, 0)} of the premium is already yours; the rest isn't worth the tail risk.")
    if math.isfinite(captured) and captured <= RULES["short_stop"]:
        return "STOP OUT", "bad", "The loss has reached the premium you collected. Buy it back before it grows."
    if in_the_money and days_left <= RULES["dte_close_long_otm"]:
        return ("ROLL OR CLOSE", "warn",
                f"In the money with {days_left} days left. Roll it further out or close before settlement.")
    if not with_view:
        kind = "put" if inst.option_type == "P" else "call"
        return "CLOSE", "bad", f"A short {kind} conflicts with a {view} view."
    return ("HOLD (THETA WORKING)", "good",
            f"Earning about {rupees(abs(theta_per_day))} a day in decay. Buy back once "
            f"{pct(RULES['short_take_profit'], 0)} is captured.")


def _option_verdict(position, inst, days_left, view, greeks_row, spot, has_stocks, today) -> dict:
    is_long = position.lots > 0
    is_call = inst.option_type == "C"
    in_the_money = spot > inst.strike if is_call else spot < inst.strike
    bullish, bearish = "BULLISH" in view, "BEARISH" in view
    if is_long:
        with_view = (is_call and bullish) or (not is_call and bearish)
    else:  # selling a put needs "not bearish", selling a call needs "not bullish"
        with_view = (not is_call and not bearish) or (is_call and not bullish)
    is_hedge = is_long and not is_call and has_stocks

    theta = float(greeks_row["theta_per_day"]) if greeks_row is not None else 0.0
    delta = float(greeks_row["delta"]) if greeks_row is not None else float("nan")
    if is_long:
        verdict, tone, action = _long_option_verdict(position, inst, days_left, in_the_money, is_hedge,
                                                     with_view, view, today)
    else:
        verdict, tone, action = _short_option_verdict(position, inst, days_left, in_the_money, with_view,
                                                      view, theta)
    why = f"P&L {pct(position.unrealised_pct, 0)} of premium, theta {rupees(theta)}/day, delta {delta:+.2f}."
    return {"verdict": verdict, "tone": tone, "action": action, "why": why}


def _describe_option(position, inst, days_left, spot) -> str:
    is_call = inst.option_type == "C"
    in_the_money = spot > inst.strike if is_call else spot < inst.strike
    root = "NIFTY" if inst.underlying == "NIFTY50" else inst.underlying
    side = "Long" if position.lots > 0 else "Short"
    return (f"{side} {abs(position.lots):g} lot {root} {inst.strike:g} {'CE' if is_call else 'PE'} "
            f"({'ITM' if in_the_money else 'OTM'}, {days_left}d)")


def position_verdicts(results: dict, data) -> pd.DataFrame:
    positions = results["positions"]
    columns = ["symbol", "position", "dte", "pnl_pct", "verdict", "tone", "action", "why"]
    if positions is None or positions.empty:
        return pd.DataFrame(columns=columns)

    view = results["index_view"]["view"]
    trailing_stop = 2.5 * results["index_view"]["technicals"]["avg_move_pts"]
    book = results["book"].set_index("symbol") if len(results["book"]) else pd.DataFrame()
    signals = results["signals"].set_index("symbol") if len(results["signals"]) else pd.DataFrame()
    has_stocks = (positions.asset_class == "equity").any()
    today = pd.Timestamp(results["as_of"])

    rows = []
    for position in positions.itertuples():
        inst = data.instruments[position.symbol]
        if inst.asset_class == "equity":
            row = {"position": f"{position.units:,.0f} shares", "dte": None,
                   **_equity_verdict(position, signals)}
        else:
            days_left = (inst.expiry - today).days
            if inst.asset_class == "future":
                description = f"{'Long' if position.lots > 0 else 'Short'} {abs(position.lots):g} lot {position.symbol}"
                verdict = _future_verdict(position, inst, days_left, view, trailing_stop, today)
            else:
                spot = float(data.prices[inst.underlying].ffill().iloc[-1])
                greeks_row = book.loc[position.symbol] if position.symbol in book.index else None
                description = _describe_option(position, inst, days_left, spot)
                verdict = _option_verdict(position, inst, days_left, view, greeks_row, spot, has_stocks, today)
            row = {"position": description, "dte": days_left, **verdict}
        rows.append({"symbol": position.symbol, "pnl_pct": position.unrealised_pct, **row})
    return pd.DataFrame(rows)[columns]


# --- One verdict per analysis ---------------------------------------------------------
# Each function reads the results and returns a Verdict (or None if it doesn't apply).


def _returns(r, data):
    s, rf = r["summary"], r["rf"]
    edge = s["cagr"] - s.get("bench_cagr", float("nan"))
    if edge > 0.01:
        return Verdict("Returns", "OUTPERFORMING NIFTY", "good",
                       f"Keep going: it beat simply owning NIFTY by {pct(edge)} a year.",
                       [f"CAGR {pct(s['cagr'])} vs NIFTY {pct(s['bench_cagr'])}; XIRR {pct(s['xirr'])}."])
    if s["cagr"] > rf:
        return Verdict("Returns", "LAGGING NIFTY", "warn",
                       "A NIFTY index fund would have done at least as well. The extra effort has to earn its keep.",
                       [f"CAGR {pct(s['cagr'])} vs NIFTY {pct(s['bench_cagr'])}, risk-free {pct(rf)}."])
    return Verdict("Returns", "BELOW RISK-FREE", "bad",
                   "T-bills would have beaten this. Stop adding money until the strategy is fixed.",
                   [f"CAGR {pct(s['cagr'])} vs risk-free {pct(rf)}."])


def _risk_adjusted(r, data):
    s = r["summary"]
    sharpe, bench_sharpe = s["sharpe"], s.get("bench_sharpe", float("nan"))
    drawdowns = f"Max drawdown {pct(s['max_drawdown'])} vs NIFTY {pct(s.get('bench_max_drawdown'))}."
    sharpes = f"Sharpe {sharpe:.2f} (NIFTY {bench_sharpe:.2f}), Sortino {s['sortino']:.2f}."
    if sharpe >= RULES["sharpe_strong"]:
        return Verdict("Risk-adjusted return", "STRONG", "good",
                       "Risk is being paid well. Keep the sizing discipline.", [sharpes, drawdowns])
    if sharpe >= RULES["sharpe_ok"]:
        return Verdict("Risk-adjusted return", "ACCEPTABLE", "neutral",
                       "Decent, not exceptional. Look for the sleeve that drags (see Attribution).",
                       [sharpes, drawdowns])
    if sharpe > bench_sharpe:
        return Verdict("Risk-adjusted return", "WEAK, BUT BEATS NIFTY", "warn",
                       "Better than the index per unit of risk, but still below a 0.5 Sharpe. "
                       "Find the sleeve that drags.", [sharpes, drawdowns])
    return Verdict("Risk-adjusted return", "WEAK", "bad",
                   "Too much risk for the return. Cut the weakest sleeve or reduce size.", [sharpes, drawdowns])


def _luck(r, data):
    s = r["summary"]
    interval = f"95% bootstrap Sharpe interval [{s['sharpe_ci_low']:.2f}, {s['sharpe_ci_high']:.2f}]."
    if s["dsr"] >= RULES["dsr_skill"]:
        return Verdict("Statistical validity", "SKILL LIKELY", "good",
                       "The record survives the luck test. Scaling up is defensible.",
                       [f"Deflated Sharpe {s['dsr']:.2f} after {s['n_trials']} trial(s).", interval])
    if s["dsr"] >= RULES["dsr_inconclusive"]:
        years_needed = s["min_track_record_years"]
        need = (f" You'd need about {years_needed:.1f} years of record; you have {s['years']:.1f}."
                if math.isfinite(years_needed) else "")
        return Verdict("Statistical validity", "INCONCLUSIVE", "warn",
                       "Could still be luck. Don't scale up yet; keep the same size and let the record build." + need,
                       [f"Deflated Sharpe {s['dsr']:.2f}, PSR {s['psr']:.2f}.", interval])
    return Verdict("Statistical validity", "COULD BE LUCK", "bad",
                   "No statistical evidence of skill yet. Treat the profits so far as luck until proven otherwise.",
                   [f"Deflated Sharpe {s['dsr']:.2f}, PSR {s['psr']:.2f}.", interval])


def _alpha(r, data):
    c = r["capm"]
    why = [f"Alpha {pct(c['alpha_annual'])} a year (Newey-West t {c['alpha_t_nw']:.2f}), beta {c['beta']:.2f}, "
           f"R-squared {c['r2']:.2f}, up/down capture {c['up_capture']:.2f}/{c['down_capture']:.2f}."]
    if c["alpha_t_nw"] >= RULES["alpha_t_sig"]:
        return Verdict("Alpha (CAPM)", "GENUINE ALPHA", "good",
                       "The returns aren't just market exposure. Protect whatever is producing them.", why)
    if c["alpha_t_nw"] <= -RULES["alpha_t_sig"]:
        return Verdict("Alpha (CAPM)", "NEGATIVE ALPHA", "bad",
                       "The active decisions are losing money. A NIFTY future or ETF would give the same beta.", why)
    if c["beta"] >= RULES["beta_high"]:
        return Verdict("Alpha (CAPM)", "LEVERAGED BETA", "warn",
                       "This is mostly a geared bet on NIFTY. Futures would give the same exposure more cheaply.", why)
    return Verdict("Alpha (CAPM)", "NO PROVEN ALPHA", "neutral",
                   "Alpha isn't statistically different from zero. Keep costs low; most of the return is beta.", why)


def _value_at_risk(r, data):
    historical = r["var"].loc["Historical", "99% (of NAV)"]
    cornish_fisher = r["var"].loc["Cornish-Fisher", "99% (of NAV)"]
    backtest = r["kupiec"].get(0.99, {})
    why = [f"1-day 99% VaR {pct(historical, 2)} ({rupees(historical * r['nav_last'])}), "
           f"Cornish-Fisher {pct(cornish_fisher, 2)}.",
           f"Kupiec backtest: {backtest.get('breaches', 'n/a')} breaches vs "
           f"{backtest.get('expected', float('nan')):.1f} expected, {backtest.get('result', 'n/a')}."]
    worst = max(historical, cornish_fisher)
    limit = RULES["var99_limit"]
    if worst > limit:
        return Verdict("Value at Risk", "ABOVE LIMIT", "bad",
                       f"Cut gross exposure by about {pct(1 - limit / worst, 0)} to get 99% VaR under "
                       f"{pct(limit, 0)} of NAV.", why)
    if backtest.get("result") == "FAIL":
        return Verdict("Value at Risk", "MODEL UNRELIABLE", "warn",
                       "VaR is breached more (or less) often than it should be. Size off Expected Shortfall instead.",
                       why)
    return Verdict("Value at Risk", "WITHIN LIMIT", "good",
                   "Daily risk is inside the limit and the model backtests well.", why)


def _greeks(r, data):
    g = r["greeks"]
    view = r["index_view"]["view"]
    net = g["net_delta_pct_nav"]
    why = [f"Net NIFTY delta {rupees(g['net_index_delta'])} ({pct(net, 0)} of NAV), theta "
           f"{rupees(g['theta_per_day'])}/day, vega {rupees(g['vega_per_volpt'])} per vol point, "
           f"gamma {rupees(g['gamma_per_1pct'])} per 1% move."]
    direction = "NET LONG" if net > 0.25 else "NET SHORT" if net < -0.25 else "NEAR NEUTRAL"
    if (direction == "NET LONG" and "BEARISH" in view) or (direction == "NET SHORT" and "BULLISH" in view):
        return Verdict("Exposure & Greeks", f"{direction} vs {view} VIEW", "bad",
                       "The book points the opposite way to the trend. Bring delta towards neutral "
                       "(sell futures or buy puts).", why)
    if g["theta_per_day"] < 0 and abs(g["theta_pct_nav"]) > 0.001:
        return Verdict("Exposure & Greeks", f"{direction}, HIGH THETA BLEED", "warn",
                       f"Long options cost {pct(-g['theta_pct_nav'] * 21, 1)} of NAV a month in decay. "
                       f"Spreads would cut that.", why)
    if direction == "NEAR NEUTRAL":
        return Verdict("Exposure & Greeks", direction, "good",
                       "The book is close to delta-neutral: a NIFTY move either way changes little.", why)
    return Verdict("Exposure & Greeks", direction, "neutral",
                   f"Exposure is consistent with a {view} view.", why)


def _concentration(r, data):
    sleeves = r["euler_sleeves"]
    if not len(sleeves):
        return None
    biggest = sleeves["share_of_risk"].idxmax()
    share = sleeves.loc[biggest, "share_of_risk"]
    hedges = [name for name, value in sleeves["vol_contribution_daily"].items() if value < 0]
    why = [f"Largest driver: {biggest}, with {pct(share, 0)} of portfolio volatility."]
    if hedges:
        why.append(f"Hedging the rest: {', '.join(hedges)}.")
    if share > RULES["max_risk_share"]:
        return Verdict("Risk concentration", "CONCENTRATED", "warn",
                       f"Trim {biggest}. No single sleeve should drive more than {pct(RULES['max_risk_share'], 0)} "
                       f"of the risk. Risk-parity weights are in the Sizing section.", why)
    return Verdict("Risk concentration", "DIVERSIFIED", "good", "No single sleeve dominates the risk.", why)


def _hedge_suggestion(r, q=config.INDEX_DIVIDEND_YIELD) -> str:
    """How many ~5% out-of-the-money puts would halve the loss in the crash scenario."""
    nav, scenarios = r["nav_last"], r["scenarios"]
    crash_loss = -scenarios.loc[CRASH, "portfolio_pnl"] if CRASH in scenarios.index else 0.0
    to_cover = crash_loss - RULES["stress_loss_limit"] * nav * 0.5
    if to_cover <= 0:
        return ""
    view, rf = r["index_view"], r["rf"]
    spot, iv = view["spot"], view["iv"]
    expiry = next_monthly_expiry(r["as_of"], 14)
    T = (pd.Timestamp(expiry) - pd.Timestamp(r["as_of"])).days / 365
    step = config.STRIKE_STEP.get(config.BENCHMARK, 50)
    strike = round(spot * 0.95 / step) * step
    lot = lot_size(config.BENCHMARK, expiry)
    cost_now = float(bs_price(spot, strike, T, rf, iv, q, "P"))
    value_in_crash = float(bs_price(spot * 0.88, strike, T, rf, iv * 2, q, "P"))
    gain_per_lot = (value_in_crash - cost_now) * lot
    lots = math.ceil(to_cover / gain_per_lot) if gain_per_lot > 0 else 0
    return (f"Buy {lots} lot(s) of {option_symbol(config.BENCHMARK, expiry, strike, 'P')} (about 5% out of "
            f"the money, roughly {cost_now:,.0f} points = {rupees(cost_now * lot)} a lot) to halve the crash loss.")


def _stress(r, data):
    grid, scenarios = r["stress_grid"], r["scenarios"]
    worst = float(np.nanmin(grid.to_numpy())) if grid.size else 0.0
    why = [f"Worst cell of the NIFTY x volatility grid: {pct(worst)} of NAV."]
    if CRASH in scenarios.index and math.isfinite(scenarios.loc[CRASH, "hedge_offset"]):
        offset = scenarios.loc[CRASH, "hedge_offset"]
        if offset >= 0:
            why.append(f"In a 12% crash the derivatives win back {pct(offset, 0)} of the stock loss.")
        else:
            why.append(f"In a 12% crash the derivatives add {pct(-offset, 0)} to the stock loss (they're net long).")

    limit = RULES["stress_loss_limit"]
    if worst < -limit:
        action = _hedge_suggestion(r) or f"A crash would cost more than {pct(limit, 0)} of NAV. Reduce exposure."
        return Verdict("Stress test", "HEDGE NEEDED", "bad", action, why)
    if worst < -limit / 2:
        return Verdict("Stress test", "MODERATE TAIL RISK", "warn",
                       "Survivable, but keep or add protective puts ahead of event risk.", why)
    return Verdict("Stress test", "WELL PROTECTED", "good", "Crash scenarios stay well inside the loss limit.", why)


def _simulation(r, data):
    mc = r["montecarlo"]
    why = [f"{mc['n_paths']:,} resampled one-year paths: P(loss) {pct(mc['prob_loss'], 0)}, median "
           f"{pct(mc['terminal_pct'][50])}, 5th percentile {pct(mc['terminal_pct'][5])}, "
           f"P(drawdown over 20%) {pct(mc['prob_dd_20'], 0)}."]
    if mc["prob_loss"] > RULES["prob_loss_limit"]:
        return Verdict("Forward simulation", "HIGH CHANCE OF A LOSING YEAR", "warn",
                       "Losing years will be common. Size so that the 5th-percentile year is one you can live with.",
                       why)
    return Verdict("Forward simulation", "FAVOURABLE ODDS", "good",
                   "If the past is any guide, most one-year paths end positive.", why)


def _sizing(r, data):
    sz = r["sizing"]
    vol_ratio = sz["ewma_vol_forecast"] / sz["vol_target"]
    half_kelly = sz["half_kelly_leverage"]
    why = [f"Forecast vol {pct(sz['ewma_vol_forecast'])} vs target {pct(sz['vol_target'], 0)}; gross exposure "
           f"{sz['current_gross_leverage']:.2f}x NAV; half-Kelly suggests {half_kelly:.2f}x."]
    if vol_ratio > RULES["oversize_ratio"]:
        return Verdict("Position sizing", "OVERSIZED", "bad",
                       f"Scale every position by {sz['vol_target_scale']:.2f}x (a {pct(1 - sz['vol_target_scale'], 0)} "
                       f"cut) to hit the volatility target.", why)
    if vol_ratio < RULES["undersize_ratio"] and math.isfinite(half_kelly) and half_kelly > sz["current_gross_leverage"]:
        return Verdict("Position sizing", "UNDERSIZED", "neutral",
                       f"There's room to add: scaling up {min(sz['vol_target_scale'], 2):.2f}x would still "
                       f"respect the volatility target.", why)
    return Verdict("Position sizing", "SIZED RIGHT", "good",
                   "Volatility is close to target. Keep the current size.", why)


def _costs(r, data):
    t = r["trading"]
    share = t.get("cost_share_of_gross", float("nan"))
    why = [f"{t['trades']} trades, {rupees(t['fees_total'])} in charges "
           f"({pct(t.get('fees_pct_nav_annual', 0), 2)} of NAV a year), "
           f"turnover {t.get('turnover_annual', 0):.1f}x a year."]
    if not math.isfinite(share):
        return Verdict("Costs & turnover", "LOSING BEFORE COSTS", "bad",
                       "The strategy loses money even before charges, so cutting costs alone won't fix it.", why)
    if share > RULES["cost_share_limit"]:
        return Verdict("Costs & turnover", "OVERTRADING", "bad",
                       f"Charges eat {pct(share, 0)} of the gross profit. Trade less often, in fewer, larger orders.",
                       why)
    return Verdict("Costs & turnover", "COSTS UNDER CONTROL", "good",
                   f"Charges take {pct(share, 0)} of gross profit.", why)


def _market_view(r, data):
    idea = r["index_view"]
    legs = ", ".join(f"{leg['action']} {leg['symbol']}" for leg in idea.get("legs", [])) or "no trade"
    return Verdict("NIFTY view & strategy", f"{idea['view']}: {idea['strategy']}", "neutral",
                   f"{idea['rationale']} Legs: {legs}.",
                   [f"Score {idea['score']:+d}; implied vol {pct(idea['iv'])} vs 20-day realised "
                    f"{pct(idea['rv20'])} ({idea['vol_regime']}).", "; ".join(idea["why"]) + "."])


SECTIONS = [_returns, _risk_adjusted, _luck, _alpha, _value_at_risk, _greeks, _concentration,
            _stress, _simulation, _sizing, _costs, _market_view]


def section_verdicts(results: dict, data=None) -> list[Verdict]:
    return [v for section in SECTIONS if (v := section(results, data)) is not None]


# --- The headline ---------------------------------------------------------------------

PERFORMANCE_AREAS = ("Returns", "Risk-adjusted return", "Statistical validity")
RISK_AREAS = {"Value at Risk", "Stress test", "Position sizing", "Risk concentration", "Exposure & Greeks"}


def overall_verdict(sections: list[Verdict]) -> dict:
    by_area = {v.area: v for v in sections}
    bad = [v for v in sections if v.tone == "bad"]
    warnings = [v for v in sections if v.tone == "warn"]
    performance_bad = sum(by_area[a].tone == "bad" for a in PERFORMANCE_AREAS if a in by_area)
    returns_bad = "Returns" in by_area and by_area["Returns"].tone == "bad"

    if performance_bad >= 2 and returns_bad:
        headline, tone = "RETHINK THE STRATEGY", "bad"
    elif any(v.area in RISK_AREAS for v in bad):
        headline, tone = "KEEP THE STRATEGY, CUT RISK NOW", "bad"
    elif bad or len(warnings) >= 3:
        headline, tone = "HOLD COURSE, FIX THE FLAGS", "warn"
    else:
        headline, tone = "ON TRACK", "good"

    return {
        "headline": headline,
        "tone": tone,
        "actions": [f"{v.area}: {v.action}" for v in bad + warnings][:4],
        "counts": {t: sum(v.tone == t for v in sections) for t in ("good", "neutral", "warn", "bad")},
    }


def verdicts_frame(sections: list[Verdict]) -> pd.DataFrame:
    return pd.DataFrame([v.as_row() for v in sections])
