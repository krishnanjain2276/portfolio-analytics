"""The HTML tearsheet: one self-contained page, verdicts first, details after.

Each ``_section_*`` function returns one block of HTML. They're kept separate so the order of
the report is easy to see (and change) in ``to_html`` at the bottom.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, config

TONE_COLOUR = {"good": "#1e7b4f", "warn": "#b06d10", "bad": "#b83227", "neutral": "#52606d"}
TONE_BACKGROUND = {"good": "#e3f4ea", "warn": "#fdf1dc", "bad": "#fbe5e3", "neutral": "#eef1f4"}

# Formatting helpers


def esc(x) -> str:
    return html.escape(str(x))


def pct(x, decimals=1) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return esc(x)
    return f"{x * 100:.{decimals}f}%" if math.isfinite(x) else "–"


def num(x, decimals=2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if math.isnan(x):
        return "–"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    return f"{x:,.{decimals}f}"


def rupees(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return esc(x)
    if not math.isfinite(x):
        return "–"
    return f"₹{x:,.0f}" if x >= 0 else f"−₹{abs(x):,.0f}"


def whole(x) -> str:
    return num(x, 0)


def badge(text, tone) -> str:
    return (f'<span class="badge" style="color:{TONE_COLOUR[tone]};background:{TONE_BACKGROUND[tone]}">'
            f"{esc(text)}</span>")


# Friendlier column headings for the raw DataFrame columns that end up in tables.
HEADINGS = {
    "pnl": "P&L", "portfolio_pnl": "Portfolio P&L", "equity_only_pnl": "Stocks only",
    "derivatives_pnl": "Derivatives", "pct_nav": "% of NAV", "hedge_offset": "Derivatives offset",
    "iv": "Implied vol", "index_exposure": "NIFTY-equivalent exposure", "delta_per_1pct": "Delta per 1%",
    "gamma_per_1pct": "Gamma per 1%", "vega_per_volpt": "Vega per vol pt", "theta_per_day": "Theta per day",
    "vol_contribution_daily": "Daily vol contribution", "share_of_risk": "Share of risk",
    "days_to_trough": "Days to trough", "days_to_recover": "Days to recover", "total_days": "Total days",
    "LR": "LR stat", "T": "Days tested", "p_value": "p-value", "rate": "Breach rate",
}


def heading(column) -> str:
    column = str(column)
    if column in HEADINGS:
        return HEADINGS[column]
    return column.replace("_", " ").capitalize() if column.islower() else column


def _looks_like_text(values) -> bool:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return bool(values) and sum(isinstance(v, str) for v in values) / len(values) > 0.5


def table(df: pd.DataFrame, formats: dict | None = None, index=True, note_column: str | None = None,
          wide=()) -> str:
    """A DataFrame as an HTML table.

    ``formats`` maps a column to a formatting function. ``note_column`` is shown as a grey
    line under each row instead of as a column, which keeps long explanations readable.
    """
    formats = formats or {}
    columns = [c for c in df.columns if c != note_column]
    text_column = {c: _looks_like_text(df[c].tolist()) for c in columns}

    head = f"<th>{esc(heading(df.index.name or ''))}</th>" if index else ""
    head += "".join(f"<th class='{'l' if text_column[c] else ''}'>{esc(heading(c))}</th>" for c in columns)

    body = []
    for row_label, row in df.iterrows():
        cells = f"<td class='l'>{esc(row_label)}</td>" if index else ""
        for c in columns:
            value = row[c]
            if c in formats:
                shown = formats[c](value)
            elif isinstance(value, (float, np.floating)):
                shown = num(value)
            elif value is None or (isinstance(value, float) and math.isnan(value)):
                shown = ""
            else:
                shown = esc(value)
            style = " style='min-width:240px'" if c in wide else ""
            cells += f"<td class='{'l' if text_column[c] else ''}'{style}>{shown}</td>"
        note = str(row[note_column]).strip() if note_column else ""
        if note:
            body.append(f"<tr class='has-note'>{cells}</tr>"
                        f"<tr class='note-row'><td colspan='{len(columns) + int(index)}'>{esc(note)}</td></tr>")
        else:
            body.append(f"<tr>{cells}</tr>")
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def heat_table(df: pd.DataFrame, low: float, high: float) -> str:
    def shade(v):
        if not isinstance(v, (int, float, np.floating)) or not math.isfinite(v):
            return ""
        strength = max(min(v / (high if v > 0 else -low), 1), -1)
        if strength >= 0:
            return f"background:rgba(30,123,79,{0.08 + 0.45 * strength:.2f})"
        return f"background:rgba(184,50,39,{0.08 + 0.45 * -strength:.2f})"

    head = f"<th>{esc(df.index.name or '')}</th>" + "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for label, row in df.iterrows():
        cells = "".join(f"<td style='{shade(v)}'>{pct(v) if pd.notna(v) else ''}</td>" for v in row.values)
        rows.append(f"<tr><td class='l'>{esc(label)}</td>{cells}</tr>")
    body = "".join(rows)
    return f"<div class='tw'><table class='heat'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def kpis(items) -> str:
    tiles = "".join(f"<div class='kpi'><div class='k'>{esc(k)}</div><div class='val'>{v}</div></div>"
                    for k, v in items)
    return f"<div class='kpis'>{tiles}</div>"


def section(anchor: str, title: str, body: str) -> str:
    return f"<h2 id='{anchor}'>{esc(title)}</h2>{body}"


# Sections


def _section_verdict(r) -> str:
    overall = r["overall"]
    colour = TONE_COLOUR[overall["tone"]]
    actions = "".join(f"<li>{esc(a)}</li>" for a in overall["actions"]) or "<li>Nothing flagged.</li>"
    counts = " · ".join(badge(f"{n} {tone}", tone) for tone, n in overall["counts"].items() if n)
    cards = "".join(
        f"<div class='v' style='border-left-color:{TONE_COLOUR[v.tone]}'>"
        f"<div class='area'>{esc(v.area)}</div>"
        f"<div class='verdict' style='color:{TONE_COLOUR[v.tone]}'>{esc(v.verdict)}</div>"
        f"<div class='act'>{esc(v.action)}</div><div class='why'>{esc(' '.join(v.why))}</div></div>"
        for v in r["section_verdicts"]
    )
    hero = (f"<div class='card hero' style='border-color:{colour}'>"
            f"<div class='head' style='color:{colour}'>{esc(overall['headline'])}</div>"
            f"<div style='margin:6px 0 4px'>{counts}</div><b>Do first:</b><ul>{actions}</ul></div>")
    return section("verdict", "Verdict", hero + f"<div class='grid'>{cards}</div>")


def _section_trade_idea(r) -> str:
    idea = r["index_view"]
    view_tone = "good" if "BULLISH" in idea["view"] else "bad" if "BEARISH" in idea["view"] else "neutral"
    tiles = kpis([
        ("View", badge(idea["view"], view_tone)), ("Trend score", f"{idea['score']:+d}"),
        ("Spot", num(idea["spot"], 1)), ("Implied vol", pct(idea["iv"])),
        ("20-day realised vol", pct(idea["rv20"])), ("Options look", badge(idea["vol_regime"], "neutral")),
    ])

    details = ""
    if idea.get("legs"):
        legs = pd.DataFrame(idea["legs"])[["action", "symbol", "type", "strike", "premium_est", "lot"]]
        legs.columns = ["Action", "Contract", "Type", "Strike", "Est. premium (pts)", "Lot size"]
        details = table(legs, {"Strike": whole, "Est. premium (pts)": lambda v: num(v, 1), "Lot size": whole},
                        index=False)
        net = idea["net_per_lot"]
        max_profit = idea["max_profit_pts"]
        max_profit_text = "unlimited" if not math.isfinite(max_profit) else rupees(max_profit * idea["lot_size"])
        breakevens = ", ".join(whole(b) for b in idea.get("breakevens", []))
        details += (f"<p>Net {'credit' if net > 0 else 'debit'} {rupees(abs(net))} a lot · max loss "
                    f"{rupees(idea['max_loss_per_lot'])} a lot · max profit {max_profit_text} a lot · "
                    f"breakeven {breakevens} · <b>{idea['lots']} lot(s)</b> fit a "
                    f"{pct(config.RISK_PER_TRADE, 0)}-of-NAV loss budget.</p>")
    futures = idea.get("futures_alt")
    if futures:
        details += (f"<p>Or: <b>{futures['action']} {futures['symbol']}</b> with a stop {futures['stop_pts']:,} "
                    f"points away, {futures['lots']} lot(s) within the risk budget.</p>")

    card = (f"<div class='card'><div style='font-size:17px;font-weight:700'>{esc(idea['strategy'])}</div>"
            f"<p>{esc(idea['rationale'])} Expiry {idea['expiry']} ({idea['dte']} days).</p>{details}"
            f"<p class='note'>Why: {esc('; '.join(idea['why']))}. Implied vol source: {esc(idea['iv_source'])}. "
            f"Premiums are Black-Scholes estimates at that vol; check live quotes before trading.</p></div>")
    return section("trade-idea", "NIFTY view and trade idea", tiles + card)


def _section_positions(r) -> str:
    verdicts = r["position_verdicts"]
    if not len(verdicts):
        return section("positions", "Verdict on every open position", "<p>No open positions.</p>")
    shown = verdicts.copy()
    shown["verdict"] = [badge(v, tone) for v, tone in zip(shown["verdict"], shown["tone"], strict=True)]
    shown = shown[["symbol", "position", "dte", "pnl_pct", "verdict", "action", "why"]]
    shown.columns = ["Symbol", "Position", "Days to expiry", "P&L %", "Verdict", "Action", "Why"]
    days = lambda v: "" if v is None or pd.isna(v) else f"{int(v)}"  # noqa: E731
    body = table(shown, {"P&L %": pct, "Verdict": str, "Days to expiry": days}, index=False,
                 note_column="Why", wide=("Action",))
    return section("positions", "Verdict on every open position", body)


def _section_stock_signals(r) -> str:
    signals = r["signals"]
    if not len(signals):
        return ""
    shown = signals.copy()

    def tone(points):
        if points >= config.VERDICT["score_act"]:
            return "good"
        return "bad" if points <= -config.VERDICT["score_act"] else "neutral"

    shown["action"] = [badge(a, tone(p)) for a, p in zip(shown["action"], shown["score"], strict=True)]
    shown = shown[["symbol", "sector", "held_units", "last", "score", "rating", "action", "stop", "target",
                   "suggested_qty_1pct_risk", "rsi14", "mom_3m", "off_52w_high", "why"]]
    shown.columns = ["Symbol", "Sector", "Held", "Last", "Score", "Rating", "Verdict", "Stop", "Target",
                     "Qty (1% risk)", "RSI", "3M", "From 52w high", "Why"]
    one_dp = lambda v: num(v, 1)  # noqa: E731
    body = table(shown, {"Held": whole, "Score": lambda v: f"{int(v):+d}", "Verdict": str,
                         "Qty (1% risk)": whole, "RSI": whole, "3M": pct, "From 52w high": pct,
                         "Last": one_dp, "Stop": one_dp, "Target": one_dp}, index=False, note_column="Why")
    body += ("<p class='note'>Score = trend (20/50/200-day averages) + 12-1 momentum + 3-month strength vs "
             "NIFTY + RSI timing, minus penalties for deep drawdowns and extreme volatility. The stop is 2.5 "
             f"average daily moves away; the quantity loses {pct(config.RISK_PER_TRADE, 0)} of NAV if the stop "
             "is hit (capped at 20% of NAV in one stock). Add any stock to the Watchlist to get a verdict on it.</p>")
    return section("stocks", "Stock signals (held and watch-list)", body)


def _section_performance(r) -> str:
    s = r["summary"]
    tiles = kpis([
        ("NAV", rupees(r["nav_last"])), ("Total P&L", rupees(s["total_pnl"])), ("CAGR (TWR)", pct(s["cagr"])),
        ("XIRR", pct(s["xirr"])), ("Volatility", pct(s["vol"])), ("Sharpe", num(s["sharpe"])),
        ("Sortino", num(s["sortino"])), ("Max drawdown", pct(s["max_drawdown"])),
        ("Calmar", num(s["calmar"])), ("Omega", num(s["omega"])),
    ])
    comparison = pd.DataFrame({
        "CAGR": [s["cagr"], s.get("bench_cagr")],
        "Volatility": [s["vol"], s.get("bench_vol")],
        "Sharpe": [s["sharpe"], s.get("bench_sharpe")],
        "Max drawdown": [s["max_drawdown"], s.get("bench_max_drawdown")],
    }, index=["Portfolio", r["benchmark"]])
    body = tiles + charts.growth_and_drawdown(r)
    body += table(comparison, {"CAGR": pct, "Volatility": pct, "Max drawdown": pct})
    body += "<h3>Monthly returns</h3>" + heat_table(r["monthly"], -0.05, 0.05)
    if len(r["drawdowns"]):
        drawdowns = table(r["drawdowns"], {"depth": pct, "days_to_recover": whole}, index=False)
        body += "<h3>Worst drawdowns</h3>" + drawdowns
    return section("performance", "Performance", body)


def _section_luck(r) -> str:
    s = r["summary"]
    body = kpis([
        ("PSR (Sharpe > 0)", pct(s["psr"], 0)), ("Deflated Sharpe", pct(s["dsr"], 0)),
        ("Strategies tried", s["n_trials"]), ("Hurdle Sharpe", num(s["dsr_benchmark_sharpe"])),
        ("Track record needed", f"{num(s['min_track_record_years'], 1)} yrs"),
        ("Sharpe 95% interval", f"{num(s['sharpe_ci_low'])} to {num(s['sharpe_ci_high'])}"),
        ("Skew", num(s["skew"])), ("Kurtosis", num(s["kurtosis"])),
    ])
    body += ("<p class='note'>PSR is the probability that the true Sharpe is above zero, allowing for skew and "
             "fat tails. The Deflated Sharpe asks the same question against the best Sharpe you'd expect by luck "
             "from the number of strategy variants you tried (the --trials option).</p>")
    return section("luck", "Could this be luck?", body)


def _section_capm(r) -> str:
    c = r["capm"]
    rows = [
        ("Alpha (annual)", pct(c["alpha_annual"], 2)), ("Alpha t-stat (Newey-West)", num(c["alpha_t_nw"])),
        ("Alpha t-stat (plain OLS, for comparison)", num(c["alpha_t_ols"])),
        ("Alpha p-value (Newey-West)", num(c["alpha_p_nw"], 3)), ("Beta", num(c["beta"])),
        ("Beta standard error (Newey-West)", num(c["beta_se_nw"], 3)), ("R-squared", num(c["r2"])),
        ("Tracking error", pct(c["tracking_error"])), ("Information ratio", num(c["information_ratio"])),
        ("Treynor ratio", pct(c["treynor"])), ("Appraisal ratio", num(c["appraisal_ratio"])),
        ("Up-market capture", num(c["up_capture"])), ("Down-market capture", num(c["down_capture"])),
        ("Newey-West lag (days)", c["nw_lag"]),
    ]
    metrics = pd.DataFrame(rows, columns=["Metric", "Value"]).set_index("Metric")
    return section("capm", "CAPM (Newey-West)", table(metrics, {"Value": str}) + charts.rolling_beta(r))


def _section_var(r) -> str:
    var = r["var"].copy()
    var.index.name = "Method"
    body = table(var, {col: (pct if "NAV" in col else rupees) for col in var.columns})
    kupiec = pd.DataFrame(r["kupiec"]).T
    kupiec.index = [f"{int(level * 100)}% VaR" for level in kupiec.index]
    kupiec.index.name = "Kupiec backtest"
    body += table(kupiec, {"rate": lambda v: pct(v, 2), "p_value": lambda v: num(v, 3), "LR": num,
                           "expected": lambda v: num(v, 1), "T": whole, "breaches": whole})
    return section("var", "Value at Risk and backtest", body + charts.return_distribution(r))


def _section_book(r) -> str:
    g = r["greeks"]
    body = kpis([
        ("Net NIFTY delta", rupees(g["net_index_delta"])), ("Delta / NAV", pct(g["net_delta_pct_nav"], 0)),
        ("Gamma per 1% move", rupees(g["gamma_per_1pct"])), ("Vega per vol point", rupees(g["vega_per_volpt"])),
        ("Theta per day", rupees(g["theta_per_day"])),
    ])
    if len(r["book"]):
        book = r["book"][["symbol", "sleeve", "units", "price", "iv", "delta", "index_exposure",
                          "delta_per_1pct", "gamma_per_1pct", "vega_per_volpt", "theta_per_day"]]
        body += table(book, {"units": whole, "price": num, "iv": pct, "delta": lambda v: num(v, 3),
                             "index_exposure": rupees, "delta_per_1pct": rupees, "gamma_per_1pct": rupees,
                             "vega_per_volpt": rupees, "theta_per_day": rupees}, index=False)
    body += "<h3>Where the risk comes from</h3>" + charts.risk_by_sleeve(r)
    if len(r["euler_sleeves"]):
        sleeves = r["euler_sleeves"].copy()
        sleeves.index.name = "Sleeve"
        body += table(sleeves, {"exposure": rupees, "vol_contribution_daily": rupees, "share_of_risk": pct})
    return section("book", "Live book and Greeks", body)


def _section_stress(r) -> str:
    body = ("<p class='note'>P&L as a share of NAV for an instant NIFTY move (rows) and a change in implied "
            "volatility (columns). Options are fully repriced, not approximated with delta.</p>")
    body += heat_table(r["stress_grid"], -0.15, 0.15)
    scenarios = r["scenarios"].copy()
    scenarios.index.name = "Scenario"
    body += table(scenarios, {"portfolio_pnl": rupees, "equity_only_pnl": rupees, "derivatives_pnl": rupees,
                              "pct_nav": pct, "hedge_offset": lambda v: pct(v, 0)})
    return section("stress", "Stress test", body)


def _section_simulation(r) -> str:
    mc = r["montecarlo"]
    body = kpis([
        ("P(losing year)", pct(mc["prob_loss"], 0)), ("P(loss over 10%)", pct(mc["prob_loss_10"], 0)),
        ("P(drawdown over 20%)", pct(mc["prob_dd_20"], 0)), ("Median year", pct(mc["terminal_pct"][50])),
        ("5th percentile year", pct(mc["terminal_pct"][5])), ("5th percentile max DD", pct(mc["mdd_pct"][5])),
    ])
    return section("simulation", "Forward simulation", body + charts.simulation(r))


def _section_sizing(r) -> str:
    sz = r["sizing"]
    body = kpis([
        ("Forecast vol (EWMA)", pct(sz["ewma_vol_forecast"])), ("Vol target", pct(sz["vol_target"], 0)),
        ("Vol-target scale", f"{num(sz['vol_target_scale'])}x"),
        ("Half-Kelly leverage", f"{num(sz['half_kelly_leverage'])}x"),
        ("Gross exposure now", f"{num(sz['current_gross_leverage'])}x"),
    ])
    if "risk_parity" in sz:
        weights = sz["risk_parity"].copy()
        weights.index.name = "Stock"
        body += ("<h3>Risk-parity weights for the stocks you hold</h3>"
                 + table(weights, {"weight": pct, "risk_contribution": pct}))
    return section("sizing", "Sizing", body)


def _section_attribution(r) -> str:
    t = r["trading"]
    body = kpis([
        ("Trades", t["trades"]), ("Charges paid", rupees(t["fees_total"])),
        ("Charges / NAV / year", pct(t.get("fees_pct_nav_annual", 0), 2)),
        ("Turnover / year", f"{num(t.get('turnover_annual', 0), 1)}x"),
        ("Charges / gross P&L", pct(t.get("cost_share_of_gross"), 0)),
    ])
    body += charts.pnl_by_sleeve(r)
    monthly = r["attr_monthly"].tail(18).copy()
    monthly.index.name = "Month"
    body += "<h3>Monthly P&L by asset class (last 18 months)</h3>"
    body += table(monthly, dict.fromkeys(monthly.columns, rupees))
    instruments = r["attr_instrument"].copy()
    instruments.index.name = "Instrument"
    body += "<h3>Best and worst instruments</h3>" + table(instruments, {"pnl": rupees})
    return section("attribution", "Attribution", body)


def _section_notes(r) -> str:
    notes = list(r["notes"])
    if r.get("modelled"):
        listed = ", ".join(r["modelled"][:30]) + (" …" if len(r["modelled"]) > 30 else "")
        notes.append(f"Contracts with modelled closes: {listed}")
    notes.append(config.DISCLAIMER)
    return section("notes", "Notes", "<ul class='note'>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>")


# The page

CSS = """
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:#f5f7f9;color:#1f2933;
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid #e4e7eb;scroll-margin-top:56px}
h3{font-size:14px;margin:18px 0 8px}
.sub{color:#7b8794}
nav{position:sticky;top:0;z-index:2;background:rgba(245,247,249,.95);border-bottom:1px solid #e4e7eb;
  margin:14px -18px 0;padding:8px 18px;overflow-x:auto;white-space:nowrap}
nav a{color:#2f5d8a;text-decoration:none;font-size:13px;margin-right:14px}
nav a:hover{text-decoration:underline}
.card{background:#fff;border:1px solid #e4e7eb;border-radius:10px;padding:16px 18px;margin:10px 0}
.hero{border-left:6px solid;padding:18px 22px}
.hero .head{font-size:22px;font-weight:700;letter-spacing:.3px}
.grid,.kpis{display:flex;flex-wrap:wrap;margin:0 -5px}
.grid>.v{flex:1 1 300px;margin:5px}
.v{background:#fff;border:1px solid #e4e7eb;border-left:5px solid;border-radius:8px;padding:12px 14px}
.v .area{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:#7b8794}
.v .verdict{font-weight:700;margin:2px 0 6px}
.v .act{font-size:13px}
.v .why{font-size:12px;color:#616e7c;margin-top:6px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:700;white-space:nowrap}
table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff}
th,td{padding:6px 8px;border-bottom:1px solid #eef1f4;text-align:right;vertical-align:top}
th{background:#f0f3f6;color:#52606d;font-weight:600}
td.l,th.l,th:first-child{text-align:left}
.tw{overflow-x:auto;border:1px solid #e4e7eb;border-radius:8px;margin:8px 0}
table.heat td{text-align:center}
img{max-width:100%;display:block;margin:6px 0}
.kpi{flex:1 1 140px;margin:5px;background:#fff;border:1px solid #e4e7eb;border-radius:8px;padding:10px 12px}
.kpi .k{font-size:11.5px;color:#7b8794;text-transform:uppercase;letter-spacing:.5px}
.kpi .val{font-size:19px;font-weight:700}
.warnbox{background:#fdf1dc;border:1px solid #f0d49a;border-radius:8px;padding:10px 14px;font-size:13px}
.note{font-size:12px;color:#616e7c}
ul{margin:6px 0 0 18px;padding:0}
tr.note-row td{border-bottom:1px solid #e4e7eb;padding-top:0;font-size:11.5px;color:#616e7c;text-align:left}
tr.has-note td{border-bottom:0}
"""

SECTIONS = [
    ("verdict", "Verdict", _section_verdict),
    ("trade-idea", "Trade idea", _section_trade_idea),
    ("positions", "Positions", _section_positions),
    ("stocks", "Stocks", _section_stock_signals),
    ("performance", "Performance", _section_performance),
    ("luck", "Luck test", _section_luck),
    ("capm", "CAPM", _section_capm),
    ("var", "VaR", _section_var),
    ("book", "Greeks", _section_book),
    ("stress", "Stress", _section_stress),
    ("simulation", "Simulation", _section_simulation),
    ("sizing", "Sizing", _section_sizing),
    ("attribution", "Attribution", _section_attribution),
    ("notes", "Notes", _section_notes),
]


def to_html(results: dict, path: str | Path, title: str = "Portfolio Tearsheet", synthetic: bool = False) -> Path:
    s = results["summary"]
    nav = "<nav>" + "".join(f"<a href='#{anchor}'>{esc(label)}</a>" for anchor, label, _ in SECTIONS) + "</nav>"
    header = (f"<h1>{esc(title)}</h1><div class='sub'>As of {results['as_of']} · {s['start']} to {s['end']} "
              f"({s['years']:.1f} years) · benchmark {results['benchmark']} · risk-free {pct(results['rf'], 2)}</div>")
    if synthetic:
        header += ("<p class='warnbox'><b>Synthetic demo data.</b> Simulated prices and a simple rules-based "
                   "strategy. These numbers are nobody's track record.</p>")

    body = "".join(build(results) for _, _, build in SECTIONS)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>{header}{nav}{body}</div></body></html>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
