"""The tearsheet's charts, drawn with matplotlib and embedded in the HTML as PNGs."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # no screen needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from . import config  # noqa: E402
from .performance import drawdown_series, wealth  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GREY = "#2f5d8a", "#c97b2a", "#7a8b99"
GREEN, RED = "#1e7b4f", "#b83227"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "axes.axisbelow": True,
})

AS_PERCENT = PercentFormatter(1.0, 0)


def to_img_tag(fig, alt: str) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=144, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{encoded}"/>'


def growth_and_drawdown(results) -> str:
    portfolio, benchmark = results["returns"], results["bench_returns"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    top.plot(wealth(portfolio).index, wealth(portfolio).values, color=BLUE, lw=1.6,
             label="Portfolio (time-weighted)")
    top.plot(wealth(benchmark).index, wealth(benchmark).values, color=GREY, lw=1.2, label=results["benchmark"])
    top.set_title("Growth of ₹1")
    top.legend(loc="upper left")

    dd = drawdown_series(portfolio)
    bottom.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.25, lw=0)
    bottom.plot(dd.index, dd.values, color=RED, lw=0.8)
    bench_dd = drawdown_series(benchmark)
    bottom.plot(bench_dd.index, bench_dd.values, color=GREY, lw=0.8, alpha=0.8)
    bottom.set_title("Drawdown")
    bottom.yaxis.set_major_formatter(AS_PERCENT)
    fig.tight_layout()
    return to_img_tag(fig, "Growth of 1 rupee and drawdown, portfolio vs NIFTY")


def rolling_beta(results) -> str:
    beta = results["rolling_beta"].dropna()
    fig, ax = plt.subplots(figsize=(9.5, 2.4))
    ax.plot(beta.index, beta.values, color=BLUE, lw=1.2)
    full_period = results["capm"]["beta"]
    ax.axhline(full_period, color=ORANGE, lw=1, ls="--", label=f"Full-period beta {full_period:.2f}")
    ax.set_title(f"Rolling {config.ROLLING_BETA_WINDOW}-day beta to {results['benchmark']}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return to_img_tag(fig, "Rolling beta")


def return_distribution(results) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.hist(results["returns"].values, bins=70, color=BLUE, alpha=0.75)
    for method, colour, style in (("Historical", ORANGE, "-"), ("Cornish-Fisher", RED, "--"),
                                  ("EWMA (RiskMetrics)", GREY, ":")):
        var = results["var"].loc[method, "99% (of NAV)"]
        ax.axvline(-var, color=colour, ls=style, lw=1.3, label=f"99% VaR {method}: {var * 100:.2f}%")
    ax.set_title("Daily returns, with where each VaR method draws the line")
    ax.xaxis.set_major_formatter(AS_PERCENT)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return to_img_tag(fig, "Distribution of daily returns")


def risk_by_sleeve(results) -> str:
    sleeves = results["euler_sleeves"]
    if sleeves is None or len(sleeves) == 0:
        return ""
    sleeves = sleeves.sort_values("share_of_risk")
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(sleeves) + 1.0))
    ax.barh(sleeves.index, sleeves["share_of_risk"],
            color=[RED if v < 0 else BLUE for v in sleeves["share_of_risk"]])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.xaxis.set_major_formatter(AS_PERCENT)
    ax.set_title("Share of portfolio volatility by sleeve (negative = hedging)")
    fig.tight_layout()
    return to_img_tag(fig, "Risk contribution by sleeve")


def simulation(results) -> str:
    mc = results["montecarlo"]
    fan = mc["fan"]
    fig, (paths, outcomes) = plt.subplots(1, 2, figsize=(9.5, 2.9))
    days = np.arange(1, len(fan) + 1)
    paths.fill_between(days, fan.p5 - 1, fan.p95 - 1, color=BLUE, alpha=0.15, lw=0, label="5–95%")
    paths.fill_between(days, fan.p25 - 1, fan.p75 - 1, color=BLUE, alpha=0.3, lw=0, label="25–75%")
    paths.plot(days, fan.p50 - 1, color=BLUE, lw=1.5, label="Median")
    paths.axhline(0, color=MUTED, lw=0.8)
    paths.set_title("One-year return paths")
    paths.set_xlabel("trading days")
    paths.yaxis.set_major_formatter(AS_PERCENT)
    paths.legend(loc="upper left", fontsize=8)

    outcomes.hist(mc["terminal"], bins=80, color=BLUE, alpha=0.75)
    outcomes.axvline(0, color=RED, lw=1)
    outcomes.set_title(f"Return after one year, P(loss) = {mc['prob_loss'] * 100:.0f}%")
    outcomes.xaxis.set_major_formatter(AS_PERCENT)
    fig.tight_layout()
    return to_img_tag(fig, "Monte Carlo simulation")


def pnl_by_sleeve(results) -> str:
    sleeves = results["attr_sleeve"].sort_values("pnl")
    fig, ax = plt.subplots(figsize=(9.5, 0.4 * len(sleeves) + 1.0))
    ax.barh(sleeves["sleeve"], sleeves["pnl"], color=[GREEN if v >= 0 else RED for v in sleeves["pnl"]])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title("Total P&L by sleeve (₹ lakh, after charges)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e5:,.1f}L"))
    fig.tight_layout()
    return to_img_tag(fig, "P&L by sleeve")
