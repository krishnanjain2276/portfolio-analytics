"""Every assumption the engine makes lives in this file.

If you disagree with a number (the risk-free rate, a lot size, a verdict threshold), change
it here and it flows through the ledger, the analytics and the report. Nothing else in the
package hard-codes a market assumption.
"""

from __future__ import annotations

import datetime as dt

# --- Market conventions ---------------------------------------------------------------

TRADING_DAYS = 252
RISK_FREE = 0.065  # roughly the Indian 91-day T-bill; override with --rf
INDEX_DIVIDEND_YIELD = 0.012  # NIFTY 50 trailing yield, used in Black-Scholes
BENCHMARK = "NIFTY50"

# All the ways people (and brokers, and Yahoo) write the index.
INDEX_ALIASES = {"NIFTY": "NIFTY50", "NIFTY 50": "NIFTY50", "^NSEI": "NIFTY50", "NIFTY50": "NIFTY50"}
STRIKE_STEP = {"NIFTY50": 50}

# NSE has changed the NIFTY lot size several times. Each entry is (first expiry date on
# which the size applies, lot size). If a contract note disagrees, the contract note wins:
# put the right multiplier in the Instruments sheet.
LOT_SIZE_SCHEDULE = {
    "NIFTY50": [
        (dt.date(2000, 1, 1), 50),
        (dt.date(2024, 4, 25), 25),
        (dt.date(2025, 1, 1), 75),
        (dt.date(2026, 1, 6), 65),
    ],
}

# Monthly expiry was the last Thursday until NSE moved it to the last Tuesday in Sept 2025.
# Python weekdays: Monday = 0, so 3 = Thursday and 1 = Tuesday.
EXPIRY_WEEKDAY_SCHEDULE = [
    (dt.date(2000, 1, 1), 3),
    (dt.date(2025, 9, 1), 1),
]

# --- Transaction charges --------------------------------------------------------------
# Only used when a trade's fee is left blank. Rates are fractions of turnover (premium
# turnover for options). These are typical discount-broker numbers, not any one broker's.

CHARGES = {
    "brokerage_delivery": 0.0,
    "brokerage_flat_per_order": 20.0,  # F&O and intraday
    "brokerage_pct_cap": 0.0003,  # ...or 0.03% of turnover if that is cheaper
    "stt_equity_delivery": 0.001,  # charged on both buy and sell
    # STT on F&O is sell-side only and has gone up twice:
    # (effective from, futures rate, options rate on premium)
    "stt_fno_schedule": [
        (dt.date(2000, 1, 1), 0.000125, 0.000625),
        (dt.date(2024, 10, 1), 0.0002, 0.001),
        (dt.date(2026, 4, 1), 0.0005, 0.0015),
    ],
    "exchange_equity": 0.0000297,
    "exchange_futures": 0.0000173,
    "exchange_options": 0.0003503,
    "sebi_fee": 0.000001,  # Rs 10 per crore
    "stamp_equity_delivery": 0.00015,  # stamp duty is buy-side only
    "stamp_futures": 0.00002,
    "stamp_options": 0.00003,
    "gst": 0.18,  # on brokerage + exchange charges + SEBI fee
}

# --- Analytics ------------------------------------------------------------------------

VAR_LEVELS = (0.95, 0.99)
VAR_WINDOW = 250  # about one trading year of history for historical VaR
EWMA_LAMBDA = 0.94  # the RiskMetrics daily decay factor
ROLLING_BETA_WINDOW = 63  # about three months
BOOTSTRAP_BLOCK = 10  # mean block length in days
BOOTSTRAP_SAMPLES = 2000
MC_PATHS = 10_000
MC_HORIZON = TRADING_DAYS
SEED = 42
DEMO_SEED = 7

STRESS_SPOT = (-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20)
STRESS_VOL = (-0.30, 0.0, 0.50, 1.00)  # relative change in implied vol, so +1.00 = doubles

# (NIFTY move, relative IV change). Stylised one-day shocks, not historical replays.
SCENARIOS = {
    "Gap down -5%, vol +50%": (-0.05, 0.50),
    "Crash -12%, vol +100%": (-0.12, 1.00),
    "Covid-style -20%, vol +150%": (-0.20, 1.50),
    "Grind lower -3%, vol +20%": (-0.03, 0.20),
    "Rally +5%, vol -20%": (0.05, -0.20),
    "Melt-up +10%, vol -30%": (0.10, -0.30),
}

KELLY_FRACTION = 0.5  # full Kelly is far too aggressive once you allow for estimation error
VOL_TARGET = 0.15
RISK_PER_TRADE = 0.01  # lose at most 1% of NAV if a stop is hit

# --- Verdict thresholds ---------------------------------------------------------------
# The verdicts are simple rules on purpose: every one can be traced back to a number here.

VERDICT = {
    # performance
    "sharpe_strong": 1.0,
    "sharpe_ok": 0.5,
    "dsr_skill": 0.95,
    "dsr_inconclusive": 0.50,
    "alpha_t_sig": 2.0,
    "beta_high": 1.2,
    # risk limits
    "var99_limit": 0.03,  # 1-day 99% VaR as a fraction of NAV
    "stress_loss_limit": 0.15,  # worst stress-grid loss as a fraction of NAV
    "max_risk_share": 0.40,  # no single sleeve should drive more of the volatility than this
    "cost_share_limit": 0.30,  # charges as a share of gross P&L
    "prob_loss_limit": 0.30,  # Monte Carlo probability of a losing year
    "oversize_ratio": 1.25,  # forecast vol / target vol
    "undersize_ratio": 0.60,
    # stock and index scoring
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "score_strong": 50,
    "score_act": 20,
    # implied vs realised vol
    "iv_rich": 1.20,
    "iv_cheap": 0.90,
    "iv_premium_assumed": 1.10,  # IV guess = this x realised vol when no option closes exist
    # managing live options
    "long_take_profit": 0.50,  # +50% on premium paid
    "long_stop": -0.50,
    "short_take_profit": 0.70,  # 70% of the premium already captured
    "short_stop": -1.00,  # losing as much as the premium received
    "dte_close_long_otm": 5,
    "dte_roll": 3,
}

DISCLAIMER = (
    "Verdicts are mechanical outputs of the rules in config.py applied to your data. They "
    "are not investment advice, they ignore anything not in the data (news, earnings, "
    "liquidity, your tax position) and they can be wrong. Check every one before acting."
)
