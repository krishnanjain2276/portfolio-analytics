"""Risk: CAPM, Value at Risk, the Greeks of the live book, where the risk comes from, and
what happens in a crash.

Conventions: VaR and Expected Shortfall are positive numbers meaning "loss", as a fraction
of NAV. Rupee exposures are "NIFTY-equivalent": a stock counts as its value times its beta,
an option as delta times spot times units.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .instruments import Instrument
from .options import bs_price, greeks, implied_vol
from .performance import daily_rate

TD = config.TRADING_DAYS

# --- CAPM with Newey-West standard errors ---------------------------------------------
# Daily portfolio returns are autocorrelated and heteroskedastic, which makes plain OLS
# t-stats look more significant than they are. Newey-West corrects for both.


def newey_west_lag(n: int) -> int:
    """The usual rule of thumb: floor(4 (n/100)^(2/9))."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def newey_west_cov(X: np.ndarray, residuals: np.ndarray, lag: int | None = None) -> np.ndarray:
    """HAC covariance of OLS coefficients with Bartlett weights. lag=0 gives White's."""
    lag = newey_west_lag(len(residuals)) if lag is None else lag
    scores = X * residuals[:, None]
    meat = scores.T @ scores
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        gamma = scores[k:].T @ scores[:-k]
        meat += weight * (gamma + gamma.T)
    bread = np.linalg.inv(X.T @ X)
    return bread @ meat @ bread


def capm(portfolio: pd.Series, benchmark: pd.Series, rf: float = config.RISK_FREE) -> dict:
    both = pd.concat([portfolio.rename("p"), benchmark.rename("b")], axis=1).dropna()
    y = (both.p - daily_rate(rf)).to_numpy()
    x = (both.b - daily_rate(rf)).to_numpy()
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef
    residuals = y - X @ coef

    se_nw = np.sqrt(np.diag(newey_west_cov(X, residuals)))
    se_ols = np.sqrt(np.diag(residuals @ residuals / (n - 2) * np.linalg.inv(X.T @ X)))
    t_nw = coef / se_nw
    p_nw = 2 * (1 - stats.t.cdf(np.abs(t_nw), n - 2))

    r_squared = 1 - (residuals @ residuals) / ((y - y.mean()) @ (y - y.mean()))
    active = both.p - both.b
    tracking_error = active.std(ddof=1) * math.sqrt(TD)
    residual_vol = residuals.std(ddof=1) * math.sqrt(TD)
    up_days, down_days = both.b > 0, both.b < 0

    def safe_ratio(a, b):
        return a / b if b else float("nan")

    return {
        "alpha_annual": alpha * TD,
        "alpha_t_nw": t_nw[0],
        "alpha_p_nw": p_nw[0],
        "alpha_t_ols": alpha / se_ols[0],
        "beta": beta,
        "beta_se_nw": se_nw[1],
        "beta_t_nw": t_nw[1],
        "r2": r_squared,
        "tracking_error": tracking_error,
        "information_ratio": safe_ratio(active.mean() * TD, tracking_error),
        "treynor": safe_ratio(y.mean() * TD, beta),
        "appraisal_ratio": safe_ratio(alpha * TD, residual_vol),
        "up_capture": safe_ratio(both.p[up_days].mean(), both.b[up_days].mean()),
        "down_capture": safe_ratio(both.p[down_days].mean(), both.b[down_days].mean()),
        "nw_lag": newey_west_lag(n),
        "n": n,
    }


def rolling_beta(portfolio: pd.Series, benchmark: pd.Series,
                 window: int = config.ROLLING_BETA_WINDOW) -> pd.Series:
    both = pd.concat([portfolio.rename("p"), benchmark.rename("b")], axis=1).dropna()
    return (both.p.rolling(window).cov(both.b) / both.b.rolling(window).var()).rename("rolling_beta")


# --- Value at Risk and Expected Shortfall ---------------------------------------------


def var_historical(returns, level):
    return float(-np.quantile(returns, 1 - level))


def var_gaussian(returns, level):
    return float(-(returns.mean() + stats.norm.ppf(1 - level) * returns.std(ddof=1)))


def var_cornish_fisher(returns, level):
    """Gaussian VaR with the quantile bent for skew and fat tails."""
    z = stats.norm.ppf(1 - level)
    S = stats.skew(returns, bias=False)
    K = stats.kurtosis(returns, fisher=True, bias=False)  # excess kurtosis
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24 - (2 * z**3 - 5 * z) * S**2 / 36
    return float(-(returns.mean() + z_cf * returns.std(ddof=1)))


def ewma_vol(returns: pd.Series, lam: float = config.EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics volatility. The value on day t is the forecast for day t+1."""
    x = returns.to_numpy()
    var = np.empty_like(x)
    var[0] = np.var(x[:30], ddof=1) if len(x) > 1 else x[0] ** 2  # seed with early sample variance
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    tomorrow = lam * var[-1] + (1 - lam) * x[-1] ** 2
    return pd.Series(np.sqrt(np.append(var[1:], tomorrow)), index=returns.index)


def var_ewma(returns, level):
    return float(-stats.norm.ppf(1 - level) * ewma_vol(returns).iloc[-1])


def es_historical(returns, level):
    var = var_historical(returns, level)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if len(tail) else var


def es_gaussian(returns, level):
    z = stats.norm.ppf(1 - level)
    return float(returns.std(ddof=1) * stats.norm.pdf(z) / (1 - level) - returns.mean())


VAR_METHODS = {
    "Historical": var_historical,
    "Gaussian": var_gaussian,
    "Cornish-Fisher": var_cornish_fisher,
    "EWMA (RiskMetrics)": var_ewma,
    "ES Historical": es_historical,
    "ES Gaussian": es_gaussian,
}


def var_table(returns: pd.Series, nav: float, levels=config.VAR_LEVELS) -> pd.DataFrame:
    rows = []
    for name, method in VAR_METHODS.items():
        row = {"method": name}
        for level in levels:
            loss = method(returns, level)
            row[f"{int(level * 100)}% (of NAV)"] = loss
            row[f"{int(level * 100)}% (Rs)"] = loss * nav
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def kupiec_test(returns: pd.Series, level: float = 0.99, window: int = config.VAR_WINDOW) -> dict:
    """Does a rolling historical VaR get breached as often as it should?

    Each day's VaR is estimated from the previous ``window`` days only, so this is a true
    out-of-sample backtest. The likelihood ratio is chi-squared with one degree of freedom.
    """
    x = returns.to_numpy()
    if len(x) <= window + 20:
        return {"T": 0, "breaches": 0, "expected": 0.0, "rate": float("nan"), "LR": float("nan"),
                "p_value": float("nan"), "result": "not enough history"}

    days_tested = len(x) - window
    breaches = sum(x[t] < np.quantile(x[t - window:t], 1 - level) for t in range(window, len(x)))
    p = 1 - level
    observed = breaches / days_tested

    log_lik_model = (days_tested - breaches) * math.log(1 - p) + breaches * math.log(p)
    log_lik_observed = 0.0
    if observed < 1:
        log_lik_observed += (days_tested - breaches) * math.log(1 - observed)
    if breaches > 0:
        log_lik_observed += breaches * math.log(observed)
    lr = -2 * (log_lik_model - log_lik_observed)
    p_value = 1 - stats.chi2.cdf(lr, 1)
    return {"T": days_tested, "breaches": int(breaches), "expected": days_tested * p,
            "rate": observed, "LR": lr, "p_value": p_value,
            "result": "PASS" if p_value > 0.05 else "FAIL"}


# --- The live book --------------------------------------------------------------------


def stock_beta(prices: pd.DataFrame, symbol: str, bench: str, window: int = config.VAR_WINDOW) -> float:
    rets = prices[[symbol, bench]].pct_change(fill_method=None).dropna().tail(window)
    if len(rets) < 30 or rets[bench].var() == 0:
        return 1.0  # not enough history to say otherwise
    return float(rets[symbol].cov(rets[bench]) / rets[bench].var())


def years_until(expiry, today) -> float:
    return max((pd.Timestamp(expiry) - pd.Timestamp(today)).days, 0) / 365.0


def live_book(units_last: pd.Series, instruments: dict[str, Instrument], prices: pd.DataFrame,
              date, rf: float = config.RISK_FREE, q: float = config.INDEX_DIVIDEND_YIELD,
              bench: str = config.BENCHMARK) -> pd.DataFrame:
    latest = prices.loc[:date].ffill().iloc[-1]
    rows = []
    for symbol, units in units_last.items():
        if units == 0:
            continue
        inst = instruments[symbol]
        price = float(latest[symbol])
        row = {
            "symbol": symbol, "asset_class": inst.asset_class, "sleeve": inst.sleeve,
            "units": units, "price": price, "underlying": inst.underlying or symbol,
            "strike": inst.strike, "option_type": inst.option_type,
            "spot": price, "T": np.nan, "iv": np.nan,
            "delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "beta": 1.0,
            "own_exposure": units * price, "index_exposure": units * price,
        }
        if inst.asset_class == "equity":
            beta = stock_beta(prices, symbol, bench) if bench in prices else 1.0
            row.update(beta=beta, index_exposure=units * price * beta)
        elif inst.asset_class == "future":
            row.update(spot=float(latest[inst.underlying]), T=years_until(inst.expiry, date))
        elif inst.asset_class == "option":
            spot = float(latest[inst.underlying])
            T = years_until(inst.expiry, date)
            iv = implied_vol(price, spot, inst.strike, T, rf, q, inst.option_type) if T > 0 else np.nan
            if np.isnan(iv):
                iv = 0.15  # price gives no usable IV (expiring or stale); a typical NIFTY level
            g = greeks(spot, inst.strike, max(T, 1 / 365), rf, iv, q, inst.option_type)
            row.update(spot=spot, T=T, iv=iv, delta=g["delta"], gamma=g["gamma"], vega=g["vega"],
                       theta=g["theta"], own_exposure=units * g["delta"] * spot,
                       index_exposure=units * g["delta"] * spot)
        rows.append(row)

    book = pd.DataFrame(rows)
    if len(book):
        # Rupee sensitivities that are easier to think about than raw Greeks
        book["delta_per_1pct"] = book["index_exposure"] * 0.01
        book["gamma_per_1pct"] = 0.5 * book["gamma"] * book["units"] * (book["spot"] * 0.01) ** 2
        book["vega_per_volpt"] = book["vega"] * book["units"] * 0.01
        book["theta_per_day"] = book["theta"] * book["units"] / 365
    return book


def book_greeks_summary(book: pd.DataFrame, nav: float) -> dict:
    if book.empty:
        return {"net_index_delta": 0.0, "net_delta_pct_nav": 0.0, "gamma_per_1pct": 0.0,
                "vega_per_volpt": 0.0, "theta_per_day": 0.0, "theta_pct_nav": 0.0}
    net_delta = book["index_exposure"].sum()
    theta = book["theta_per_day"].sum()
    return {
        "net_index_delta": net_delta,
        "net_delta_pct_nav": net_delta / nav,
        "gamma_per_1pct": book["gamma_per_1pct"].sum(),
        "vega_per_volpt": book["vega_per_volpt"].sum(),
        "theta_per_day": theta,
        "theta_pct_nav": theta / nav,
    }


# --- Where the risk comes from (Euler decomposition) ----------------------------------


def _risk_factor(position) -> str:
    """Stocks are their own risk factor; F&O positions load on their underlying."""
    return position["symbol"] if position["asset_class"] in ("equity", "index") else position["underlying"]


def euler_decomposition(book: pd.DataFrame, prices: pd.DataFrame, bench: str = config.BENCHMARK,
                        window: int = config.VAR_WINDOW) -> tuple[pd.DataFrame, float]:
    """Split daily portfolio volatility (in rupees) into one contribution per position.

    With exposures e and covariance S, sigma = sqrt(e'Se) and position i contributes
    e_i (Se)_i / sigma. The contributions add up to sigma exactly, and a negative one means
    that position is hedging the rest.
    """
    if book.empty:
        return pd.DataFrame(), 0.0
    factors = list(dict.fromkeys(_risk_factor(p) for _, p in book.iterrows()))
    cov = prices[factors].pct_change(fill_method=None).tail(window).cov().fillna(0.0).to_numpy()

    loadings = np.zeros((len(book), len(factors)))
    for i, (_, position) in enumerate(book.iterrows()):
        loadings[i, factors.index(_risk_factor(position))] = position["own_exposure"]

    total = loadings.sum(axis=0)
    sigma = math.sqrt(max(total @ cov @ total, 0.0))
    contributions = loadings @ (cov @ total) / sigma if sigma > 0 else np.zeros(len(book))

    out = book[["symbol", "sleeve", "asset_class"]].copy()
    out["exposure"] = loadings.sum(axis=1)
    out["vol_contribution_daily"] = contributions
    out["share_of_risk"] = contributions / sigma if sigma > 0 else 0.0
    return out, sigma


# --- Stress testing -------------------------------------------------------------------


def revalue(book: pd.DataFrame, spot_shock: float, vol_shock: float, rf=config.RISK_FREE,
            q=config.INDEX_DIVIDEND_YIELD) -> pd.Series:
    """P&L of each position after an instant NIFTY move and a relative change in IV.

    Options are fully repriced rather than approximated with delta, so a long put's payoff
    in a crash (convexity) is captured properly. Stocks move by beta times the index.
    """
    pnl = []
    for _, p in book.iterrows():
        if p["asset_class"] == "equity":
            pnl.append(p["units"] * p["price"] * p["beta"] * spot_shock)
        elif p["asset_class"] in ("future", "index"):
            pnl.append(p["units"] * p["price"] * spot_shock)
        else:
            T = max(p["T"], 1 / 365)
            before = bs_price(p["spot"], p["strike"], T, rf, p["iv"], q, p["option_type"])
            shocked_iv = max(p["iv"] * (1 + vol_shock), 0.01)
            after = bs_price(p["spot"] * (1 + spot_shock), p["strike"], T, rf, shocked_iv, q,
                             p["option_type"])
            pnl.append(p["units"] * (after - before))
    return pd.Series(pnl, index=book.index, dtype=float)


def stress_grid(book: pd.DataFrame, nav: float, spot=config.STRESS_SPOT, vol=config.STRESS_VOL,
                rf=config.RISK_FREE) -> pd.DataFrame:
    grid = pd.DataFrame(0.0, index=[f"{s:+.0%}" for s in spot],
                        columns=[f"IV {v:+.0%}" for v in vol])
    grid.index.name = "NIFTY move"
    if book.empty:
        return grid
    for s in spot:
        for v in vol:
            grid.loc[f"{s:+.0%}", f"IV {v:+.0%}"] = revalue(book, s, v, rf).sum() / nav
    return grid


def scenario_table(book: pd.DataFrame, nav: float, scenarios=config.SCENARIOS,
                   rf=config.RISK_FREE) -> pd.DataFrame:
    rows = []
    for name, (spot_shock, vol_shock) in scenarios.items():
        if book.empty:
            total = equities = 0.0
        else:
            pnl = revalue(book, spot_shock, vol_shock, rf)
            equities = pnl[book.asset_class == "equity"].sum()
            total = pnl.sum()
        rows.append({
            "scenario": name,
            "portfolio_pnl": total,
            "equity_only_pnl": equities,
            "derivatives_pnl": total - equities,
            "pct_nav": total / nav,
            # 1.0 = derivatives cancel the whole stock loss, negative = they add to it
            "hedge_offset": (1 - total / equities) if equities < 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("scenario")
