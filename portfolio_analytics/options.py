"""Black-Scholes-Merton pricing, Greeks and implied volatility.

European options with a continuous dividend yield ``q``. Everything works element-wise on
NumPy arrays, which the stress test relies on. Time ``T`` is in years.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Black-Scholes divides by sqrt(T), so expiring options get a tiny positive T instead of 0.
_TINY_T = 1e-8


def _d1_d2(S, K, T, r, sigma, q):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    T = np.maximum(T, _TINY_T)
    sigma = np.maximum(sigma, 1e-8)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t, T


def _is_call(option_type):
    return np.char.upper(np.asarray(option_type, dtype=str)) == "C"


def _as_float_if_scalar(value):
    return float(value) if np.ndim(value) == 0 else value


def bs_price(S, K, T, r, sigma, q=0.0, option_type="C"):
    """Option price. At or after expiry (T <= 0) it is just the intrinsic value."""
    T_given = np.asarray(T, dtype=float)
    d1, d2, T = _d1_d2(S, K, T, r, sigma, q)
    S, K = np.asarray(S, float), np.asarray(K, float)
    is_call = _is_call(option_type)

    call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    intrinsic = np.where(is_call, np.maximum(S - K, 0), np.maximum(K - S, 0))

    price = np.where(T_given <= 0, intrinsic, np.where(is_call, call, put))
    return _as_float_if_scalar(price)


def greeks(S, K, T, r, sigma, q=0.0, option_type="C") -> dict:
    """Delta, gamma, vega, theta and rho.

    Units: vega per 1.00 change in vol (so divide by 100 for "per vol point"), theta per
    year (divide by 365 for per day), rho per 1.00 change in the rate.
    """
    d1, d2, T = _d1_d2(S, K, T, r, sigma, q)
    S, K, sigma = np.asarray(S, float), np.asarray(K, float), np.asarray(sigma, float)
    is_call = _is_call(option_type)
    density = norm.pdf(d1)
    div_discount, rate_discount = np.exp(-q * T), np.exp(-r * T)

    delta = np.where(is_call, div_discount * norm.cdf(d1), div_discount * (norm.cdf(d1) - 1))
    gamma = div_discount * density / (S * sigma * np.sqrt(T))
    vega = S * div_discount * density * np.sqrt(T)

    decay = -S * div_discount * density * sigma / (2 * np.sqrt(T))
    theta_call = decay - r * K * rate_discount * norm.cdf(d2) + q * S * div_discount * norm.cdf(d1)
    theta_put = decay + r * K * rate_discount * norm.cdf(-d2) - q * S * div_discount * norm.cdf(-d1)
    theta = np.where(is_call, theta_call, theta_put)

    rho = np.where(is_call, K * T * rate_discount * norm.cdf(d2), -K * T * rate_discount * norm.cdf(-d2))

    results = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
    return {name: _as_float_if_scalar(value) for name, value in results.items()}


def no_arbitrage_bounds(S, K, T, r, q=0.0, option_type="C") -> tuple[float, float]:
    div_discount, rate_discount = math.exp(-q * T), math.exp(-r * T)
    if option_type.upper() == "C":
        return max(S * div_discount - K * rate_discount, 0.0), S * div_discount
    return max(K * rate_discount - S * div_discount, 0.0), K * rate_discount


def implied_vol(price, S, K, T, r, q=0.0, option_type="C", low=1e-4, high=5.0) -> float:
    """Back out volatility from a price with Brent's method.

    Returns NaN when no volatility could explain the price (it breaks the no-arbitrage
    bounds), which usually means a stale or mistyped price.
    """
    if T <= 0 or price <= 0 or S <= 0:
        return float("nan")
    lower, upper = no_arbitrage_bounds(S, K, T, r, q, option_type)
    if price < lower - 1e-10 or price > upper + 1e-10:
        return float("nan")

    def pricing_error(sigma):
        return bs_price(S, K, T, r, sigma, q, option_type) - price

    try:
        if pricing_error(low) > 0:  # cheaper than even a near-zero vol allows
            return low
        if pricing_error(high) < 0:
            return float("nan")
        return brentq(pricing_error, low, high, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return float("nan")


def forward_price(S, T, r, q=0.0):
    return np.asarray(S, float) * np.exp((r - q) * np.maximum(np.asarray(T, float), 0))
