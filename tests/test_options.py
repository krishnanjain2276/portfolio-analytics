import math

import pytest

from portfolio_analytics.options import bs_price, greeks, implied_vol

S, K, T, r, q, SIGMA = 25000.0, 25200.0, 30 / 365, 0.065, 0.012, 0.16


def test_put_call_parity():
    call = bs_price(S, K, T, r, SIGMA, q, "C")
    put = bs_price(S, K, T, r, SIGMA, q, "P")
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), rel=1e-10)


@pytest.mark.parametrize("option_type", ["C", "P"])
@pytest.mark.parametrize("strike", [23500, 25000, 26500])
@pytest.mark.parametrize("vol", [0.08, 0.20, 0.60])
def test_implied_vol_recovers_the_vol_used_to_price(option_type, strike, vol):
    price = bs_price(S, strike, T, r, vol, q, option_type)
    assert implied_vol(price, S, strike, T, r, q, option_type) == pytest.approx(vol, abs=1e-6)


def test_impossible_prices_have_no_implied_vol():
    assert math.isnan(implied_vol(S * 1.1, S, K, T, r, q, "C"))  # dearer than the stock itself
    assert math.isnan(implied_vol(0.0, S, K, T, r, q, "C"))


@pytest.mark.parametrize("option_type", ["C", "P"])
def test_greeks_match_finite_differences(option_type):
    g = greeks(S, K, T, r, SIGMA, q, option_type)

    def price(spot=S, t=T, rate=r, vol=SIGMA):
        return bs_price(spot, K, t, rate, vol, q, option_type)

    h = 1.0
    assert g["delta"] == pytest.approx((price(spot=S + h) - price(spot=S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((price(spot=S + h) - 2 * price() + price(spot=S - h)) / h**2, rel=1e-3)
    assert g["vega"] == pytest.approx((price(vol=SIGMA + 1e-5) - price(vol=SIGMA - 1e-5)) / 2e-5, rel=1e-5)
    assert g["rho"] == pytest.approx((price(rate=r + 1e-6) - price(rate=r - 1e-6)) / 2e-6, rel=1e-4)
    assert g["theta"] == pytest.approx(-(price(t=T + 1e-6) - price(t=T - 1e-6)) / 2e-6, rel=1e-4)


def test_an_expired_option_is_worth_its_intrinsic_value():
    assert bs_price(105.0, 100.0, 0.0, r, SIGMA, q, "C") == pytest.approx(5.0)
    assert bs_price(105.0, 100.0, 0.0, r, SIGMA, q, "P") == pytest.approx(0.0)
