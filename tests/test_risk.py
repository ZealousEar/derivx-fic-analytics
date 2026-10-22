import numpy as np
import pytest

from derivx.rates.curves import Instrument
from derivx.rates.examples import example_market
from derivx.risk.sensitivities import bumped_market, key_rate_pv01, ladder, parallel_gamma, parallel_pv01, solve_hedge


def _receiver(market, index, notional=1e6):
    """PV of receiving the curve's own quote ``index`` (a par swap today)."""

    quote = market.curve.quotes[index]
    return lambda m: notional * Instrument(quote, m.asof, m.curve.conventions).npv(m.curve)


def test_parallel_pv01_is_full_revaluation_not_a_frozen_forward_bump():
    market = example_market()
    pv = _receiver(market, 11)  # 10Y OIS receiver, $1mm
    pv01 = parallel_pv01(pv, market)
    up, down = pv(bumped_market(market, "curve", 1.0)), pv(bumped_market(market, "curve", -1.0))
    assert pv01 == pytest.approx((up - down) / 2)
    assert -900 < pv01 < -800  # receiver loses about one annuity (~8.4) bp of notional
    assert parallel_gamma(pv, market) > 0


def test_key_rates_sum_to_parallel_and_hedge_flattens_the_ladder():
    market = example_market()
    pv = lambda m: _receiver(market, 9)(m) - 0.5 * _receiver(market, 13)(m)  # noqa: E731
    rows = key_rate_pv01(pv, market)
    # Equal up to second-order terms: the annuity itself moves with the other quotes.
    assert sum(r["pv01"] for r in rows) == pytest.approx(parallel_pv01(pv, market), rel=1e-5)
    hedge_cols = [ladder(key_rate_pv01(_receiver(market, i, 1.0), market)) for i in (9, 13)]
    result = solve_hedge(ladder(rows), np.column_stack(hedge_cols), ["5Y", "20Y"])
    assert result["risk_removed"] > 0.999999
    assert result["multiples"]["5Y"] == pytest.approx(-1e6, rel=1e-6)


def test_bumps_leave_past_fixings_alone():
    market = example_market()
    bumped = bumped_market(market, "curve", 5.0)
    assert bumped.fixings is market.curve
    with pytest.raises(ValueError, match="which"):
        parallel_pv01(lambda m: 0.0, market, which="ois")
