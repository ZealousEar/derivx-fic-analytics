"""QuantLib par OIS rates off the Bank of England's fitted UK OIS curve
(tests/test_market_quotes.py).

The recorded BoE fixture (tests/fixtures/rates/boe_latest.zip) is parsed
with derivx, then QuantLib builds its own curve from the published numbers
only: a ZeroCurve through the continuously compounded spot rates (nodes on
whole days from the curve date, ACT/365F, linear in the zero rate, the first
rate held back to the curve date), and prices ``ql.MakeOIS`` SONIA swaps
from their tenors (QuantLib's own schedules: spot T+0, annual,
UK calendar, payment lag 0). derivx must reproduce the fair rates from the
same inputs with :func:`derivx.rates.market_quotes.par_from_zero_rows`.
"""

from __future__ import annotations

from pathlib import Path

import QuantLib as ql

from derivx.data.rates_sources import parse_boe_ois_zip
from derivx.rates.market_quotes import OIS_PAR_TENORS

from .common import CALENDARS, qd, set_evaluation_date

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "rates" / "boe_latest.zip"
TENORS = tuple(t for t in OIS_PAR_TENORS if t != "25Y")  # 25Y pays past the last BoE node; extrapolation differs


def generate() -> dict:
    frame = parse_boe_ois_zip(FIXTURE.read_bytes())
    cases = []
    for day in sorted(frame["asof"].unique()):
        rows = frame[frame["asof"] == day].sort_values("tenor_years")
        set_evaluation_date(day)
        days = [round(y * 365.0) for y in rows["tenor_years"]]
        dates = [qd(day)] + [qd(day) + d for d in days]
        zeros = [float(rows["value_decimal"].iloc[0])] + [float(z) for z in rows["value_decimal"]]
        curve = ql.ZeroCurve(dates, zeros, ql.Actual365Fixed(), CALENDARS["LONDON"], ql.Linear(), ql.Continuous)
        handle = ql.YieldTermStructureHandle(curve)
        index = ql.Sonia(handle)
        rates = {}
        for tenor in TENORS:
            swap = ql.MakeOIS(
                ql.Period(tenor),
                index,
                0.0,
                settlementDays=0,
                paymentLag=0,
                calendar=CALENDARS["LONDON"],
                endOfMonth=False,
                pricingEngine=ql.DiscountingSwapEngine(handle),
            )
            rates[tenor] = swap.fairRate()
        cases.append({"asof": day.isoformat(), "par": rates})
    return {"fixture": "tests/fixtures/rates/boe_latest.zip", "cases": cases}
