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

import datetime as dt
from pathlib import Path

import QuantLib as ql

from derivx.data.rates_sources import parse_boe_ois_zip
from derivx.rates.market_quotes import OIS_PAR_TENORS

from .common import CALENDARS, qd, set_evaluation_date

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "rates" / "boe_latest.zip"
TENORS = tuple(t for t in OIS_PAR_TENORS if t != "25Y")  # 25Y pays past the last BoE node; extrapolation differs
# Par OIS rates the history store derives from the committed sample on dates the BoE fixture does not cover:
# the EUR proxy (ECB AAA spot curve, ESTR: spot T+2 TARGET, payment lag 1) with negative rates, and the
# pre-2016 BoE curve that stops at 5Y.
STORE_CASES = (
    ("EUR_AAA_ECB", "2020-03-31"),
    ("EUR_AAA_ECB", "2021-12-31"),
    ("GBP_OIS_BOE", "2013-05-31"),
)
INDEX = {"EUR ESTR": ql.Estr, "GBP SONIA": ql.Sonia}


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
    return {"fixture": "tests/fixtures/rates/boe_latest.zip", "cases": cases, "store_cases": store_cases()}


def store_cases() -> list:
    """QuantLib par OIS rates off the committed history sample's zero curves
    (same curve construction as above; extrapolation on, so the last tenor
    may pay a few days past the last published maturity)."""

    from derivx.data import rates_history as rh
    from derivx.rates.conventions import parse_tenor
    from derivx.rates.curves import get_conventions

    store = rh.bundled_store()
    out = []
    for dataset, text in STORE_CASES:
        day = dt.date.fromisoformat(text)
        conv = get_conventions(rh.dataset_info(dataset).ois_conventions)
        zeros = store.curve(dataset, day, "zero_cc")
        rows = sorted((parse_tenor(t).years, float(z)) for t, z in zeros.items())
        set_evaluation_date(day)
        calendar = CALENDARS[conv.calendar]
        dates = [qd(day)] + [qd(day) + round(y * 365.0) for y, _ in rows]
        curve = ql.ZeroCurve(dates, [rows[0][1]] + [z for _, z in rows], ql.Actual365Fixed(), calendar, ql.Linear(),
                             ql.Continuous)  # fmt: skip
        curve.enableExtrapolation()
        handle = ql.YieldTermStructureHandle(curve)
        index = INDEX[conv.name](handle)
        first = rows[0][0]
        rates = {}
        for tenor in rh.PAR_TENORS:
            if parse_tenor(tenor).years < first - 1e-9 or tenor not in store.ois_par(dataset, day):
                continue
            swap = ql.MakeOIS(
                ql.Period(tenor),
                index,
                0.0,
                settlementDays=conv.spot_lag,
                paymentLag=conv.payment_lag,
                calendar=calendar,
                endOfMonth=False,
                pricingEngine=ql.DiscountingSwapEngine(handle),
            )
            rates[tenor] = swap.fairRate()
        out.append({"dataset": dataset, "asof": text, "par": rates})
    return out
