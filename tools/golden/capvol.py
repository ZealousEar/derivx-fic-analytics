"""QuantLib references for caplet vol stripping (tests/test_capvol.py).

* ``stripper1``: ``ql.OptionletStripper1`` on a ``CapFloorTermVolSurface``
  of flat cap vols quoted at every caplet maturity (so QuantLib's term-vol
  interpolation never enters), with normal or shifted-lognormal vols: the
  caplet vols it strips must equal derivx's piecewise-constant strip of the
  same quotes. Single-curve USD (a 3M term index on the SOFR example curve)
  and dual-curve EUR (EURIBOR 6M projecting, €STR discounting, the
  tools/golden/multicurve.py base set).
* ``repricing``: QuantLib NPVs of coarse cap quotes (1Y ... 30Y, ATM and
  fixed strikes) at their flat vols with ``BachelierCapFloorEngine``; the
  caplet vols derivx strips from the quotes must reprice them.

Flat vols are synthetic test inputs (a smooth function of maturity and
strike), not market data.
"""

from __future__ import annotations

import math
from typing import Dict, List

import QuantLib as ql

from derivx.rates.capfloor import CapFloor
from derivx.rates.examples import EXAMPLE_ASOF, example_market

from . import multicurve as mc
from .common import CALENDARS, pyd, qd, ql_curve, set_evaluation_date

USD_TERMS = {"index_months": 3, "day_count": "ACT/360", "calendar": "USD", "fixing_lag": 2}
EUR_TERMS = {"index_months": 6, "day_count": "ACT/360", "calendar": "TARGET", "fixing_lag": 2,
             "projection": "EURIBOR 6M"}  # fmt: skip
COARSE = ("1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")


def flat_vol(years: float, strike: float, model: str) -> float:
    """Synthetic flat cap vol: a hump in maturity plus a smile in strike."""

    hump = 1.0 + 0.12 * math.sin(years / 2.5) - 0.006 * years
    if model == "bachelier":
        return 0.0095 * hump + 0.08 * (strike - 0.035) ** 2
    return 0.20 * hump + 6.0 * (strike - 0.035) ** 2


def _years(tenor: str) -> float:
    return int(tenor[:-1]) / 12.0 if tenor.endswith("M") else float(tenor[:-1])


def _usd_index(handle) -> ql.IborIndex:
    return ql.IborIndex("DERIVX-TERM", ql.Period(3, ql.Months), 2, ql.USDCurrency(), CALENDARS["USD"],
                        ql.ModifiedFollowing, False, ql.Actual360(), handle)  # fmt: skip


def stripper1_cases() -> List[Dict[str, object]]:
    out = []
    specs = [
        ("usd", "bachelier", 0.0, [0.025, 0.035, 0.045], 3, 5),
        ("usd", "shifted_black", 0.02, [0.03, 0.04], 3, 4),
        ("eur", "bachelier", 0.0, [0.02, 0.03], 6, 10),
    ]
    for name, model, shift, strikes, months, years in specs:
        if name == "usd":
            market = example_market()
            asof = market.asof
            set_evaluation_date(asof)
            disc = ql_curve(market.curve)
            index = _usd_index(disc)
            cal = CALENDARS["USD"]
        else:
            asof = mc.SETS["base"][0]
            market = mc.build(asof, mc.quote_set("base"))
            set_evaluation_date(asof)
            disc = ql_curve(market.curve)
            index = ql.Euribor6M(ql_curve(market.projection("EURIBOR 6M")))
            cal = CALENDARS["TARGET"]
        tenors = [f"{months * k}M" for k in range(2, years * 12 // months + 1)]
        vols = [[flat_vol(_years(t), k, model) for k in strikes] for t in tenors]
        matrix = ql.Matrix(len(tenors), len(strikes))
        for i, row in enumerate(vols):
            for j, v in enumerate(row):
                matrix[i][j] = v
        surface = ql.CapFloorTermVolSurface(0, cal, ql.ModifiedFollowing, [ql.Period(t) for t in tenors], strikes,
                                            matrix, ql.Actual365Fixed())  # fmt: skip
        vol_type = ql.Normal if model == "bachelier" else ql.ShiftedLognormal
        stripper = ql.OptionletStripper1(surface, index, ql.nullDouble(), 1e-14, 500, disc, vol_type, shift)
        dates = list(stripper.optionletFixingDates())
        caplet_vols = [list(stripper.optionletVolatilities(i)) for i in range(len(dates))]
        out.append({
            "market": name, "model": model, "shift": shift, "strikes": strikes, "tenors": tenors, "flat_vols": vols,
            "fixing_dates": [pyd(d).isoformat() for d in dates], "caplet_vols": caplet_vols,
        })  # fmt: skip
    set_evaluation_date(EXAMPLE_ASOF)
    return out


def repricing_cases() -> List[Dict[str, object]]:
    """Flat-vol NPVs of coarse quotes on the USD example curve (unit notional)."""

    market = example_market()
    asof = market.asof
    set_evaluation_date(asof)
    disc = ql_curve(market.curve)
    index = _usd_index(disc)
    out = []
    for strike in (None, 0.03, 0.045):
        rows = []
        for tenor in COARSE:
            vol = flat_vol(_years(tenor), 0.035 if strike is None else strike, "bachelier")
            cf = CapFloor(1.0, "cap", 0.03 if strike is None else strike, tenor, vol, **USD_TERMS)
            periods = cf.periods(asof)
            schedule = ql.Schedule(ql.DateVector([qd(periods[0].start)] + [qd(p.end) for p in periods]))
            leg = ql.IborLeg([1.0], schedule, index, ql.Actual360(), ql.ModifiedFollowing, [2])
            probe = ql.Cap(leg, [0.03])
            k = probe.atmRate(disc.currentLink()) if strike is None else strike
            cap = ql.Cap(leg, [k])
            surface = ql.ConstantOptionletVolatility(qd(asof), CALENDARS["USD"], ql.Following, vol,
                                                     ql.Actual365Fixed(), ql.Normal, 0.0)  # fmt: skip
            cap.setPricingEngine(ql.BachelierCapFloorEngine(disc, ql.OptionletVolatilityStructureHandle(surface)))
            rows.append({"tenor": tenor, "strike": k, "vol": vol, "npv": cap.NPV()})
        out.append({"strike": strike, "quotes": rows})
    return out


def generate() -> dict:
    return {"asof": EXAMPLE_ASOF.isoformat(), "stripper1": stripper1_cases(), "repricing": repricing_cases()}
