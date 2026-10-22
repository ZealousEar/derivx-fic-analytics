"""QuantLib references for swaptions and rates SABR (tests/test_swaptions.py,
tests/test_swaption_vol.py).

* ``cases``: European swaptions priced by QuantLib's ``BlackSwaptionEngine``
  (plain and shifted, the shift on a ``ConstantSwaptionVolatility``) and
  ``BachelierSwaptionEngine`` on exact copies of the derivx curves. QuantLib
  builds the underlying itself: for USD a SOFR ``OvernightIndexedSwap``
  (annual ACT/360, payment lag 2) on a schedule QuantLib generates from the
  expiry (calendar advance, modified following) and spot lag; for EUR
  ``MakeSwaption`` on ``EuriborSwapIsdaFixA`` (annual 30/360 against
  EURIBOR 6M, or 3M for a 1Y tenor), the EURIBOR index projecting on the
  derivx EURIBOR curve and every flow discounted on the derivx €STR curve
  (dual curve). Records dates, NPV, annuity, ATM forward, delta, vega,
  ``Swaption.impliedVolatility`` of 1.15 x the NPV in the same model and of
  the NPV in the other vol type, and the time decay to the next business
  day from ``blackFormula`` / ``bachelierBlackFormula``.
* ``sabr``: ``ql.shiftedSabrVolatility`` (lognormal and normal) on rates
  parameter sets, including a negative forward; ``recovery``: smiles QuantLib
  generates from known SABR parameters at the grid offsets, which the cube
  calibration must recover.

Curves: the illustrative SOFR example curve, and the €STR / EURIBOR curve
sets of tools/golden/multicurve.py (synthetic, not market data).
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List

import QuantLib as ql

from derivx.rates.examples import EXAMPLE_ASOF, example_market
from derivx.rates.swaption_vol import GRID_OFFSETS_BP
from derivx.rates.swaptions import Swaption

from . import multicurve as mc
from .common import CALENDARS, pyd, qd, ql_curve, set_evaluation_date

NOTIONAL = 10_000_000.0
MODELS = {"bachelier": (0.0095, 0.0), "black": (0.27, 0.0), "shifted_black": (0.18, 0.02)}
EUR_NEGATIVE_MODELS = {"bachelier": (0.0060, 0.0), "shifted_black": (0.22, 0.03)}
USD_POINTS = [("1M", "1Y"), ("3M", "10Y"), ("1Y", "5Y"), ("2Y", "2Y"), ("5Y", "10Y"), ("10Y", "30Y"), ("6M", "30Y")]
EUR_POINTS = [("1M", "2Y"), ("1Y", "5Y"), ("5Y", "10Y"), ("10Y", "20Y")]
EUR_3M_POINTS = [("6M", "1Y"), ("2Y", "1Y")]
NEGATIVE_POINTS = [("1Y", "5Y"), ("5Y", "10Y")]
SABR_SETS = [  # (F, T, alpha, beta, nu, rho, shift)
    (0.036, 1.0, 0.0105, 0.0, 0.45, -0.10, 0.03),
    (0.024, 10.0, 0.0075, 0.0, 0.28, 0.15, 0.03),
    (-0.004, 2.0, 0.0060, 0.0, 0.50, 0.20, 0.03),
    (0.030, 1.0 / 12.0, 0.0110, 0.0, 1.00, -0.30, 0.02),
    (0.036, 5.0, 0.0400, 0.5, 0.30, -0.20, 0.03),
    (0.010, 1.0, 0.0350, 0.5, 0.60, 0.10, 0.02),
]
RECOVERY = [  # (fit, F, T, alpha, beta, nu, rho, shift)
    ("normal", 0.0358, 2.0, 0.0102, 0.0, 0.38, -0.12, 0.03),
    ("lognormal", 0.0263, 5.0, 0.0370, 0.5, 0.33, -0.25, 0.03),
]


def _vol_handle(asof: dt.date, cal: ql.Calendar, model: str, vol: float, shift: float):
    vol_type = ql.Normal if model == "bachelier" else ql.ShiftedLognormal
    surface = ql.ConstantSwaptionVolatility(qd(asof), cal, ql.ModifiedFollowing, vol, ql.Actual365Fixed(), vol_type,
                                            shift if model == "shifted_black" else 0.0)  # fmt: skip
    return ql.SwaptionVolatilityStructureHandle(surface)


def _engine(disc: ql.YieldTermStructureHandle, handle, model: str):
    if model == "bachelier":
        return ql.BachelierSwaptionEngine(disc, handle)
    return ql.BlackSwaptionEngine(disc, handle)


def _usd_swaption(asof: dt.date, expiry: str, tenor: str, strike: float, payer: bool, disc) -> ql.Swaption:
    """SOFR OIS swaption with QuantLib's own dates."""

    cal = CALENDARS["USD"]
    fixing = cal.advance(cal.adjust(qd(asof)), ql.Period(expiry), ql.ModifiedFollowing)
    start = cal.advance(fixing, 2, ql.Days)
    end = start + ql.Period(tenor)
    if end - start <= 12 * 31:  # one annual period (derivx's and the curve's rule for short swaps)
        schedule = ql.Schedule(ql.DateVector([cal.adjust(start, ql.ModifiedFollowing),
                                              cal.adjust(end, ql.ModifiedFollowing)]))  # fmt: skip
    else:
        schedule = ql.Schedule(start, end, ql.Period(12, ql.Months), cal, ql.ModifiedFollowing, ql.ModifiedFollowing,
                               ql.DateGeneration.Backward, False)  # fmt: skip
    side = ql.Swap.Payer if payer else ql.Swap.Receiver
    swap = ql.OvernightIndexedSwap(side, NOTIONAL, schedule, strike, ql.Actual360(), ql.Sofr(disc), 0.0, 2,
                                   ql.Following, cal)  # fmt: skip
    return ql.Swaption(swap, ql.EuropeanExercise(fixing))


def _eur_swaption(asof: dt.date, expiry: str, tenor: str, strike: float, payer: bool, disc, proj) -> ql.Swaption:
    index = ql.EuriborSwapIsdaFixA(ql.Period(tenor), proj, disc)
    side = ql.Swap.Payer if payer else ql.Swap.Receiver
    return ql.MakeSwaption(index, ql.Period(expiry), strike, underlyingType=side, nominal=NOTIONAL)


def _formula(model: str, payer: bool, strike: float, forward: float, std: float, shift: float) -> float:
    kind = ql.Option.Call if payer else ql.Option.Put
    if model == "bachelier":
        return ql.bachelierBlackFormula(kind, strike, forward, std)
    return ql.blackFormula(kind, strike, forward, std, 1.0, shift)


def _case(market_name: str, swaption: ql.Swaption, spec: Dict[str, object], disc, asof: dt.date) -> Dict[str, object]:
    model, vol, shift = spec["model"], spec["vol"], spec["shift"]
    cal = CALENDARS["USD"] if market_name == "usd" else CALENDARS["TARGET"]
    swaption.setPricingEngine(_engine(disc, _vol_handle(asof, cal, model, vol, shift), model))
    npv = swaption.NPV()
    payer = spec["kind"] == "payer"
    vol_type = ql.Normal if model == "bachelier" else ql.ShiftedLognormal
    implied = swaption.impliedVolatility(1.15 * npv, disc, vol, 1e-14, 1000, 1e-8, 10.0 if model != "bachelier"
                                         else 0.5, vol_type, shift if model == "shifted_black" else 0.0)  # fmt: skip
    other_type, other_shift = (ql.ShiftedLognormal, 0.02) if model == "bachelier" else (ql.Normal, 0.0)
    forward, annuity = swaption.atmForward(), swaption.annuity()
    other = None
    if other_type == ql.Normal or (forward + other_shift > 0 and spec["strike"] + other_shift > 0):
        other = swaption.impliedVolatility(npv, disc, 0.2 if other_type != ql.Normal else 0.01, 1e-14, 1000, 1e-8,
                                           10.0, other_type, other_shift)  # fmt: skip
    exercise = swaption.exercise().dates()[0]
    T = ql.Actual365Fixed().yearFraction(qd(asof), exercise)
    days = cal.advance(qd(asof), 1, ql.Days) - qd(asof)
    later = max(T - days / 365.0, 0.0)
    decay = annuity * _formula(model, payer, spec["strike"], forward, vol * later**0.5, shift) - npv
    under = swaption.underlying()
    fixed_leg = under.fixedLeg()
    return {
        **spec,
        "market": market_name,
        "expiry_date": pyd(exercise).isoformat(),
        "start": pyd(under.startDate()).isoformat(),
        "maturity": pyd(under.maturityDate()).isoformat(),
        "fixed_pay": [pyd(cf.date()).isoformat() for cf in fixed_leg],
        "npv": npv,
        "annuity": annuity,
        "forward": forward,
        "delta": swaption.delta(),
        "vega": swaption.vega(),
        "implied": {"price": 1.15 * npv, "vol": implied},
        "implied_other": None if other is None else {"model": "bachelier" if other_type == ql.Normal
                                                     else "shifted_black", "shift": other_shift, "vol": other},
        "decay": decay,
        "decay_days": days,
    }  # fmt: skip


def _derivx_forward(market, expiry: str, tenor: str, index) -> float:
    return Swaption(1.0, expiry, tenor, 0.0, index=index).inputs(market).forward


def swaption_cases() -> List[Dict[str, object]]:
    out = []
    usd = example_market()
    set_evaluation_date(EXAMPLE_ASOF)
    disc = ql_curve(usd.curve)
    for expiry, tenor in USD_POINTS:
        atm = _derivx_forward(usd, expiry, tenor, None)
        for offset in (-100.0, 0.0, 75.0):
            for kind in ("payer", "receiver"):
                for model, (vol, shift) in MODELS.items():
                    strike = round(atm + offset * 1e-4, 6)
                    spec = {"expiry": expiry, "tenor": tenor, "strike": strike, "kind": kind, "model": model,
                            "vol": vol, "shift": shift, "index": None}  # fmt: skip
                    sw = _usd_swaption(EXAMPLE_ASOF, expiry, tenor, strike, kind == "payer", disc)
                    out.append(_case("usd", sw, spec, disc, EXAMPLE_ASOF))
    for name, points, models in (("eur", EUR_POINTS + EUR_3M_POINTS, MODELS),
                                 ("eur_negative", NEGATIVE_POINTS, EUR_NEGATIVE_MODELS)):  # fmt: skip
        set_name = "base" if name == "eur" else "negative"
        asof = mc.SETS[set_name][0]
        market = mc.build(asof, mc.quote_set(set_name))
        set_evaluation_date(asof)
        disc = ql_curve(market.curve)
        for expiry, tenor in points:
            index = "EURIBOR 3M" if tenor == "1Y" else "EURIBOR 6M"
            proj = ql_curve(market.projection(index))
            atm = _derivx_forward(market, expiry, tenor, index)
            for offset in (0.0,) if tenor == "1Y" else (-50.0, 0.0, 100.0 if name == "eur" else 50.0):
                for kind in ("payer", "receiver"):
                    for model, (vol, shift) in models.items():
                        if model == "black" and atm + offset * 1e-4 <= 0:
                            continue
                        strike = round(atm + offset * 1e-4, 6)
                        spec = {"expiry": expiry, "tenor": tenor, "strike": strike, "kind": kind, "model": model,
                                "vol": vol, "shift": shift, "index": index}  # fmt: skip
                        sw = _eur_swaption(asof, expiry, tenor, strike, kind == "payer", disc, proj)
                        out.append(_case(name, sw, spec, disc, asof))
    set_evaluation_date(EXAMPLE_ASOF)
    return out


def sabr_cases() -> List[Dict[str, object]]:
    out = []
    for F, T, alpha, beta, nu, rho, shift in SABR_SETS:
        strikes = [F + o * 1e-4 for o in GRID_OFFSETS_BP if F + o * 1e-4 + shift > 0]
        row = {"F": F, "T": T, "alpha": alpha, "beta": beta, "nu": nu, "rho": rho, "shift": shift, "strikes": strikes}
        for name, vt in (("lognormal", ql.ShiftedLognormal), ("normal", ql.Normal)):
            row[name] = [ql.shiftedSabrVolatility(k, F, T, alpha, beta, nu, rho, shift, vt) for k in strikes]
        out.append(row)
    return out


def recovery_cases() -> List[Dict[str, object]]:
    out = []
    for fit, F, T, alpha, beta, nu, rho, shift in RECOVERY:
        vt = ql.Normal if fit == "normal" else ql.ShiftedLognormal
        vols = [ql.shiftedSabrVolatility(F + o * 1e-4, F, T, alpha, beta, nu, rho, shift, vt) for o in GRID_OFFSETS_BP]
        out.append({"fit": fit, "F": F, "T": T, "alpha": alpha, "beta": beta, "nu": nu, "rho": rho, "shift": shift,
                    "offsets": list(GRID_OFFSETS_BP), "vols": vols})  # fmt: skip
    return out


def generate() -> dict:
    return {
        "asof": EXAMPLE_ASOF.isoformat(),
        "sets": {name: {"asof": mc.SETS[name][0].isoformat(), "quotes": mc.quote_set(name)}
                 for name in ("base", "negative")},
        "cases": swaption_cases(),
        "sabr": sabr_cases(),
        "recovery": recovery_cases(),
    }  # fmt: skip
