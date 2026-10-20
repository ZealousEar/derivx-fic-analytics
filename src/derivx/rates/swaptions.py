"""European swaptions: payer, receiver and straddle, physically settled.

A payer swaption gives the right, on its expiry date, to enter the
underlying swap paying the fixed ``strike``; a receiver the right to
receive it; a straddle is one of each at the same strike. Physically
settled, its value under the annuity measure is::

    PV = N * A * E[(w (S_T - K))^+],    A = sum_i tau_i DF(pay_i)

over the underlying's fixed leg (``w = +1`` payer, ``-1`` receiver). ``A``
(the annuity, per unit notional and unit rate) discounts on the OIS curve
``market.curve``, and the forward swap rate ``S`` is the underlying's float
leg value over ``A``: an OIS leg (``index=None``, the curve's own
conventions: SOFR, €STR, SONIA) is projected on the OIS curve, a term-index
leg (``index="EURIBOR 6M"``) on that index's projection curve, every flow
discounted on OIS (multi-curve). These are the par rate and annuity of
:class:`~derivx.rates.swaps.VanillaSwap`, so the underlying is the swap the
SWAP tab prices.

The expectation is taken under one of three models (``model``):

* ``bachelier``     - ``S_T = S + vol W_T``: normal vol in rate units
  (1bp = 1e-4 per sqrt(year)); any sign of S and K
* ``black``         - lognormal S: vol a decimal (0.20 = 20%), needs S, K > 0
* ``shifted_black`` - lognormal ``S + shift``: needs S, K > -shift

Dates follow QuantLib's ``MakeSwaption``: the expiry is the option tenor
after the valuation date (moved to a business day first), modified
following on the underlying's calendar; the underlying swap starts its spot
lag of business days after the expiry and runs ``tenor`` from there.
Option time is ACT/365F from the valuation date to the expiry. On the
expiry date a swaption is worth its intrinsic value (it is exercised into
the swap if in the money); after it, it is no longer a swaption and pricing
raises.

Signs and units: every value is in currency for the notional, signed to the
holder (``notional < 0`` is a sold swaption). Risk is P&L for a + move:

* ``delta_fwd``   - per +1bp of the forward swap rate, annuity held
* ``gamma_fwd``   - change in ``delta_fwd`` per +1bp (P&L ~ delta x + gamma x^2 / 2, x in bp)
* ``vega_normal_bp`` - per +1bp of normal vol; ``vega_lognormal_pt`` per
  +1 vol point (1%) of (shifted) lognormal vol, each at the vol of that type
  that gives the same price
* ``decay``       - time decay to the next business day with the forward,
  annuity and vol held (the full 1-day theta, curves rolled, is
  :func:`derivx.risk.scenarios.theta`)
* ``delta_notional`` - notional of the payer swap with the same forward
  delta (``N * dV/dS``; negative means a receiver swap)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from derivx.rates.capfloor import MODELS, VOL_UNIT, model_delta, model_gamma, model_price, model_vega
from derivx.rates.conventions import Calendar, Date, parse_tenor
from derivx.rates.curves import CurveConventions, Market, get_conventions
from derivx.rates.indices import get_index
from derivx.rates.swaps import VanillaSwap, _periods
from derivx.risk.sensitivities import Position
from derivx.vol.implied import STATUS as IMPLIED_STATUS
from derivx.vol.implied import implied_black_vol, implied_normal_vol, lognormal_to_normal_vol, normal_to_lognormal_vol

KINDS = ("payer", "receiver", "straddle")
EXPIRY_UNITS = ("W", "M", "Y")
DEFAULT_CONVENTIONS = "USD SOFR"


@dataclass(frozen=True)
class SwaptionInputs:
    """Market inputs of a swaption on one valuation date.

    ``annuity`` is per unit notional and unit rate (sum of fixed accrual x
    OIS discount factor); ``forward`` the forward swap rate (decimal);
    ``expiry_years`` ACT/365F from the valuation date to the expiry."""

    expiry_date: Date
    start: Date
    maturity: Date
    expiry_years: float
    forward: float
    annuity: float


LEGS = {"payer": (True,), "receiver": (False,), "straddle": (True, False)}  # call (payer) flags per kind


@dataclass(frozen=True)
class Swaption:
    """European swaption on a fixed/float swap (see the module docstring).

    ``expiry`` is the option tenor (``"1Y"``) counted from the valuation
    date; give ``expiry_date`` instead (or call :meth:`pinned`) to keep the
    expiry fixed as the market moves on. ``tenor`` is the underlying swap
    tenor. ``strike`` and ``shift`` are decimal rates; ``vol`` is in the
    model's units (normal: rate units, 0.0085 = 85bp; lognormal: decimal).
    ``index`` names the float leg's term index (multi-curve); ``None`` is an
    OIS swap on ``conventions`` (default: the market's discount curve's).
    """

    notional: float
    expiry: Optional[str]
    tenor: str
    strike: float
    kind: str = "payer"
    vol: float = 0.0
    model: str = "bachelier"
    shift: float = 0.0
    index: Optional[str] = None
    conventions: Optional[str] = None
    expiry_date: Optional[Date] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind).strip().lower())
        object.__setattr__(self, "model", str(self.model).strip().lower())
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        if not math.isfinite(self.notional) or self.notional == 0.0:
            raise ValueError("notional must be a non-zero number (negative for a sold swaption)")
        if not math.isfinite(self.strike):
            raise ValueError("strike must be a finite decimal rate")
        if not math.isfinite(self.vol) or self.vol < 0.0:
            raise ValueError(f"vol must be a non-negative number in the model's units, got {self.vol!r}")
        if not math.isfinite(self.shift) or self.shift < 0.0:
            raise ValueError("shift must be zero or a positive decimal")
        if self.model == "black" and self.strike <= 0.0:
            raise ValueError(
                f"Black needs a positive strike (got {self.strike:.4%}); use the normal or shifted Black model"
            )
        if self.model == "shifted_black" and self.strike + self.shift <= 0.0:
            raise ValueError(
                f"shifted Black needs strike + shift > 0 (strike {self.strike:.4%}, shift {self.shift:.4%})"
            )
        if self.expiry_date is None:
            if not self.expiry:
                raise ValueError("give an option expiry tenor (e.g. 1Y) or an expiry date")
            if parse_tenor(self.expiry).unit not in EXPIRY_UNITS:
                raise ValueError(f"option expiry must be weeks, months or years, got {self.expiry!r}")
        tenor = parse_tenor(self.tenor)
        if tenor.months is None:
            raise ValueError(f"the underlying swap tenor must be months or years, got {self.tenor!r}")
        if self.index is not None:
            object.__setattr__(self, "index", get_index(self.index).name)
            if self.conventions is not None:
                raise ValueError("give either a term index or OIS conventions for the underlying, not both")
        elif self.conventions is not None:
            get_conventions(self.conventions)

    # -- conventions ----------------------------------------------------------

    def _conventions(self, market: Market) -> CurveConventions:
        if self.conventions is not None:
            return get_conventions(self.conventions)
        conv = market.curve.conventions
        return conv if conv is not None else get_conventions(DEFAULT_CONVENTIONS)

    def calendar(self, market: Market) -> Calendar:
        return get_index(self.index).cal if self.index is not None else self._conventions(market).cal

    @property
    def vol_model(self) -> str:
        """``normal`` or ``lognormal``: which market vol shift moves it."""

        return "normal" if self.model == "bachelier" else "lognormal"

    @property
    def label(self) -> str:
        side = {"payer": "PAYER", "receiver": "RECEIVER", "straddle": "STRADDLE"}[self.kind]
        expiry = self.expiry_date.isoformat() if self.expiry_date is not None else self.expiry
        under = self.index or "OIS"
        return f"{'' if self.notional > 0 else 'SOLD '}{expiry}x{self.tenor} {side} {self.strike:.3%} {under}"

    def expiry_on(self, market: Market) -> Date:
        """Expiry date for a valuation date: ``expiry_date`` as given, or the
        option tenor after it (modified following, as ``MakeSwaption``)."""

        if self.expiry_date is not None:
            return self.expiry_date
        cal = self.calendar(market)
        return cal.advance(cal.adjust(market.asof, "F"), self.expiry, "MF")

    def pinned(self, market: Market) -> "Swaption":
        """Same trade with its expiry date fixed, so it keeps its dates when
        valued on a later date (scenario horizons, theta)."""

        if self.expiry_date is not None:
            return self
        return replace(self, expiry_date=self.expiry_on(market), expiry=None)

    def underlying(self, market: Market) -> VanillaSwap:
        """The swap entered on exercise (payer swap for a payer swaption),
        for the notional's size, starting spot from the expiry."""

        conv = self._conventions(market) if self.index is None else None
        return _underlying(abs(self.notional), self.kind == "receiver", self.tenor, self.strike, self.index, conv,
                           self.expiry_on(market))  # fmt: skip

    # -- valuation ------------------------------------------------------------

    def inputs(self, market: Market) -> SwaptionInputs:
        """Expiry, forward swap rate and annuity on ``market``. Raises once
        the swaption has expired."""

        expiry = self.expiry_on(market)
        if expiry < market.asof:
            raise ValueError(
                f"the swaption expired on {expiry} (valuation date {market.asof}); physically settled, it is now the "
                "underlying swap if it was exercised: value that on the SWAP tab"
            )
        swap = self.underlying(market)
        forward, annuity = forward_and_annuity(swap, market)
        periods = _periods(swap, market.asof)
        return SwaptionInputs(
            expiry_date=expiry,
            start=periods[0].start,
            maturity=max(p.end for p in periods),
            expiry_years=(expiry - market.asof).days / 365.0,
            forward=forward,
            annuity=annuity,
        )

    def market_vol(self, market: Market) -> float:
        """Vol after the market's scenario vol shift (floored at zero, as the
        cap pricer does)."""

        bump = market.vol_shift_normal if self.model == "bachelier" else market.vol_shift_lognormal
        return max(self.vol + bump, 0.0)

    def unit_value(self, inputs: SwaptionInputs, vol: float, expiry_years: Optional[float] = None) -> float:
        """Undiscounted option value per unit annuity at ``vol`` (model units)."""

        T = inputs.expiry_years if expiry_years is None else expiry_years
        legs = LEGS[self.kind]
        try:
            return sum(model_price(self.model, inputs.forward, self.strike, T, vol, self.shift, c) for c in legs)
        except ValueError as err:
            raise ValueError(f"{self.label}: {err}") from None

    def value(self, inputs: SwaptionInputs, vol: float) -> float:
        """PV in currency at ``vol`` (model units) on resolved inputs."""

        return self.notional * inputs.annuity * self.unit_value(inputs, vol)

    def pv(self, market: Market) -> float:
        """Present value to the holder, in currency."""

        return self.value(self.inputs(market), self.market_vol(market))

    def premium_bp(self, market: Market) -> float:
        """PV in bp of the notional (signed to the holder)."""

        return self.pv(market) / abs(self.notional) * 1e4

    def atm_strike(self, market: Market) -> float:
        """The forward swap rate: the strike at which payer and receiver are
        worth the same."""

        return self.inputs(market).forward

    def cash(self, market: Market, start: Date, end: Date) -> float:
        """A swaption pays nothing before expiry (the premium is paid at
        trade); valuing it after expiry raises (see :meth:`inputs`)."""

        return 0.0

    # -- vols -----------------------------------------------------------------

    def equivalent_vols(self, inputs: SwaptionInputs, vol: float) -> Dict[str, float]:
        """Normal vol and (shifted, by ``self.shift``) lognormal vol with the
        same price as ``vol`` in the trade's model, by exact price matching;
        NaN where one does not exist (e.g. lognormal for F + shift <= 0)."""

        F, K, T = inputs.forward, self.strike, inputs.expiry_years
        if T <= 0.0 or vol == 0.0:
            return {"normal": 0.0 if vol == 0.0 else float("nan"), "lognormal": 0.0 if vol == 0.0 else float("nan")}
        if self.model == "bachelier":
            normal = vol
            if F + self.shift > 0 and K + self.shift > 0:
                lognormal = float(normal_to_lognormal_vol(F, K, T, vol, shift=self.shift))
            else:
                lognormal = float("nan")
        else:
            lognormal = vol
            shift = self.shift if self.model == "shifted_black" else 0.0
            normal = float(lognormal_to_normal_vol(F, K, T, vol, shift=shift))
        return {"normal": normal, "lognormal": lognormal}

    def implied_vol(
        self, pv: float, market: Market, model: Optional[str] = None, shift: Optional[float] = None
    ) -> float:
        """Vol (in ``model``'s units, default the trade's) that reprices the
        swaption to ``pv`` in currency (signed to the holder like
        :meth:`pv`). Raises ``ValueError`` with the reason when no vol does
        (price below intrinsic, above the infinite-vol bound, expired)."""

        model = self.model if model is None else str(model).lower()
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}")
        shift = (self.shift if shift is None else float(shift)) if model == "shifted_black" else 0.0
        if not math.isfinite(pv):
            raise ValueError("premium must be a finite number")
        inputs = self.inputs(market)
        if inputs.expiry_years <= 0.0:
            raise ValueError("the swaption expires today: there is no time value to imply a vol from")
        unit = pv / (self.notional * inputs.annuity)
        legs = LEGS[self.kind]
        F, K, T = inputs.forward, self.strike, inputs.expiry_years
        if self.kind == "straddle":
            return _implied_straddle(unit, F, K, T, model, shift)
        solve = implied_normal_vol if model == "bachelier" else implied_black_vol
        kwargs = {} if model == "bachelier" else {"shift": shift}
        result = solve(unit, F, K, T, is_call=legs[0], **kwargs)
        if not bool(result.ok):
            status = str(result.status)
            raise ValueError(f"no {model} vol reproduces a premium of {pv:,.2f}: {IMPLIED_STATUS.get(status, status)}")
        return float(result.vol)

    # -- risk -----------------------------------------------------------------

    def greeks(self, market: Market, decay_days: Optional[int] = None) -> Dict[str, float]:
        """Value and analytic risk on ``market`` (see the module docstring
        for units and signs). ``decay_days`` defaults to the calendar days to
        the next business day of the underlying's calendar."""

        inputs = self.inputs(market)
        vol = self.market_vol(market)
        F, K, T, N, A = inputs.forward, self.strike, inputs.expiry_years, self.notional, inputs.annuity
        legs = LEGS[self.kind]
        try:
            delta = sum(model_delta(self.model, F, K, T, vol, self.shift, c) for c in legs)
            gamma = model_gamma(self.model, F, K, T, vol, self.shift) * len(legs)  # same for payer and receiver
            vega = model_vega(self.model, F, K, T, vol, self.shift) * len(legs)
        except ValueError as err:
            raise ValueError(f"{self.label}: {err}") from None
        if decay_days is None:
            cal = self.calendar(market)
            decay_days = (cal.add_business_days(market.asof, 1) - market.asof).days
        later = max(T - decay_days / 365.0, 0.0)
        pv = N * A * self.unit_value(inputs, vol)
        decay = N * A * self.unit_value(inputs, vol, later) - pv
        eq = self.equivalent_vols(inputs, vol)
        vega_n = vega_ln = float("nan")
        if T > 0.0:
            if np.isfinite(eq["normal"]):
                per_leg = model_vega("bachelier", F, K, T, eq["normal"])
                vega_n = N * A * per_leg * len(legs) * VOL_UNIT["bachelier"]
            if np.isfinite(eq["lognormal"]) and F + self.shift > 0 and K + self.shift > 0:
                per_leg = model_vega("shifted_black", F, K, T, eq["lognormal"], self.shift)
                vega_ln = N * A * per_leg * len(legs) * VOL_UNIT["black"]
        return {
            "pv": pv,
            "premium_bp": pv / abs(N) * 1e4,
            "forward": F,
            "strike": K,
            "atm_offset_bp": (K - F) * 1e4,
            "annuity_bp": abs(N) * A * 1e-4,
            "expiry_years": T,
            "vol": vol,
            "normal_vol": eq["normal"],
            "lognormal_vol": eq["lognormal"],
            "delta_fwd": N * A * delta * 1e-4,
            "delta_notional": N * delta,
            "gamma_fwd": N * A * gamma * 1e-8,
            "vega": N * A * vega * VOL_UNIT[self.model],
            "vega_normal_bp": vega_n,
            "vega_lognormal_pt": vega_ln,
            "decay": decay,
            "decay_days": float(decay_days),
        }


def _implied_straddle(unit: float, F: float, K: float, T: float, model: str, shift: float) -> float:
    """Vol of a straddle (payer + receiver) worth ``unit`` per unit annuity:
    the price is increasing in vol from |F - K| (zero vol)."""

    from scipy.optimize import brentq

    def value(s: float) -> float:
        return model_price(model, F, K, T, s, shift, True) + model_price(model, F, K, T, s, shift, False)

    lower = value(0.0)
    if not math.isfinite(unit) or unit <= lower * (1.0 + 1e-14):
        raise ValueError(f"a straddle premium must be above its intrinsic value {lower:.6g} per unit annuity")
    hi, cap = (0.01, 10.0) if model == "bachelier" else (0.5, 100.0)
    while value(hi) < unit:
        hi *= 2.0
        if hi > cap:
            raise ValueError("the straddle premium implies an implausibly high vol: check the premium")
    return float(brentq(lambda s: value(s) - unit, 0.0, hi, xtol=1e-16, rtol=4 * np.finfo(float).eps, maxiter=500))


@lru_cache(maxsize=4096)
def _underlying(
    notional: float,
    receive_fixed: bool,
    tenor: str,
    strike: float,
    index: Optional[str],
    conv: Optional[CurveConventions],
    expiry: Date,
) -> VanillaSwap:
    """Swap entered on exercise: a term-index swap starting at the index's
    value date of the expiry, or an OIS swap ``spot_lag`` business days
    after it."""

    if index is not None:
        term = get_index(index)
        return VanillaSwap.from_index(term, tenor, strike, notional, receive_fixed, start=term.value_date(expiry))
    start = conv.cal.add_business_days(expiry, conv.spot_lag)
    return VanillaSwap.from_conventions(conv, tenor, strike, notional, receive_fixed, kind="OIS", start=start)


def forward_and_annuity(swap: VanillaSwap, market: Market) -> Tuple[float, float]:
    """(forward swap rate, annuity per unit notional) of a swap's remaining
    cash flows: the par rate and sum of fixed accrual x OIS discount factor,
    in one discount-curve call."""

    periods = _periods(swap, market.asof)
    live = [p for p in periods if p.pay > market.asof]
    fixed = [p for p in live if p.leg == "fixed"]
    floats = [p for p in live if p.leg == "float"]
    if not fixed or not floats:
        raise ValueError("the underlying swap has no payments left")
    fixed_df = np.asarray(market.curve.df([p.pay for p in fixed]), dtype=float)
    float_df = np.asarray(market.curve.df([p.pay for p in floats]), dtype=float)
    annuity = float(np.dot([p.accrual for p in fixed], fixed_df))
    rates = swap._float_rates(market, floats)
    float_pv = float(np.dot(rates * np.array([p.accrual for p in floats]), float_df))
    return float_pv / annuity, annuity


def swaption_position(swaption: Swaption, market: Optional[Market] = None, name: Optional[str] = None) -> Position:
    """Wrap a swaption for the risk and scenario tools. With ``market`` the
    expiry is pinned to that valuation date (see :meth:`Swaption.pinned`)."""

    trade = swaption.pinned(market) if market is not None else swaption
    return Position(
        name=name or swaption.label,
        pv=trade.pv,
        kind="swaption",
        vol_model=trade.vol_model,
        cash=trade.cash,
        details={
            "type": trade.kind,
            "notional": trade.notional,
            "expiry": swaption.expiry,
            "expiry_date": trade.expiry_date,
            "tenor": trade.tenor,
            "strike": trade.strike,
            "model": trade.model,
            "vol": trade.vol,
            "shift": trade.shift,
            "index": trade.index,
            "trade": trade,
        },
    )


def atm_straddle_approx_bp(normal_vol: float, expiry_years: float) -> float:
    """Bachelier ATM straddle per unit annuity, in bp: ``vol sqrt(2T/pi)``
    (exact for the normal model at K = F)."""

    return normal_vol * math.sqrt(2.0 * expiry_years / math.pi) * 1e4


__all__ = [
    "KINDS",
    "Swaption",
    "SwaptionInputs",
    "atm_straddle_approx_bp",
    "forward_and_annuity",
    "swaption_position",
]
