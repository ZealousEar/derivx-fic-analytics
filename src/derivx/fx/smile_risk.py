"""Smile vegas of an FX option: by market quote and by SABR parameter.

* :func:`quote_vegas` - the P&L of an option priced on the tenor's smile
  when one market quote (ATM, RR25, BF25 and, when quoted, RR10, BF10)
  rises by one vol point and the smile is rebuilt from the bumped quotes
  with the same model and conventions (:func:`derivx.fx.smile.build_smile`),
  at the same strike. Central differences of +-``h`` (5 vol bp), scaled to
  one vol point (first order). The ATM line is the vega to a parallel move
  of the quoted smile (RR and BF are spreads to ATM); RR is the skew
  (vanna) exposure and BF the wing (volga) exposure. Quotes the model does
  not fit (vanna-volga checks the 10-delta pair but goes through the
  25-delta pillars only) have zero vega and say so.
* :func:`sabr_parameter_vegas` - dV/dp for the SABR smile's alpha, rho and
  nu (beta = 1 fixed), at the Garman-Kohlhagen price on the smile vol.

Units: premium in quote currency per one unit of base currency on the
spot date (multiply by the base notional), per +1 vol point (0.01) or per
+1.00 / +0.01 of a parameter; signed as P&L of a long option.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd

from derivx.fx.pricing import FxMarket, gk_price
from derivx.fx.smile import FxSmile, SmileConventions, TenorQuote, build_smile
from derivx.vol.sabr import hagan_sabr_iv

QUOTE_BUMP = 5e-4  # +-5 vol bp central difference, scaled to one vol point


def _quote_names(quote: TenorQuote) -> List[str]:
    names = ["atm", "rr25", "bf25"]
    if quote.has_10d:
        names += ["rr10", "bf10"]
    return names


def _price(m: FxMarket, smile: FxSmile, K: float, call: bool) -> float:
    vol = smile.vol(K)
    if not (np.isfinite(vol) and vol > 0):
        raise ValueError(f"the {smile.model} smile has no vol at K = {K:.6g}")
    return float(gk_price(m, K, vol, call))


def quote_vegas(
    m: FxMarket,
    quote: TenorQuote,
    conventions: SmileConventions,
    model: str,
    K: float,
    call: bool,
    h: float = QUOTE_BUMP,
) -> pd.DataFrame:
    """Vega of an option at strike ``K`` to each market quote of its tenor
    (module docs). Columns: quote, quoted (decimal), vega (per +1 vol point
    on that quote), used (whether the model fits that quote)."""

    if not (np.isfinite(K) and K > 0):
        raise ValueError(f"strike must be > 0, got {K}")
    if m.tau_e <= 0:
        raise ValueError("at expiry an option has no vega")
    base = build_smile(m, quote, conventions, model)
    used = {q.lower() for q in base.used_quotes}
    rows: List[Dict[str, object]] = []
    for name in _quote_names(quote):
        value = float(getattr(quote, name))
        if name not in used:
            rows.append({"quote": name.upper(), "quoted": value, "vega": 0.0, "used": False})
            continue
        up = build_smile(m, replace(quote, **{name: value + h}), conventions, model)
        dn = build_smile(m, replace(quote, **{name: value - h}), conventions, model)
        d = (_price(m, up, K, call) - _price(m, dn, K, call)) / (2.0 * h)
        rows.append({"quote": name.upper(), "quoted": value, "vega": 0.01 * d, "used": True})
    return pd.DataFrame(rows)


def sabr_parameter_vegas(smile: FxSmile, K: float, call: bool) -> pd.DataFrame:
    """dV/dp for alpha, rho and nu of a SABR smile (beta fixed) at strike
    ``K`` (central differences). Columns: parameter, value, dV_dp (per
    +1.00), per_0.01. ``ValueError`` for a vanna-volga smile."""

    if smile.model != "SABR":
        raise ValueError("parameter vegas need a SABR smile (vanna-volga has no model parameters)")
    m = smile.market
    if m.tau_e <= 0:
        raise ValueError("at expiry an option has no vega")
    p = {k: float(smile.params[k]) for k in ("alpha", "beta", "rho", "nu")}

    def price(q: Dict[str, float]) -> float:
        vol = hagan_sabr_iv(m.forward, K, m.tau_e, q["alpha"], q["beta"], q["rho"], q["nu"])
        return float(gk_price(m, K, vol, call))

    rows = []
    for name in ("alpha", "rho", "nu"):
        h = 1e-6 * max(abs(p[name]), 1e-3)
        up, dn = dict(p), dict(p)
        up[name] += h
        dn[name] = max(dn[name] - h, 0.0) if name == "nu" else dn[name] - h
        d = (price(up) - price(dn)) / (up[name] - dn[name])
        rows.append({"parameter": name, "value": p[name], "dV_dp": d, "per_0.01": 0.01 * d})
    return pd.DataFrame(rows)
