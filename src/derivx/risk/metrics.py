from __future__ import annotations

import warnings
from typing import Dict

import numpy as np

from ..rates.term_structure import ZeroCurve


def scen_shift_curve(
    curve: ZeroCurve, parallel_bp: float = 0.0, twist_bp: float = 0.0
) -> ZeroCurve:
    """Deprecated: shift zero rates in bp (``twist`` ramps linearly from 0 at
    t=0, so it also moves the curve in parallel). Use
    :mod:`derivx.risk.scenarios`, which shocks quotes and rebuilds the curve.
    """

    warnings.warn(
        "scen_shift_curve is deprecated; use derivx.risk.scenarios", DeprecationWarning, stacklevel=2
    )

    ts = curve.ts.copy()
    bp_to_dec = 1e-4
    denom = ts.max() if ts.size and ts.max() > 0 else 1.0
    shift = (parallel_bp * bp_to_dec) + (twist_bp * bp_to_dec) * (ts / denom)
    dfs = curve.dfs * np.exp(-shift * ts)
    return ZeroCurve(ts=ts, dfs=dfs)


def dv01_pv01_buckets(instrument_spec: Dict, curve: ZeroCurve, bump: float = 1e-4) -> Dict[str, float]:
    """Compute DV01/PV01 via bump-and-revalue for a generic instrument spec.

    instrument_spec should define a callable 'pv(curve) -> float'.

    Deprecated: this is dPV/dr per unit of rate on a zero-rate bump. Use
    :func:`derivx.risk.sensitivities.parallel_pv01` (P&L per +1bp of quotes).
    """

    warnings.warn(
        "dv01_pv01_buckets is deprecated; use derivx.risk.sensitivities", DeprecationWarning, stacklevel=2
    )

    pv = float(instrument_spec["pv"](curve))
    bumped = ZeroCurve(ts=curve.ts, dfs=curve.dfs * np.exp(-bump * curve.ts))
    pv_b = float(instrument_spec["pv"](bumped))
    dv01 = (pv_b - pv) / bump
    return {"PV": pv, "DV01": dv01, "PV01": dv01}


