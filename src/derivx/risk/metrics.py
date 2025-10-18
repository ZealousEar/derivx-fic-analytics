from __future__ import annotations

from typing import Dict

import numpy as np

from ..rates.term_structure import ZeroCurve


def scen_shift_curve(
    curve: ZeroCurve, parallel_bp: float = 0.0, twist_bp: float = 0.0
) -> ZeroCurve:
    """Apply parallel and twist shifts (in bp) to a curve by adjusting dfs."""

    ts = curve.ts.copy()
    bp_to_dec = 1e-4
    denom = ts.max() if ts.size and ts.max() > 0 else 1.0
    shift = (parallel_bp * bp_to_dec) + (twist_bp * bp_to_dec) * (ts / denom)
    dfs = curve.dfs * np.exp(-shift * ts)
    return ZeroCurve(ts=ts, dfs=dfs)


def dv01_pv01_buckets(instrument_spec: Dict, curve: ZeroCurve, bump: float = 1e-4) -> Dict[str, float]:
    """Compute DV01/PV01 via bump-and-revalue for a generic instrument spec.

    instrument_spec should define a callable 'pv(curve) -> float'.
    """

    pv = float(instrument_spec["pv"](curve))
    bumped = ZeroCurve(ts=curve.ts, dfs=curve.dfs * np.exp(-bump * curve.ts))
    pv_b = float(instrument_spec["pv"](bumped))
    dv01 = (pv_b - pv) / bump
    return {"PV": pv, "DV01": dv01, "PV01": dv01}


