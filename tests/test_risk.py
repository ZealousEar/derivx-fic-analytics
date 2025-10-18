import numpy as np

from derivx.rates.term_structure import ZeroCurve
from derivx.rates.black_ir import price_cap_floor
from derivx.risk.metrics import scen_shift_curve, dv01_pv01_buckets


def _flat_curve(rate: float, ts: np.ndarray) -> ZeroCurve:
    dfs = np.exp(-rate * ts)
    return ZeroCurve(ts=ts, dfs=dfs)


def test_dv01_matches_bump_cap():
    ts = np.linspace(0.5, 5.0, 10)
    curve = _flat_curve(0.03, ts)
    vol_curve = ZeroCurve(ts=ts, dfs=np.full_like(ts, 0.2))

    def pv_fn(cv: ZeroCurve) -> float:
        res = price_cap_floor(cv, 0.03, ts, vol_curve)
        return float(res["PV"])

    inst = {"pv": pv_fn}
    res = dv01_pv01_buckets(inst, curve, bump=1e-4)

    bump = 1e-4
    bumped_curve = scen_shift_curve(curve, parallel_bp=bump * 1e4)
    pv_bump = pv_fn(bumped_curve)
    dv_fd = (pv_bump - pv_fn(curve)) / bump
    assert abs(res["DV01"] - dv_fd) < 2e-4


def test_scenario_shift_parallel():
    ts = np.linspace(0.5, 5.0, 10)
    curve = _flat_curve(0.02, ts)
    shifted = scen_shift_curve(curve, parallel_bp=100)
    # Parallel +100bp should reduce dfs noticeably
    assert np.all(shifted.dfs < curve.dfs)

