"""Daily smile-parameter history and its stability: SVI and SABR fitted to
every (trade date, expiry) slice of a history of option-chain snapshots,
and the day-to-day parameter changes, with jumps flagged.

History (:func:`parameter_history`): each snapshot is cleaned with
:func:`derivx.calibration.production.clean_chain` (OTM quotes, one per
strike, put-call-parity forwards, vols re-inverted from mids with the given
discounting), kept within ``|k| <= std_band * atm_vol * sqrt(tau)``, and
needs ``min_points`` quotes with two on each side of the forward. Then:

* SVI (raw, total variance for the slice's tau; soft-L1 loss at 50 vol bp;
  with ``warm_start`` each fit starts from the previous snapshot's
  parameters for the same expiry, as the walk-forward does, else from the
  usual multi-start seeds, so independent fits can land on different but
  near-equivalent optima) - parameters ``a, b, rho, m, sigma``, plus the
  smile's own ATM vol, skew
  ``d sigma / dk`` and curvature ``d2 sigma / dk2`` at k = 0
  (:func:`derivx.vol.smile_risk.svi_atm_skew_curvature`);
* SABR (Hagan lognormal, ``beta`` fixed, default 1) - ``alpha, rho, nu``.

Each row is one (trade date, expiry, model, parameter) value with the
fit's RMSE in vol bp; slices that cannot be fitted are listed with the
reason.

Stability (:func:`parameter_jumps`): for the same expiry (the same listed
contract, so tau shrinks by the days elapsed) on consecutive snapshots of
the history, the change of each parameter. A change is a *jump* when it is
more than ``z`` robust standard deviations from the median change of that
(model, parameter) across the history - robust standard deviation = 1.4826
x the median absolute deviation, floored at ``floor`` x the parameter's
median absolute level (so a near-constant parameter does not flag noise) -
see :data:`ABS_FLOOR` for absolute floors. Raw SVI parameters trade off
against each other (a, b and sigma), so they jump even when the smile does
not: compare with the jumps of the smile's ATM vol, skew and curvature,
which are what a trader marks.

Units: SVI parameters are total-variance parameters (a in variance units,
b per unit k, rho and m dimensionless / in k, sigma in k); ATM vol and
SABR rho, nu are decimals per annum; skew per unit k; curvature per unit
k^2; SABR alpha with beta = 1 is a lognormal vol (decimal). Changes are in
the parameter's own units.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from derivx.calibration.production import clean_chain
from derivx.data.calendar import trading_days_between
from derivx.vol.discounting import Discounting
from derivx.vol.quotes import apply_std_band
from derivx.vol.sabr import calibrate_sabr_smile
from derivx.vol.smile_risk import svi_atm_skew_curvature
from derivx.vol.svi import fit_svi_strike_slice

HISTORY_MODELS = ("svi", "sabr")
SVI_SHAPE = ("atm_vol", "skew", "curvature")
PARAMETERS: Dict[str, Tuple[str, ...]] = {
    "svi": ("a", "b", "rho", "m", "sigma") + SVI_SHAPE,
    "sabr": ("alpha", "rho", "nu"),
}
ABS_FLOOR: Dict[str, float] = {"atm_vol": 5e-4, "alpha": 5e-4}  # 5 vol bp: below this no ATM move is a jump
# display order of parameters per model (including the rates-vol history's normal SABR)
PARAMETER_ORDER: Dict[str, Tuple[str, ...]] = {**PARAMETERS, "normal_sabr": ("atm_normal_bp", "alpha", "rho", "nu")}


def _days(chains: Mapping[object, pd.DataFrame]) -> List[Tuple[pd.Timestamp, pd.DataFrame]]:
    out = []
    for key, df in chains.items():
        if df is None or df.empty:
            continue
        td = pd.Timestamp(key).normalize() if "trade_date" not in df.columns else None
        if td is None:
            tds = pd.to_datetime(df["trade_date"]).dt.normalize().unique()
            if len(tds) != 1:
                raise ValueError(f"snapshot {key} holds {len(tds)} trade dates: pass one snapshot per trade date")
            td = pd.Timestamp(tds[0])
        else:
            df = df.assign(trade_date=td)
        out.append((td, df))
    out.sort(key=lambda p: p[0])
    dates = [d for d, _ in out]
    if len(set(dates)) != len(dates):
        raise ValueError("two snapshots share a trade date")
    return out


def parameter_history(
    chains: Mapping[object, pd.DataFrame],
    *,
    discount: Optional[Discounting] = None,
    rate: float = 0.0,
    models: Tuple[str, ...] = HISTORY_MODELS,
    sabr_beta: float = 1.0,
    std_band: Optional[float] = 3.0,
    min_points: int = 8,
    min_days: float = 2.0,
    max_days: Optional[float] = None,
    warm_start: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit ``models`` to every slice of ``chains`` (``{trade_date:
    snapshot}``; see the module docs). Returns ``(history, skipped)``:
    history columns trade_date, expiry, days, tau, forward, model,
    parameter, value, rmse_bp; skipped columns trade_date, expiry, model,
    reason."""

    unknown = [m for m in models if m not in PARAMETERS]
    if unknown or not models:
        raise ValueError(f"models must be a non-empty subset of {HISTORY_MODELS}, got {list(models)}")
    if min_points <= 5:
        raise ValueError("min_points must exceed the 5 SVI parameters (use >= 6)")
    rows: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    warm: Dict[pd.Timestamp, Dict[str, float]] = {}
    for td, df in _days(chains):
        cleaned, bad = clean_chain(df, rate=rate, min_days=1.0, discount=discount)
        skipped += [{"trade_date": td, "expiry": s.expiry, "model": "", "reason": s.reason} for s in bad]
        for _, exp, _, clean in cleaned:
            if clean.days < min_days or (max_days is not None and clean.days > max_days):
                continue
            banded = apply_std_band(clean, std_band) if std_band is not None and not clean.quotes.empty else clean
            q = banded.quotes
            n_left, n_right = int((q["k"] < 0).sum()), int((q["k"] > 0).sum())
            base = {"trade_date": td, "expiry": pd.Timestamp(exp)}
            if len(q) < min_points or n_left < 2 or n_right < 2:
                skipped.append(
                    {
                        **base,
                        "model": "",
                        "reason": f"{len(q)} quotes ({n_left} below / {n_right} above F); "
                        f"need {min_points} with 2 per side",
                    }
                )
                continue
            K, vols = q["strike"].to_numpy(float), q["implied_vol"].to_numpy(float)
            F, T = clean.forward, clean.tau
            meta = {**base, "days": int(clean.days), "tau": T, "forward": F}
            if "svi" in models:
                try:
                    start = warm.get(pd.Timestamp(exp)) if warm_start else None
                    fit = fit_svi_strike_slice(K, vols, F=F, T=T, x0=start)
                    warm[pd.Timestamp(exp)] = fit.params
                    values = {**fit.params, **svi_atm_skew_curvature(fit.params, T)}
                    rows += [
                        {**meta, "model": "svi", "parameter": p, "value": float(values[p]), "rmse_bp": fit.rmse_bp}
                        for p in PARAMETERS["svi"]
                    ]
                except ValueError as exc:
                    skipped.append({**base, "model": "svi", "reason": f"SVI fit failed: {exc}"})
            if "sabr" in models:
                try:
                    fit = calibrate_sabr_smile(K, vols, T=T, F=F, beta=sabr_beta)
                    rows += [
                        {**meta, "model": "sabr", "parameter": p, "value": float(fit[p]), "rmse_bp": fit.rmse_bp}
                        for p in PARAMETERS["sabr"]
                    ]
                except ValueError as exc:
                    skipped.append({**base, "model": "sabr", "reason": f"SABR fit failed: {exc}"})
    history = pd.DataFrame(
        rows, columns=["trade_date", "expiry", "days", "tau", "forward", "model", "parameter", "value", "rmse_bp"]
    )
    return history, pd.DataFrame(skipped, columns=["trade_date", "expiry", "model", "reason"])


def parameter_jumps(history: pd.DataFrame, z: float = 4.0, floor: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Day-to-day changes of each (expiry, model, parameter) series of a
    :func:`parameter_history` table and the jumps among them (module docs).
    Returns ``(changes, summary)``: changes columns expiry, model,
    parameter, from_date, to_date, trading_days, value_from, value_to,
    change, robust_z, jump; summary per (model, parameter): n_changes,
    median_abs_change, robust_sd, max_abs_change, jumps, jump_share."""

    if not z > 0 or not floor >= 0:
        raise ValueError("z must be > 0 and floor >= 0")
    cols = [
        "expiry",
        "model",
        "parameter",
        "from_date",
        "to_date",
        "trading_days",
        "value_from",
        "value_to",
        "change",
        "robust_z",
        "jump",
    ]
    if history is None or history.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame()
    dates = sorted(pd.to_datetime(history["trade_date"]).unique())
    nxt = {pd.Timestamp(a): pd.Timestamp(b) for a, b in zip(dates, dates[1:])}
    frames = []
    for (exp, model, param), g in history.groupby(["expiry", "model", "parameter"], sort=True):
        g = g.sort_values("trade_date")
        d = pd.to_datetime(g["trade_date"]).to_numpy()
        v = g["value"].to_numpy(float)
        for i in range(len(g) - 1):
            a, b = pd.Timestamp(d[i]), pd.Timestamp(d[i + 1])
            if nxt.get(a) != b:  # the expiry was not fitted on the snapshot in between: no day-to-day change
                continue
            frames.append(
                {
                    "expiry": exp,
                    "model": model,
                    "parameter": param,
                    "from_date": a,
                    "to_date": b,
                    "trading_days": trading_days_between(a, b),
                    "value_from": v[i],
                    "value_to": v[i + 1],
                    "change": v[i + 1] - v[i],
                }
            )
    changes = pd.DataFrame(frames, columns=cols[:-2])
    if changes.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame()
    level = history.groupby(["model", "parameter"])["value"].apply(lambda s: float(np.median(np.abs(s))))
    rz, rows = np.full(len(changes), np.nan), []
    for (model, param), g in changes.groupby(["model", "parameter"], sort=False):
        c = g["change"].to_numpy(float)
        med = float(np.median(c))
        mad = float(np.median(np.abs(c - med)))
        scale = max(1.4826 * mad, floor * float(level.get((model, param), 0.0)), ABS_FLOOR.get(param, 0.0), 1e-12)
        zz = (c - med) / scale
        rz[g.index.to_numpy()] = zz
        jumps = int((np.abs(zz) > z).sum())
        rows.append(
            {
                "model": model,
                "parameter": param,
                "n_changes": int(c.size),
                "median_abs_change": float(np.median(np.abs(c))),
                "robust_sd": scale,
                "max_abs_change": float(np.max(np.abs(c))),
                "jumps": jumps,
                "jump_share": jumps / c.size,
            }
        )
    changes["robust_z"] = rz
    changes["jump"] = np.abs(rz) > z
    order = {m: i for i, m in enumerate(PARAMETER_ORDER)}
    summary = pd.DataFrame(rows)
    summary["_m"] = summary["model"].map(lambda m: order.get(m, len(order)))
    summary["_p"] = [
        PARAMETER_ORDER[m].index(p) if p in PARAMETER_ORDER.get(m, ()) else 99
        for m, p in zip(summary["model"], summary["parameter"])
    ]
    summary = summary.sort_values(["_m", "_p"]).drop(columns=["_m", "_p"]).reset_index(drop=True)
    return changes[cols], summary
