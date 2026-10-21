"""Dated rates-vol history (swaption or caplet normal vols by expiry x
tenor x strike offset): parsing, a SYNTHETIC demo, a walk-forward test of
SABR smile forecasts, a smile-model comparison and the daily SABR
parameter history.

Data: there is no free source of dated swaption or cap vol quotes (brokers
and vendors license them), so the history is the user's upload: the
:func:`derivx.rates.swaption_vol.parse_vol_grid` table (long ``expiry,
tenor, offset_bp, vol`` or wide by offset) plus a ``date`` column, and
optionally a ``forward`` column (the ATM forward swap rate of the point on
that date, decimal or percent). :func:`synthetic_vol_history` generates a
SYNTHETIC demo history (labelled as such everywhere; not market data).

Times: a point's expiry time is its nominal tenor in years (``1Y`` = 1.0),
since the history carries no calendar; it shrinks by nothing from one date
to the next (a constant-expiry grid, as quoted). Normal SABR with beta = 0
depends on the rate level only through Hagan's ``sqrt(F K) ln(F/K)``, which
is ``F - K`` to second order, so forwards are optional for it (a nominal 3%
forward is used without them: the fitted smile in offsets moves by a
fraction of a bp); the shifted-lognormal fit and the sticky-strike
forecasts need them.

Walk-forward (:func:`run_vol_walk_forward`): for each origin date and point
with a target date ``horizon`` rows later, SABR is fitted to the origin
smile (:func:`derivx.rates.swaption_vol.calibrate_point`: alpha from the
ATM quote, rho and nu fitted), and every target quote at the same point and
offset is forecast by

* ``persistence``           - the origin quote at the same offset (benchmark);
* ``sabr_sticky_moneyness`` - the origin SABR smile at the same offset from
  ATM (a smoothed persistence);
* ``sabr_sticky_strike``    - with forwards on both dates: the origin SABR
  smile at the target's absolute strike ``F_target + offset`` (benchmark:
  ``quotes_sticky_strike``, the origin quotes linear in strike, flat beyond).

Scores in bp of normal vol, as in :mod:`derivx.validation.backtest`
(skill, days won with a Wilson interval, Diebold-Mariano on the daily
mean squared error differential); regimes are terciles of the date's mean
ATM normal vol across points (labels from the whole sample).

Model comparison (:func:`compare_vol_models`): per date and point, normal
SABR (beta 0), shifted-lognormal SABR (beta 0.5, shift 3%; needs forwards)
and a natural cubic spline in offset, scored in sample and by leave-one-out
over the interior offsets (bp of normal vol).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from derivx.rates.conventions import parse_tenor
from derivx.rates.swaption_vol import (
    DEFAULT_SHIFT,
    GRID_OFFSETS_BP,
    GridIssue,
    GridPoint,
    VolGrid,
    calibrate_point,
    parse_vol_grid,
    smile_vol,
    tenor_label,
)
from derivx.validation.backtest import diebold_mariano, regime_buckets, wilson_interval
from derivx.vol.sabr import sabr_alpha_from_atm, sabr_vol

BP = 1e4
NOMINAL_FORWARD = 0.03  # used only by normal SABR with beta 0 (offset-only) when no forward is given
SYNTHETIC_SOURCE = "SYNTHETIC demo vol history (normal SABR random walk): not market data"
PREDICTORS: Dict[str, str] = {
    "persistence": "origin quote at the same point and offset: no change",
    "sabr_sticky_moneyness": "origin SABR smile at the same offset from ATM",
    "quotes_sticky_strike": "origin quotes, linear in strike, at the target's absolute strike (needs forwards)",
    "sabr_sticky_strike": "origin SABR smile at the target's absolute strike (needs forwards)",
}
BENCHMARK: Dict[str, Optional[str]] = {
    "persistence": None,
    "sabr_sticky_moneyness": "persistence",
    "quotes_sticky_strike": "persistence",
    "sabr_sticky_strike": "quotes_sticky_strike",
}
VOL_MODELS: Dict[str, str] = {
    "normal_sabr": "normal SABR, beta 0 (alpha from ATM; rho, nu fitted)",
    "lognormal_sabr": "shifted-lognormal SABR, beta 0.5, shift 3% (needs forwards)",
    "spline": "natural cubic spline in offset: interpolates, scored out of sample only",
}


@dataclass
class VolHistory:
    """Grids by date (oldest first), forwards ``{date: {(expiry, tenor):
    decimal}}`` where given, where the quotes come from and whether they
    are synthetic."""

    grids: Dict[dt.date, VolGrid]
    forwards: Dict[dt.date, Dict[Tuple[str, str], float]] = field(default_factory=dict)
    source: str = "user upload"
    synthetic: bool = False

    @property
    def dates(self) -> List[dt.date]:
        return sorted(self.grids)

    @property
    def points(self) -> List[Tuple[str, str]]:
        pts = {(r[0], r[1]) for g in self.grids.values() for r in g.rows}
        return sorted(pts, key=lambda p: (parse_tenor(p[0]).years, parse_tenor(p[1]).years))

    def has_forwards(self, day: dt.date, point: Tuple[str, str]) -> bool:
        return point in self.forwards.get(day, {})

    def point(self, day: dt.date, point: Tuple[str, str]) -> GridPoint:
        F = self.forwards.get(day, {}).get(point, NOMINAL_FORWARD)
        return GridPoint(point[0], point[1], parse_tenor(point[0]).years, parse_tenor(point[1]).years, F, 1.0)

    def frame(self) -> pd.DataFrame:
        rows = []
        for d in self.dates:
            fw = self.forwards.get(d, {})
            for e, t, o, v in self.grids[d].rows:
                rows.append(
                    {
                        "date": d,
                        "expiry": e,
                        "tenor": t,
                        "offset_bp": o,
                        "vol_bp": v * BP,
                        "forward": fw.get((e, t), np.nan),
                    }
                )
        return pd.DataFrame(rows)


def parse_vol_history(
    raw: pd.DataFrame, units: str = "bp", forward_units: str = "auto", source: str = "user upload"
) -> Tuple[Optional[VolHistory], List[GridIssue]]:
    """A dated normal-vol table -> :class:`VolHistory` (module docs).
    ``units`` of the vols: bp or decimal; ``forward_units``: decimal,
    percent or auto (|values| > 1 read as percent, said in a warning).
    Every date must parse as a grid (errors are listed per date) and the
    history needs at least two dates."""

    issues: List[GridIssue] = []
    if raw is None or raw.empty:
        return None, [GridIssue("error", "the vol history is empty")]
    frame = raw.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    lower = {c.lower(): c for c in frame.columns}
    if "date" not in lower:
        return None, [GridIssue("error", "the vol history needs a 'date' column (YYYY-MM-DD) next to the vol grid")]
    dates = pd.to_datetime(frame[lower["date"]], errors="coerce")
    if dates.isna().any():
        rows = [str(i + 1) for i in np.flatnonzero(dates.isna().to_numpy())[:5]]
        return None, [GridIssue("error", f"unreadable date on row(s) {', '.join(rows)}: use YYYY-MM-DD")]
    frame["_day"] = dates.dt.date
    fcol = lower.get("forward")
    fwd_scale = 1.0
    if fcol is not None:
        f = pd.to_numeric(frame[fcol], errors="coerce")
        if f.isna().any():
            return None, [GridIssue("error", "the 'forward' column has blank or non-numeric values")]
        if forward_units == "percent" or (forward_units == "auto" and f.abs().max() > 1.0):
            fwd_scale = 0.01
            if forward_units == "auto":
                issues.append(GridIssue("warning", "forwards read as percent (values above 1)"))
        elif forward_units not in ("decimal", "auto"):
            return None, [GridIssue("error", f"forward units must be decimal, percent or auto, got {forward_units!r}")]
        frame[fcol] = f * fwd_scale
    grids: Dict[dt.date, VolGrid] = {}
    forwards: Dict[dt.date, Dict[Tuple[str, str], float]] = {}
    for day, g in frame.groupby("_day", sort=True):
        body = g.drop(columns=[c for c in (lower["date"], "_day", fcol) if c is not None])
        grid, grid_issues = parse_vol_grid(body, "normal", units, 0.0, source)
        issues += [GridIssue(i.level, f"{day}: {i.message}") for i in grid_issues]
        if grid is None:
            continue
        grids[day] = grid
        if fcol is not None:
            fw: Dict[Tuple[str, str], float] = {}
            for (e, t), sub in g.groupby([lower["expiry"], lower["tenor"]]):
                vals = sub[fcol].unique()
                key = (tenor_label(e), tenor_label(t))
                if len(vals) > 1:
                    issues.append(GridIssue("error", f"{day} {key[0]}x{key[1]}: several forwards ({vals[:3]})"))
                fw[key] = float(vals[0])
            forwards[day] = fw
    if any(i.level == "error" for i in issues):
        return None, issues
    if len(grids) < 2:
        return None, issues + [GridIssue("error", f"{len(grids)} date(s): a history needs at least two")]
    return VolHistory(grids, forwards, source, synthetic=False), issues


def synthetic_vol_history(
    n_days: int = 30,
    *,
    seed: int = 20261020,
    end: str = "2026-10-19",
    expiries: Tuple[str, ...] = ("3M", "1Y", "5Y"),
    tenors: Tuple[str, ...] = ("2Y", "10Y"),
    noise_bp: float = 0.5,
) -> VolHistory:
    """SYNTHETIC vol history (not market data): per point a normal SABR
    smile (beta 0) whose ATM normal vol follows a log random walk (2%
    daily), rho and nu mean-revert, the forward follows a 5bp daily random
    walk; quotes at :data:`GRID_OFFSETS_BP` plus i.i.d. ``noise_bp`` noise,
    on business days ending ``end``. Deterministic for a seed."""

    if n_days < 2:
        raise ValueError("a synthetic history needs at least two days")
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp(end), periods=int(n_days)).date
    state = {}
    for i, e in enumerate(expiries):
        for j, t in enumerate(tenors):
            state[(e, t)] = {
                "atm": 0.0105 - 0.0008 * i - 0.0004 * j,
                "rho": -0.1 + 0.1 * j,
                "nu": 0.45 - 0.08 * i,
                "F": 0.036 + 0.002 * j,
            }
    grids, forwards = {}, {}
    for k, day in enumerate(days):
        rows, fw = [], {}
        for (e, t), s in state.items():
            if k:
                s["atm"] *= float(np.exp(0.02 * rng.standard_normal()))
                s["rho"] = float(np.clip(s["rho"] + 0.2 * (-0.05 - s["rho"]) + 0.03 * rng.standard_normal(), -0.9, 0.9))
                s["nu"] = float(max(s["nu"] + 0.2 * (0.35 - s["nu"]) + 0.02 * rng.standard_normal(), 0.05))
                s["F"] += 5e-4 * float(rng.standard_normal())
            T = parse_tenor(e).years
            K = s["F"] + np.asarray(GRID_OFFSETS_BP) / BP
            alpha = sabr_alpha_from_atm(s["F"], T, s["atm"], 0.0, s["rho"], s["nu"], DEFAULT_SHIFT, "normal")
            vols = np.asarray(sabr_vol(s["F"], K, T, alpha, 0.0, s["rho"], s["nu"], DEFAULT_SHIFT, "normal"))
            vols = vols + noise_bp / BP * rng.standard_normal(vols.size) * (np.asarray(GRID_OFFSETS_BP) != 0)
            rows += [(e, t, float(o), round(float(v) * BP, 2) / BP) for o, v in zip(GRID_OFFSETS_BP, vols)]
            fw[(e, t)] = round(s["F"], 6)
        grids[day] = VolGrid(tuple(rows), "normal", 0.0, SYNTHETIC_SOURCE)
        forwards[day] = fw
    return VolHistory(grids, forwards, SYNTHETIC_SOURCE, synthetic=True)


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


@dataclass
class VolForecastResult:
    """``errors``: one row per scored quote (bp); ``daily``, ``summary``,
    ``by_point``, ``regimes``, ``by_regime``; ``skipped`` with reasons."""

    horizon: int
    errors: pd.DataFrame
    daily: pd.DataFrame
    summary: pd.DataFrame
    by_point: pd.DataFrame
    regimes: pd.DataFrame
    by_regime: pd.DataFrame
    skipped: pd.DataFrame

    @property
    def empty(self) -> bool:
        return self.errors.empty


def _rmse(e) -> float:
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")


def _atm_level(grid: VolGrid) -> float:
    atm = [r[3] for r in grid.rows if r[2] == 0.0]
    return float(np.mean(atm)) if atm else float("nan")


def run_vol_walk_forward(history: VolHistory, horizon: int = 1, beta: float = 0.0) -> VolForecastResult:
    """Walk-forward of SABR smile forecasts (module docs). ``horizon`` in
    rows of the history (dates); normal SABR with fixed ``beta`` (0)."""

    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1 date")
    dates = history.dates
    rows, skipped = [], []
    level = pd.Series({d: _atm_level(history.grids[d]) for d in dates}, dtype=float)
    regimes = pd.DataFrame({"date": dates, "atm_normal_bp": level.to_numpy() * BP})
    if not regimes.empty:
        regimes = pd.concat([regimes, regime_buckets(regimes["atm_normal_bp"])], axis=1)
    bucket_of = dict(zip(regimes["date"], regimes["bucket"]))
    for i, origin in enumerate(dates[:-h] if h < len(dates) else []):
        target = dates[i + h]
        g0, g1 = history.grids[origin], history.grids[target]
        for point in history.points:
            label = f"{point[0]}x{point[1]}"
            off0, vol0 = g0.smile(*point)
            off1, vol1 = g1.smile(*point)
            if off0.size == 0 or off1.size == 0:
                skipped.append({"origin": origin, "point": label, "reason": "point not quoted on both dates"})
                continue
            gp = history.point(origin, point)
            try:
                fit = calibrate_point(g0, gp, "normal", beta=beta, shift=DEFAULT_SHIFT)
            except ValueError as exc:
                skipped.append({"origin": origin, "point": label, "reason": f"SABR fit failed: {exc}"})
                continue
            if fit.status != "fitted":
                skipped.append({"origin": origin, "point": label, "reason": f"SABR {fit.status}: {fit.message}"})
                continue
            F0 = gp.forward
            pred_m = np.asarray(fit.vol(F0 + off1 / BP), dtype=float)
            persist = np.interp(off1, off0, vol0, left=np.nan, right=np.nan)
            exact = np.isin(off1, off0)
            persist = np.where(exact, persist, np.nan)  # persistence only where the same offset was quoted
            has_f = history.has_forwards(origin, point) and history.has_forwards(target, point)
            if has_f:
                F1 = history.forwards[target][point]
                strikes = F1 + off1 / BP
                pred_s = np.asarray(fit.vol(strikes), dtype=float)
                quotes_s = np.interp(strikes, F0 + off0 / BP, vol0)
            else:
                pred_s = quotes_s = np.full(off1.size, np.nan)
            for j, o in enumerate(off1):
                rows.append(
                    {
                        "origin": origin,
                        "target": target,
                        "point": label,
                        "offset_bp": float(o),
                        "actual_bp": vol1[j] * BP,
                        "origin_bp": persist[j] * BP,
                        "err_bp_persistence": (persist[j] - vol1[j]) * BP,
                        "err_bp_sabr_sticky_moneyness": (pred_m[j] - vol1[j]) * BP,
                        "err_bp_quotes_sticky_strike": (quotes_s[j] - vol1[j]) * BP,
                        "err_bp_sabr_sticky_strike": (pred_s[j] - vol1[j]) * BP,
                        "sabr_rmse_bp": fit.rmse_bp,
                        "regime": bucket_of.get(origin, "n/a"),
                    }
                )
    errors = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped, columns=["origin", "point", "reason"])
    if errors.empty:
        return VolForecastResult(
            h, errors, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), regimes, pd.DataFrame(), skipped_df
        )
    daily = (
        errors.groupby(["origin", "target"], sort=True)
        .agg(regime=("regime", "first"), **{f"rmse_bp_{p}": (f"err_bp_{p}", _rmse) for p in PREDICTORS})
        .reset_index()
    )
    by_point = pd.DataFrame(
        [
            {"point": p, **{f"rmse_bp_{n}": _rmse(g[f"err_bp_{n}"]) for n in PREDICTORS}}
            for p, g in errors.groupby("point", sort=False)
        ]
    )
    by_regime = []
    for bucket in ("low", "mid", "high", "n/a"):
        g = errors[errors["regime"] == bucket]
        if not g.empty:
            by_regime.append(
                {
                    "regime": bucket,
                    "origin_days": int(g["origin"].nunique()),
                    **{f"rmse_bp_{n}": _rmse(g[f"err_bp_{n}"]) for n in PREDICTORS},
                }
            )
    return VolForecastResult(
        h, errors, daily, _summary(errors, h), by_point, regimes, pd.DataFrame(by_regime), skipped_df
    )


def _summary(errors: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for name, description in PREDICTORS.items():
        e = errors[f"err_bp_{name}"].to_numpy(float)
        row: Dict[str, object] = {
            "predictor": name,
            "rmse_bp": _rmse(e),
            "n_quotes": int(np.isfinite(e).sum()),
            "n_days": int(errors.loc[np.isfinite(e), "target"].nunique()),
        }
        bench = BENCHMARK[name]
        if bench is None or row["n_quotes"] == 0:
            row.update(
                benchmark=bench or "",
                benchmark_rmse_bp=np.nan,
                skill=np.nan,
                dm_t=np.nan,
                day_win_rate=np.nan,
                day_win_lo=np.nan,
                day_win_hi=np.nan,
            )
        else:
            b = errors[f"err_bp_{bench}"].to_numpy(float)
            both = np.isfinite(e) & np.isfinite(b)
            sq = pd.DataFrame({"day": errors["target"].to_numpy()[both], "f": e[both] ** 2, "b": b[both] ** 2})
            daily = sq.groupby("day", sort=True)[["f", "b"]].mean()
            diff = daily["f"] - daily["b"]
            wins, n = int((diff < 0).sum()), int(diff.size)
            lo, hi = wilson_interval(wins, n)
            rb, rf = _rmse(b[both]), _rmse(e[both])
            row.update(
                benchmark=bench,
                benchmark_rmse_bp=rb,
                skill=1.0 - rf / rb if rb > 0 else np.nan,
                dm_t=diebold_mariano(diff.to_numpy(float), horizon=horizon),
                day_win_rate=wins / n if n else np.nan,
                day_win_lo=lo,
                day_win_hi=hi,
            )
        row["description"] = description
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Model comparison and parameter history
# --------------------------------------------------------------------------


def _fit_values(model: str, grid: VolGrid, gp: GridPoint, offsets: np.ndarray, vols: np.ndarray, train=None):
    """Fitted normal vols at ``offsets`` from a fit on the ``train`` quotes."""

    train = np.ones(offsets.size, dtype=bool) if train is None else train
    if model == "spline":
        if train.sum() < 3:
            raise ValueError("a spline needs 3 quotes")
        return CubicSpline(offsets[train], vols[train], bc_type="natural")(offsets)
    sub = VolGrid(
        tuple((gp.expiry, gp.tenor, float(o), float(v)) for o, v in zip(offsets[train], vols[train])),
        "normal",
        0.0,
        grid.source,
    )
    if 0.0 not in set(offsets[train]):
        raise ValueError("the ATM quote is held out")
    fit = calibrate_point(
        sub,
        gp,
        "normal" if model == "normal_sabr" else "lognormal",
        beta=0.0 if model == "normal_sabr" else 0.5,
        shift=DEFAULT_SHIFT,
    )
    if fit.status == "failed":
        raise ValueError(fit.message)
    return np.asarray(smile_vol(gp.forward, gp.forward + offsets / BP, gp.expiry_years, fit.params, fit.fit), float)


def compare_vol_models(history: VolHistory, dates: Optional[List[dt.date]] = None, models=tuple(VOL_MODELS)):
    """Per date and point: each model's in-sample RMSE and leave-one-out
    RMSE over interior offsets except ATM (ATM pins SABR's alpha), in bp
    of normal vol. Returns ``(scores, skipped)``; ``scores`` has
    ``trade_date`` and ``expiry`` (the point label) keys as
    :func:`derivx.models.comparison.summarise_comparison` expects."""

    rows, skipped = [], []
    for day in dates or history.dates:
        grid = history.grids[day]
        for point in history.points:
            label = f"{point[0]}x{point[1]}"
            offsets, vols = grid.smile(*point)
            if offsets.size < 5:
                skipped.append({"date": day, "point": label, "model": "", "reason": f"{offsets.size} quotes; need 5"})
                continue
            gp = history.point(day, point)
            for model in models:
                if model == "lognormal_sabr" and not history.has_forwards(day, point):
                    skipped.append(
                        {
                            "date": day,
                            "point": label,
                            "model": model,
                            "reason": "no forward: the shifted-lognormal fit depends on the rate level",
                        }
                    )
                    continue
                try:
                    fitted = _fit_values(model, grid, gp, offsets, vols)
                    cv = []
                    for i in range(1, offsets.size - 1):
                        if offsets[i] == 0.0:
                            continue
                        train = np.ones(offsets.size, dtype=bool)
                        train[i] = False
                        try:
                            cv.append(_fit_values(model, grid, gp, offsets, vols, train)[i] - vols[i])
                        except ValueError:
                            continue
                except ValueError as exc:
                    skipped.append({"date": day, "point": label, "model": model, "reason": str(exc)})
                    continue
                e = (fitted - vols) * BP
                cv = np.asarray(cv) * BP
                rows.append(
                    {
                        "trade_date": day,
                        "expiry": label,
                        "model": model,
                        "rmse_bp": _rmse(e),
                        "rmse_cv_bp": _rmse(cv),
                        "max_abs_bp": float(np.nanmax(np.abs(e))),
                        "n_points": int(offsets.size),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(skipped, columns=["date", "point", "model", "reason"])


def sabr_parameter_history(history: VolHistory, beta: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Normal SABR (alpha from ATM, rho and nu fitted) on every date and
    point, in the long format of
    :func:`derivx.validation.stability.parameter_history` (``expiry`` is the
    point label, model ``normal_sabr``; alpha in rate units)."""

    rows, skipped = [], []
    for day in history.dates:
        grid = history.grids[day]
        for point in history.points:
            label = f"{point[0]}x{point[1]}"
            if grid.smile(*point)[0].size == 0:
                continue
            gp = history.point(day, point)
            try:
                fit = calibrate_point(grid, gp, "normal", beta=beta, shift=DEFAULT_SHIFT)
            except ValueError as exc:
                skipped.append({"trade_date": day, "expiry": label, "model": "normal_sabr", "reason": str(exc)})
                continue
            if fit.status != "fitted":
                skipped.append(
                    {
                        "trade_date": day,
                        "expiry": label,
                        "model": "normal_sabr",
                        "reason": f"{fit.status}: {fit.message}",
                    }
                )
                continue
            atm = fit.atm_normal
            for name, value in (("atm_normal_bp", atm * BP), ("alpha", fit.alpha), ("rho", fit.rho), ("nu", fit.nu)):
                rows.append(
                    {
                        "trade_date": pd.Timestamp(day),
                        "expiry": label,
                        "days": np.nan,
                        "tau": gp.expiry_years,
                        "forward": gp.forward,
                        "model": "normal_sabr",
                        "parameter": name,
                        "value": float(value),
                        "rmse_bp": fit.rmse_bp,
                    }
                )
    cols = ["trade_date", "expiry", "days", "tau", "forward", "model", "parameter", "value", "rmse_bp"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(skipped, columns=["trade_date", "expiry", "model", "reason"])


def history_from_mapping(grids: Mapping[dt.date, VolGrid]) -> VolHistory:
    """A :class:`VolHistory` from grids already parsed (no forwards)."""

    return VolHistory(dict(grids))
