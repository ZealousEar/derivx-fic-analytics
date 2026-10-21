"""Swaption vol cube: quotes by expiry x tenor x strike offset, SABR per
point, an ATM normal-vol matrix and vega buckets.

Quotes (:class:`VolGrid`) are vols by option expiry, underlying swap tenor
and strike offset from the ATM forward swap rate, in bp (``-200 ... +200``;
0 is ATM). Swaption desks quote normal (Bachelier) vols; lognormal quotes
are accepted with their shift and converted to normal vols by exact price
matching (:func:`derivx.vol.implied.lognormal_to_normal_vol`). Vol quotes
are not available from a free source: the defaults are ILLUSTRATIVE
(:mod:`derivx.rates.vol_examples`) and users upload or paste their own.

Calibration (:func:`calibrate_cube`) fits Hagan SABR to each point's smile
with :func:`derivx.vol.sabr.calibrate_sabr_smile`, alpha solved from the
ATM quote so ATM reprices exactly and rho, nu fitted:

* ``fit="normal"``    - normal SABR (default beta = 0) fitted to the normal vols;
* ``fit="lognormal"`` - shifted-lognormal SABR (default beta = 0.5) fitted
  to the shifted Black vols of the quotes (Black on ``F + shift``).

Both use shift ``shift`` (``F + shift`` and ``K + shift`` must be positive;
quotes at strikes below ``-shift`` are dropped and counted). Residuals are
model minus quote in bp of normal vol, whichever space was fitted.

The cube (:class:`SwaptionVolCube`) interpolates the ATM normal vol in
variance-time: total variance ``sigma_N^2 T`` linear in expiry between
quoted expiries (flat vol before the first and after the last) and linear
in tenor (flat outside), and SABR rho and nu bilinearly in (expiry, tenor),
flat outside; alpha is then re-solved from the interpolated ATM vol at the
target forward, so the cube reprices its ATM matrix at any forward.

Expiries are ACT/365F years from the valuation date to the expiry date
(``MakeSwaption`` dates, see :mod:`derivx.rates.swaptions`); tenors are
nominal years (10Y = 10.0). Vols in rate units (1bp = 1e-4) unless a name
ends in ``_bp``. Vega buckets are P&L per +1bp of normal vol on one ATM
matrix point, smile shape (rho, nu) held.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from derivx.rates.conventions import parse_tenor
from derivx.rates.curves import Market
from derivx.rates.swaptions import Swaption
from derivx.vol.implied import bachelier_price, lognormal_to_normal_vol, normal_to_lognormal_vol
from derivx.vol.sabr import calibrate_sabr_smile, sabr_alpha_from_atm, sabr_vol

GRID_EXPIRIES = ("1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y")
GRID_TENORS = ("1Y", "2Y", "5Y", "10Y", "20Y", "30Y")
GRID_OFFSETS_BP = (-200.0, -100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0, 200.0)
FITS = ("normal", "lognormal")
DEFAULT_BETA = {"normal": 0.0, "lognormal": 0.5}
DEFAULT_SHIFT = 0.03  # 3%: room for strikes 200bp below forwards down to +1%, and for negative forwards to -3%
NORMAL_BP_RANGE = (1.0, 1000.0)  # sane normal vol quotes, bp
LOGNORMAL_PCT_RANGE = (0.5, 400.0)  # sane (shifted) lognormal vol quotes, percent
MAX_OFFSET_BP = 1000.0
MIN_SMILE_QUOTES = 3  # ATM plus two others: rho and nu are fitted, alpha comes from ATM


# --------------------------------------------------------------------------
# Quotes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VolGrid:
    """Swaption vol quotes: rows ``(expiry, tenor, offset_bp, vol)``.

    ``vol_type`` is ``normal`` (rate units, 0.0085 = 85bp) or ``lognormal``
    (Black vol of ``F + shift``, decimal). ``source`` says where the quotes
    come from (``ILLUSTRATIVE ...`` for the built-in defaults)."""

    rows: Tuple[Tuple[str, str, float, float], ...]
    vol_type: str = "normal"
    shift: float = 0.0
    source: str = "user"

    def __post_init__(self) -> None:
        if self.vol_type not in FITS:
            raise ValueError(f"vol_type must be one of {FITS}, got {self.vol_type!r}")
        if not math.isfinite(self.shift) or self.shift < 0.0:
            raise ValueError("the quotes' shift must be zero or a positive decimal")
        if not self.rows:
            raise ValueError("the vol grid has no quotes")
        for what, labels in (("option expiry", self.expiries), ("swap tenor", self.tenors)):
            years: Dict[float, str] = {}
            for label in labels:
                y = parse_tenor(label).years
                if y in years:
                    raise ValueError(
                        f"{years[y]} and {label} are the same {what}: quote it once (one label per {what})"
                    )
                years[y] = label

    @property
    def expiries(self) -> Tuple[str, ...]:
        return tuple(sorted({r[0] for r in self.rows}, key=lambda t: (parse_tenor(t).years, t)))

    @property
    def tenors(self) -> Tuple[str, ...]:
        return tuple(sorted({r[1] for r in self.rows}, key=lambda t: (parse_tenor(t).years, t)))

    def smile(self, expiry: str, tenor: str) -> Tuple[np.ndarray, np.ndarray]:
        """(offsets in bp, vols) quoted at one point, by offset."""

        rows = sorted((r[2], r[3]) for r in self.rows if r[0] == expiry and r[1] == tenor)
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.rows), columns=["expiry", "tenor", "offset_bp", "vol"])

    def wide(self, units: float = 1e4) -> pd.DataFrame:
        """One row per (expiry, tenor), one column per offset (vols times
        ``units``: 1e4 gives bp for normal vols, 100 percent for lognormal)."""

        frame = self.frame()
        frame["vol"] = frame["vol"] * units
        table = frame.pivot_table(index=["expiry", "tenor"], columns="offset_bp", values="vol", aggfunc="first")
        order = [(e, t) for e in self.expiries for t in self.tenors if (e, t) in table.index]
        table = table.loc[order]
        table.columns = [_offset_label(c) for c in table.columns]
        return table.reset_index()


def _offset_label(offset: float) -> str:
    return "ATM" if offset == 0 else f"{offset:+.0f}"


def _offset_value(label: object) -> float:
    """Strike offset in bp from a label: ``-50``, ``+25bp``, ``ATM``,
    ``ATM+25`` (a typographic minus counts as a minus)."""

    if isinstance(label, (int, float, np.integer, np.floating)) and not isinstance(label, bool):
        return float(label)
    text = str(label).strip().upper().replace("BP", "").replace("\u2212", "-").replace(" ", "")
    if text in ("ATM", "0", "+0", "-0"):
        return 0.0
    if text.startswith("ATM"):
        text = text[3:]
    return float(text)


def tenor_label(text: object) -> str:
    """Canonical label of an option expiry or swap tenor: whole years as
    ``Y`` (``12M`` -> ``1Y``, ``1.5Y`` -> ``18M``), other months as ``M``,
    weeks as ``W``, so one point cannot be quoted under two names."""

    tenor = parse_tenor(str(text))
    months = tenor.months
    if months is None:
        return str(tenor)
    return f"{months // 12}Y" if months % 12 == 0 else f"{months}M"


@dataclass(frozen=True)
class GridIssue:
    level: str  # "error" or "warning"
    message: str


def parse_vol_grid(
    raw: pd.DataFrame,
    vol_type: str = "normal",
    units: str = "bp",
    shift: float = 0.0,
    source: str = "user",
) -> Tuple[Optional[VolGrid], List[GridIssue]]:
    """Read swaption vol quotes from a table, long or wide.

    * long: columns ``expiry``, ``tenor``, ``offset_bp`` (or ``offset``) and ``vol``;
    * wide: ``expiry``, ``tenor`` and one column per offset (``-200``, ...,
      ``ATM`` or ``0``, ..., ``+200``).

    ``units``: ``bp`` or ``decimal`` for normal vols; ``percent`` or
    ``decimal`` for lognormal vols (of ``F + shift``). Returns (grid or None,
    issues); any error means no grid. Every (expiry, tenor) needs an ATM
    quote, and the ATM points must fill the whole expiry x tenor grid (the
    cube interpolates between them)."""

    issues: List[GridIssue] = []
    seen: set = set()

    def problem(message: str) -> None:
        if message not in seen:  # a wide row repeats a bad expiry or tenor once per offset column
            seen.add(message)
            issues.append(GridIssue("error", message))

    if vol_type not in FITS:
        return None, [GridIssue("error", f"vol type must be one of {FITS}, got {vol_type!r}")]
    allowed = {"normal": ("bp", "decimal"), "lognormal": ("percent", "decimal")}[vol_type]
    if units not in allowed:
        return None, [GridIssue("error", f"{vol_type} vols are in {' or '.join(allowed)}, got {units!r}")]
    scale = {"bp": 1e-4, "percent": 0.01, "decimal": 1.0}[units]
    if raw is None or raw.empty:
        return None, [GridIssue("error", "the vol table is empty")]
    frame = raw.copy().reset_index(drop=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    lower = {c.lower(): c for c in frame.columns}
    if "expiry" not in lower or "tenor" not in lower:
        return None, [GridIssue("error", "the vol table needs 'expiry' and 'tenor' columns (e.g. 1Y and 10Y)")]
    table_row = pd.Series(np.arange(1, len(frame) + 1), index=frame.index)  # row number in the user's table
    offset_col = lower.get("offset_bp") or lower.get("offset")
    if offset_col is not None and "vol" in lower:
        long = pd.DataFrame(
            {"row": table_row, "expiry": frame[lower["expiry"]], "tenor": frame[lower["tenor"]],
             "offset": frame[offset_col], "vol": frame[lower["vol"]]}
        )  # fmt: skip
    else:
        value_cols = [c for c in frame.columns if c.lower() not in ("expiry", "tenor", "source")]
        if not value_cols:
            return None, [GridIssue("error", "no vol columns: add offset columns (-200 ... ATM ... +200) or "
                                             "'offset_bp' and 'vol' columns")]  # fmt: skip
        offsets = {}
        for col in value_cols:
            try:
                offsets[col] = _offset_value(col)
            except ValueError:
                return None, [GridIssue("error", f"column {col!r} is not a strike offset in bp (e.g. -50, ATM, +100)")]
        wide = frame.assign(_table_row=table_row)
        long = wide.melt(id_vars=["_table_row", lower["expiry"], lower["tenor"]], value_vars=value_cols,
                         var_name="offset", value_name="vol")  # fmt: skip
        long = long.rename(columns={"_table_row": "row", lower["expiry"]: "expiry", lower["tenor"]: "tenor"})
        long["offset"] = long["offset"].map(offsets)
    rows: Dict[Tuple[str, str, float], float] = {}
    typed_as: Dict[Tuple[str, str], str] = {}  # point -> labels as first entered
    lo, hi = (NORMAL_BP_RANGE[0] * 1e-4, NORMAL_BP_RANGE[1] * 1e-4) if vol_type == "normal" else (
        LOGNORMAL_PCT_RANGE[0] * 0.01, LOGNORMAL_PCT_RANGE[1] * 0.01)  # fmt: skip
    shown = "bp" if vol_type == "normal" else "%"
    show = 1e4 if vol_type == "normal" else 100.0
    for row in long.itertuples(index=False):
        n = int(row.row)
        vol_cell = row.vol
        if vol_cell is None or (isinstance(vol_cell, float) and math.isnan(vol_cell)) or str(vol_cell).strip() == "":
            continue  # blank cell: no quote at this offset
        try:
            expiry = parse_tenor(str(row.expiry))
            tenor = parse_tenor(str(row.tenor))
        except ValueError as exc:
            problem(f"row {n}: {exc}")
            continue
        if expiry.unit not in ("W", "M", "Y"):
            problem(f"row {n}: option expiry {row.expiry!r} must be weeks, months or years")
            continue
        if tenor.months is None:
            problem(f"row {n}: swap tenor {row.tenor!r} must be months or years")
            continue
        point = f"{tenor_label(row.expiry)}x{tenor_label(row.tenor)}"
        try:
            offset = _offset_value(row.offset)
        except ValueError:
            problem(f"row {n} ({point}): strike offset {row.offset!r} is not a number of bp (e.g. -50, ATM, +100)")
            continue
        try:
            vol = float(str(vol_cell).replace("%", "").strip()) * scale
        except ValueError:
            problem(f"row {n} ({point} {_offset_label(offset)}): vol {vol_cell!r} is not a number")
            continue
        if not math.isfinite(offset) or abs(offset) > MAX_OFFSET_BP:
            problem(f"row {n} ({point}): strike offset {offset} bp is outside +/-{MAX_OFFSET_BP:.0f}bp")
            continue
        if not math.isfinite(vol):
            problem(f"row {n} ({point} {_offset_label(offset)}): vol {vol_cell!r} is not a finite number")
            continue
        if not lo <= vol <= hi:
            problem(
                f"row {n} ({point} {_offset_label(offset)}): vol {vol * show:.4g}{shown} is outside "
                f"{lo * show:g}-{hi * show:g}{shown}; check the units ({units})"
            )
            continue
        key = (tenor_label(row.expiry), tenor_label(row.tenor), offset)
        typed = f"{str(row.expiry).strip().upper()}x{str(row.tenor).strip().upper()}"
        if key in rows:
            first = typed_as[key[:2]]
            alias = f" ({first} and {typed} are the same point)" if typed != first else ""
            problem(f"row {n}: a second quote for {key[0]}x{key[1]} {_offset_label(offset)}{alias}")
            continue
        typed_as.setdefault(key[:2], typed)
        rows[key] = vol
    if not rows and not issues:
        issues.append(GridIssue("error", "the vol table has no quotes"))
    if any(i.level == "error" for i in issues):
        return None, issues
    points = sorted({(e, t) for e, t, _ in rows}, key=lambda p: (parse_tenor(p[0]).years, parse_tenor(p[1]).years))
    no_atm = [f"{e}x{t}" for e, t in points if (e, t, 0.0) not in rows]
    if no_atm:
        return None, issues + [GridIssue("error", f"no ATM quote (offset 0) at {', '.join(no_atm[:8])}")]
    expiries = sorted({e for e, _ in points}, key=lambda t: parse_tenor(t).years)
    tenors = sorted({t for _, t in points}, key=lambda t: parse_tenor(t).years)
    missing = [f"{e}x{t}" for e in expiries for t in tenors if (e, t) not in points]
    if missing:
        return None, issues + [
            GridIssue("error", f"the ATM quotes must fill the expiry x tenor grid; missing {', '.join(missing[:8])}")
        ]
    thin = [f"{e}x{t}" for e, t in points if sum(1 for k in rows if k[:2] == (e, t)) < MIN_SMILE_QUOTES]
    if thin:
        issues.append(
            GridIssue(
                "warning",
                f"{len(thin)} point(s) have fewer than {MIN_SMILE_QUOTES} strikes ({', '.join(thin[:6])}): their "
                "smile is not fitted (flat in normal vol at the ATM quote: SABR nu = 0, rho = 0)",
            )
        )
    ordered = tuple((e, t, o, v) for (e, t, o), v in sorted(rows.items(), key=lambda kv: (
        parse_tenor(kv[0][0]).years, parse_tenor(kv[0][1]).years, kv[0][2])))  # fmt: skip
    return VolGrid(ordered, vol_type, float(shift), source), issues


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    """Market inputs of one quoted (expiry, tenor): ACT/365F years to
    expiry, forward swap rate (decimal) and annuity per unit notional."""

    expiry: str
    tenor: str
    expiry_years: float
    tenor_years: float
    forward: float
    annuity: float


def grid_points(
    grid: VolGrid, market: Market, index: Optional[str] = None, conventions: Optional[str] = None
) -> Tuple[GridPoint, ...]:
    """Expiry times and forward swap rates of every quoted point on
    ``market`` (underlying: an OIS swap, or one against ``index``)."""

    out = []
    for e in grid.expiries:
        for t in grid.tenors:
            probe = Swaption(1.0, e, t, 0.0, index=index, conventions=conventions)
            inputs = probe.inputs(market)
            out.append(GridPoint(e, t, inputs.expiry_years, parse_tenor(t).years, inputs.forward, inputs.annuity))
    return tuple(out)


@dataclass(frozen=True)
class SmileFit:
    """SABR fit of one point. Vols in rate units; ``*_bp`` in bp of normal
    vol. ``status`` is ``fitted``, ``flat`` (too few strikes: nu = rho = 0)
    or ``failed`` (with ``message``; the point then uses a flat smile)."""

    point: GridPoint
    offsets_bp: np.ndarray = field(repr=False)
    strikes: np.ndarray = field(repr=False)
    market_normal: np.ndarray = field(repr=False)
    model_normal: np.ndarray = field(repr=False)
    alpha: float
    beta: float
    rho: float
    nu: float
    shift: float
    fit: str
    status: str
    message: str = ""
    n_dropped: int = 0
    at_bound: Tuple[str, ...] = ()

    @property
    def atm_normal(self) -> float:
        return float(self.market_normal[self.offsets_bp == 0.0][0])

    @property
    def residual_bp(self) -> np.ndarray:
        return (self.model_normal - self.market_normal) * 1e4

    @property
    def rmse_bp(self) -> float:
        r = self.residual_bp[np.isfinite(self.residual_bp)]
        return float(np.sqrt(np.mean(r * r))) if r.size else float("nan")

    @property
    def max_abs_bp(self) -> float:
        r = self.residual_bp[np.isfinite(self.residual_bp)]
        return float(np.max(np.abs(r))) if r.size else float("nan")

    @property
    def params(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "rho": self.rho, "nu": self.nu, "shift": self.shift}

    def vol(self, K, vol_type: str = "normal"):
        """Fitted smile at strikes ``K`` (absolute, decimal) as normal vols
        (rate units) or as the fit's own lognormal vols."""

        return smile_vol(self.point.forward, K, self.point.expiry_years, self.params, self.fit, vol_type)


def smile_vol(F: float, K, T: float, params: Mapping[str, float], fit: str, vol_type: str = "normal"):
    """SABR vol at strikes ``K``: a ``normal`` fit gives Hagan's normal
    expansion; a ``lognormal`` fit gives the shifted Black vol, converted
    to normal by exact price matching when ``vol_type="normal"``."""

    a, b, r, n, s = (float(params[k]) for k in ("alpha", "beta", "rho", "nu", "shift"))
    if fit == "normal":
        if vol_type != "normal":
            raise ValueError("a normal-SABR fit gives normal vols")
        return sabr_vol(F, K, T, a, b, r, n, s, "normal")
    black = sabr_vol(F, K, T, a, b, r, n, s, "lognormal")
    return black if vol_type == "lognormal" else lognormal_to_normal_vol(F, K, T, black, shift=s)


def _quotes_normal(grid: VolGrid, point: GridPoint) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets, vols = grid.smile(point.expiry, point.tenor)
    strikes = point.forward + offsets * 1e-4
    if grid.vol_type == "normal":
        return offsets, strikes, vols
    normal = np.asarray(lognormal_to_normal_vol(point.forward, strikes, point.expiry_years, vols, shift=grid.shift))
    return offsets, strikes, normal


def calibrate_point(
    grid: VolGrid,
    point: GridPoint,
    fit: str = "normal",
    beta: Optional[float] = None,
    shift: float = DEFAULT_SHIFT,
    weights: Optional[str] = None,
) -> SmileFit:
    """SABR fit of one point's smile (see the module docstring)."""

    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    beta = DEFAULT_BETA[fit] if beta is None else float(beta)
    F, T = point.forward, point.expiry_years
    if F + shift <= 0:
        raise ValueError(
            f"{point.expiry}x{point.tenor}: forward {F:.3%} is below -shift ({-shift:.2%}); raise the SABR shift"
        )
    offsets, strikes, normal = _quotes_normal(grid, point)
    atm_n = float(normal[offsets == 0.0][0])
    target = normal if fit == "normal" else normal_to_lognormal_vol(F, strikes, T, normal, shift=shift)
    target = np.asarray(target, dtype=float)
    atm = float(target[offsets == 0.0][0])
    usable = int(np.sum(np.isfinite(target) & (strikes + shift > 0)))
    base = dict(point=point, offsets_bp=offsets, strikes=strikes, market_normal=normal, beta=beta, shift=shift, fit=fit)

    def flat(status: str, message: str = "", dropped: int = 0) -> SmileFit:
        alpha = sabr_alpha_from_atm(F, T, atm, beta, 0.0, 0.0, shift, fit)
        params = {"alpha": alpha, "beta": beta, "rho": 0.0, "nu": 0.0, "shift": shift}
        model = _model_normal(F, strikes, T, params, fit)
        return SmileFit(**base, model_normal=model, alpha=alpha, rho=0.0, nu=0.0, status=status, message=message,
                        n_dropped=dropped)  # fmt: skip

    if not math.isfinite(atm):
        raise ValueError(f"{point.expiry}x{point.tenor}: the ATM quote has no {fit} vol equivalent")
    if usable < MIN_SMILE_QUOTES:
        return flat("flat", f"{usable} usable strike(s): smile not fitted", int(offsets.size - usable))
    try:
        cal = calibrate_sabr_smile(
            strikes, target, T, F, beta, shift, vol_type=fit, weights=weights, alpha_from_atm=True, atm_vol=atm
        )
    except ValueError as exc:
        return flat("failed", str(exc), int(offsets.size - usable))
    params = {"alpha": cal.alpha, "beta": cal.beta, "rho": cal.rho, "nu": cal.nu, "shift": shift}
    try:
        model = _model_normal(F, strikes, T, params, fit)
    except ValueError as exc:
        return flat("failed", f"the fitted smile breaks down: {exc}", cal.n_dropped)
    if not np.isfinite(atm_n):
        raise ValueError(f"{point.expiry}x{point.tenor}: no ATM normal vol")
    return SmileFit(
        **base,
        model_normal=model,
        alpha=cal.alpha,
        rho=cal.rho,
        nu=cal.nu,
        status="fitted" if cal.success else "failed",
        message="" if cal.success else cal.message,
        n_dropped=cal.n_dropped,
        at_bound=tuple(k for k, v in cal.at_bound.items() if v),
    )


def _model_normal(F: float, strikes: np.ndarray, T: float, params: Mapping[str, float], fit: str) -> np.ndarray:
    """Model normal vols at the quoted strikes (NaN below ``-shift``)."""

    out = np.full(strikes.shape, np.nan)
    ok = strikes + params["shift"] > 0
    if np.any(ok):
        out[ok] = np.asarray(smile_vol(F, strikes[ok], T, params, fit, "normal"), dtype=float)
    return out


# --------------------------------------------------------------------------
# Cube
# --------------------------------------------------------------------------


def _segment(xs: np.ndarray, x: float) -> Tuple[int, int, float]:
    """(left, right, weight of right) for linear interpolation at ``x``,
    flat outside ``xs``."""

    if x <= xs[0]:
        return 0, 0, 0.0
    if x >= xs[-1]:
        n = xs.size - 1
        return n, n, 0.0
    j = int(np.searchsorted(xs, x))
    w = (x - xs[j - 1]) / (xs[j] - xs[j - 1])
    return j - 1, j, float(w)


def variance_time_atm(expiries: np.ndarray, tenors: np.ndarray, atm: np.ndarray, T: float, L: float) -> float:
    """ATM normal vol at (``T`` years to expiry, ``L``-year tenor) from an
    expiry x tenor matrix: total variance ``vol^2 T`` linear in expiry
    between quoted expiries (flat vol outside), then linear in tenor (flat
    outside); ``sqrt(variance / T)``."""

    if not (math.isfinite(T) and T > 0):
        raise ValueError(f"expiry must be a positive number of years, got {T}")
    i0, i1, wi = _segment(expiries, T)
    j0, j1, wj = _segment(tenors, L)

    def column(j: int) -> float:
        if i0 == i1:  # flat vol beyond the quoted expiries
            return atm[i0, j] ** 2 * T
        v0, v1 = atm[i0, j] ** 2 * expiries[i0], atm[i1, j] ** 2 * expiries[i1]
        return v0 + wi * (v1 - v0)

    variance = (1.0 - wj) * column(j0) + wj * column(j1)
    return math.sqrt(variance / T)


def _bilinear(expiries: np.ndarray, tenors: np.ndarray, grid: np.ndarray, T: float, L: float) -> float:
    i0, i1, wi = _segment(expiries, T)
    j0, j1, wj = _segment(tenors, L)
    top = (1 - wj) * grid[i0, j0] + wj * grid[i0, j1]
    bottom = (1 - wj) * grid[i1, j0] + wj * grid[i1, j1]
    return float((1 - wi) * top + wi * bottom)


@dataclass(frozen=True)
class SwaptionVolCube:
    """SABR smiles on an expiry x tenor grid with variance-time ATM
    interpolation (see the module docstring)."""

    grid: VolGrid
    fits: Tuple[SmileFit, ...]
    fit: str
    beta: float
    shift: float
    index: Optional[str] = None
    atm_bumps: Optional[np.ndarray] = field(default=None, repr=False)  # normal-vol bumps of the ATM matrix

    @property
    def expiries(self) -> Tuple[str, ...]:
        return self.grid.expiries

    @property
    def tenors(self) -> Tuple[str, ...]:
        return self.grid.tenors

    @property
    def expiry_years(self) -> np.ndarray:
        return np.array([f.point.expiry_years for f in self.fits[:: len(self.tenors)]])

    @property
    def tenor_years(self) -> np.ndarray:
        return np.array([f.point.tenor_years for f in self.fits[: len(self.tenors)]])

    def _matrix(self, attr: str) -> np.ndarray:
        values = np.array([getattr(f, attr) for f in self.fits], dtype=float)
        return values.reshape(len(self.expiries), len(self.tenors))

    @property
    def atm_normal(self) -> np.ndarray:
        """ATM normal vol matrix (expiry x tenor), rate units, bumps included."""

        base = self._matrix("atm_normal")
        return base if self.atm_bumps is None else base + self.atm_bumps

    @property
    def rho(self) -> np.ndarray:
        return self._matrix("rho")

    @property
    def nu(self) -> np.ndarray:
        return self._matrix("nu")

    def point(self, expiry: str, tenor: str) -> SmileFit:
        for f in self.fits:
            if f.point.expiry == expiry and f.point.tenor == tenor:
                return f
        raise ValueError(f"{expiry}x{tenor} is not a quoted point of the cube")

    def atm_vol(self, T: float, L: float) -> float:
        """Interpolated ATM normal vol (rate units)."""

        return variance_time_atm(self.expiry_years, self.tenor_years, self.atm_normal, T, L)

    def params_at(self, T: float, L: float, F: float) -> Dict[str, float]:
        """SABR parameters at (expiry years, tenor years) and forward ``F``:
        rho, nu bilinear; alpha from the interpolated ATM vol."""

        if F + self.shift <= 0:
            raise ValueError(f"forward {F:.3%} is below -shift ({-self.shift:.2%}): raise the cube's SABR shift")
        rho = _bilinear(self.expiry_years, self.tenor_years, self.rho, T, L)
        nu = _bilinear(self.expiry_years, self.tenor_years, self.nu, T, L)
        atm_n = self.atm_vol(T, L)
        atm = atm_n if self.fit == "normal" else float(normal_to_lognormal_vol(F, F, T, atm_n, shift=self.shift))
        alpha = sabr_alpha_from_atm(F, T, atm, self.beta, rho, nu, self.shift, self.fit)
        return {"alpha": alpha, "beta": self.beta, "rho": rho, "nu": nu, "shift": self.shift}

    def normal_vol(self, T: float, L: float, F: float, K):
        """Cube normal vol (rate units) at strike(s) ``K``."""

        K_arr = np.asarray(K, dtype=float)
        if np.any(K_arr + self.shift <= 0):
            low = float(np.min(K_arr))
            raise ValueError(
                f"strike {low:.3%} is below -shift ({-self.shift:.2%}): the SABR cube has no vol there; raise the "
                "SABR shift or price at a flat vol"
            )
        out = smile_vol(F, K_arr, T, self.params_at(T, L, F), self.fit, "normal")
        return float(out) if np.ndim(out) == 0 else np.asarray(out)

    def bumped(self, i: int, j: int, bp: float = 1.0) -> "SwaptionVolCube":
        """The cube with ATM point (i, j) moved by ``bp`` of normal vol
        (rho and nu held)."""

        bumps = np.zeros((len(self.expiries), len(self.tenors))) if self.atm_bumps is None else self.atm_bumps.copy()
        bumps[i, j] += bp * 1e-4
        return replace(self, atm_bumps=bumps)

    def parallel(self, bp: float) -> "SwaptionVolCube":
        bumps = np.full((len(self.expiries), len(self.tenors)), bp * 1e-4)
        if self.atm_bumps is not None:
            bumps += self.atm_bumps
        return replace(self, atm_bumps=bumps)

    def fit_table(self) -> pd.DataFrame:
        """One row per point: forward, ATM vol, SABR params, residuals (bp)."""

        rows = []
        for f in self.fits:
            rows.append(
                {
                    "expiry": f.point.expiry,
                    "tenor": f.point.tenor,
                    "expiry_years": f.point.expiry_years,
                    "forward_pct": f.point.forward * 100.0,
                    "atm_normal_bp": f.atm_normal * 1e4,
                    "alpha": f.alpha,
                    "beta": f.beta,
                    "rho": f.rho,
                    "nu": f.nu,
                    "shift_pct": f.shift * 100.0,
                    "rmse_bp": f.rmse_bp,
                    "max_abs_bp": f.max_abs_bp,
                    "quotes": int(f.offsets_bp.size),
                    "dropped": f.n_dropped,
                    "status": f.status,
                    "at_bound": ", ".join(f.at_bound),
                    "message": f.message,
                }
            )
        return pd.DataFrame(rows)


def calibrate_cube(
    grid: VolGrid,
    points: Sequence[GridPoint],
    fit: str = "normal",
    beta: Optional[float] = None,
    shift: float = DEFAULT_SHIFT,
    weights: Optional[str] = None,
    index: Optional[str] = None,
) -> SwaptionVolCube:
    """Fit SABR at every point of ``grid`` (``points`` from
    :func:`grid_points`, in the grid's expiry-major order)."""

    if fit not in FITS:
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    if not math.isfinite(shift) or shift < 0:
        raise ValueError("the SABR shift must be zero or a positive decimal")
    expected = [(e, t) for e in grid.expiries for t in grid.tenors]
    if [(p.expiry, p.tenor) for p in points] != expected:
        raise ValueError("grid points must come from grid_points(grid, market) for this grid")
    beta = DEFAULT_BETA[fit] if beta is None else float(beta)
    fits = tuple(calibrate_point(grid, p, fit, beta, shift, weights) for p in points)
    return SwaptionVolCube(grid, fits, fit, beta, shift, index)


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def butterfly_check(fit: SmileFit, n: int = 161) -> Dict[str, object]:
    """Risk-neutral density of the fitted smile, ``d2 C / dK2`` of the
    Bachelier price of its normal vols, on a strike grid across the quoted
    offsets; negative density is a butterfly arbitrage. Returns the minimum
    density (per unit rate, undiscounted) and the offsets (bp) where it is
    negative."""

    F, T = fit.point.forward, fit.point.expiry_years
    lo = max(float(fit.offsets_bp.min()), (-fit.shift - F) * 1e4 + 1.0)
    offsets = np.linspace(lo, float(fit.offsets_bp.max()), n)
    K = F + offsets * 1e-4
    h = 1e-4  # 1bp strike step
    try:
        vols = [np.asarray(fit.vol(K + d), dtype=float) for d in (-h, 0.0, h)]
    except ValueError as exc:
        return {"min_density": float("nan"), "negative_offsets_bp": [], "message": str(exc)}
    prices = [np.asarray(bachelier_price(F, K + d, T, v), dtype=float) for d, v in zip((-h, 0.0, h), vols)]
    density = (prices[0] - 2 * prices[1] + prices[2]) / (h * h)
    tol = -1e-8 * float(np.max(np.abs(density)))
    bad = offsets[density < tol]
    return {"min_density": float(density.min()), "negative_offsets_bp": [float(x) for x in bad], "message": ""}


def calendar_check(cube: SwaptionVolCube) -> List[Dict[str, object]]:
    """Points where ATM total normal variance ``vol^2 T`` falls from one
    expiry to the next at the same tenor. A diagnostic, not a strict
    arbitrage: different expiries have different underlying swaps."""

    atm, T = cube.atm_normal, cube.expiry_years
    out = []
    for j, tenor in enumerate(cube.tenors):
        w = atm[:, j] ** 2 * T
        for i in range(1, len(T)):
            if w[i] < w[i - 1] * (1 - 1e-12):
                out.append({"tenor": tenor, "from": cube.expiries[i - 1], "to": cube.expiries[i],
                            "variance_drop_pct": (1 - w[i] / w[i - 1]) * 100.0})  # fmt: skip
    return out


# --------------------------------------------------------------------------
# Trades on the cube and vega buckets
# --------------------------------------------------------------------------


def cube_vol(cube: SwaptionVolCube, inputs_T: float, tenor: str, forward: float, strike: float, model: str,
             shift: float = 0.0) -> float:  # fmt: skip
    """Vol in ``model``'s units for a trade: the cube's normal vol, or the
    (shifted) lognormal vol with the same price."""

    L = parse_tenor(tenor).years
    normal = cube.normal_vol(inputs_T, L, forward, strike)
    if model == "bachelier":
        return float(normal)
    s = shift if model == "shifted_black" else 0.0
    out = float(normal_to_lognormal_vol(forward, strike, inputs_T, normal, shift=s))
    if not math.isfinite(out):
        raise ValueError(f"no {model} vol has the cube's price at strike {strike:.3%} (forward {forward:.3%})")
    return out


def swaption_on_cube(
    cube: SwaptionVolCube,
    market: Market,
    notional: float,
    expiry: str,
    tenor: str,
    strike: Optional[float] = None,
    offset_bp: float = 0.0,
    kind: str = "payer",
    model: str = "bachelier",
    shift: float = 0.0,
    conventions: Optional[str] = None,
) -> Swaption:
    """A swaption with its vol read from the cube (in ``model``'s units) at
    ``strike``, or at ATM + ``offset_bp`` when ``strike`` is None."""

    probe = Swaption(notional, expiry, tenor, 0.0, kind, index=cube.index, conventions=conventions)
    inputs = probe.inputs(market)
    K = inputs.forward + offset_bp * 1e-4 if strike is None else float(strike)
    vol = cube_vol(cube, inputs.expiry_years, tenor, inputs.forward, K, model, shift)
    return Swaption(notional, expiry, tenor, K, kind, vol, model, shift, cube.index, conventions)


def vega_buckets(trades: Sequence[Swaption], cube: SwaptionVolCube, market: Market, bp: float = 1.0) -> np.ndarray:
    """P&L of the trades for +``bp`` of normal vol on each ATM point of the
    cube (expiry x tenor matrix, currency), smile shape held. Each trade's
    vol moves by the change in the cube's vol at its own strike (converted
    to its model's units), on top of the vol it carries."""

    out = np.zeros((len(cube.expiries), len(cube.tenors)))
    for trade in trades:
        inputs = trade.inputs(market)
        T, F, K = inputs.expiry_years, inputs.forward, trade.strike
        if T <= 0:
            continue
        base_cube = cube_vol(cube, T, trade.tenor, F, K, trade.model, trade.shift)
        base_pv = trade.value(inputs, trade.market_vol(market))
        for i in range(len(cube.expiries)):
            for j in range(len(cube.tenors)):
                moved = cube_vol(cube.bumped(i, j, bp), T, trade.tenor, F, K, trade.model, trade.shift)
                if moved == base_cube:
                    continue
                vol = max(trade.market_vol(market) + moved - base_cube, 0.0)
                out[i, j] += trade.value(inputs, vol) - base_pv
    return out
