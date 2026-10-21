"""Discounting of option premiums from the trade date to each expiry: a
flat rate, a discount curve (the US Treasury curve of the trade date), or
the discount factor implied per expiry by put-call parity.

Put-call parity (European exercise): ``C(K) - P(K) = DF (F - K)``, so across
the strikes of one expiry ``C - P`` is a line in K with slope ``-DF`` and
intercept ``DF F``. :func:`parity_regression` fits that line by weighted
least squares on the (up to 40) strikes nearest the forward that have both
a call and a put two-sided quote (weights ``1 / (spread_call^2 +
spread_put^2)``, so wide in-the-money quotes count little; one trimming
pass at 3 residual standard deviations) and returns F, DF, the implied
rate ``r = -ln(DF) / tau`` and the standard error of r. Near expiry the
slope cannot pin r down (its standard error is hundreds of bp within a
month on noisy quotes), but there the DF barely matters, and the gate
hands those expiries to the reference curve.

American exercise (single stocks and ETFs such as SPY) adds an
early-exercise premium to the in-the-money leg that grows with moneyness,
which biases the slope: on the bundled SPY chain the regression implies
rates from -30% to +1.5% across expiries (DF above 1 at short expiries).
Parity discount factors are therefore meant for European-style options
(:func:`european_style` lists index tickers with European exercise); for
American-style chains the Treasury curve is the default.

:class:`Discounting` modes:

* ``rate``   - ``DF = exp(-rate tau)``, a flat continuously compounded rate
  (an input, not market data);
* ``curve``  - the trade date's discount curve (:class:`CurveDiscount`,
  e.g. :func:`treasury_discount`): ``DF = P(expiry) / P(trade date)``;
  without a curve for a trade date the flat ``rate`` is used and the slice
  says so;
* ``parity`` - the parity regression's DF where it passes every gate (at
  least ``min_pairs`` call/put pairs, ``se(r) <= max_rate_se``, ``|r -
  r_ref| <= max_gap`` with ``r_ref`` the curve rate, or the flat rate
  without a curve, and a forward within ``max_carry_gap`` of spot), else the
  reference DF, with the reason recorded per expiry.

Units: rates are continuously compounded per-annum decimals; ``tau`` is
ACT/365F years from the trade date to the expiry (whole calendar days, as
in :mod:`derivx.vol.quotes`); premiums in the underlying's currency.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

MODES: Dict[str, str] = {
    "curve": "the trade date's discount curve (US Treasury par curve, bootstrapped)",
    "parity": "put-call parity per expiry (European options), else the reference curve or rate",
    "rate": "a flat continuously compounded rate (an input)",
}
# Index options with European exercise (Cboe/Nasdaq); Yahoo spells some with a caret.
EUROPEAN_UNDERLYINGS = frozenset(
    {"SPX", "SPXW", "XSP", "NDX", "NDXP", "XND", "RUT", "MRUT", "VIX", "DJX", "XEO", "SYNTH"}
)
MAX_CURVE_AGE_DAYS = 7  # a Treasury curve older than this before the trade date is not used


def european_style(ticker: Optional[str]) -> bool:
    """True for index tickers whose listed options are European-style
    (``^SPX``, ``SPX``, ``XSP``, ``NDX``, ``RUT``, ``VIX``, ...) and for the
    SYNTHETIC demo chains (priced with Black-76). Single stocks and ETFs
    (SPY, QQQ, IWM, ...) list American options: False."""

    return bool(ticker) and str(ticker).strip().upper().lstrip("^") in EUROPEAN_UNDERLYINGS


def _day(value) -> dt.date:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize().date()


# --------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _curve(asof: dt.date, times: Tuple[float, ...], dfs: Tuple[float, ...], interpolation: str):
    from derivx.rates.curves import Curve

    return Curve(asof, times, dfs, interpolation)


@dataclass(frozen=True)
class CurveDiscount:
    """A discount curve as of ``asof``: pillar ``times`` (ACT/365F years
    from ``asof``) and discount factors, interpolated as ``interpolation``
    (see :class:`derivx.rates.curves.Curve`). Frozen and hashable, so it can
    key caches. ``label`` names its data (source and date)."""

    asof: dt.date
    times: Tuple[float, ...]
    dfs: Tuple[float, ...]
    interpolation: str = "loglinear"
    label: str = ""

    @classmethod
    def from_curve(cls, curve, label: str) -> "CurveDiscount":
        return cls(
            curve.asof,
            tuple(float(t) for t in curve.ts),
            tuple(float(d) for d in curve.dfs),
            curve.interpolation,
            label,
        )

    def discount(self, trade_date, expiry) -> float:
        """``P(expiry) / P(trade_date)``: the DF from the trade date to the
        expiry. ``ValueError`` when the trade date is before the curve date
        (a curve is never used before it was published)."""

        td, ex = _day(trade_date), _day(expiry)
        if td < self.asof:
            raise ValueError(f"the trade date {td} is before the curve date {self.asof}")
        curve = _curve(self.asof, self.times, self.dfs, self.interpolation)
        return float(curve.df(ex) / curve.df(td))

    def describe(self) -> Dict[str, object]:
        return {
            "asof": self.asof.isoformat(),
            "label": self.label,
            "interpolation": self.interpolation,
            "pillars": len(self.times) - 1,
        }


def treasury_discount(store, trade_date, max_age_days: int = MAX_CURVE_AGE_DAYS) -> Optional[CurveDiscount]:
    """The US Treasury curve published on or before ``trade_date`` (within
    ``max_age_days``) from a :class:`derivx.data.rates_history.HistoryStore`,
    bootstrapped from its par yields (bills to 1Y as bond-equivalent yields,
    2Y-30Y as semiannual par coupons; :func:`derivx.rates.curves.bootstrap`
    with ``USD UST`` conventions, log-linear DFs). None when the store has
    no curve that recent: the caller falls back to a flat rate and says so.

    The curve of the trade date itself is used when the store has it; a
    snapshot fetched before the Treasury's evening publication normally
    gets the previous business day's curve."""

    from derivx.rates.curves import bootstrap
    from derivx.rates.market_quotes import treasury_quotes

    day = _day(trade_date)
    asof = store.latest_on_or_before("UST_PAR", day)
    if asof is None or (day - asof).days > int(max_age_days):
        return None
    curve = bootstrap(asof, treasury_quotes(store.curve("UST_PAR", asof).to_dict()), "USD UST", "loglinear")
    origin = store.origin("UST_PAR", asof)
    return CurveDiscount.from_curve(curve, f"US Treasury par curve of {asof} ({origin}), bootstrapped")


# --------------------------------------------------------------------------
# Put-call parity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ParityEstimate:
    """Result of :func:`parity_regression`. ``ok`` says whether every gate
    passed; ``reason`` says which failed. ``rate`` / ``rate_se`` are cc
    decimals; NaN when the regression could not be run."""

    forward: float
    discount_factor: float
    rate: float
    rate_se: float
    n_pairs: int
    n_trimmed: int
    residual_rms: float
    ok: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "parity_forward": self.forward,
            "parity_df": self.discount_factor,
            "parity_rate": self.rate,
            "parity_rate_se": self.rate_se,
            "parity_pairs": self.n_pairs,
            "parity_trimmed": self.n_trimmed,
            "parity_residual_rms": self.residual_rms,
            "parity_ok": self.ok,
            "parity_reason": self.reason,
        }


def _failed(reason: str, n: int = 0) -> ParityEstimate:
    nan = float("nan")
    return ParityEstimate(nan, nan, nan, nan, n, 0, nan, False, reason)


def _pairs(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Per strike with both a call and a put two-sided quote: mids and
    spreads (median over duplicate quotes)."""

    if not {"strike", "bid", "ask", "option_type"} <= set(slice_df.columns):
        return pd.DataFrame(columns=["c", "p", "sc", "sp"])
    raw = slice_df["option_type"].astype(str).str.strip().str.lower().str[0]
    bid = pd.to_numeric(slice_df["bid"], errors="coerce")
    ask = pd.to_numeric(slice_df["ask"], errors="coerce")
    df = pd.DataFrame(
        {
            "K": pd.to_numeric(slice_df["strike"], errors="coerce"),
            "t": raw.to_numpy(),
            "mid": ((bid + ask) / 2.0).to_numpy(),
            "spr": (ask - bid).to_numpy(),
        }
    )
    df = df[np.isfinite(df["K"]) & (df["K"] > 0) & (bid.to_numpy() > 0) & (ask.to_numpy() >= bid.to_numpy())]
    calls = df[df["t"] == "c"].groupby("K")[["mid", "spr"]].median()
    puts = df[df["t"] == "p"].groupby("K")[["mid", "spr"]].median()
    both = calls.index.intersection(puts.index)
    return pd.DataFrame(
        {
            "c": calls.loc[both, "mid"],
            "p": puts.loc[both, "mid"],
            "sc": calls.loc[both, "spr"],
            "sp": puts.loc[both, "spr"],
        },
        index=both,
    ).sort_index()


def parity_regression(
    slice_df: pd.DataFrame,
    tau: float,
    centre: float,
    *,
    reference_rate: Optional[float] = None,
    spot: Optional[float] = None,
    n_strikes: int = 40,
    min_pairs: int = 5,
    max_rate_se: float = 0.0025,
    max_gap: Optional[float] = 0.01,
    max_carry_gap: float = 0.10,
) -> ParityEstimate:
    """Forward and discount factor of one expiry from put-call parity (see
    the module docs). ``centre`` (e.g. a first parity forward) picks the
    ``n_strikes`` call/put pairs nearest it; ``reference_rate`` (cc) is the
    rate the gate ``|r - reference_rate| <= max_gap`` compares with (None
    or ``max_gap=None`` skips that gate); ``spot`` enables the forward gate
    ``|F / spot - 1| <= max_carry_gap``."""

    if not np.isfinite(tau) or tau <= 0:
        return _failed("tau <= 0: no discounting to estimate")
    pairs = _pairs(slice_df)
    if len(pairs) < max(int(min_pairs), 3):
        return _failed(
            f"{len(pairs)} strike(s) with both a call and a put quote; need {max(int(min_pairs), 3)}", len(pairs)
        )
    K_all = pairs.index.to_numpy(float)
    near = np.argsort(np.abs(K_all - float(centre)), kind="stable")[: max(int(n_strikes), int(min_pairs))]
    sel = pairs.iloc[np.sort(near)]
    K = sel.index.to_numpy(float)
    y = (sel["c"] - sel["p"]).to_numpy(float)
    var = (sel["sc"].to_numpy(float) ** 2 + sel["sp"].to_numpy(float) ** 2) / 12.0  # uniform error within the spread
    w = 1.0 / np.maximum(var, 1e-12)
    keep = np.ones(K.size, dtype=bool)
    trimmed = 0
    for _ in range(2):
        X = np.c_[np.ones(int(keep.sum())), -K[keep]]
        sw = np.sqrt(w[keep])
        coef, *_ = np.linalg.lstsq(X * sw[:, None], y[keep] * sw, rcond=None)
        resid = y - (coef[0] - coef[1] * K)
        dof = max(int(keep.sum()) - 2, 1)
        s2 = float(np.sum(w[keep] * resid[keep] ** 2) / dof)
        z = np.abs(resid) * np.sqrt(w) / np.sqrt(max(s2, 1e-300))
        drop = keep & (z > 3.0)
        if not drop.any() or int((keep & ~drop).sum()) < max(int(min_pairs), 3):
            break
        trimmed += int(drop.sum())
        keep &= ~drop
    n = int(keep.sum())
    a, D = float(coef[0]), float(coef[1])
    if not np.isfinite(D) or D <= 0:
        return _failed(f"the parity slope gives a non-positive discount factor ({D:.6g})", n)
    X = np.c_[np.ones(n), -K[keep]]
    try:
        cov = s2 * np.linalg.inv((X * w[keep][:, None]).T @ X)
    except np.linalg.LinAlgError:
        return _failed("the strikes do not identify the parity line (singular design)", n)
    se_D = float(np.sqrt(max(cov[1, 1], 0.0)))
    F = a / D
    r = float(-np.log(D) / tau)
    se_r = float(se_D / (D * tau))
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
    est = dict(
        forward=float(F), discount_factor=D, rate=r, rate_se=se_r, n_pairs=n, n_trimmed=trimmed, residual_rms=rms
    )
    reasons = []
    if n < int(min_pairs):
        reasons.append(f"{n} pairs after trimming < {int(min_pairs)}")
    if not np.isfinite(se_r) or se_r > max_rate_se:
        reasons.append(f"rate standard error {se_r * 1e4:.0f}bp > {max_rate_se * 1e4:.0f}bp")
    if max_gap is not None and reference_rate is not None and abs(r - reference_rate) > max_gap:
        reasons.append(
            f"implied rate {r:.2%} is {(r - reference_rate) * 1e4:+.0f}bp from the reference {reference_rate:.2%} "
            f"(limit {max_gap * 1e4:.0f}bp: American exercise, stale or wide quotes?)"
        )
    if spot is not None and np.isfinite(spot) and spot > 0 and abs(F / spot - 1.0) > max_carry_gap:
        reasons.append(f"parity forward {F:.6g} is {F / spot - 1:+.1%} from spot")
    if reasons:
        return ParityEstimate(**est, ok=False, reason="; ".join(reasons))
    note = f"; {trimmed} pair(s) trimmed" if trimmed else ""
    return ParityEstimate(**est, ok=True, reason=f"parity regression on {n} call/put pairs{note}")


# --------------------------------------------------------------------------
# Discounting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscountChoice:
    """How one expiry is discounted: ``discount_factor`` from the trade date
    to the expiry, ``method`` ('rate', 'curve', 'parity' or 'given'),
    ``source`` (a label), the parity ``forward`` when the parity DF was
    used, and the regression details in ``parity`` (empty when not run)."""

    discount_factor: float
    method: str
    source: str
    forward: Optional[float] = None
    parity: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Discounting:
    """How a chain is discounted (see the module docs). ``curves`` maps
    trade dates (ISO strings) to :class:`CurveDiscount`; ``rate`` is the flat
    cc rate of mode 'rate' and the fallback where no curve covers a trade
    date. The parity gates are ``min_pairs``, ``max_rate_se`` and
    ``max_gap`` (decimals)."""

    mode: str = "rate"
    rate: float = 0.0
    curves: Tuple[Tuple[str, CurveDiscount], ...] = ()
    min_pairs: int = 5
    max_rate_se: float = 0.0025
    max_gap: float = 0.01
    n_strikes: int = 40

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"discounting mode must be one of {sorted(MODES)}, got {self.mode!r}")
        if not np.isfinite(self.rate) or not -0.5 < self.rate < 1.0:
            raise ValueError(f"the flat rate must be a cc decimal between -50% and 100%, got {self.rate}")
        if self.min_pairs < 3 or self.max_rate_se <= 0 or self.max_gap <= 0:
            raise ValueError("parity gates need min_pairs >= 3, max_rate_se > 0 and max_gap > 0")
        if self.mode == "curve" and not self.curves:
            raise ValueError("mode 'curve' needs at least one curve (e.g. treasury_discount(store, trade_date))")

    @classmethod
    def flat(cls, rate: float) -> "Discounting":
        return cls("rate", float(rate))

    def curve_for(self, trade_date) -> Optional[CurveDiscount]:
        day = _day(trade_date).isoformat()
        for key, curve in self.curves:
            if key == day:
                return curve
        return None

    def reference(self, trade_date, expiry, tau: float) -> Tuple[float, str, str]:
        """(DF, method, source) of the curve, or of the flat rate when the
        mode is 'rate' or no curve covers ``trade_date``."""

        curve = self.curve_for(trade_date) if self.mode != "rate" else None
        if curve is not None:
            return curve.discount(trade_date, expiry), "curve", curve.label
        tau = max(float(tau), 0.0)
        if self.mode == "rate":
            return float(np.exp(-self.rate * tau)), "rate", f"flat rate {self.rate:.4%} cc (input)"
        return (
            float(np.exp(-self.rate * tau)),
            "rate",
            f"flat rate {self.rate:.4%} cc (input): no Treasury curve within {MAX_CURVE_AGE_DAYS} days before "
            f"{_day(trade_date)}",
        )

    def choose(
        self, slice_df: pd.DataFrame, trade_date, expiry, tau: float, spot: float, centre: float
    ) -> DiscountChoice:
        """The discount of one expiry slice (``slice_df``: its usable quotes;
        ``centre``: a first forward estimate to pick strikes around)."""

        df_ref, method, source = self.reference(trade_date, expiry, tau)
        if self.mode != "parity" or tau <= 0:
            return DiscountChoice(df_ref, method, source)
        r_ref = -np.log(df_ref) / tau
        est = parity_regression(
            slice_df,
            tau,
            centre,
            reference_rate=r_ref,
            spot=spot,
            n_strikes=self.n_strikes,
            min_pairs=self.min_pairs,
            max_rate_se=self.max_rate_se,
            max_gap=self.max_gap,
        )
        if est.ok:
            return DiscountChoice(est.discount_factor, "parity", est.reason, est.forward, est.as_dict())
        return DiscountChoice(df_ref, method, f"{source} (parity not used: {est.reason})", None, est.as_dict())

    def describe(self) -> Dict[str, object]:
        """JSON-able description (for manifests and cache keys)."""

        out: Dict[str, object] = {"mode": self.mode, "what": MODES[self.mode], "flat_rate_cc": self.rate}
        if self.curves:
            out["curves"] = {k: c.describe() for k, c in self.curves}
        if self.mode == "parity":
            out["parity_gates"] = {
                "min_pairs": self.min_pairs,
                "max_rate_se": self.max_rate_se,
                "max_gap": self.max_gap,
                "n_strikes": self.n_strikes,
            }
        return out


def treasury_discounting(store, trade_dates, mode: str = "curve", rate: float = 0.0, **gates) -> Discounting:
    """:class:`Discounting` with the Treasury curve of each trade date
    (:func:`treasury_discount`; trade dates without a recent curve use
    ``rate``). ``mode`` is 'curve' or 'parity'."""

    if mode not in ("curve", "parity"):
        raise ValueError("treasury_discounting builds mode 'curve' or 'parity'")
    curves = []
    for td in sorted({_day(d) for d in trade_dates}):
        curve = treasury_discount(store, td)
        if curve is not None:
            curves.append((td.isoformat(), curve))
    if mode == "curve" and not curves:
        days = sorted({_day(d) for d in trade_dates})
        raise ValueError(
            f"the rates store has no US Treasury curve within {MAX_CURVE_AGE_DAYS} days before "
            f"{', '.join(str(d) for d in days[:4])}: choose a flat rate"
        )
    return Discounting(mode, float(rate), tuple(curves), **gates)
