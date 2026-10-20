"""Quote hygiene and forwards for one option-chain slice (one trade date,
one expiry), before any smile is fitted.

Input columns (Yahoo-style, as produced by
:func:`derivx.data.market_data.fetch_option_chain`): ``strike``,
``implied_vol`` (per-annum decimal), ``expiry``; optional ``trade_date``,
``option_type`` ('call'/'put' or 'C'/'P'), ``bid``, ``ask``, ``mid``,
``underlying_price`` (spot), ``last_trade_date`` / ``lastTradeDate``.

Conventions: time to expiry ``tau = days / 365`` with ``days`` the whole
calendar days from the trade date to the expiry (ACT/365F at day
granularity, so a same-day expiry has tau = 0; a timezone-aware date keeps
its local calendar day); ``rate`` and ``div_yield`` are continuously
compounded per-annum decimals; forward log-moneyness ``k = ln(K / F)``.

Implied vols: by default (``recompute_iv='auto'``) a slice with ``bid``,
``ask`` and ``option_type`` columns is re-inverted from its mids with
Black-76 on the parity forward and ``df = exp(-rate tau)``; the quoted
``implied_vol`` is used only when there are no executable quotes. Yahoo's
quoted vols come from the last trade with its own rate and dividend
assumptions and disagree between the call and the put at the same strike
(on a real SPY chain by 1 vol point at 1 month and 7-10 vol points beyond 6
months), which a smile fit cannot absorb. Recomputed vol levels depend on
the discount rate: pass the real ``rate`` (with ``rate=0`` the prices are
read about ``rate * tau`` too low in relative terms, which on a real SPY
chain put the vols 35-55 vol bp low at 1 year for a 4% rate, under 5 vol bp
within 2 months).

:func:`clean_option_slice` drops, with a reason per row (see
:data:`REASONS`): expired or same-day expiries, non-finite or non-positive
strikes and vols, Yahoo IV placeholders (iv <= 1e-4 or > 5), zero bids,
crossed quotes (ask < bid), stale quotes, in-the-money options (puts are
kept below the forward, calls above: the OTM option at a strike carries the
same vol with a tighter spread), and duplicates. It infers the forward from
put-call parity (:func:`infer_forward`) and returns one quote per strike.
A chain-level caller can pass ``forward=`` when parity fails for a slice
(e.g. one interpolated from the other expiries with
:func:`interpolate_forward`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from derivx.vol.implied import implied_black_vol

REASONS: Dict[str, str] = {
    "expired": "expiry on or before the trade date, or closer than min_days",
    "bad_strike": "strike missing, non-finite or <= 0",
    "iv_missing": "implied vol missing or non-finite",
    "iv_placeholder": "implied vol <= 1e-4 or > 5 (Yahoo placeholder values)",
    "iv_unsolvable": "recompute_iv: no Black vol reproduces the mid (outside the no-arbitrage bounds)",
    "zero_bid": "bid <= 0 (no executable market)",
    "crossed": "ask < bid",
    "stale": "last trade older than max_stale_days before the trade date",
    "itm": "in the money: the OTM option at the same strike is kept (puts below F, calls above)",
    "duplicate": "same strike and option type quoted more than once (kept the tightest spread)",
    "outside_band": "outside the standardised moneyness band |k| / (atm_vol sqrt(tau)) <= n_std",
}

IV_MIN = 1e-4
IV_MAX = 5.0


def _ts(value) -> pd.Timestamp:
    """Calendar date of ``value`` (a timezone-aware time keeps its local
    date: 20:30 New York time on the 19th is the 19th, not the 20th UTC)."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def day_count(trade_date, expiry) -> int:
    """Whole calendar days from ``trade_date`` to ``expiry`` (negative when
    the expiry is in the past)."""

    return int((_ts(expiry) - _ts(trade_date)).days)


def year_fraction(trade_date, expiry) -> float:
    """ACT/365F years from ``trade_date`` to ``expiry`` (dates at day
    granularity; negative when the expiry is in the past)."""

    return day_count(trade_date, expiry) / 365.0


def _option_types(df: pd.DataFrame) -> Optional[np.ndarray]:
    """'call' / 'put' per row, or None when the column is missing."""

    if "option_type" not in df.columns:
        return None
    raw = df["option_type"].astype(str).str.strip().str.lower()
    out = np.where(raw.str.startswith("c"), "call", np.where(raw.str.startswith("p"), "put", ""))
    if np.any(out == ""):
        bad = sorted(set(raw[out == ""]))[:5]
        raise ValueError(f"option_type must be call/put (or C/P); got {bad}")
    return out


def _mid(df: pd.DataFrame) -> pd.Series:
    if "bid" in df.columns and "ask" in df.columns:
        bid, ask = df["bid"].astype(float), df["ask"].astype(float)
        mid = (bid + ask) / 2.0
        return mid.where((bid > 0) & (ask >= bid))
    if "mid" in df.columns:
        return df["mid"].astype(float)
    return pd.Series(np.nan, index=df.index)


def infer_forward(
    slice_df: pd.DataFrame,
    spot: float,
    tau: float,
    rate: float = 0.0,
    div_yield: float = 0.0,
    n_strikes: int = 5,
    max_dispersion: float = 0.01,
    max_carry_gap: Optional[float] = 0.10,
) -> Tuple[float, str, int]:
    """Forward for one expiry from put-call parity, ``F = K + (C - P) /
    df`` with ``df = exp(-rate tau)``, per strike that has both a call and
    a put mid (from bid/ask, else ``mid``). Near the money ``C - P`` is
    small, so the result is insensitive to the assumed rate.

    The ``n_strikes`` strikes nearest the carry forward ``spot exp((rate -
    div_yield) tau)`` are taken, then re-selected once around the median of
    their estimates (the carry forward can be far off when ``rate`` and
    ``div_yield`` are unknown). Estimates more than ``max_dispersion / 2``
    (relative) from that median are dropped as inconsistent (stale or wide
    quotes); the forward is the median of the rest. Falls back to the carry
    forward when fewer than 2 pairs exist, when fewer than 2 (or fewer than
    half) of the estimates are consistent, or when the parity forward is
    more than ``max_carry_gap`` away from the carry forward (None disables
    that check). Returns ``(forward, source, n_used)``; ``source`` says
    which rule was used and why (``n_used = 0`` for the carry fallback)."""

    if not np.isfinite(spot) or spot <= 0:
        raise ValueError(f"spot (underlying_price) must be > 0, got {spot}")
    carry = float(spot * np.exp((rate - div_yield) * tau))
    types = _option_types(slice_df)
    if types is None:
        return carry, "carry: no option_type column for put-call parity", 0
    df = slice_df.assign(_type=types, _mid=_mid(slice_df).to_numpy())
    df = df[np.isfinite(df["_mid"]) & (df["_mid"] > 0) & np.isfinite(df["strike"].astype(float))]
    calls = df[df["_type"] == "call"].groupby("strike")["_mid"].median()
    puts = df[df["_type"] == "put"].groupby("strike")["_mid"].median()
    both = calls.index.intersection(puts.index)
    if len(both) < 2:
        return carry, f"carry: {len(both)} call/put pair(s) for put-call parity", 0
    strikes = np.asarray(both, dtype=float)
    disc = np.exp(-rate * tau)
    est_all = strikes + (calls.loc[both].to_numpy() - puts.loc[both].to_numpy()) / disc
    n = max(2, int(n_strikes))
    centre = carry
    for _ in range(2):  # nearest the carry forward, then nearest the first parity estimate
        near = np.argsort(np.abs(strikes - centre), kind="stable")[:n]
        centre = float(np.median(est_all[near]))
    est = est_all[near]
    if not np.isfinite(centre) or centre <= 0:
        return carry, "carry: parity estimates are not positive", 0
    consistent = np.abs(est - centre) <= 0.5 * max_dispersion * centre
    n_ok = int(consistent.sum())
    if n_ok < 2 or 2 * n_ok < est.size:
        spread = float((est.max() - est.min()) / centre)
        return (
            carry,
            f"carry: parity estimates disagree ({spread:.2%} range over {est.size} strikes, {n_ok} within "
            f"+-{0.5 * max_dispersion:.2%} of their median)",
            0,
        )
    fwd = float(np.median(est[consistent]))
    if max_carry_gap is not None and abs(fwd / carry - 1.0) > max_carry_gap:
        return carry, f"carry: parity forward {fwd:.6g} is {fwd / carry - 1:+.1%} from carry", 0
    dropped = est.size - n_ok
    note = f"; {dropped} inconsistent strike(s) dropped" if dropped else ""
    return fwd, f"put-call parity (median of {n_ok} strikes{note})", n_ok


def interpolate_forward(tau: float, spot: float, known: Sequence[Tuple[float, float]]) -> float:
    """Forward to ``tau`` years from the forwards ``known = [(tau_i, F_i)]``
    of other expiries of the same trade date (e.g. their parity forwards):
    ``ln F`` linear in tau through ``(0, ln spot)`` and the known points;
    beyond the last one the average carry ``ln(F_last / spot) / tau_last``
    is extended. Raises ``ValueError`` without a usable known forward."""

    if not np.isfinite(spot) or spot <= 0:
        raise ValueError(f"spot must be > 0, got {spot}")
    pts = sorted((float(t), float(f)) for t, f in known if np.isfinite(t) and t > 0 and np.isfinite(f) and f > 0)
    if not pts:
        raise ValueError("interpolate_forward needs at least one known forward with tau > 0 and F > 0")
    ts = np.r_[0.0, [t for t, _ in pts]]
    logf = np.r_[np.log(spot), np.log([f for _, f in pts])]
    if tau <= ts[-1]:
        return float(np.exp(np.interp(tau, ts, logf)))
    carry = (logf[-1] - logf[0]) / ts[-1]
    return float(np.exp(logf[0] + carry * tau))


@dataclass
class CleanSlice:
    """One cleaned expiry slice: ``quotes`` has one row per strike with
    columns ``strike``, ``k`` (= ln(K/F)), ``implied_vol``, ``option_type``
    (or 'smile' when the input had none) and, when available, ``bid``,
    ``ask``, ``mid`` (and ``iv_bid`` / ``iv_ask`` for recomputed vols);
    ``removed`` holds dropped input rows with a ``reason`` column (see
    :data:`REASONS`). ``forward_method`` is 'parity', 'carry' or 'given'
    (:func:`derivx.calibration.production.calibrate_chain` relabels a
    forward it interpolated from other expiries 'interpolated');
    ``iv_source`` is 'mid' (recomputed from bid/ask mids) or 'quoted'."""

    trade_date: pd.Timestamp
    expiry: pd.Timestamp
    tau: float
    spot: float
    forward: float
    forward_source: str
    discount_factor: float
    quotes: pd.DataFrame
    removed: pd.DataFrame
    n_input: int
    notes: list = field(default_factory=list)
    forward_method: str = "parity"
    iv_source: str = "quoted"
    rate: float = 0.0

    @property
    def days(self) -> int:
        """Whole calendar days from the trade date to the expiry."""

        return day_count(self.trade_date, self.expiry)

    @property
    def counts(self) -> Dict[str, int]:
        if self.removed.empty:
            return {}
        return {str(k): int(v) for k, v in self.removed["reason"].value_counts().items()}

    def summary(self) -> Dict[str, object]:
        return {
            "trade_date": self.trade_date.date().isoformat(),
            "expiry": self.expiry.date().isoformat(),
            "tau": self.tau,
            "spot": self.spot,
            "forward": self.forward,
            "forward_source": self.forward_source,
            "iv_source": self.iv_source,
            "n_input": self.n_input,
            "n_kept": int(len(self.quotes)),
            "removed": self.counts,
        }


def _first_reason(masks) -> np.ndarray:
    n = len(masks[0][1])
    reason = np.array([""] * n, dtype=object)
    for name, mask in masks:
        reason = np.where((reason == "") & np.asarray(mask, dtype=bool), name, reason)
    return reason


def clean_option_slice(
    slice_df: pd.DataFrame,
    *,
    trade_date=None,
    spot: Optional[float] = None,
    rate: float = 0.0,
    div_yield: float = 0.0,
    min_days: float = 1.0,
    iv_min: float = IV_MIN,
    iv_max: float = IV_MAX,
    max_stale_days: Optional[float] = 4.0,
    otm_only: bool = True,
    recompute_iv: Union[bool, str] = "auto",
    forward: Optional[float] = None,
    forward_source: Optional[str] = None,
) -> CleanSlice:
    """Clean one (trade_date, expiry) slice; see the module docs.

    ``trade_date`` defaults to the slice's ``trade_date`` column (there is
    no 'today' fallback: one of the two is required) and ``spot`` to its
    ``underlying_price``. Rows expiring fewer than ``min_days`` whole days
    after the trade date (default 1: same-day and expired) are dropped.
    ``max_stale_days`` (None disables) applies when a last-trade column
    exists. With ``otm_only`` and an ``option_type`` column, ITM options are
    dropped and a call and put at the forward strike are averaged.

    ``recompute_iv`` (``'auto'``: whenever ``bid``, ``ask`` and
    ``option_type`` exist; ``True`` requires them; ``False`` keeps the
    quoted vols) replaces the quoted ``implied_vol`` with Black-76 vols of
    the mid on the forward and ``df = exp(-rate tau)``, and adds ``iv_bid``
    / ``iv_ask`` (NaN where the bid has no time value), e.g. for 1/spread
    weights; quotes no vol reproduces are dropped as 'iv_unsolvable'. The
    discount rate matters here (unlike for the parity forward): pass the
    real ``rate``.

    ``forward`` (with an optional ``forward_source`` label) replaces the
    put-call-parity inference, e.g. when the caller interpolated it from
    the other expiries of the chain.

    Raises ``ValueError`` when the slice mixes several trade dates,
    expiries or spots, or misses required columns."""

    for col in ("strike", "implied_vol", "expiry"):
        if col not in slice_df.columns:
            raise ValueError(f"option slice needs a {col!r} column")
    if slice_df.empty:
        raise ValueError("option slice is empty")
    df = slice_df.copy()
    expiries = pd.to_datetime(df["expiry"]).map(_ts).unique()
    if len(expiries) != 1:
        raise ValueError(f"clean_option_slice takes one expiry; got {len(expiries)}: group by expiry first")
    expiry = pd.Timestamp(expiries[0])
    if trade_date is None:
        if "trade_date" not in df.columns:
            raise ValueError("pass trade_date or include a 'trade_date' column: tau is measured from it")
        dates = pd.to_datetime(df["trade_date"]).map(_ts).unique()
        if len(dates) != 1:
            raise ValueError(f"slice mixes {len(dates)} trade dates: group by (trade_date, expiry) first")
        trade_date = dates[0]
    trade_date = _ts(trade_date)
    if spot is None:
        if "underlying_price" not in df.columns:
            raise ValueError("pass spot or include an 'underlying_price' column")
        spots = pd.to_numeric(df["underlying_price"], errors="coerce").dropna().unique()
        if len(spots) == 0:
            raise ValueError("underlying_price is missing for every row")
        if (spots.max() - spots.min()) > 1e-9 * abs(spots.max()):
            raise ValueError("slice has several underlying prices: group by trade date / snapshot first")
        spot = float(spots[0])
    spot = float(spot)
    days = day_count(trade_date, expiry)
    tau = days / 365.0
    types = _option_types(df)
    strike = pd.to_numeric(df["strike"], errors="coerce").to_numpy(dtype=float)
    iv = pd.to_numeric(df["implied_vol"], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    false = np.zeros(n, dtype=bool)
    bid = pd.to_numeric(df["bid"], errors="coerce").to_numpy(float) if "bid" in df.columns else None
    ask = pd.to_numeric(df["ask"], errors="coerce").to_numpy(float) if "ask" in df.columns else None
    stale = false
    last_col = next((c for c in ("last_trade_date", "lastTradeDate") if c in df.columns), None)
    if max_stale_days is not None and last_col is not None:
        last = pd.to_datetime(df[last_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
        age = (trade_date - last).dt.days.to_numpy(dtype=float)
        stale = np.isfinite(age) & (age > max_stale_days)
    has_quotes = bid is not None and ask is not None and types is not None
    if isinstance(recompute_iv, str):
        if recompute_iv != "auto":
            raise ValueError(f"recompute_iv must be True, False or 'auto', got {recompute_iv!r}")
        recompute = has_quotes
    else:
        recompute = bool(recompute_iv)
        if recompute and not has_quotes:
            raise ValueError("recompute_iv needs 'bid', 'ask' and 'option_type' columns")
    masks = [
        ("expired", np.full(n, days < min_days)),
        ("bad_strike", ~np.isfinite(strike) | (strike <= 0)),
        ("zero_bid", false if bid is None else ~(bid > 0)),
        ("crossed", false if (bid is None or ask is None) else (ask < bid)),
        ("stale", stale),
    ]
    reason = _first_reason(masks)
    good = reason == ""
    disc = float(np.exp(-rate * max(tau, 0.0)))
    notes = []
    if forward is not None:
        forward = float(forward)
        if not np.isfinite(forward) or forward <= 0:
            raise ValueError(f"forward must be > 0, got {forward}")
        source, method = forward_source or "given by the caller", "given"
    elif good.any() and tau > 0:
        forward, source, n_used = infer_forward(df[good], spot, tau, rate, div_yield)
        method = "parity" if n_used else "carry"
    else:
        forward, source = float(spot * np.exp((rate - div_yield) * max(tau, 0.0))), "carry (no valid quotes)"
        method = "carry"
    iv_bid = iv_ask = None
    iv_source = "quoted"
    if recompute and tau > 0:
        is_call = types == "call"
        solved = implied_black_vol((bid + ask) / 2.0, forward, strike, tau, is_call, disc)
        reason = np.where((reason == "") & ~solved.ok, "iv_unsolvable", reason)
        iv = np.asarray(solved.vol, dtype=float)
        with np.errstate(invalid="ignore"):
            iv_bid = np.asarray(
                implied_black_vol(np.where(bid > 0, bid, np.nan), forward, strike, tau, is_call, disc).vol
            )
            iv_ask = np.asarray(implied_black_vol(ask, forward, strike, tau, is_call, disc).vol)
        iv_source = "mid"
        notes.append(f"implied vols recomputed from mids (Black-76 on the forward, rate {rate:.4%} cc)")
    iv_masks = [
        ("iv_missing", ~np.isfinite(iv)),
        ("iv_placeholder", np.isfinite(iv) & ((iv <= iv_min) | (iv > iv_max))),
    ]
    iv_reason = _first_reason(iv_masks)
    reason = np.where(reason == "", iv_reason, reason)
    if types is not None and otm_only:
        itm = np.where(types == "call", strike < forward, strike > forward)
        reason = np.where((reason == "") & itm, "itm", reason)
    elif types is None:
        notes.append("no option_type column: quotes treated as one smile (no OTM selection)")
    # dedupe: same strike and type -> tightest relative spread
    kept = np.flatnonzero(reason == "")
    label = types if types is not None else np.array(["smile"] * n, dtype=object)
    if kept.size:
        spread = np.full(n, np.inf)
        if bid is not None and ask is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                spread = np.where((ask + bid) > 0, (ask - bid) / ((ask + bid) / 2.0), np.inf)
        sub = pd.DataFrame({"i": kept, "strike": strike[kept], "type": label[kept], "spread": spread[kept]})
        sub = sub.sort_values(["strike", "type", "spread", "i"])
        dup = sub.duplicated(["strike", "type"], keep="first")
        reason[sub.loc[dup, "i"].to_numpy()] = "duplicate"
    kept = np.flatnonzero(reason == "")
    rows = pd.DataFrame(
        {
            "strike": strike[kept],
            "implied_vol": iv[kept],
            "option_type": label[kept],
        }
    )
    for name, arr in (("bid", bid), ("ask", ask), ("iv_bid", iv_bid), ("iv_ask", iv_ask)):
        if arr is not None:
            rows[name] = arr[kept]
    if "bid" in rows and "ask" in rows:
        rows["mid"] = (rows["bid"] + rows["ask"]) / 2.0
    elif "mid" in df.columns:
        rows["mid"] = pd.to_numeric(df["mid"], errors="coerce").to_numpy(float)[kept]
    # a call and a put left at the same strike (at the forward, or no OTM selection): average them
    if not rows.empty and rows["strike"].duplicated().any():
        agg = {c: "mean" for c in rows.columns if c not in ("strike", "option_type")}
        agg["option_type"] = lambda s: "/".join(sorted(set(s)))
        rows = rows.groupby("strike", as_index=False).agg(agg)
        notes.append("call and put at the same strike averaged")
    rows = rows.sort_values("strike").reset_index(drop=True)
    rows.insert(1, "k", np.log(rows["strike"].to_numpy(float) / forward))
    removed = df.iloc[np.flatnonzero(reason != "")].copy()
    removed["reason"] = reason[reason != ""]
    return CleanSlice(
        trade_date=trade_date,
        expiry=expiry,
        tau=tau,
        spot=spot,
        forward=float(forward),
        forward_source=source,
        discount_factor=disc,
        quotes=rows,
        removed=removed,
        n_input=n,
        notes=notes,
        forward_method=method,
        iv_source=iv_source,
        rate=float(rate),
    )


def atm_vol(quotes: pd.DataFrame) -> float:
    """Vol at k = 0 from a cleaned smile (``k``, ``implied_vol`` columns):
    linear in k between the quotes either side, nearest quote outside."""

    if quotes.empty:
        raise ValueError("no quotes to read an ATM vol from")
    q = quotes.sort_values("k")
    return float(np.interp(0.0, q["k"].to_numpy(float), q["implied_vol"].to_numpy(float)))


def apply_std_band(clean: CleanSlice, n_std: float = 3.0) -> CleanSlice:
    """Keep quotes with ``|k| <= n_std * atm_vol * sqrt(tau)`` (standardised
    moneyness: the same band in standard deviations at every maturity);
    the rest move to ``removed`` with reason 'outside_band'."""

    if n_std <= 0:
        raise ValueError("n_std must be > 0")
    if clean.quotes.empty or clean.tau <= 0:
        return clean
    width = n_std * atm_vol(clean.quotes) * np.sqrt(clean.tau)
    inside = np.abs(clean.quotes["k"].to_numpy(float)) <= width
    out = clean.quotes[~inside].assign(reason="outside_band")
    removed = pd.concat([clean.removed, out], ignore_index=True) if not out.empty else clean.removed
    return CleanSlice(
        trade_date=clean.trade_date,
        expiry=clean.expiry,
        tau=clean.tau,
        spot=clean.spot,
        forward=clean.forward,
        forward_source=clean.forward_source,
        discount_factor=clean.discount_factor,
        quotes=clean.quotes[inside].reset_index(drop=True),
        removed=removed,
        n_input=clean.n_input,
        notes=list(clean.notes) + [f"band |k| <= {width:.4f} ({n_std:g} std)"],
        forward_method=clean.forward_method,
        iv_source=clean.iv_source,
        rate=clean.rate,
    )
