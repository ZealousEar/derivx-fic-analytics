"""A stand-in for ``yfinance.Ticker`` serving a recorded chain, for tests
that must never touch the network. Like Yahoo, it serves only the current
chain; it can raise rate-limit errors on chosen requests and records every
request made."""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Optional

import pandas as pd
from yfinance.exceptions import YFRateLimitError

from derivx.data.samples import load_sample_chain

Options = namedtuple("Options", ["calls", "puts", "underlying"])
SPOT_TIME = int(pd.Timestamp("2026-10-20T20:00:00Z").timestamp())


def _yahoo_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={
            "contract_symbol": "contractSymbol",
            "last_trade_date": "lastTradeDate",
            "last_price": "lastPrice",
            "open_interest": "openInterest",
            "implied_vol": "impliedVolatility",
        }
    )
    cols = ["contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask", "volume", "openInterest"]
    return out[cols + ["impliedVolatility"]].reset_index(drop=True)


class FakeTicker:
    """``failures``: request name ('options' or an expiry) -> number of
    rate-limit errors to raise before serving it (-1: always)."""

    def __init__(
        self,
        symbol: str = "SPY",
        failures: Optional[Dict[str, int]] = None,
        spot: float = 769.05,
        spot_time: Optional[int] = SPOT_TIME,
        fatal: Optional[Dict[str, Exception]] = None,
    ):
        self.symbol = symbol
        self.failures = dict(failures or {})
        self.fatal = dict(fatal or {})
        self.spot, self.spot_time = spot, spot_time
        self.requests: List[str] = []
        self.chain = load_sample_chain().single()

    def _maybe_fail(self, name: str) -> None:
        self.requests.append(name)
        if name in self.fatal:
            raise self.fatal[name]
        left = self.failures.get(name, 0)
        if left:
            if left > 0:
                self.failures[name] = left - 1
            raise YFRateLimitError()

    @property
    def options(self):
        self._maybe_fail("options")
        return tuple(sorted(self.chain["expiry"].dt.strftime("%Y-%m-%d").unique()))

    def option_chain(self, expiry: str):
        self._maybe_fail(expiry)
        g = self.chain[self.chain["expiry"].dt.strftime("%Y-%m-%d") == expiry]
        underlying = {"regularMarketPrice": self.spot}
        if self.spot_time is not None:
            underlying["regularMarketTime"] = self.spot_time
        return Options(
            _yahoo_frame(g[g["option_type"] == "call"]), _yahoo_frame(g[g["option_type"] == "put"]), underlying
        )


class Clock:
    """Deterministic UTC clock advancing one second per reading."""

    def __init__(self, start: str = "2026-10-20T21:00:00Z"):
        self.t = pd.Timestamp(start)

    def __call__(self) -> pd.Timestamp:
        self.t = self.t + pd.Timedelta(seconds=1)
        return self.t
