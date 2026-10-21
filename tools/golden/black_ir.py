"""QuantLib reference values for the legacy Black-76 caplet
(``derivx.rates.black_ir.price_caplet_black``, tests/test_black_ir.py).

``cases``: ``ql.blackFormula`` for calls (caplets) and puts (floorlets) on a
forward rate ``F`` with strike ``K``, total standard deviation
``sigma * sqrt(T)`` and discount factor ``df``, per unit notional and unit
accrual. Forwards from 50bp to 8%, strikes from half to twice the forward,
3 months to 20 years, vols from 5% to 120%; ``df = exp(-F (T + 0.25))``
(payment one quarter after the fixing).
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List

import QuantLib as ql

FORWARDS = [0.005, 0.03, 0.08]
STRIKE_RATIOS = [0.5, 1.0, 2.0]
EXPIRIES = [0.25, 1.0, 5.0, 20.0]
VOLS = [0.05, 0.3, 1.2]


def generate() -> Dict[str, object]:
    cases: List[Dict[str, float]] = []
    for (F, ratio, T), vol in zip(
        itertools.product(FORWARDS, STRIKE_RATIOS, EXPIRIES), itertools.cycle(VOLS + VOLS[::-1])
    ):
        K = F * ratio
        df = math.exp(-F * (T + 0.25))
        std = vol * math.sqrt(T)
        call = ql.blackFormula(ql.Option.Call, K, F, std, df)
        put = ql.blackFormula(ql.Option.Put, K, F, std, df)
        cases.append({"F": F, "K": K, "T": T, "sigma": vol, "df": df, "caplet": call, "floorlet": put})
    return {"cases": cases}
