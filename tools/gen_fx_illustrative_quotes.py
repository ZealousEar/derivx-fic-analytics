"""Print the ILLUSTRATIVE FX vol quotes of src/derivx/fx/examples.py.

Not market data: a beta = 1 SABR smile per pair and tenor, with a skew rho
and a vol of vol nu (scaling like T^-0.4) chosen per pair, on the trade
date 2026-10-20 with the bundled ECB spots and ILLUSTRATIVE_RATES, quoted
as ATM DNS, RR and market-strangle BF at 25 and 10 delta in the pair's
market delta convention, rounded to 0.01 vol points (ATM to 0.05).

    PYTHONPATH=src python tools/gen_fx_illustrative_quotes.py
"""

from __future__ import annotations

import datetime as dt

from scipy.optimize import brentq

from derivx.fx.conventions import OPTION_TENORS, PAIRS, fx_dates, market_delta_type
from derivx.fx.examples import ILLUSTRATIVE_RATES
from derivx.fx.market_data import bundled_ecb_rates
from derivx.fx.pricing import FxMarket, atm_strike
from derivx.fx.smile import TenorQuote, _sabr_alpha, market_strangle, sabr_beta1_vol, smile_delta_strikes

TRADE = dt.date(2026, 10, 20)
# pair: (ATM vol at 3M in vol points, SABR rho, SABR nu at 1Y)
SHAPE = {
    "EURUSD": (7.5, -0.10, 0.55),
    "GBPUSD": (8.0, -0.20, 0.60),
    "AUDUSD": (9.5, -0.35, 0.65),
    "NZDUSD": (10.0, -0.35, 0.65),
    "USDJPY": (10.0, -0.30, 0.70),
    "USDCHF": (7.5, -0.05, 0.60),
    "USDCAD": (6.0, 0.15, 0.55),
    "EURGBP": (5.5, 0.10, 0.60),
    "EURJPY": (9.5, -0.35, 0.70),
    "EURCHF": (5.0, -0.30, 0.80),
}
ATM_TERM = {
    "ON": 0.92,
    "1W": 0.94,
    "2W": 0.95,
    "3W": 0.96,
    "1M": 0.97,
    "2M": 0.99,
    "3M": 1.0,
    "6M": 1.03,
    "9M": 1.05,
    "1Y": 1.06,
    "18M": 1.08,
    "2Y": 1.09,
}


def quotes(code: str) -> list:
    pair, ecb = PAIRS[code], bundled_ecb_rates()
    atm3m, rho, nu1y = SHAPE[code]
    rows = []
    for tenor in OPTION_TENORS:
        d = fx_dates(code, TRADE, tenor=tenor)
        m = FxMarket(ecb.cross(code), ILLUSTRATIVE_RATES[pair.quote], ILLUSTRATIVE_RATES[pair.base], d.tau_e, d.tau_d)
        dtype = market_delta_type(code, d)
        atm = round(atm3m * ATM_TERM[tenor] * 20) / 20 / 100
        nu = nu1y * max(d.tau_e, 1 / 52) ** -0.4
        k_atm = atm_strike(m, atm, "DNS", dtype.endswith("_pa"))
        alpha = _sabr_alpha(m.forward, d.tau_e, k_atm, atm, rho, nu)

        def smile(K):
            return sabr_beta1_vol(m.forward, K, d.tau_e, alpha, rho, nu)

        row = [tenor, round(atm * 100, 2)]
        for lv in (25, 10):
            _, vols = smile_delta_strikes(m, smile, [lv / 100, -lv / 100], [True, False], dtype, atm)
            rr = vols[0] - vols[1]

            def gap(bf):
                q = TenorQuote(tenor, atm, rr, bf) if lv == 25 else TenorQuote(tenor, atm, 0.0, 0.0, rr, bf)
                ms = market_strangle(m, q, lv, dtype)
                return ms.smile_value(m, smile) - ms.value

            row += [round(rr * 100, 2), round(brentq(gap, -0.01, 0.05) * 100, 2)]
        rows.append(tuple(row))
    return rows


if __name__ == "__main__":
    for code in PAIRS:
        print(f'    "{code}": (')
        for r in quotes(code):
            print(f'        ("{r[0]}", {r[1]:.2f}, {r[2]:.2f}, {r[3]:.2f}, {r[4]:.2f}, {r[5]:.2f}),')
        print("    ),")
