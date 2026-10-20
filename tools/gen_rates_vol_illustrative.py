"""Print the ILLUSTRATIVE rates vol quotes of src/derivx/rates/vol_examples.py.

Not market data. Swaption normal vols by expiry x tenor x strike offset:
an ATM matrix ``level x f(expiry) x g(tenor)`` per currency, and a smile
from normal SABR (beta = 0, shift 3%) at a reference forward per currency,
with rho rising with tenor and nu falling like 1/sqrt(expiry) (capped at 1), alpha set so
the ATM quote is matched; rounded to 0.1bp. Flat cap normal vols by
maturity: an ATM hump ``level x h(maturity)``, and on a fixed strike grid
the ATM vol plus 8bp per (1%)^2 and minus 3bp per 1% of strike above the
reference rate; rounded to 0.1bp.

    PYTHONPATH=src python tools/gen_rates_vol_illustrative.py
"""

from __future__ import annotations

import numpy as np

from derivx.rates.conventions import parse_tenor
from derivx.rates.swaption_vol import GRID_EXPIRIES, GRID_OFFSETS_BP, GRID_TENORS
from derivx.vol.sabr import hagan_sabr_normal_vol, sabr_alpha_from_atm

LEVEL = {"USD": 100.0, "EUR": 74.0, "GBP": 95.0}  # ATM normal vol scale, bp
REFERENCE_FORWARD = {"USD": 0.036, "EUR": 0.024, "GBP": 0.038}
SHIFT = 0.03
EXPIRY_SHAPE = {"1M": 1.06, "3M": 1.08, "6M": 1.09, "1Y": 1.08, "2Y": 1.04, "3Y": 1.00, "5Y": 0.94, "7Y": 0.89,
                "10Y": 0.83}  # fmt: skip
TENOR_SHAPE = {"1Y": 1.00, "2Y": 1.03, "5Y": 1.02, "10Y": 0.97, "20Y": 0.90, "30Y": 0.86}
CAP_MATURITIES = ("1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y")
CAP_SHAPE = (0.92, 1.00, 1.04, 1.05, 1.04, 1.01, 0.96, 0.93, 0.89, 0.85, 0.82, 0.80)
CAP_STRIKES = {
    "USD": (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0),
    "EUR": (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    "GBP": (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.5),
}
CAP_REFERENCE = {"USD": 3.5, "EUR": 2.5, "GBP": 4.0}  # percent


def swaption_rows(ccy: str) -> list:
    rows = []
    F = REFERENCE_FORWARD[ccy]
    for e in GRID_EXPIRIES:
        T = parse_tenor(e).years
        for t in GRID_TENORS:
            L = parse_tenor(t).years
            atm = round(LEVEL[ccy] * EXPIRY_SHAPE[e] * TENOR_SHAPE[t], 1) * 1e-4
            rho = float(np.clip(-0.15 + 0.008 * L + 0.01 * T, -0.3, 0.3))
            nu = min(1.0, 0.20 + 0.25 / np.sqrt(T))
            alpha = sabr_alpha_from_atm(F, T, atm, 0.0, rho, nu, SHIFT, "normal")
            K = F + np.array(GRID_OFFSETS_BP) * 1e-4
            vols = hagan_sabr_normal_vol(F, K, T, alpha, 0.0, rho, nu, SHIFT) * 1e4
            vols[np.array(GRID_OFFSETS_BP) == 0.0] = atm * 1e4
            rows.append((e, t, tuple(round(float(v), 1) for v in vols)))
    return rows


def cap_rows(ccy: str) -> tuple:
    atm = tuple(round(LEVEL[ccy] * h, 1) for h in CAP_SHAPE)
    grid = []
    for m, a in zip(CAP_MATURITIES, atm):
        x = np.array(CAP_STRIKES[ccy]) - CAP_REFERENCE[ccy]
        grid.append((m, tuple(round(float(a + 8.0 * xi * xi - 3.0 * xi), 1) for xi in x)))
    return atm, grid


if __name__ == "__main__":
    print("ILLUSTRATIVE_SWAPTION_NORMAL_BP = {")
    for ccy in LEVEL:
        print(f'    "{ccy}": (')
        for e, t, vols in swaption_rows(ccy):
            print(f'        ("{e}", "{t}", ({", ".join(f"{v:.1f}" for v in vols)})),')
        print("    ),")
    print("}")
    print("ILLUSTRATIVE_CAP_ATM_NORMAL_BP = {")
    for ccy in LEVEL:
        atm, _ = cap_rows(ccy)
        print(f'    "{ccy}": ({", ".join(f"{v:.1f}" for v in atm)}),')
    print("}")
    print("ILLUSTRATIVE_CAP_STRIKE_NORMAL_BP = {")
    for ccy in LEVEL:
        _, grid = cap_rows(ccy)
        print(f'    "{ccy}": (')
        for m, vols in grid:
            print(f'        ("{m}", ({", ".join(f"{v:.1f}" for v in vols)})),')
        print("    ),")
    print("}")
