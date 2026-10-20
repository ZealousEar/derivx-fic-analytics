import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from derivx.vol.svi import svi_iv

APP = Path(__file__).resolve().parents[1] / "src" / "derivx" / "ui" / "streamlit_app.py"
EXPIRY = pd.Timestamp("2024-03-15")


def _chain(trade_date: dt.date, shift: float) -> pd.DataFrame:
    strikes = np.linspace(80, 120, 17)  # 13 inside the tab's 0.85-1.15 band: more than 5 SVI params
    forward = 100.0 + shift
    params = {"a": 0.02, "b": 0.3, "rho": -0.2, "m": 0.0, "s": 0.4}
    return pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "expiry": EXPIRY,
            "strike": strikes,
            "implied_vol": svi_iv(strikes, params, F=forward, T=1.0),  # params: total variance at T = 1
            "underlying_price": forward,
            "bid": 1.0,
            "ask": 1.2,
            "mid": 1.1,
            "volume": 50,
            "open_interest": 100,
            "option_type": np.where(strikes < forward, "put", "call"),
        }
    )


@pytest.mark.parametrize("n_days", [3, 4, 5, 6])
def test_walkforward_tab_runs_for_any_supported_day_count(n_days):
    start = dt.date(2024, 1, 2)
    days = {
        pd.Timestamp(start + dt.timedelta(days=i)): _chain(start + dt.timedelta(days=i), 0.5 * i)
        for i in range(n_days)
    }
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.session_state["walkforward_chains"] = days
    at.run()
    at.sidebar.radio[0].set_value("[ ANALYTICS ]").run()
    assert not at.exception

    sliders = [s for s in at.slider if s.label == "Calibration window (days)"]
    if n_days <= 5:  # only one window fits: shown as a caption, not a slider
        assert not sliders
    else:
        assert sliders[0].max == n_days - 2
        sliders[0].set_value(sliders[0].max).run()

    # The largest window offered must still leave a day to score against.
    next(b for b in at.button if b.label == "Run walk-forward test").click().run()
    assert not at.exception
    assert not at.session_state["walkforward_metrics"].empty
