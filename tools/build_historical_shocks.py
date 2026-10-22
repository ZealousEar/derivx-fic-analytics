"""Superseded by tools/build_rates_snapshot.py.

The historical scenario episodes are no longer a precomputed CSV: the scenario
engine measures them from the rates history sample in
data/examples/rates_history (Treasury, ICE swap, BoE OIS and ECB curves), so
every curve replays its own history. This entry point runs the new builder:

    python tools/build_historical_shocks.py [--skip-treasury]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_rates_snapshot import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
