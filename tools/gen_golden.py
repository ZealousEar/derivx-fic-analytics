"""Regenerate the QuantLib golden fixtures in tests/golden/.

Dev only (needs ``pip install QuantLib``); the test suite reads the committed
JSON and does not need QuantLib. Each module in tools/golden/ with a
``generate() -> dict`` function writes tests/golden/<module>.json:

    python tools/gen_golden.py            # every fixture
    python tools/gen_golden.py swaps      # one fixture
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "tests" / "golden"


def main(names: list[str]) -> None:
    import golden
    import QuantLib as ql

    available = sorted(m.name for m in pkgutil.iter_modules(golden.__path__) if m.name != "common")
    for name in names or available:
        if name not in available:
            raise SystemExit(f"unknown fixture {name!r}; available: {available}")
        data = importlib.import_module(f"golden.{name}").generate()
        payload = {"generator": f"tools/golden/{name}.py", "quantlib": ql.__version__, "data": data}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
        print(f"wrote tests/golden/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
