import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture
def golden():
    """Load tests/golden/<name>.json (QuantLib reference values; regenerate
    with ``python tools/gen_golden.py``)."""

    def load(name: str):
        return json.loads((GOLDEN_DIR / f"{name}.json").read_text())["data"]

    return load
