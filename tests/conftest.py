import json
import urllib.error
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def golden():
    """Load tests/golden/<name>.json (QuantLib reference values; regenerate
    with ``python tools/gen_golden.py``)."""

    def load(name: str):
        return json.loads((GOLDEN_DIR / f"{name}.json").read_text())["data"]

    return load


@pytest.fixture(autouse=True)
def no_network(monkeypatch, tmp_path_factory):
    """Tests never touch the network or the developer's download cache:
    fetchers run offline against an empty temporary cache unless a test
    serves recorded fixtures through ``rates_sources._urlopen`` itself."""

    from derivx.data import rates_sources

    def blocked(request, timeout):
        raise urllib.error.URLError(f"network disabled in tests ({request.full_url})")

    monkeypatch.setattr(rates_sources, "_urlopen", blocked)
    monkeypatch.setenv("DERIVX_OFFLINE", "1")
    monkeypatch.setenv("DERIVX_CACHE_DIR", str(tmp_path_factory.mktemp("derivx-cache")))
