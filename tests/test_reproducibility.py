"""G14 (1): no unseeded randomness anywhere in the library or the UI."""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "derivx"
# np.random.<anything> except default_rng(<seed>) / Generator / SeedSequence, the stdlib random
# module, and default_rng() with no seed.
PATTERNS = {
    "legacy numpy global RNG": re.compile(r"\bnp\.random\.(?!default_rng\(\s*[^)\s]|Generator\b|SeedSequence\b)"),
    "numpy.random import": re.compile(
        r"^\s*(from\s+numpy\s+import\s+random|from\s+numpy\.random\s+import|import\s+numpy\.random\b)"
    ),
    "stdlib random": re.compile(r"^\s*(import\s+random\b|from\s+random\s+import)"),
    "unseeded default_rng": re.compile(r"default_rng\(\s*((seed\s*=\s*)?None\s*)?\)"),
}


def test_no_unseeded_randomness_in_src():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        for no, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            for name, pattern in PATTERNS.items():
                if pattern.search(code):
                    offenders.append(f"{path.relative_to(SRC.parent)}:{no}: {name}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_the_scanner_catches_the_old_smile_noise():
    old = "ivs = np.full_like(Ks, sigma) + 0.01 * np.random.randn(len(Ks))"
    assert PATTERNS["legacy numpy global RNG"].search(old)
    assert not PATTERNS["legacy numpy global RNG"].search("rng = np.random.default_rng(20261020)")
    assert PATTERNS["unseeded default_rng"].search("rng = np.random.default_rng()")
    assert PATTERNS["unseeded default_rng"].search("rng = np.random.default_rng(None)")  # OS entropy too
    assert PATTERNS["unseeded default_rng"].search("rng = np.random.default_rng(seed=None)")
    assert not PATTERNS["unseeded default_rng"].search("rng = np.random.default_rng(seed)")
    assert PATTERNS["numpy.random import"].search("import numpy.random as npr")
