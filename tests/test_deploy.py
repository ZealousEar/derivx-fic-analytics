"""Deployment and dependency configuration (UIE-16, UIE-17).

Every way of running the app (Dockerfile, render.yaml, Procfile,
devcontainer, Streamlit Community Cloud) reads the committed
``.streamlit/config.toml`` and keeps Streamlit's XSRF and CORS protection
on; the Docker image copies only paths that exist and checks its health;
requirements.txt, the Docker lock file and pyproject.toml agree. The last
test starts a real Streamlit server on the committed config and checks that
an upload without an XSRF token is refused and that no foreign origin is
allowed.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: tomli ships with pytest there
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".streamlit" / "config.toml"
DEPLOY_FILES = ["Dockerfile", "Procfile", "render.yaml", ".devcontainer/devcontainer.json", ".streamlit/config.toml"]
DISABLING = [
    re.compile(r"enable(?:XsrfProtection|CORS)\s*[=: ]\s*[\"']?(?:false|0)\b", re.IGNORECASE),
    re.compile(r"STREAMLIT_SERVER_ENABLE_(?:XSRF_PROTECTION|CORS)", re.IGNORECASE),
]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _deps() -> Dict[str, Requirement]:
    reqs = [Requirement(r) for r in _pyproject()["project"]["dependencies"]]
    return {canonicalize_name(r.name): r for r in reqs}


def _dockerfile() -> List[Tuple[str, str]]:
    """(INSTRUCTION, arguments) with continuation lines joined."""

    text = re.sub(r"\\\n", " ", (ROOT / "Dockerfile").read_text())
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            word, _, rest = line.partition(" ")
            out.append((word.upper(), rest.strip()))
    return out


# -- Streamlit config (also run inside the Docker image) ---------------------


def test_committed_streamlit_config_keeps_protection_on_and_uses_known_options():
    from streamlit import config as st_config

    cfg = tomllib.loads(CONFIG.read_text())
    known = set(st_config.get_config_options())
    keys = [f"{section}.{name}" for section, opts in cfg.items() for name in opts]
    assert keys and set(keys) <= known, f"unknown Streamlit options: {set(keys) - known}"
    assert cfg["server"]["enableXsrfProtection"] is True
    assert cfg["server"]["enableCORS"] is True
    # the app paints itself black (ui/theme.py): a light base theme left a white header band and
    # #31333F labels on black (UIE-08)
    assert cfg["theme"]["base"] == "dark" and cfg["theme"]["backgroundColor"] == "#000000"
    for key in keys:
        assert not re.search(r"secret|password|token|apikey|api_key", key, re.IGNORECASE)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _request(url: str, method: str = "GET", headers=None, data: bytes = None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # local server: never via a proxy
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with opener.open(req, timeout=5) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read()


def test_a_server_on_the_committed_config_refuses_tokenless_uploads_and_foreign_origins(tmp_path):
    script = tmp_path / "upload_app.py"
    script.write_text('import streamlit as st\nst.file_uploader("chain")\n')
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("STREAMLIT_")}
    env.update(STREAMLIT_SERVER_HEADLESS="true", STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    cmd = [sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)]
    cmd += ["--server.address", "127.0.0.1", "--server.fileWatcherType", "none"]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                status, _, body = _request(f"{base}/_stcore/health")
                if status == 200:
                    break
            except OSError:
                pass
            assert proc.poll() is None, proc.stdout.read().decode(errors="replace")
            assert time.monotonic() < deadline, "Streamlit did not become healthy within 60 s"
            time.sleep(0.25)
        assert body.strip() == b"ok"

        status, headers, _ = _request(f"{base}/_stcore/health", headers={"Origin": "https://evil.example"})
        assert status == 200
        allowed = {k.lower(): v for k, v in headers.items()}.get("access-control-allow-origin")
        assert allowed in (None, ""), f"CORS is off: foreign origin allowed ({allowed})"

        boundary = "derivxboundary"
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="chain.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\nstrike,iv\r\n100,0.2\r\n--{boundary}--\r\n"
        ).encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        status, _, _ = _request(f"{base}/_stcore/upload_file/session/file", "PUT", headers, body)
        assert status == 403, f"an upload without an XSRF token got HTTP {status} (403 expected: XSRF is off)"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()


# -- deploy files --------------------------------------------------------------


def test_no_deploy_path_turns_off_xsrf_or_cors():
    paths = [ROOT / p for p in DEPLOY_FILES]
    paths += sorted((ROOT / ".github" / "workflows").glob("*.y*ml")) + sorted(ROOT.glob("*.sh"))
    for path in paths:
        text = path.read_text()
        for pattern in DISABLING:
            assert not pattern.search(text), f"{path.relative_to(ROOT)} disables XSRF/CORS: {pattern.pattern}"


def test_setup_sh_is_gone_and_the_procfile_runs_streamlit_on_the_host_port():
    assert not (ROOT / "setup.sh").exists()  # it wrote ~/.streamlit/config.toml with dash-only echo escapes
    lines = [line for line in (ROOT / "Procfile").read_text().splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith("web: streamlit run src/derivx/ui/streamlit_app.py ")
    assert "--server.port=$PORT" in lines[0] and "setup.sh" not in lines[0]


def _dockerignore() -> List[str]:
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _ignored(path: str, patterns: List[str]) -> bool:
    """Docker's rule: the last matching pattern wins; ``!`` re-includes."""

    ignored = False
    parts = path.rstrip("/").split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    for pattern in patterns:
        neg = pattern.startswith("!")
        pat = pattern[1:] if neg else pattern
        pat = pat.rstrip("/").removeprefix("**/")
        if any(fnmatch.fnmatch(p, pat) or fnmatch.fnmatch(p.split("/")[-1], pat) for p in prefixes):
            ignored = not neg
    return ignored


def test_dockerfile_copies_existing_paths_checks_health_and_runs_as_non_root():
    steps = _dockerfile()
    patterns = _dockerignore()
    copies = [args.split() for word, args in steps if word == "COPY"]
    assert copies
    for args in copies:
        sources = [a for a in args if not a.startswith("--")][:-1]
        for src in sources:
            assert (ROOT / src).exists(), f"COPY {src}: no such path in the repository"
            assert not _ignored(src, patterns), f"COPY {src} is excluded by .dockerignore"
    copied = {a for args in copies for a in args}
    for needed in ("requirements-lock.txt", ".streamlit/config.toml", "data/examples/", "src/"):
        assert needed in copied, f"the image needs {needed}"
    assert any(w == "HEALTHCHECK" and "/_stcore/health" in a for w, a in steps)
    users = [a for w, a in steps if w == "USER"]
    assert users and all(u not in ("root", "0") for u in users)
    env = " ".join(a for w, a in steps if w == "ENV")
    port = re.search(r"STREAMLIT_SERVER_PORT=(\d+)", env).group(1)
    assert [a for w, a in steps if w == "EXPOSE"] == [port]
    assert "STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=type" in env
    text = (ROOT / "Dockerfile").read_text()
    assert "~/.streamlit" not in text and "setup.py" not in text and "apt-get" not in text
    stages = [a.split()[-1] for w, a in steps if w == "FROM"]
    assert stages[-1] == "runtime" and "test" in stages  # `docker build .` gives the runtime image


def test_dockerignore_keeps_secrets_and_vcs_out_of_the_build_context():
    patterns = _dockerignore()
    for path in (".git", ".streamlit/secrets.toml", ".worktrees/x", ".venv", "src/derivx/__pycache__/a.pyc"):
        assert _ignored(path, patterns), path
    for path in (".streamlit/config.toml", "src/derivx/ui/streamlit_app.py", "data/examples/rates_quotes.csv"):
        assert not _ignored(path, patterns), path


def test_render_and_devcontainer_use_the_image_defaults():
    render = (ROOT / "render.yaml").read_text()
    assert "runtime: docker" in render and "healthCheckPath: /_stcore/health" in render
    port = re.search(r"STREAMLIT_SERVER_PORT=(\d+)", " ".join(a for w, a in _dockerfile() if w == "ENV")).group(1)
    assert re.search(rf"key: STREAMLIT_SERVER_PORT\s+value: {port}\b", render)
    raw = (ROOT / ".devcontainer" / "devcontainer.json").read_text()
    dev = json.loads(re.sub(r"^\s*//.*$", "", raw, flags=re.MULTILINE))  # JSONC: drop comment lines
    assert dev["postAttachCommand"]["server"] == "streamlit run src/derivx/ui/streamlit_app.py"
    assert ".[dev,notebooks]" in dev["updateContentCommand"]
    assert dev["portsAttributes"]["8501"]["onAutoForward"] == "openBrowser"


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_only_the_streamlit_config_is_committed_from_dot_streamlit():
    def ignored(path: str) -> bool:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT).returncode == 0

    assert not ignored(".streamlit/config.toml")
    assert ignored(".streamlit/secrets.toml") and ignored(".streamlit/credentials.toml")


# -- dependencies --------------------------------------------------------------


def test_requirements_txt_repeats_pyproject_and_installs_the_package():
    lines = [line.split("#")[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()]
    lines = [line for line in lines if line]
    assert "-e ." in lines  # the app reads data/examples next to src/
    reqs = {canonicalize_name(Requirement(r).name): Requirement(r) for r in lines if not r.startswith("-")}
    deps = _deps()
    assert set(reqs) == set(deps)
    for name, req in reqs.items():
        assert req.specifier == deps[name].specifier, f"{name}: {req.specifier} vs pyproject {deps[name].specifier}"
    assert "pytest" not in reqs and "quantlib" not in reqs  # test tools are extras


def test_every_runtime_dependency_is_bounded_and_streamlit_stays_on_1_x():
    deps = _deps()
    for name, req in deps.items():
        assert any(s.operator in (">=", "==", "~=") for s in req.specifier), f"{name} has no lower bound"
    assert str(deps["streamlit"].specifier) in ("<2,>=1.59", ">=1.59,<2")
    for name in ("numpy", "pandas", "scipy", "plotly", "streamlit", "yfinance"):
        assert any(s.operator == "<" for s in deps[name].specifier), f"{name} has no upper bound"
    project = _pyproject()["project"]
    assert project["requires-python"] == ">=3.10"
    pythons = {
        c.rsplit(" ", 1)[-1] for c in project["classifiers"] if c.startswith("Programming Language :: Python :: 3.")
    }
    assert pythons == {"3.10", "3.11", "3.12", "3.13"}


def test_the_docker_lock_pins_every_dependency_inside_its_range():
    text = (ROOT / "requirements-lock.txt").read_text()
    assert "uv pip compile pyproject.toml --python-version 3.11" in text  # how to regenerate it
    pins = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            name, sep, version = line.partition("==")
            assert sep and ";" not in version, f"unpinned lock line: {line}"
            pins[canonicalize_name(name)] = Version(version.strip())
    for name, req in _deps().items():
        assert name in pins, f"{name} missing from the lock"
        assert req.specifier.contains(pins[name], prereleases=False), f"{name}=={pins[name]} outside {req.specifier}"
    assert "quantlib" not in pins and "pytest" not in pins
    image_python = re.search(r"ARG PYTHON_VERSION=([\d.]+)", (ROOT / "Dockerfile").read_text()).group(1)
    assert f"--python-version {image_python}" in text


def test_dev_extra_pins_the_quantlib_oracle_and_the_linter():
    extras = _pyproject()["project"]["optional-dependencies"]
    dev = {canonicalize_name(Requirement(r).name): Requirement(r) for r in extras["dev"]}
    assert str(dev["quantlib"].specifier) == "==1.44"  # the version tests/golden was generated with
    assert "ruff" in dev and "pytest" in dev
    assert {canonicalize_name(Requirement(r).name) for r in extras["notebooks"]} >= {"matplotlib", "nbmake"}
    ruff = _pyproject()["tool"]["ruff"]
    assert ruff["line-length"] == 120 and _pyproject()["tool"]["black"]["line-length"] == 120
    assert set(ruff["lint"]["select"]) == {"E", "F", "W", "I"}
