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
import warnings
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


@pytest.fixture(scope="module")
def committed_config_server(tmp_path_factory):
    """A real Streamlit server started from the repository root, so it reads the committed
    .streamlit/config.toml (no STREAMLIT_* variables): yields ``(base_url, port)``."""

    script = tmp_path_factory.mktemp("app") / "upload_app.py"
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
        yield base, port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        proc.stdout.close()


def test_a_server_on_the_committed_config_refuses_tokenless_uploads_and_foreign_origins(committed_config_server):
    base, _ = committed_config_server
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


def _healthcheck_command() -> List[str]:
    """The image's HEALTHCHECK, an exec-form CMD array."""

    args = next(a for w, a in _dockerfile() if w == "HEALTHCHECK")
    return json.loads(args[args.index("CMD") + len("CMD") :].strip())


@pytest.mark.skipif(not (ROOT / "Dockerfile").exists(), reason="the image's test stage has no Dockerfile")
def test_the_image_healthcheck_passes_on_a_healthy_server_even_with_a_proxy_configured(committed_config_server):
    _, port = committed_config_server
    cmd = _healthcheck_command()
    assert cmd[0] == "python" and "/_stcore/health" in " ".join(cmd)
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
    dead_proxy = f"http://127.0.0.1:{_free_port()}"  # nothing listens there
    proxies = {"HTTP_PROXY": dead_proxy, "http_proxy": dead_proxy, "HTTPS_PROXY": dead_proxy, "NO_PROXY": ""}
    for extra in ({}, proxies):  # a host may pass its outbound proxy into the container
        run = subprocess.run(
            [sys.executable, *cmd[1:]], env={**env, **extra, "STREAMLIT_SERVER_PORT": str(port)}, timeout=30
        )
        assert run.returncode == 0, f"healthcheck failed on a healthy server (proxy settings: {extra})"
    # and it is not vacuous: no server on the port -> unhealthy
    run = subprocess.run(
        [sys.executable, *cmd[1:]], env={**env, "STREAMLIT_SERVER_PORT": str(_free_port())}, capture_output=True
    )
    assert run.returncode != 0


# -- deploy files --------------------------------------------------------------


def test_no_deploy_path_turns_off_xsrf_or_cors():
    paths = [ROOT / p for p in DEPLOY_FILES]
    paths += sorted((ROOT / ".github" / "workflows").glob("*.y*ml")) + sorted(ROOT.glob("*.sh"))
    for path in paths:
        text = path.read_text()
        for pattern in DISABLING:
            assert not pattern.search(text), f"{path.relative_to(ROOT)} disables XSRF/CORS: {pattern.pattern}"


def test_the_ci_docker_job_checks_xsrf_and_cors_on_the_running_container():
    text = (ROOT / ".github" / "workflows" / "ci.yaml").read_text()
    assert 'test "$code" = 403' in text and "/_stcore/upload_file/" in text
    assert "Origin: https://evil.example" in text
    # `! curl ... | grep` passes when curl itself fails (no pipefail; `!` is exempt from `bash -e`), so the
    # headers are captured first and a match fails the step explicitly (review finding)
    assert not re.search(r"^\s*!\s*curl", text, re.MULTILINE)
    assert "headers=$(curl -fsS" in text and "exit 1" in text


def _ci_jobs() -> Dict[str, str]:
    """Job id -> its text, for the jobs of .github/workflows/ci.yaml."""

    body = (ROOT / ".github" / "workflows" / "ci.yaml").read_text().split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2]))


def _dev_pins() -> Dict[str, str]:
    dev = _pyproject()["project"]["optional-dependencies"]["dev"]
    return {canonicalize_name(Requirement(r).name): str(Requirement(r).specifier) for r in dev}


def test_ci_type_checks_the_package_with_the_dev_extras_mypy_on_the_locked_versions():
    """UIE-18: mypy was in the dev extra with no config and never ran."""

    jobs = _ci_jobs()
    runs_mypy = [job for job, body in jobs.items() if re.search(r"^\s*run: mypy\s*$", body, re.MULTILINE)]
    assert runs_mypy, "no CI job runs mypy"
    body = jobs[runs_mypy[0]]
    pin = re.search(r'"mypy==([^"]+)"', body)
    assert pin and _dev_pins()["mypy"] == f"=={pin.group(1)}"  # CI and `pip install -e .[dev]` agree
    assert "-r requirements-lock.txt" in body  # the tested resolution, not whatever stubs are newest
    ruff = re.search(r'"ruff==([^"]+)"', jobs["lint"])
    assert ruff and _dev_pins()["ruff"] == f"=={ruff.group(1)}"
    cfg = _pyproject()["tool"]["mypy"]
    assert cfg["files"] == ["src/derivx"] and cfg["python_version"] == "3.10"
    # Lenient, not off: no module is skipped, and the codes that flag real defects stay on everywhere.
    overrides = cfg.get("overrides", [])
    assert "ignore_errors" not in cfg and all("ignore_errors" not in o for o in overrides)
    off = set(cfg["disable_error_code"]) | {c for o in overrides for c in o.get("disable_error_code", [])}
    on = {"call-arg", "name-defined", "no-redef", "var-annotated", "has-type", "override", "return", "import"}
    assert not off & on
    assert all(o["disable_error_code"] == ["attr-defined"] for o in overrides)  # the attr-defined baseline only


def test_every_ci_test_run_turns_deprecation_and_runtime_warnings_into_errors():
    """UIE-18: the suite passes with both, so CI keeps it that way (the
    notebooks job executes notebooks in a kernel, where -W does not reach)."""

    runs = [
        (job, line)
        for job, body in _ci_jobs().items()
        for line in body.splitlines()
        if re.search(r"-m pytest\b|^\s*(run:\s*)?pytest\b", line) and "--nbmake" not in line
    ]
    assert {job for job, _ in runs} >= {"test", "oldest"}
    runs += [("Dockerfile test stage", args) for word, args in _dockerfile() if word == "RUN" and "-m pytest" in args]
    assert len(runs) >= 3
    for where, line in runs:
        assert "-W error::DeprecationWarning" in line and "-W error::RuntimeWarning" in line, (where, line)


def test_setup_sh_is_gone_and_the_procfile_runs_streamlit_on_the_host_port():
    assert not (ROOT / "setup.sh").exists()  # it wrote ~/.streamlit/config.toml with dash-only echo escapes
    lines = [line for line in (ROOT / "Procfile").read_text().splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith("web: streamlit run src/derivx/ui/streamlit_app.py ")
    assert "--server.port=$PORT" in lines[0] and "setup.sh" not in lines[0]
    # a public host, like the Docker image and Render: viewers see an exception's type, not a traceback
    # with server paths (review finding: the Procfile alone showed full tracebacks)
    assert "--client.showErrorDetails=type" in lines[0]


def test_every_public_deploy_path_redacts_tracebacks():
    env = " ".join(a for w, a in _dockerfile() if w == "ENV")
    assert "STREAMLIT_CLIENT_SHOW_ERROR_DETAILS=type" in env
    assert re.search(r"key: STREAMLIT_CLIENT_SHOW_ERROR_DETAILS\s+value: type\b", (ROOT / "render.yaml").read_text())
    assert "--client.showErrorDetails=type" in (ROOT / "Procfile").read_text()
    # the committed config keeps Streamlit's default (full details) for local `streamlit run` and the devcontainer
    assert "showErrorDetails" not in tomllib.loads(CONFIG.read_text()).get("client", {})


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
    # Yahoo changes its endpoints and only new yfinance releases follow: the range must admit the
    # maintained 1.x line (1.0 shipped after 0.2.66; a <0.3 cap froze the app on a 2025 release)
    assert deps["yfinance"].specifier.contains("1.7.0") and deps["yfinance"].specifier.contains("0.2.66")
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


def test_package_metadata_uses_an_spdx_license_that_current_setuptools_accepts():
    """setuptools deprecates the ``license = {text = ...}`` table and License classifiers and stops
    supporting them after 2027-02-18; every deploy path builds this package (``-e .``) with the newest
    setuptools, so the old metadata would eventually break all of them (review finding)."""

    pyproject = _pyproject()
    project = pyproject["project"]
    assert project["license"] == "MIT" and project["license-files"] == ["LICENSE"]
    assert (ROOT / "LICENSE").is_file()
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
    backend = {canonicalize_name(Requirement(r).name): Requirement(r) for r in pyproject["build-system"]["requires"]}
    assert not backend["setuptools"].specifier.contains("76.1.0") and backend["setuptools"].specifier.contains("77.0.3")


def test_the_installed_setuptools_reads_the_package_metadata_without_deprecations():
    setuptools = pytest.importorskip("setuptools")
    if Version(setuptools.__version__) < Version("77"):
        pytest.skip(f"setuptools {setuptools.__version__} predates PEP 639 license expressions")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from setuptools.config.pyprojecttoml import apply_configuration
        from setuptools.dist import Distribution

        dist = apply_configuration(Distribution(), ROOT / "pyproject.toml")
    deprecations = [str(w.message).splitlines()[0] for w in caught if "Setuptools" in type(w.message).__name__]
    assert deprecations == []
    assert dist.metadata.license_expression == "MIT"


def test_dev_extra_pins_the_quantlib_oracle_and_the_linter():
    extras = _pyproject()["project"]["optional-dependencies"]
    dev = {canonicalize_name(Requirement(r).name): Requirement(r) for r in extras["dev"]}
    assert str(dev["quantlib"].specifier) == "==1.44"  # the version tests/golden was generated with
    assert "ruff" in dev and "pytest" in dev
    assert {canonicalize_name(Requirement(r).name) for r in extras["notebooks"]} >= {"matplotlib", "nbmake"}
    ruff = _pyproject()["tool"]["ruff"]
    assert ruff["line-length"] == 120 and _pyproject()["tool"]["black"]["line-length"] == 120
    assert set(ruff["lint"]["select"]) == {"E", "F", "W", "I"}
