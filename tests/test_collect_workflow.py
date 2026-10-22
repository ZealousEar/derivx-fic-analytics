"""``.github/workflows/collect-data.yml``: it parses, runs on a weekday
schedule after the options close in both EDT and EST and by hand (tickers,
dry run), has write access to contents only (plus re-enabling its own
schedule), one run at a time, and never splices inputs into its scripts.
Its branch and publish scripts are run here as they are, with git, against
a local bare repository: the first run that saves a session creates the
orphan ``data-snapshots`` branch, later runs append, a run with nothing new
commits nothing, a rejected push is rebuilt on the branch's new tip. No
network (the staged snapshots are recorded fakes)."""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

import pytest
import yaml
from _snapshot_store import real_store

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "collect-data.yml"


def _workflow() -> Dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _steps() -> Dict[str, Dict]:
    (job,) = _workflow()["jobs"].values()
    return {step.get("name", step.get("uses", "")): step for step in job["steps"]}


def _step(prefix: str) -> Dict:
    matches = [s for name, s in _steps().items() if name.startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


def test_triggers_permissions_and_concurrency():
    wf = _workflow()
    on = wf.get("on", wf.get(True))  # YAML 1.1 reads a bare `on` key as True
    (schedule,) = on["schedule"]
    assert schedule["cron"] == "35 21 * * 1-5"
    minute, hour, *_ = schedule["cron"].split()
    utc = dt.time(int(hour), int(minute))
    for day in (dt.date(2026, 7, 15), dt.date(2026, 12, 15)):  # EDT and EST
        run = dt.datetime.combine(day, utc, dt.timezone.utc).astimezone(ZoneInfo("America/New_York"))
        assert dt.time(16, 25) <= run.time() <= dt.time(18, 0), run  # after the 16:15 close + 10 min, same evening
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["tickers"]["type"] == "string" and inputs["tickers"]["default"] == "SPY"
    assert inputs["dry_run"]["type"] == "boolean" and inputs["dry_run"]["default"] is False
    assert wf["permissions"] == {"contents": "write"}
    assert wf["concurrency"] == {"group": "collect-data", "cancel-in-progress": False}
    (job,) = wf["jobs"].values()
    assert job["permissions"] == {"contents": "write", "actions": "write"}
    assert job["timeout-minutes"] <= 30


def test_steps_check_out_install_collect_publish_and_fail_visibly():
    checkout = _step("Check out the code")
    assert checkout["uses"] == "actions/checkout@v4"
    assert checkout["with"] == {"ref": "${{ github.event.repository.default_branch }}", "path": "code"}
    install = _step("Install the locked")["run"]
    assert "pip install -r requirements-lock.txt" in install and "--no-deps -e ." in install
    branch = _step("Check out data-snapshots")["run"]
    assert "ls-remote --exit-code --heads origin" in branch and "--orphan" in branch
    collect = _step("Collect")
    assert (
        "python -m derivx snapshots collect" in collect["run"] and '--summary "$GITHUB_STEP_SUMMARY"' in collect["run"]
    )
    assert "--staging ../staging --store ../store" in collect["run"] and 'echo "code=$code"' in collect["run"]
    publish = _step("Copy the new files")
    assert publish["if"] == "env.DRY_RUN != 'true'"
    assert (
        "snapshots merge ../staging ../store" in publish["run"]
        and "snapshots index --store ../store --check" in publish["run"]
    )
    assert 'user.name="github-actions[bot]"' in publish["run"] and "HEAD:refs/heads/$DATA_BRANCH" in publish["run"]
    fail = _step("Fail when a trading day produced nothing")
    assert fail["if"] == "steps.collect.outputs.code == '1'" and "exit 1" in fail["run"]
    keep = _step("Keep the schedule enabled")
    assert (
        keep["if"] == "always() && github.event_name == 'schedule'"
        and "gh workflow enable collect-data.yml" in keep["run"]
    )
    env = _workflow()["jobs"]["collect"]["env"]
    assert env["DATA_BRANCH"] == "data-snapshots"
    assert env["DRY_RUN"] == "${{ github.event_name == 'workflow_dispatch' && inputs.dry_run && 'true' || 'false' }}"


def test_inputs_never_reach_a_script_unquoted():
    for name, step in _steps().items():
        assert "${{" not in step.get("run", ""), f"{name}: an expression is spliced into a shell script"
    assert re.search(r"TICKERS: \$\{\{ inputs\.tickers \|\| 'SPY' \}\}", WORKFLOW.read_text())


# -- the branch and publish scripts, run against a local bare repository -----------------------


def _git(*args, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _bash(script: str, cwd: Path, env: Dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-eo", "pipefail", "-c", script], cwd=cwd, env=env, capture_output=True, text=True)


@pytest.fixture
def workspace(tmp_path):
    if shutil.which("git") is None or shutil.which("bash") is None:  # pragma: no cover - CI runners have both
        pytest.fail("git and bash are needed to run the workflow's scripts")
    origin = tmp_path / "origin.git"
    _git("init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    seed = tmp_path / "seed"
    _git("init", "-q", "-b", "main", str(seed), cwd=tmp_path)
    (seed / "README.md").write_text("code\n")
    _git("add", "README.md", cwd=seed)
    _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "code", cwd=seed)
    _git("push", "-q", str(origin), "main", cwd=seed)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "python"  # `python` in the scripts: this interpreter (a symlink would leave its venv)
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    wrapper.chmod(0o755)
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "PYTHONPATH": str(ROOT / "src"),
        "DATA_BRANCH": "data-snapshots",
        "GITHUB_STEP_SUMMARY": str(summary),
        "GITHUB_RUN_ID": "1",
        "GITHUB_SHA": "0" * 40,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(tmp_path),
    }
    return tmp_path, origin, env, summary


def _run(tmp_path: Path, origin: Path, env: Dict[str, str], run: str, sessions) -> subprocess.CompletedProcess:
    """One workflow run in a fresh workspace: clone the code, check out the
    data branch, stage ``sessions`` (what the collect step would write) and
    publish."""

    ws = tmp_path / run
    _git("clone", "-q", str(origin), str(ws / "code"), cwd=tmp_path)
    branch = _bash(_step("Check out data-snapshots")["run"], ws / "code", env)
    assert branch.returncode == 0, branch.stderr
    if sessions:
        real_store(ws / "staging", sessions)
    return _bash(_step("Copy the new files")["run"], ws / "store", env)


def test_publish_creates_the_orphan_branch_then_appends_and_skips_empty_runs(workspace):
    tmp_path, origin, env, summary = workspace
    first = _run(tmp_path, origin, env, "run1", ["2026-10-20"])
    assert first.returncode == 0, first.stdout + first.stderr
    log = _git("log", "--format=%an|%s|%P", "data-snapshots", cwd=origin).splitlines()
    assert log == ["github-actions[bot]|data: option chains SPY_2026-10-20|"]  # a root commit: an orphan branch
    files = set(_git("ls-tree", "-r", "--name-only", "data-snapshots", cwd=origin).split())
    assert files == {"README.md", "options/index.csv", "options/SPY/2026/SPY_2026-10-20.csv.gz",
                     "options/SPY/2026/SPY_2026-10-20.json"}  # fmt: skip
    assert "main" in _git("branch", "--list", "main", cwd=origin)  # the code branch is untouched
    second = _run(tmp_path, origin, env, "run2", ["2026-10-20", "2026-10-21"])  # 10-20 is kept, 10-21 added
    assert second.returncode == 0, second.stdout + second.stderr
    log = _git("log", "--format=%s", "data-snapshots", cwd=origin).splitlines()
    assert log == ["data: option chains SPY_2026-10-21", "data: option chains SPY_2026-10-20"]
    first_file = _git("rev-parse", "data-snapshots~1:options/SPY/2026/SPY_2026-10-20.csv.gz", cwd=origin)
    assert _git("rev-parse", "data-snapshots:options/SPY/2026/SPY_2026-10-20.csv.gz", cwd=origin) == first_file
    index = _git("show", "data-snapshots:options/index.csv", cwd=origin).splitlines()
    assert [row.split(",")[0] for row in index[1:]] == ["2026-10-20", "2026-10-21"]
    third = _run(tmp_path, origin, env, "run3", [])  # a holiday: nothing staged
    assert third.returncode == 0 and "Nothing new to commit" in third.stdout
    assert len(_git("log", "--format=%s", "data-snapshots", cwd=origin).splitlines()) == 2
    assert "Pushed to data-snapshots: SPY_2026-10-21" in summary.read_text()


def test_a_rejected_push_is_rebuilt_on_the_new_tip(workspace):
    tmp_path, origin, env, summary = workspace
    assert _run(tmp_path, origin, env, "run1", ["2026-10-20"]).returncode == 0
    ws = tmp_path / "run2"
    _git("clone", "-q", str(origin), str(ws / "code"), cwd=tmp_path)
    assert _bash(_step("Check out data-snapshots")["run"], ws / "code", env).returncode == 0
    real_store(ws / "staging", ["2026-10-20", "2026-10-21"])
    # after this run checked the branch out, someone else pushes to it: the run's first push is rejected
    other = tmp_path / "other"
    _git("clone", "-q", "-b", "data-snapshots", str(origin), str(other), cwd=tmp_path)
    (other / "README.md").write_text("notes on the data branch\n")
    _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-am", "docs: notes", cwd=other)
    _git("push", "-q", "origin", "data-snapshots", cwd=other)
    result = _bash(_step("Copy the new files")["run"], ws / "store", env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "push rejected (attempt 1)" in result.stdout
    # the data commit is rebuilt on top of the other push: nothing of either is lost, nothing is mislabelled
    log = _git("log", "--format=%an|%s", "data-snapshots", cwd=origin).splitlines()
    assert log == ["github-actions[bot]|data: option chains SPY_2026-10-21", "t|docs: notes",
                   "github-actions[bot]|data: option chains SPY_2026-10-20"]  # fmt: skip
    assert _git("show", "data-snapshots:README.md", cwd=origin) == "notes on the data branch\n"
    index = _git("show", "data-snapshots:options/index.csv", cwd=origin).splitlines()
    assert [row.split(",")[0] for row in index[1:]] == ["2026-10-20", "2026-10-21"]
    assert "Pushed to data-snapshots: SPY_2026-10-21" in summary.read_text()


def test_a_run_with_nothing_due_on_a_new_repository_creates_no_branch(workspace):
    tmp_path, origin, env, _ = workspace
    run = _run(tmp_path, origin, env, "run1", [])
    assert run.returncode == 0 and "Nothing new to commit" in run.stdout
    assert _git("branch", "--list", "data-snapshots", cwd=origin) == ""


def test_a_corrupt_staged_file_fails_the_publish_and_commits_nothing(workspace):
    tmp_path, origin, env, _ = workspace
    ws = tmp_path / "bad"
    _git("clone", "-q", str(origin), str(ws / "code"), cwd=tmp_path)
    assert _bash(_step("Check out data-snapshots")["run"], ws / "code", env).returncode == 0
    store = real_store(ws / "staging", ["2026-10-20"])
    path = store.root / "options" / "SPY" / "2026" / "SPY_2026-10-20.csv.gz"
    path.write_bytes(path.read_bytes()[:-5])
    result = _bash(_step("Copy the new files")["run"], ws / "store", env)
    assert result.returncode == 1 and "could not be merged" in result.stdout
    assert _git("branch", "--list", "data-snapshots", cwd=origin) == ""
