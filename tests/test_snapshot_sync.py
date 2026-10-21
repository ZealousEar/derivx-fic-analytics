"""``derivx snapshots sync`` / :func:`derivx.data.snapshots.sync`: the
published store mirrored from an HTTP server (a local fake of
raw.githubusercontent.com serving a fixture tree) into the cache, with every
download checked against the index; incremental re-syncs, offline mode,
fallbacks when the server is down, and reading a local checkout. The only
network used is a server on 127.0.0.1 started by the test."""

from __future__ import annotations

import functools
import http.server
import threading
import urllib.request

import pytest
from _snapshot_store import real_store

from derivx import cli
from derivx.data import rates_sources as rs
from derivx.data import snapshots as snap

SESSIONS = ["2026-10-20", "2026-10-21", "2026-10-23"]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep the test output clean
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A fixture store served over HTTP on 127.0.0.1 (the layout of the
    data branch at raw.githubusercontent.com/<repo>/<branch>/)."""

    tree = tmp_path / "published"
    real_store(tree, SESSIONS)
    handler = functools.partial(_Quiet, directory=str(tree))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never the outbound proxy

    def local_only(request, timeout):
        if not request.full_url.startswith(f"http://127.0.0.1:{httpd.server_port}/"):
            raise AssertionError(f"unexpected network access: {request.full_url}")
        return direct.open(request, timeout=timeout)

    monkeypatch.setattr(rs, "_urlopen", local_only)
    monkeypatch.delenv("DERIVX_OFFLINE", raising=False)
    requests = []
    original = _Quiet.do_GET

    def counting(self):
        requests.append(self.path)
        return original(self)

    monkeypatch.setattr(_Quiet, "do_GET", counting)
    yield {"url": f"http://127.0.0.1:{httpd.server_port}", "tree": tree, "requests": requests, "httpd": httpd}
    httpd.shutdown()
    httpd.server_close()


def _sync(server, tmp_path, **kw):
    return snap.sync("owner/repo", "data-snapshots", base_url=server["url"], cache_root=tmp_path / "cache",
                     retries=0, **kw)  # fmt: skip


def test_sync_mirrors_the_store_and_checks_every_file(server, tmp_path):
    result = _sync(server, tmp_path)
    assert (result.downloaded, result.cached, result.failed, result.offline) == (3, 0, [], False)
    root = tmp_path / "cache" / "snapshots" / "owner__repo" / "data-snapshots"
    assert result.store.root == root and result.store.verify() == []
    assert [d.isoformat() for d in result.store.sessions("SPY")] == SESSIONS
    cov = result.store.coverage("SPY")
    assert [d.isoformat() for d in cov.gaps] == ["2026-10-22"]
    ds = result.store.load("SPY")
    assert len(ds.chains) == 3 and "owner/repo@data-snapshots" in ds.source
    assert result.synced_at is not None and (root / "sync.json").exists()
    # a second sync downloads only the index
    server["requests"].clear()
    again = _sync(server, tmp_path)
    assert (again.downloaded, again.cached) == (0, 3) and server["requests"] == ["/options/index.csv"]


def test_a_new_session_on_the_branch_is_the_only_download(server, tmp_path):
    _sync(server, tmp_path)
    published = snap.SnapshotStore(server["tree"])
    published.merge_from(real_store(tmp_path / "staging", ["2026-10-26"]))
    server["requests"].clear()
    result = _sync(server, tmp_path)
    assert (result.downloaded, result.cached) == (1, 3)
    assert sorted(server["requests"]) == ["/options/SPY/2026/SPY_2026-10-26.csv.gz",
                                          "/options/SPY/2026/SPY_2026-10-26.json", "/options/index.csv"]  # fmt: skip


def test_a_download_that_does_not_match_the_index_is_not_kept(server, tmp_path):
    bad = server["tree"] / "options" / "SPY" / "2026" / "SPY_2026-10-21.csv.gz"
    bad.write_bytes(bad.read_bytes()[:-20])
    result = _sync(server, tmp_path)
    assert result.downloaded == 2 and len(result.failed) == 1
    assert "SPY_2026-10-21.csv.gz: the download's SHA-256 differs from the index" in result.failed[0]
    assert [d.isoformat() for d in result.store.sessions("SPY")] == ["2026-10-20", "2026-10-23"]
    assert not result.store.file(snap.relative_path("SPY", "2026-10-21")).exists()
    assert result.store.verify() == []


def test_offline_reads_the_cache_and_a_server_outage_falls_back_to_it(server, tmp_path):
    with pytest.raises(snap.SnapshotError, match="has not been synced"):
        _sync(server, tmp_path, offline=True)
    _sync(server, tmp_path)
    server["requests"].clear()
    offline = _sync(server, tmp_path, offline=True)
    assert offline.offline and server["requests"] == [] and len(offline.store.sessions("SPY")) == 3
    assert "offline: the cached copy of owner/repo@data-snapshots synced" in offline.notices[0]
    (server["tree"] / "options" / "index.csv").unlink()  # the branch is gone or the host serves 404
    down = _sync(server, tmp_path)
    assert down.offline and "index download failed (HTTP 404" in down.notices[0]
    assert len(down.store.sessions("SPY")) == 3


def test_without_a_cache_an_unreachable_host_is_a_clear_error(tmp_path):
    # the hermetic test setup blocks every request: the default URL is raw.githubusercontent.com
    with pytest.raises(snap.SnapshotError, match=r"raw\.githubusercontent\.com unreachable.*nothing is cached"):
        snap.sync("owner/repo", offline=False, cache_root=tmp_path, retries=0)
    with pytest.raises(ValueError, match="OWNER/NAME"):
        snap.sync("not a repo", cache_root=tmp_path)


def test_repo_and_branch_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(snap.REPO_ENV, "someone/fork")
    monkeypatch.setenv(snap.BRANCH_ENV, "my-data")
    assert (snap.default_repo(), snap.default_branch()) == ("someone/fork", "my-data")
    assert snap.sync_root(snap.default_repo(), snap.default_branch(), tmp_path) == (
        tmp_path / "snapshots" / "someone__fork" / "my-data"
    )


def test_sync_of_one_ticker_keeps_the_others(server, tmp_path):
    _sync(server, tmp_path)
    result = _sync(server, tmp_path, tickers=["QQQ"])  # not on the branch
    assert result.downloaded == 0 and len(result.store.sessions("SPY")) == 3


def test_cli_sync_list_and_local_checkout_round_trip(server, tmp_path, capsys):
    common = ["--repo", "owner/repo", "--cache", str(tmp_path / "cache")]
    assert cli.main(["snapshots", "sync", *common, "--base-url", server["url"]]) == 0
    out = capsys.readouterr().out
    assert "owner/repo@data-snapshots" in out and "3 downloaded, 0 already cached" in out
    assert "SPY: 3 session(s) 2026-10-20 .. 2026-10-23; 1 missing trading day(s): 2026-10-22" in out
    server["requests"].clear()
    assert cli.main(["snapshots", "list", *common, "--sessions"]) == 0  # never downloads
    out = capsys.readouterr().out
    assert server["requests"] == [] and "SPY: 2026-10-20 2026-10-21 2026-10-23" in out
    assert cli.main(["snapshots", "sync", "--local", str(server["tree"])]) == 0
    assert "local checkout" in capsys.readouterr().out
    assert cli.main(["snapshots", "list", "--repo", "nobody/nothing", "--cache", str(tmp_path / "c2")]) == 1
    assert "has not been synced" in capsys.readouterr().err
