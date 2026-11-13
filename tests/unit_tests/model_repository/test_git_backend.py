"""Tests for the Git object-store adapter."""

import io
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryFile
from types import SimpleNamespace

import pytest

from simtools.model_repository import git_backend
from simtools.model_repository.git_backend import (
    GitCommandObjectStore,
    GitNotInstalledError,
    run_git,
    split_nul,
)

BLOB = b"x" * 100000


@pytest.fixture
def repository(tmp_test_directory, make_git_repository):
    root = Path(tmp_test_directory) / "git-backend-repository"
    root.mkdir()
    return make_git_repository(
        root, {"models/file name.json": b"{}", "a.json": BLOB, "models/sub/b.json": b"[]"}
    )


@pytest.fixture
def store(repository):
    root, _ = repository
    object_store = GitCommandObjectStore(root)
    yield object_store
    object_store.close()


def test_git_object_store_reads_repository(repository, store):
    root, commit = repository
    subprocess.run(["git", "-C", str(root), "tag", "v1"], check=True)
    bare = root.parent / "model.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(root), str(bare)], check=True)
    bare_store = GitCommandObjectStore(bare)
    assert bare_store.resolve_revision("v1") == commit
    assert bare_store.iter_files(commit, "models") == [
        "models/file name.json",
        "models/sub/b.json",
    ]
    bare_store.close()

    assert store.iter_files(commit, ".") == [
        "a.json",
        "models/file name.json",
        "models/sub/b.json",
    ]
    assert store.read_blob(commit, "models/file name.json") == b"{}"
    with store.open_blob(commit, "models/sub/b.json") as stream:
        assert stream.read() == b"[]"
    assert store.read_blob(commit, "a.json") == BLOB


def test_git_object_store_reports_invalid_requests(repository, store):
    _, commit = repository
    with pytest.raises(ValueError, match="Git revision 'missing' not found"):
        store.resolve_revision("missing")
    with pytest.raises(ValueError, match="Git revision '--bad' not found"):
        store.resolve_revision("--bad")
    with pytest.raises(FileNotFoundError, match="Git path 'missing'"):
        store.read_blob(commit, "missing")
    with pytest.raises(FileNotFoundError, match="Git path"):
        store.read_blob("0" * 40, "a.json")
    with pytest.raises(FileNotFoundError, match="Git path"):
        store.read_blob(commit, "bad\nname")
    with pytest.raises(FileNotFoundError, match="Git path"):
        store.read_blob(commit, "bad\0name")
    with pytest.raises(ValueError, match="not a file"):
        store.read_blob(commit, "models")
    with pytest.raises(ValueError, match="not a directory"):
        store.iter_files(commit, "a.json")
    with pytest.raises(FileNotFoundError, match="Git path"):
        store.iter_files(commit, "missing")


def test_git_object_store_rejects_invalid_repository(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GitCommandObjectStore(tmp_test_directory / "missing")
    with pytest.raises(ValueError, match="Not a readable Git model repository"):
        GitCommandObjectStore(tmp_test_directory)


def test_git_executable_is_required(monkeypatch, repository):
    root, commit = repository
    object_store = GitCommandObjectStore(root)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(GitNotInstalledError, match="Git is required"):
        object_store.read_blob(commit, "a.json")
    assert object_store._process is None
    with pytest.raises(GitNotInstalledError, match="Git is required"):
        run_git(root, "status")


def test_git_object_store_reuses_and_restarts_process(repository, store):
    _, commit = repository
    assert store.read_blob(commit, "a.json") == BLOB
    process = store._process
    assert store.read_blob(commit, "a.json") == BLOB
    assert store._process is process

    process.kill()
    process.wait()
    assert store.read_blob(commit, "a.json") == BLOB
    assert store._process is not process

    store.close()
    assert store._process is None
    assert store.read_blob(commit, "a.json") == BLOB


def test_git_object_store_reports_repeated_process_failure(repository, store, monkeypatch):
    _, commit = repository

    def _fail(_):
        raise git_backend._GitProtocolError("broken")

    monkeypatch.setattr(store, "_exchange", _fail)
    with pytest.raises(RuntimeError, match=r"Git process failed.*broken"):
        store.read_blob(commit, "a.json")


@pytest.mark.parametrize(
    "response", [b"", b"garbage\n", b"sha blob notanumber\n", b"sha blob 10\nab"]
)
def test_git_object_store_rejects_malformed_responses(store, response):
    store._process = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(response))
    with pytest.raises(git_backend._GitProtocolError):
        store._exchange("spec")
    store._process = None


def test_git_object_store_includes_git_error_output(repository, store):
    _, commit = repository
    store.read_blob(commit, "a.json")
    store._stderr.write(b"fatal: disk on fire\n")
    assert "disk on fire" in git_backend._tail(store._stderr)
    store._stderr.close()
    assert not git_backend._tail(store._stderr)


def test_git_object_store_forked_child_starts_own_process(repository, store, monkeypatch):
    _, commit = repository
    store.read_blob(commit, "a.json")
    parent_process, parent_stderr = store._process, store._stderr
    monkeypatch.setattr(store, "_owner_pid", -1)
    assert store.read_blob(commit, "a.json") == BLOB
    assert store._process is not parent_process
    assert parent_process.poll() is None
    parent_process.stdin.close()
    parent_process.stdout.close()
    parent_process.wait()
    parent_stderr.close()


def test_terminate_kills_unresponsive_process(monkeypatch):
    process = subprocess.Popen(["sleep", "30"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stderr = TemporaryFile()
    monkeypatch.setattr(git_backend, "_STOP_TIMEOUT_SECONDS", 0.01)
    git_backend._terminate(process, stderr)
    assert process.poll() is not None
    assert stderr.closed


def test_iter_files_reports_listing_failure(repository, store, monkeypatch):
    _, commit = repository

    def _fail(*_):
        raise subprocess.CalledProcessError(128, "git", stderr=b"fatal: broken tree")

    monkeypatch.setattr(git_backend, "run_git", _fail)
    with pytest.raises(RuntimeError, match=r"Listing Git path.*broken tree"):
        store.iter_files(commit, "models")


def test_split_nul_keeps_non_utf8_names():
    assert split_nul(b"a\0b c\0\0") == ["a", "b c"]
    assert split_nul(b"\xff\0") == ["\udcff"]


def test_start_failure_releases_resources(repository, store, monkeypatch):
    _, commit = repository

    def _fail(*_, **__):
        raise PermissionError("cannot execute git")

    monkeypatch.setattr(subprocess, "Popen", _fail)
    with pytest.raises(PermissionError, match="cannot execute git"):
        store.read_blob(commit, "a.json")
    assert store._process is None


def test_iter_files_skips_non_blob_entries(tmp_test_directory, make_git_repository):
    root = Path(tmp_test_directory) / "git-backend-gitlink"
    root.mkdir()
    root, _ = make_git_repository(root, {"models/a.json": b"{}"})
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{'1' * 40},models/submodule",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@e.invalid",
            "commit",
            "-qm",
            "gitlink",
        ],
        check=True,
    )
    object_store = GitCommandObjectStore(root)
    commit = object_store.resolve_revision("HEAD")
    assert object_store.iter_files(commit, "models") == ["models/a.json"]
    with pytest.raises(ValueError, match="not a file"):
        object_store.read_blob(commit, "models/submodule")
    object_store.close()


def test_open_blob_streams_in_chunks(repository, store):
    _, commit = repository
    with store.open_blob(commit, "a.json") as stream:
        assert stream.read(60000) == BLOB[:60000]
        assert stream.read(60000) == BLOB[60000:]
        assert stream.read(10) == b""


def test_open_blob_does_not_hold_blob_in_memory(repository, store):
    _, commit = repository
    stream = store.open_blob(commit, "a.json")
    assert not isinstance(stream, io.BytesIO)
    stream.close()
    stream.raw.close()  # closing again is harmless
    assert stream.closed
    assert stream.raw._process.poll() is not None


def test_open_blob_reports_invalid_paths(repository, store, monkeypatch):
    _, commit = repository
    with pytest.raises(FileNotFoundError, match="Git path 'missing'"):
        store.open_blob(commit, "missing")
    with pytest.raises(ValueError, match="not a file"):
        store.open_blob(commit, "models")
    monkeypatch.setenv("PATH", "")
    with pytest.raises(GitNotInstalledError, match="Git is required"):
        git_backend._spawn_git(store.repository_path, "status")


def test_blob_stream_reports_git_failure():
    process = subprocess.Popen(
        ["sh", "-c", "echo fatal: broken object >&2; exit 3"], stdout=subprocess.PIPE
    )
    stderr = TemporaryFile()
    stream = io.BufferedReader(git_backend._BlobStream(process, stderr, "blob"))
    process.wait()
    stderr.write(b"fatal: broken object\n")
    with pytest.raises(OSError, match="Reading blob failed: fatal: broken object"):
        stream.read()
    stream.raw.readable()
    stream.close()


def test_lock_is_reset_in_forked_child(repository, store):
    store._lock.acquire()
    git_backend._reset_locks_after_fork()
    assert store._lock.acquire(blocking=False)
    store._lock.release()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_does_not_deadlock_on_inherited_lock(repository, store):
    _, commit = repository
    store._lock.acquire()  # simulate another thread holding the lock during fork
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        code = 1
        with suppress(Exception):
            code = 0 if store.read_blob(commit, "a.json") == BLOB else 2
        os._exit(code)
    store._lock.release()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    pytest.fail("forked child deadlocked on the inherited lock")
