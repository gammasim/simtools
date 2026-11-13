"""Git object-store adapter for simulation models."""

import io
import os
import subprocess
import threading
import weakref
from abc import ABC, abstractmethod
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryFile

_STOP_TIMEOUT_SECONDS = 5
_STDERR_TAIL_BYTES = 2000
_ROOT_PATHS = ("", ".")


class GitNotInstalledError(RuntimeError):
    """Raised when the Git executable is not installed."""


class GitObjectStore(ABC):
    """Interface used by ``GitStorage``."""

    @abstractmethod
    def resolve_revision(self, revision):
        """Resolve a ref or commit to a full commit ID."""

    @abstractmethod
    def iter_files(self, commit, prefix):
        """Return file paths below ``prefix`` for ``commit``."""

    @abstractmethod
    def read_blob(self, commit, path):
        """Return one blob as bytes."""

    @abstractmethod
    def open_blob(self, commit, path):
        """Return a binary stream for one blob."""

    def close(self):
        """Release resources held by the object store."""


def _stderr_text(exc):
    """Return the stripped Git error output of a failed command."""
    return (exc.stderr or b"").decode("utf-8", errors="replace").strip()


def _git_command(repository, *args):
    """Return a Git command line for a configured repository without owner validation."""
    return ["git", "-c", "safe.directory=*", "-C", str(repository), *args]


def run_git(repository, *args):
    """Run Git against a configured repository without owner validation.

    ``subprocess.CalledProcessError`` is raised on failure; ``stderr`` holds Git's message.
    """
    try:
        return subprocess.run(
            _git_command(repository, *args), check=True, capture_output=True
        ).stdout
    except FileNotFoundError as exc:
        raise GitNotInstalledError("Git is required to read Git repositories") from exc


def _spawn_git(repository, *args, stdin=None):
    """Start a long-lived Git process writing errors to a temporary file."""
    stderr = TemporaryFile()
    try:
        # long-lived by design; callers stop it through _terminate
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            _git_command(repository, *args),
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
    except FileNotFoundError as exc:
        stderr.close()
        raise GitNotInstalledError("Git is required to read Git repositories") from exc
    except OSError:
        stderr.close()
        raise
    return process, stderr


def split_nul(data):
    """Split NUL-terminated Git output into non-empty, decoded entries."""
    return [entry.decode("utf-8", errors="surrogateescape") for entry in data.split(b"\0") if entry]


class _GitProtocolError(Exception):
    """Raised when the persistent Git process fails or answers unexpectedly."""


def _terminate(process, stderr):
    """Stop a Git process and release its pipes."""
    if process.stdin is not None:
        with suppress(OSError):
            process.stdin.close()
    with suppress(OSError):
        process.stdout.close()
    try:
        process.wait(timeout=_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    stderr.close()


def _tail(stderr, size=_STDERR_TAIL_BYTES):
    """Return the end of a process' error output."""
    try:
        stderr.seek(0, os.SEEK_END)
        stderr.seek(max(0, stderr.tell() - size))
        return stderr.read().decode("utf-8", errors="replace").strip()
    except OSError, ValueError, AttributeError:
        return ""


_STORES = weakref.WeakSet()


def _reset_locks_after_fork():
    """Give each store in a forked child a free lock (the owning thread does not exist there)."""
    for store in list(_STORES):
        store._lock = threading.Lock()  # pylint: disable=protected-access


os.register_at_fork(after_in_child=_reset_locks_after_fork)


class _BlobStream(io.RawIOBase):
    """Binary stream over the output of a ``git cat-file blob`` process."""

    def __init__(self, process, stderr, description):
        super().__init__()
        self._process = process
        self._stderr = stderr
        self._description = description

    def readable(self):
        """Return True; the stream is read-only."""
        return True

    def readinto(self, buffer):
        """Read available bytes; fail if Git ends with an error."""
        count = self._process.stdout.readinto(buffer)
        if count == 0 and self._process.wait() != 0:
            raise OSError(f"Reading {self._description} failed: {_tail(self._stderr)}")
        return count

    def close(self):
        """Stop the Git process and release its resources."""
        if not self.closed:
            _terminate(self._process, self._stderr)
        super().close()


def _object_spec(commit, path):
    """Return the Git object name for a repository path at a commit."""
    return f"{commit}:" if path in _ROOT_PATHS else f"{commit}:{path}"


class GitCommandObjectStore(GitObjectStore):
    """Read objects from a local normal, bare, or mirror repository.

    Small object reads share one persistent ``git cat-file --batch`` process, so reading many
    files costs one round trip each instead of one process each. The process is started on
    first use, restarted once after a failure, and stopped by ``close()`` or garbage
    collection. Calls are serialized by a lock and are safe across threads; a forked child
    starts its own process. ``open_blob`` streams through a separate process and never holds
    the blob in memory.
    """

    def __init__(self, repository_path):
        """Open a local Git repository without creating a working tree."""
        self.repository_path = Path(repository_path).expanduser().resolve()
        if not self.repository_path.exists():
            raise FileNotFoundError(f"Git model repository does not exist: {self.repository_path}")
        try:
            run_git(self.repository_path, "rev-parse", "--git-dir")
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"Not a readable Git model repository: {self.repository_path}: {_stderr_text(exc)}"
            ) from exc
        self._lock = threading.Lock()
        self._process = None
        self._stderr = None
        self._finalizer = None
        self._owner_pid = None
        _STORES.add(self)

    def close(self):
        """Stop the persistent Git process; it is restarted on the next read."""
        with self._lock:
            self._stop()

    def _stop(self):
        """Stop the persistent process (caller holds the lock)."""
        if self._finalizer is not None:
            if os.getpid() == self._owner_pid:
                self._finalizer()
            else:
                self._finalizer.detach()  # the process belongs to the parent
        self._process = self._stderr = self._finalizer = None

    def _start(self):
        """Start the persistent process (caller holds the lock)."""
        process, stderr = _spawn_git(
            self.repository_path, "cat-file", "--batch", stdin=subprocess.PIPE
        )
        self._process, self._stderr = process, stderr
        self._owner_pid = os.getpid()
        self._finalizer = weakref.finalize(self, _terminate, process, stderr)

    def _request(self, spec):
        """Return (type, content) for an object spec, or None if it does not exist."""
        if "\n" in spec or "\0" in spec:
            return None
        with self._lock:
            try:
                return self._attempt(spec)
            except _GitProtocolError:
                pass  # a dead or confused process is replaced and the request retried once
            try:
                return self._attempt(spec)
            except _GitProtocolError as exc:
                raise RuntimeError(
                    f"Git process failed while reading {spec!r} in {self.repository_path}: {exc}"
                ) from exc

    def _attempt(self, spec):
        """Run one request, starting the process if needed (caller holds the lock)."""
        if self._process is not None and os.getpid() != self._owner_pid:
            self._stop()
        if self._process is None:
            self._start()
        try:
            return self._exchange(spec)
        except (OSError, ValueError, _GitProtocolError) as exc:
            detail = _tail(self._stderr) or str(exc)
            self._stop()
            raise _GitProtocolError(detail) from exc

    def _exchange(self, spec):
        """Send one request to the persistent process and parse its response."""
        self._process.stdin.write(spec.encode("utf-8", errors="surrogateescape") + b"\n")
        self._process.stdin.flush()
        header = self._process.stdout.readline()
        if not header:
            raise _GitProtocolError("Git process ended unexpectedly")
        fields = header.split()
        if fields[-1:] in ([b"missing"], [b"ambiguous"]):
            return None
        if fields[-1:] == [b"submodule"]:
            return "submodule", b""  # a gitlink has no object content here
        try:
            kind, size = fields[1].decode("ascii"), int(fields[2])
        except (IndexError, ValueError, UnicodeDecodeError) as exc:
            raise _GitProtocolError(f"Unexpected Git response: {header[:100]!r}") from exc
        content = self._process.stdout.read(size + 1)
        if len(content) != size + 1 or not content.endswith(b"\n"):
            raise _GitProtocolError(f"Truncated Git response for {size} byte object")
        return kind, content[:size]

    def resolve_revision(self, revision):
        """Resolve a tag, ref, or SHA to a full commit SHA."""
        try:
            result = run_git(
                self.repository_path,
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{revision}^{{commit}}",
            )
        except subprocess.CalledProcessError as exc:
            raise ValueError(
                f"Git revision {revision!r} not found or does not identify a commit "
                f"in {self.repository_path}: {_stderr_text(exc)}"
            ) from exc
        return result.decode().strip()

    def _not_found(self, commit, path):
        """Return the error for a path missing at a commit."""
        return FileNotFoundError(
            f"Git path {path!r} not found at commit {commit} in {self.repository_path}"
        )

    @staticmethod
    def _check_kind(kind, expected, path):
        """Raise if an object is not of the expected type."""
        if kind != expected:
            label = "directory" if expected == "tree" else "file"
            raise ValueError(f"Git path is not a {label}: {path}")

    def _object(self, commit, path, expected):
        """Return object content after checking it exists and has the expected type."""
        found = self._request(_object_spec(commit, path))
        if found is None:
            raise self._not_found(commit, path)
        self._check_kind(found[0], expected, path)
        return found[1]

    def iter_files(self, commit, prefix):
        """Return sorted blob paths below a tree prefix."""
        self._object(commit, prefix, "tree")
        try:
            entries = run_git(
                self.repository_path,
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                _object_spec(commit, prefix),
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Listing Git path {str(prefix)!r} at commit {commit} failed "
                f"in {self.repository_path}: {_stderr_text(exc)}"
            ) from exc
        base = Path() if prefix in _ROOT_PATHS else Path(prefix)
        paths = []
        for entry in split_nul(entries):
            metadata, _, name = entry.partition("\t")
            if metadata.split()[1:2] == ["blob"]:
                paths.append((base / name).as_posix())
        return sorted(paths)

    def read_blob(self, commit, path):
        """Read a blob into memory."""
        return self._object(commit, path, "blob")

    def open_blob(self, commit, path):
        """Open a blob as a binary stream; the content is not held in memory."""
        spec = _object_spec(commit, path)
        try:
            kind = run_git(self.repository_path, "cat-file", "-t", spec).decode().strip()
        except subprocess.CalledProcessError as exc:
            raise self._not_found(commit, path) from exc
        self._check_kind(kind, "blob", path)
        process, stderr = _spawn_git(self.repository_path, "cat-file", "blob", spec)
        description = f"{path!r} at commit {commit} in {self.repository_path}"
        return io.BufferedReader(_BlobStream(process, stderr, description))
