"""Read-only storage implementations for simulation model files."""

import io
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol

from simtools.model_repository.git_backend import Pygit2ObjectStore


class ModelStorage(Protocol):
    """Common read-only interface for model repository storage."""

    name: str
    config: dict

    def read_bytes(self, path: str) -> bytes:
        """Read a repository-relative file as bytes."""
        ...

    def open(self, path: str) -> BinaryIO:
        """Open a repository-relative file as a binary stream."""
        ...

    def iter_files(self, prefix: str) -> list[str]:
        """List repository-relative files below a directory prefix."""
        ...

    def exists(self, path: str) -> bool:
        """Return whether a repository-relative file exists."""
        ...


def _relative_path(path: str) -> PurePosixPath:
    """Validate and convert a repository-relative POSIX path."""
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Model storage path must be repository-relative: {path!r}")
    return relative


class DirectoryStorage:
    """Read model files from a directory tree."""

    def __init__(self, root):
        """Create storage rooted at a model checkout directory."""
        self.root = Path(root).expanduser().resolve()
        self.name = str(self.root)
        self.config = {"type": "filesystem", "path": str(self.root)}

    def _path(self, path):
        return self.root.joinpath(*_relative_path(path).parts)

    def read_bytes(self, path):
        """Read a repository-relative file as bytes."""
        return self._path(path).read_bytes()

    def open(self, path):
        """Open a repository-relative file as a binary stream."""
        return self._path(path).open("rb")

    def iter_files(self, prefix):
        """List repository-relative files below a directory prefix."""
        directory = self._path(prefix)
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        )

    def exists(self, path):
        """Return whether a repository-relative file exists."""
        return self._path(path).is_file()


class GitStorage:
    """Read model files from a resolved commit in a local Git repository."""

    def __init__(self, repository, revision, object_store=None):
        """Create storage for one revision of a local repository."""
        self.repository = Path(repository).expanduser().resolve()
        self.object_store = object_store or Pygit2ObjectStore(self.repository)
        self.revision = self.object_store.resolve_revision(revision)
        self.name = f"{self.repository}@{self.revision}"
        self.config = {
            "type": "git",
            "repository": str(self.repository),
            "commit": self.revision,
        }

    def read_bytes(self, path):
        """Read a repository-relative file as bytes."""
        return self.object_store.read_blob(self.revision, _relative_path(path).as_posix())

    def open(self, path):
        """Open a repository-relative file as a binary stream."""
        return self.object_store.open_blob(self.revision, _relative_path(path).as_posix())

    def iter_files(self, prefix):
        """List repository-relative files below a directory prefix."""
        return self.object_store.iter_files(self.revision, _relative_path(prefix).as_posix())

    def exists(self, path):
        """Return whether a repository-relative file exists (directories are not files)."""
        relative = _relative_path(path).as_posix()
        try:
            self.object_store.read_blob(self.revision, relative)
        except FileNotFoundError:
            return False
        except ValueError:  # the path names a tree (directory), not a blob
            return False
        return True


class MemoryStorage:
    """Read model files from an in-memory mapping of paths to bytes."""

    def __init__(self, files: dict[str, bytes]):
        """Create storage from repository-relative paths and byte contents."""
        self.files = {_relative_path(path).as_posix(): data for path, data in files.items()}
        self.name = "memory"
        self.config = {"type": "memory", "files": self.files}

    def read_bytes(self, path):
        """Read a repository-relative file as bytes."""
        relative = _relative_path(path).as_posix()
        try:
            return self.files[relative]
        except KeyError as exc:
            raise FileNotFoundError(f"Model file not found: {relative}") from exc

    def open(self, path):
        """Open a repository-relative file as a binary stream."""
        return io.BytesIO(self.read_bytes(path))

    def iter_files(self, prefix):
        """List repository-relative files below a directory prefix."""
        directory = _relative_path(prefix).as_posix()
        if directory == ".":
            return sorted(self.files)
        return sorted(path for path in self.files if path.startswith(directory.rstrip("/") + "/"))

    def exists(self, path):
        """Return whether a repository-relative file exists."""
        return _relative_path(path).as_posix() in self.files
