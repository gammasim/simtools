"""Small, replaceable Git object-store adapter for simulation models."""

from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath


class GitModelSourceDependencyError(RuntimeError):
    """Raised when the Git model source is selected without pygit2."""


class GitObjectStore(ABC):
    """Library-neutral interface used by ``GitModelSource``."""

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


class Pygit2ObjectStore(GitObjectStore):
    """Read objects from a local normal, bare, or mirror repository."""

    def __init__(self, repository_path):
        """Open a local Git repository without creating a working tree."""
        try:
            import pygit2  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise GitModelSourceDependencyError(
                "The Git simulation-model source requires pygit2; "
                "install simtools with the `git` extra."
            ) from exc

        self.repository_path = Path(repository_path).expanduser().resolve()
        if not self.repository_path.exists():
            raise FileNotFoundError(f"Git model repository does not exist: {self.repository_path}")
        try:
            self._repository = pygit2.Repository(str(self.repository_path))
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Not a readable Git model repository: {self.repository_path}"
            ) from exc
        self._pygit2 = pygit2

    def resolve_revision(self, revision):
        """Resolve a tag, ref, or SHA to a full commit SHA."""
        try:
            obj = self._repository.revparse_single(str(revision))
            commit = obj.peel(self._pygit2.Commit)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Git revision {revision!r} not found or does not identify a commit "
                f"in {self.repository_path}"
            ) from exc
        return str(commit.id)

    def _commit_tree(self, commit):
        """Return the tree for a resolved commit."""
        try:
            return self._repository[str(commit)].tree
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Git commit {commit!r} not found in {self.repository_path}") from exc

    def _tree_object(self, commit, path):
        """Return a tree object by its POSIX repository path."""
        obj = self._commit_tree(commit)
        for part in PurePosixPath(path).parts:
            try:
                obj = self._dereference_tree_entry(obj[part])
            except (KeyError, ValueError, TypeError) as exc:
                raise FileNotFoundError(
                    f"Git path {path!r} not found at commit {commit} in {self.repository_path}"
                ) from exc
        return obj

    def _dereference_tree_entry(self, entry):
        """Return the Git object represented by a pygit2 tree entry."""
        if hasattr(entry, "id"):
            return self._repository[entry.id]
        return entry

    def iter_files(self, commit, prefix):
        """Return sorted regular-file paths below a tree prefix."""
        prefix = PurePosixPath(prefix)
        tree = self._commit_tree(commit)
        for part in prefix.parts:
            try:
                tree = self._dereference_tree_entry(tree[part])
            except (KeyError, ValueError, TypeError) as exc:
                raise FileNotFoundError(
                    f"Git path {prefix!s} not found at commit {commit} in {self.repository_path}"
                ) from exc
        if tree.type_str != "tree":
            raise ValueError(f"Git path is not a directory: {prefix}")
        paths = []
        self._append_files(tree, prefix, paths)
        return sorted(paths)

    def _append_files(self, tree, prefix, paths):
        """Recursively collect paths from a tree."""
        for obj in tree:
            path = prefix / obj.name
            if obj.type_str == "tree":
                self._append_files(self._dereference_tree_entry(obj), path, paths)
            elif obj.type_str == "blob":
                paths.append(path.as_posix())

    def _blob(self, commit, path):
        """Return a blob object and provide a source-specific error."""
        obj = self._tree_object(commit, path)
        if obj.type_str != "blob":
            raise ValueError(f"Git path is not a file: {path}")
        return obj

    def read_blob(self, commit, path):
        """Read a blob into memory."""
        return self._blob(commit, path).data

    def open_blob(self, commit, path):
        """Open a blob as a binary stream without materializing it in memory."""
        return self._pygit2.BlobIO(self._blob(commit, path))
