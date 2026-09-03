"""Tests for the pygit2 object-store adapter."""

import io
import sys
from types import SimpleNamespace

import pytest

from simtools.model_repository.git_backend import (
    GitModelSourceDependencyError,
    Pygit2ObjectStore,
)


class _Node:
    """Minimal pygit2 tree or blob object."""

    def __init__(self, name, type_str, children=None, data=None):
        self.name = name
        self.type_str = type_str
        self.children = children or {}
        self.data = data

    def __getitem__(self, name):
        return self.children[name]

    def __iter__(self):
        return iter(self.children.values())


class _Repository:
    """Minimal pygit2 repository object."""

    def __init__(self, commit, tree):
        self.commit = commit
        self.tree = tree

    def revparse_single(self, _):
        """Return a revision object that resolves to the configured commit."""
        return SimpleNamespace(peel=lambda _: self.commit)

    def __getitem__(self, _):
        return SimpleNamespace(tree=self.tree)


def _raise_invalid_repository(_):
    """Raise the repository-open error used by the invalid-repository test."""
    raise OSError("invalid repository")


def _install_pygit2(monkeypatch, repository):
    """Install a minimal pygit2 substitute and return the resolved commit."""
    pygit2 = SimpleNamespace(
        Commit=object(),
        Repository=lambda _: repository,
        BlobIO=lambda blob: io.BytesIO(blob.data),
    )
    monkeypatch.setitem(sys.modules, "pygit2", pygit2)


def test_pygit2_object_store_requires_optional_dependency(monkeypatch, tmp_test_directory):
    """Selecting a Git source explains how to install its dependency."""
    monkeypatch.setitem(sys.modules, "pygit2", None)

    with pytest.raises(
        GitModelSourceDependencyError, match="install simtools with the `git` extra"
    ):
        Pygit2ObjectStore(tmp_test_directory)


def test_pygit2_object_store_rejects_missing_repository(monkeypatch, tmp_test_directory):
    """A nonexistent repository path fails before opening pygit2."""
    _install_pygit2(monkeypatch, repository=None)

    with pytest.raises(FileNotFoundError, match="Git model repository does not exist"):
        Pygit2ObjectStore(tmp_test_directory / "missing.git")


def test_pygit2_object_store_rejects_unreadable_repository(monkeypatch, tmp_test_directory):
    """A path that pygit2 cannot open reports a readable error."""
    pygit2 = SimpleNamespace(
        Commit=object(),
        Repository=_raise_invalid_repository,
        BlobIO=lambda blob: io.BytesIO(blob.data),
    )
    monkeypatch.setitem(sys.modules, "pygit2", pygit2)

    with pytest.raises(ValueError, match="Not a readable Git model repository"):
        Pygit2ObjectStore(tmp_test_directory)


def test_pygit2_object_store_reads_tree_blobs(monkeypatch, tmp_test_directory):
    """Resolve revisions and read recursively discovered blob objects."""
    blob = _Node("model.json", "blob", data=b"{}")
    nested_blob = _Node("nested.json", "blob", data=b"[]")
    nested_tree = _Node("nested", "tree", {"nested.json": nested_blob})
    models_tree = _Node("models", "tree", {"model.json": blob, "nested": nested_tree})
    root_tree = _Node("", "tree", {"models": models_tree})
    commit = SimpleNamespace(id="a" * 40)
    repository = _Repository(commit, root_tree)
    _install_pygit2(monkeypatch, repository)
    store = Pygit2ObjectStore(tmp_test_directory)

    resolved = store.resolve_revision("v1")

    assert resolved == "a" * 40
    assert store.iter_files(resolved, "models") == [
        "models/model.json",
        "models/nested/nested.json",
    ]
    assert store.read_blob(resolved, "models/model.json") == b"{}"
    assert store.open_blob(resolved, "models/nested/nested.json").read() == b"[]"


def test_pygit2_object_store_reports_invalid_paths(monkeypatch, tmp_test_directory):
    """Missing and non-file paths provide source-specific errors."""
    tree = _Node("models", "tree")
    root_tree = _Node("", "tree", {"models": tree})
    repository = _Repository(SimpleNamespace(id="a" * 40), root_tree)
    _install_pygit2(monkeypatch, repository)
    store = Pygit2ObjectStore(tmp_test_directory)

    with pytest.raises(FileNotFoundError, match="Git path missing"):
        store.iter_files("a" * 40, "missing")
    with pytest.raises(ValueError, match="Git path is not a file"):
        store.read_blob("a" * 40, "models")


def test_pygit2_object_store_reports_invalid_revisions_and_commits(
    monkeypatch, mocker, tmp_test_directory
):
    """Invalid revisions and commits report source-specific errors."""
    root_tree = _Node("", "tree")
    repository = _Repository(SimpleNamespace(id="a" * 40), root_tree)
    _install_pygit2(monkeypatch, repository)
    store = Pygit2ObjectStore(tmp_test_directory)
    mocker.patch.object(repository, "revparse_single", side_effect=KeyError)

    with pytest.raises(ValueError, match="Git revision 'missing' not found"):
        store.resolve_revision("missing")

    mocker.patch.object(_Repository, "__getitem__", side_effect=KeyError)
    with pytest.raises(ValueError, match="Git commit 'b"):
        store.iter_files("b" * 40, "models")


def test_pygit2_object_store_reports_missing_or_file_prefix(monkeypatch, tmp_test_directory):
    """Directory and blob APIs reject paths of the wrong kind."""
    blob = _Node("model.json", "blob", data=b"{}")
    root_tree = _Node("", "tree", {"model.json": blob})
    repository = _Repository(SimpleNamespace(id="a" * 40), root_tree)
    _install_pygit2(monkeypatch, repository)
    store = Pygit2ObjectStore(tmp_test_directory)

    with pytest.raises(ValueError, match="Git path is not a directory"):
        store.iter_files("a" * 40, "model.json")
    with pytest.raises(FileNotFoundError, match="Git path 'missing'"):
        store.read_blob("a" * 40, "missing")
