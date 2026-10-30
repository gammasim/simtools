"""Contract tests for model repository storage implementations."""

from pathlib import Path

import pygit2
import pytest

from simtools.model_repository.storage import DirectoryStorage, GitStorage, MemoryStorage

FILES = {
    "simulation-models/model_parameters/global/example/example-1.json": b'{"value": 1}',
    "simulation-models/productions/Prod1/info.yml": b"name: Prod1\n",
    "simulation-models/productions/Prod1/parameters.json": b"[]",
}


def _make_git_repository(root, files):
    """Create a one-commit repository containing the provided model files."""
    repository = pygit2.init_repository(root)
    for relative, content in files.items():
        file_path = root / relative
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        repository.index.add(relative)
    repository.index.write()
    tree = repository.index.write_tree()
    signature = pygit2.Signature("Test", "test@example.invalid")
    commit = repository.create_commit("HEAD", signature, signature, "model fixture", tree, [])
    return repository, str(commit)


@pytest.fixture(params=["directory", "git", "memory"])
def storage(request, tmp_test_directory):
    """Provide each storage with identical generated file content."""
    if request.param == "directory":
        root = Path(tmp_test_directory) / "directory-model"
        for relative, content in FILES.items():
            file_path = root / relative
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
        return DirectoryStorage(root)
    if request.param == "git":
        root = Path(tmp_test_directory) / "git-model"
        root.mkdir()
        _, commit = _make_git_repository(root, FILES)
        return GitStorage(root, commit)
    return MemoryStorage(FILES)


def test_storage_reads_identical_files(storage):
    """Each implementation reads files, streams, and lists them consistently."""
    path = "simulation-models/productions/Prod1/info.yml"

    assert storage.read_bytes(path) == FILES[path]
    with storage.open(path) as stream:
        assert stream.read() == FILES[path]
    assert storage.iter_files("simulation-models/productions") == sorted(
        name for name in FILES if name.startswith("simulation-models/productions/")
    )
    assert storage.exists(path)
    assert not storage.exists("simulation-models/missing.json")
    with pytest.raises(FileNotFoundError, match="missing"):
        storage.read_bytes("simulation-models/missing.json")
    assert storage.config
    assert storage.name


@pytest.mark.parametrize("root_prefix", ["", "."])
def test_storage_lists_all_files_from_repository_root(storage, root_prefix):
    assert storage.iter_files(root_prefix) == sorted(FILES)
