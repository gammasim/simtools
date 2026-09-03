"""Tests for the model-reader factory."""

from pathlib import Path
from unittest.mock import Mock, PropertyMock

import pytest

from simtools.application.model_reader import (
    create_model_reader,
    create_model_reader_from_source_config,
    require_model_reader,
)
from simtools.db.mongo_db import MongoDBDependencyError
from simtools.model_repository.reader import SimulationModelReader
from simtools.settings import config


def _model_repository_root(tmp_test_directory):
    """Create the directories required for a filesystem model source."""
    root = Path(tmp_test_directory) / "models"
    simulation_models = root / "simulation-models"
    (simulation_models / "productions").mkdir(parents=True)
    (simulation_models / "model_parameters").mkdir()
    return root


def test_create_model_reader_uses_filesystem_path(tmp_test_directory, mocker):
    """A repository path selects the filesystem source without constructing a DB handler."""
    root = _model_repository_root(tmp_test_directory)
    database_handler = mocker.patch("simtools.db.db_handler.DatabaseHandler")

    reader = create_model_reader(simulation_models_path=root)

    assert reader.source_name == str(root.resolve())
    database_handler.assert_not_called()


def test_create_model_reader_constructs_database_handler_when_needed(mocker):
    """Without a path, the factory constructs and adapts the database handler."""
    handler = Mock(model_source_name="simulation-model-db")
    database_handler = mocker.patch("simtools.db.db_handler.DatabaseHandler", return_value=handler)

    reader = create_model_reader()

    assert reader.source_name == "simulation-model-db"
    database_handler.assert_called_once_with()


def test_create_model_reader_uses_environment_path(monkeypatch, tmp_test_directory):
    """The environment selects a filesystem source when no path is passed explicitly."""
    root = _model_repository_root(tmp_test_directory)
    monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_PATH", str(root))

    reader = create_model_reader()

    assert reader.source_name == str(root.resolve())


def test_create_model_reader_selects_git_source(monkeypatch, mocker, tmp_test_directory):
    """A Git path and revision select the Git source without MongoDB."""
    git_path = Path(tmp_test_directory) / "models.git"
    git_reader = Mock()
    from_git = mocker.patch.object(SimulationModelReader, "from_git", return_value=git_reader)
    monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_GIT_PATH", str(git_path))
    monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_GIT_REVISION", "v1")
    database_handler = mocker.patch("simtools.db.db_handler.DatabaseHandler")

    assert create_model_reader() is git_reader
    from_git.assert_called_once_with(str(git_path), "v1")
    database_handler.assert_not_called()


@pytest.mark.parametrize(
    "catalog_model",
    [
        {"git-revision": "a" * 40},
        {"default-tag": "v1.2.3"},
        {"default-version": "1.2.3"},
    ],
)
def test_create_model_reader_uses_catalog_git_revision(catalog_model, mocker):
    """A missing Git revision falls back through the dependency catalog."""
    mocker.patch(
        "simtools.application.model_reader.dependency_versions.load_dependency_catalog",
        return_value={"model-database": catalog_model},
    )
    from_git = mocker.patch.object(SimulationModelReader, "from_git", return_value=Mock())

    create_model_reader(simulation_models_git_path="models.git")

    from_git.assert_called_once_with("models.git", next(iter(catalog_model.values())))


def test_create_model_reader_rejects_catalog_without_git_revision(mocker):
    """A Git source cannot start when the catalog has no usable revision."""
    mocker.patch(
        "simtools.application.model_reader.dependency_versions.load_dependency_catalog",
        return_value={"model-database": {}},
    )

    with pytest.raises(ValueError, match="Git simulation-model revision is required"):
        create_model_reader(simulation_models_git_path="models.git")


def test_create_model_reader_rejects_two_repository_sources(tmp_test_directory):
    """Filesystem and Git sources cannot be selected simultaneously."""
    with pytest.raises(ValueError, match="cannot be configured together"):
        create_model_reader(
            simulation_models_path=tmp_test_directory,
            simulation_models_git_path=tmp_test_directory,
            simulation_models_git_revision="v1",
        )


def test_create_model_reader_from_source_config_preserves_mongodb_name(mocker):
    """Worker source reconstruction keeps the selected MongoDB name."""
    handler = Mock()
    type(handler).model_source_name = PropertyMock(side_effect=lambda: handler.db_name)
    mocker.patch("simtools.application.model_reader._create_database_handler", return_value=handler)

    reader = create_model_reader_from_source_config({"type": "mongodb", "name": "worker-db"})

    assert handler.db_name == "worker-db"
    assert reader.source_name == "worker-db"


def test_create_model_reader_from_source_config_reopens_git_revision(mocker):
    """Workers reconstruct Git readers from the serialized commit."""
    reader = Mock()
    from_git = mocker.patch.object(SimulationModelReader, "from_git", return_value=reader)

    result = create_model_reader_from_source_config(
        {"type": "git", "repository": "/models.git", "commit": "a" * 40}
    )

    assert result is reader
    from_git.assert_called_once_with("/models.git", "a" * 40)


@pytest.mark.parametrize("source_config", [{"type": "git"}, {"type": "git", "repository": "repo"}])
def test_create_model_reader_from_source_config_rejects_incomplete_git_config(source_config):
    """Worker Git configurations must include both repository and commit."""
    with pytest.raises(ValueError, match="requires repository and commit"):
        create_model_reader_from_source_config(source_config)


@pytest.mark.parametrize(
    ("source_config", "message"),
    [
        (None, "must be a dictionary"),
        ({"type": "filesystem"}, "requires a path"),
        ({"type": "unsupported"}, "Unsupported model source type"),
    ],
)
def test_create_model_reader_from_source_config_rejects_invalid_config(source_config, message):
    """Invalid worker source configurations fail with actionable errors."""
    with pytest.raises(ValueError, match=message):
        create_model_reader_from_source_config(source_config)


def test_create_model_reader_from_source_config_uses_filesystem_path(mocker):
    """A filesystem source configuration delegates to the normal factory."""
    reader = Mock()
    create_reader = mocker.patch(
        "simtools.application.model_reader.create_model_reader", return_value=reader
    )

    assert (
        create_model_reader_from_source_config({"type": "filesystem", "path": "models"}) is reader
    )

    create_reader.assert_called_once_with("models")


def test_create_model_reader_from_source_config_allows_unnamed_mongodb(mocker):
    """A MongoDB source configuration may use the handler's default database name."""
    handler = Mock(model_source_name="default-db")
    mocker.patch("simtools.application.model_reader._create_database_handler", return_value=handler)

    reader = create_model_reader_from_source_config({"type": "mongodb"})

    assert reader.source_name == "default-db"


def test_create_model_reader_reports_missing_mongodb_dependency(mocker):
    """The MongoDB fallback reports how to install its optional dependency."""
    mocker.patch(
        "simtools.db.db_handler.DatabaseHandler",
        side_effect=MongoDBDependencyError("MongoDB unavailable"),
    )

    with pytest.raises(RuntimeError, match="install with the `mongodb` extra") as error:
        create_model_reader()

    assert isinstance(error.value.__cause__, MongoDBDependencyError)


def test_require_model_reader_prefers_explicit_reader():
    """An explicit reader is returned independently of application configuration."""
    reader = Mock()

    assert require_model_reader(reader) is reader


def test_require_model_reader_uses_configured_reader():
    """Library callers use the reader selected during application setup."""
    reader = Mock()
    previous_reader = config.model_reader
    config.set_model_reader(reader)
    try:
        assert require_model_reader() is reader
    finally:
        config.set_model_reader(previous_reader)


def test_require_model_reader_requires_selection():
    """Calls outside an application must explicitly select a reader."""
    previous_reader = config.model_reader
    config.set_model_reader(None)
    try:
        with pytest.raises(RuntimeError, match="No simulation model reader is configured"):
            require_model_reader()
    finally:
        config.set_model_reader(previous_reader)
