"""Tests for the model-reader factory."""

from pathlib import Path
from unittest.mock import Mock, PropertyMock

import pytest

from simtools.application.model_reader import (
    create_model_reader,
    create_model_reader_from_source_config,
    require_model_reader,
)
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


def test_create_model_reader_from_source_config_preserves_mongodb_name(mocker):
    """Worker source reconstruction keeps the selected MongoDB name."""
    handler = Mock()
    type(handler).model_source_name = PropertyMock(side_effect=lambda: handler.db_name)
    mocker.patch("simtools.application.model_reader._create_database_handler", return_value=handler)

    reader = create_model_reader_from_source_config({"type": "mongodb", "name": "worker-db"})

    assert handler.db_name == "worker-db"
    assert reader.source_name == "worker-db"


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
