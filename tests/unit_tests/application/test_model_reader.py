"""Tests for the model-reader factory."""

from pathlib import Path
from unittest.mock import Mock

from simtools.application.model_reader import create_model_reader


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
