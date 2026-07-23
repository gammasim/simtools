#!/usr/bin/python3

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import simtools.applications.db_upload_model_repository as app
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.constants import DEFAULT_SIMULATION_MODELS


def test_add_arguments_includes_repository_dir_option(tmp_test_directory):
    parser = CommandLineParser()

    parser.add_argument_definitions(app._ARGUMENTS)

    repository_dir = str(Path(tmp_test_directory) / "simulation-models")
    args = parser.parse_args(["--repository_dir", repository_dir])
    assert args.repository_dir == repository_dir


@patch("simtools.applications.db_upload_model_repository.db_model_upload.add_complete_model")
@patch("simtools.applications.db_upload_model_repository.db_handler.DatabaseHandler")
@patch("simtools.applications.db_upload_model_repository.config.load")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_forwards_repository_dir(
    mock_application_start,
    mock_config_load,
    mock_database_handler,
    mock_add_complete_model,
    tmp_test_directory,
):
    repository_dir = str(Path(tmp_test_directory) / "simulation-models")
    app_context = SimpleNamespace(
        args={
            "db_simulation_model": "CTAO-Simulation-Model",
            "db_simulation_model_version": "6.0.2",
            "branch": "main",
            "tmp_dir": "tmp_model_parameters",
            "max_attempts": 3,
            "repository_dir": repository_dir,
        },
        db_config={},
    )
    mock_application_start.return_value = app_context

    db = Mock()
    mock_database_handler.return_value = db

    app.main()

    mock_config_load.assert_called_once()
    db.print_connection_info.assert_called_once()
    mock_add_complete_model.assert_called_once_with(
        tmp_dir="tmp_model_parameters",
        db=db,
        db_simulation_model="CTAO-Simulation-Model",
        db_simulation_model_version="6.0.2",
        repository_url=None,
        repository_branch="main",
        max_attempts=3,
        repository_dir=repository_dir,
    )


@patch("simtools.applications.db_upload_model_repository.db_model_upload.add_complete_model")
@patch("simtools.applications.db_upload_model_repository.db_handler.DatabaseHandler")
@patch("simtools.applications.db_upload_model_repository.config.load")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_uses_default_repository_url_without_repository_dir(
    mock_application_start,
    mock_config_load,
    mock_database_handler,
    mock_add_complete_model,
):
    app_context = SimpleNamespace(
        args={
            "db_simulation_model": "CTAO-Simulation-Model",
            "db_simulation_model_version": "6.0.2",
            "branch": "main",
            "tmp_dir": "tmp_model_parameters",
            "max_attempts": 3,
            "repository_dir": None,
        },
        db_config={},
    )
    mock_application_start.return_value = app_context
    mock_database_handler.return_value = Mock()

    app.main()

    mock_config_load.assert_called_once()
    mock_add_complete_model.assert_called_once_with(
        tmp_dir="tmp_model_parameters",
        db=mock_database_handler.return_value,
        db_simulation_model="CTAO-Simulation-Model",
        db_simulation_model_version="6.0.2",
        repository_url=DEFAULT_SIMULATION_MODELS,
        repository_branch="main",
        max_attempts=3,
        repository_dir=None,
    )
