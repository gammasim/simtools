#!/usr/bin/python3

import logging
import sys
from copy import copy
from unittest.mock import MagicMock

import astropy.units as u
import pytest
import yaml

from simtools import settings
from simtools.configuration.configurator import Configurator
from simtools.io import io_handler

logger = logging.getLogger()


@pytest.fixture
def configurator(tmp_test_directory, _mock_settings_env_vars):
    config = Configurator()
    config.default_config(
        (
            "--output_path",
            str(tmp_test_directory),
        )
    )
    return config


def test_command_line_precedence_over_config_file(tmp_test_directory, monkeypatch):
    """Test that command-line arguments override config file settings (issue #2123)."""
    # Create a config file with label='config_label' and log_level='debug'
    _config_dict = {
        "label": "config_label",
        "log_level": "debug",
    }
    _config_file = tmp_test_directory / "configuration-precedence-test.yml"
    with open(_config_file, "w") as output:
        yaml.safe_dump(_config_dict, output, sort_keys=False)

    # Initialize configurator with command-line args that differ from config file
    configurator = Configurator()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_configurator.py",
            "--config",
            str(_config_file),
            "--label",
            "cli_label",
            "--log_level",
            "info",
        ],
    )
    config, _ = configurator.initialize(require_command_line=False, output=True)

    # Command-line values should take precedence
    assert config["label"] == "cli_label"
    assert config["log_level"] == "info"


def test_config_file_applies_when_no_command_line(tmp_test_directory, monkeypatch):
    """Test that config file values apply when no command-line override is provided."""
    # Create a config file with label='config_label' and log_level='debug'
    _config_dict = {
        "label": "config_label",
        "log_level": "debug",
    }
    _config_file = tmp_test_directory / "configuration-no-cli-test.yml"
    with open(_config_file, "w") as output:
        yaml.safe_dump(_config_dict, output, sort_keys=False)

    # Initialize configurator with only config file (no CLI overrides for these keys)
    configurator = Configurator()
    monkeypatch.setattr(sys, "argv", ["test_configurator.py", "--config", str(_config_file)])
    config, _ = configurator.initialize(require_command_line=False, output=True)

    # Config file values should be used
    assert config["label"] == "config_label"
    assert config["log_level"] == "debug"


def test_config_from_file_preserves_selected_by_version_keys(tmp_test_directory):
    config_dict = {
        "applications": [
            {
                "application": "simtools-simulate-prod-htcondor-generator",
                "configuration": {
                    "model_version": ["6.3.0", "7.0.0"],
                    "array_layout_name": {
                        "by_version": {
                            "<7.0.0": "alpha",
                            ">=7.0.0": "CTAO-North-Alpha",
                        }
                    },
                },
            }
        ]
    }
    config_file = tmp_test_directory / "configuration-preserve-by-version.yml"
    with open(config_file, "w", encoding="utf-8") as output:
        yaml.safe_dump(config_dict, output, sort_keys=False)

    config_builder = Configurator()
    loaded_config = config_builder._config_from_file(
        config_file,
        preserve_by_version_keys=["array_layout_name"],
    )

    assert loaded_config["model_version"] == ["6.3.0", "7.0.0"]
    assert loaded_config["array_layout_name"] == {
        "by_version": {
            "<7.0.0": "alpha",
            ">=7.0.0": "CTAO-North-Alpha",
        }
    }


def test_initialize_io_handler(configurator, tmp_test_directory):
    # io_handler is a Singleton, so configurator changes should
    # be reflected in the io_handler
    _io_handler = io_handler.IOHandler()

    configurator.config["output_path"] = tmp_test_directory
    configurator._initialize_io_handler()

    assert _io_handler.output_path.get("default") == tmp_test_directory


def test_arglist_from_config():
    _tmp_dict = {"a": 1.0, "b": None, "c": True, "d": ["d1", "d2", "d3"], "e": 5.0 * u.m}

    assert [
        "--a",
        "1.0",
        "--c",
        "--d",
        "d1",
        "d2",
        "d3",
        "--e",
        "5.0 m",
    ] == Configurator._arglist_from_config(_tmp_dict)

    assert [] == Configurator._arglist_from_config({})

    assert [] == Configurator._arglist_from_config(None)
    assert [] == Configurator._arglist_from_config(5.0)

    assert ["--a", "1.0", "--b", "None", "--c"] == Configurator._arglist_from_config(
        ["--a", "1.0", "--b", None, "--c"]
    )


def test_convert_string_none_to_none():
    assert {} == Configurator._convert_string_none_to_none({})

    _tmp_dict = {
        "a": 1.0,
        "b": None,
        "c": True,
        "d": "None",
    }
    _tmp_none = copy(_tmp_dict)
    _tmp_none["d"] = None

    assert _tmp_none == Configurator._convert_string_none_to_none(_tmp_dict)


def test_get_db_parameters_from_env(configurator, args_dict):
    configurator.parser.initialize_db_config_arguments()
    configurator._fill_config([])
    configurator.config["env_file"] = "this_file_does_not_exist.env"
    _env_config = configurator._config_from_env(configurator.config["env_file"])
    configurator._fill_config(configurator._arglist_from_config(_env_config))

    args_dict["db_api_user"] = "db_user"
    args_dict["db_api_pw"] = "12345"
    args_dict["db_api_port"] = 42
    args_dict["db_server"] = "abc@def.de"
    args_dict["db_simulation_model"] = "sim_model"
    args_dict["db_simulation_model_version"] = "v0.0.0"
    args_dict["env_file"] = "this_file_does_not_exist.env"

    # remove user defined parameters from comparison (depends on environment)
    expected_config = {k: v for k, v in args_dict.items() if not k.startswith("user_")}
    actual_config = {k: v for k, v in configurator.config.items() if not k.startswith("user_")}
    actual_config.pop("db_api_authentication_database")  # depends on user setup; ignore here

    expected_config["sim_telarray_path"] = settings.config.sim_telarray_path

    assert expected_config == actual_config


def test_initialize_output(configurator):
    configurator.parser.initialize_output_arguments()
    configurator._fill_config([])

    # output file for testing
    configurator.config["test"] = True
    configurator._initialize_output()
    assert configurator.config["output_file"] == "TEST.ecsv"

    # output is not configured (and not activity_id)
    configurator.config["test"] = False
    configurator.config["output_file"] = None
    configurator.config.pop("activity_id", None)
    with pytest.raises(KeyError):
        configurator._initialize_output()

    # output is not configured (but activity_id)
    configurator.config["activity_id"] = "A-ID"
    configurator.config["label"] = "test_label"
    configurator._initialize_output()
    assert configurator.config["output_file"] == "A-ID-test_label.ecsv"

    # output file is configured
    configurator.config["test"] = False
    configurator.config["output_file"] = "unit_test.txt"
    configurator._initialize_output()
    assert configurator.config["output_file"] == "unit_test.txt"


def test_default_config_with_site():
    configurator = Configurator(config={})
    configurator.default_config(arg_list=["--site", "North"])
    assert "site" in configurator.config
    assert "telescope" not in configurator.config


def test_default_config_with_telescope():
    configurator = Configurator(config={})
    configurator.default_config(arg_list=["--telescope", "LSTN-01"])
    assert "telescope" in configurator.config
    assert "site" in configurator.config


def test_get_db_parameters():
    # default config
    configurator = Configurator(config={})
    configurator.default_config(add_db_config=True)
    db_params = configurator._get_db_parameters()
    assert db_params == {
        "db_api_authentication_database": None,
        "db_api_port": None,
        "db_api_pw": None,
        "db_api_user": None,
        "db_server": None,
        "db_simulation_model": None,
        "db_simulation_model_version": None,
    }

    # filled with one entry only
    configurator = Configurator(config={})
    configurator.default_config(add_db_config=True)
    configurator.config = {
        "db_api_port": 1234,
    }
    db_params = configurator._get_db_parameters()
    assert db_params == {
        "db_api_port": 1234,
    }

    # filled config
    configurator = Configurator()
    configurator.default_config(add_db_config=True)
    configurator.config = {
        "db_api_user": "user",
        "db_api_pw": "password",
        "db_api_port": 1234,
        "db_simulation_model": "Staging-CTA-Simulation-Model",
        "db_server": "localhost",
    }

    db_params = configurator._get_db_parameters()
    assert db_params == {
        "db_api_user": "user",
        "db_api_pw": "password",
        "db_api_port": 1234,
        "db_server": "localhost",
        "db_simulation_model": "Staging-CTA-Simulation-Model",
    }

    # empty config
    configurator = Configurator(config={})
    configurator.default_config(add_db_config=True)
    configurator.config = {}
    db_params = configurator._get_db_parameters()
    assert db_params == {}


def test_reset_requirements_parameter():
    configurator = Configurator()
    configurator.parser.add_argument("--arg", required=True)
    configurator.config["arg"] = True

    configurator._reset_required_arguments()

    for action in configurator.parser._actions:
        assert action.required is False


def test_reset_required_arguments_group():
    configurator = Configurator()
    group = configurator.parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--arg1")
    group.add_argument("--arg2")
    configurator.config["arg1"] = True

    configurator._reset_required_arguments()

    for group in configurator.parser._mutually_exclusive_groups:
        assert group.required is False


def test_set_model_versions(configurator):
    assert "model_version" not in configurator.config

    model_version_1 = "5.0.0"
    model_version_2 = "6.0.0"
    configurator.config["model_version"] = None
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] is None

    configurator.config["model_version"] = [model_version_1]
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] == model_version_1

    configurator.config["model_version"] = [model_version_1, model_version_2]
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] == [model_version_1, model_version_2]

    configurator.config["model_version"] = model_version_1
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] == model_version_1


def test_initialize(configurator):
    configurator.parser.initialize_default_arguments = MagicMock()
    configurator._get_cli_arglist = MagicMock(return_value=[])
    configurator._config_from_env = MagicMock(return_value={})
    configurator._config_from_file = MagicMock(return_value={})
    configurator._fill_config = MagicMock()
    configurator._reset_required_arguments = MagicMock()
    configurator._initialize_model_versions = MagicMock()
    configurator._initialize_io_handler = MagicMock()
    configurator._initialize_output = MagicMock()
    configurator._get_db_parameters = MagicMock(return_value={"db_param": "test"})

    # Call initialize with default parameters
    config, db_dict = configurator.initialize()

    # Assert that the methods were called
    configurator.parser.initialize_default_arguments.assert_called_once()
    configurator._get_cli_arglist.assert_called_once_with(require_command_line=True)
    configurator._config_from_env.assert_called_once_with(".env")
    configurator._config_from_file.assert_called_once_with(None)
    configurator._fill_config.assert_called_once_with([])
    configurator._initialize_model_versions.assert_called_once()
    configurator._initialize_io_handler.assert_called_once()
    configurator._initialize_output.assert_not_called()
    configurator._get_db_parameters.assert_called_once()
    configurator._get_db_parameters.reset_mock()

    # Assert that activity_id and label are set
    assert "activity_id" in config
    assert config["label"] == configurator.label
    assert db_dict == {"db_param": "test"}

    # Call initialize with custom parameters
    configurator.initialize(
        require_command_line=False,
        paths=False,
        output=True,
        simulation_model=["site"],
        simulation_configuration={"test": "test"},
        db_config=True,
    )

    # Assert that the methods were called with the correct parameters
    configurator.parser.initialize_default_arguments.assert_called()
    configurator._get_cli_arglist.assert_called()
    configurator._config_from_env.assert_called()
    configurator._config_from_file.assert_called()
    configurator._fill_config.assert_called()
    configurator._initialize_model_versions.assert_called()
    configurator._initialize_io_handler.assert_called()
    configurator._initialize_output.assert_called_once()
    configurator._get_db_parameters.assert_called_once()

    # test activity_id and label
    configurator.config_class_init = {"activity_id": "test_activity_id", "label": "test_label"}
    configurator._fill_config.side_effect = lambda _: configurator.config.update(
        {"activity_id": "test_activity_id", "label": "test_label"}
    )
    configurator.initialize()
    assert configurator.config["activity_id"] == "test_activity_id"
    assert configurator.config["label"] == "test_label"
