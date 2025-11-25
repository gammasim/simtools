#!/usr/bin/python3

import logging
import os
from copy import copy
from pathlib import Path
from unittest.mock import MagicMock

import astropy.units as u
import pytest
import yaml

from simtools import settings
from simtools.configuration.configurator import (
    Configurator,
    InvalidConfigurationParameterError,
)
from simtools.io import io_handler

logger = logging.getLogger()


@pytest.fixture
def configurator(tmp_test_directory, _mock_settings_env_vars):
    config = Configurator()
    config.default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--simtel_path",
            str(settings.config.sim_telarray_path),
        )
    )
    return config


def test_fill_from_command_line(configurator, args_dict):
    assert args_dict == configurator.config
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)
    assert args_dict == configurator.config

    with pytest.raises(SystemExit):
        configurator._fill_from_command_line(arg_list=[], require_command_line=True)

    configurator._fill_from_command_line(arg_list=["--data_path", Path("abc")])
    _tmp_config = copy(dict(args_dict))
    _tmp_config["data_path"] = Path("abc")
    assert _tmp_config == configurator.config

    with pytest.raises(SystemExit):
        configurator._fill_from_command_line(arg_list=["--data_pth", Path("abc")])

    with pytest.raises(SystemExit):
        configurator._fill_from_command_line(arg_list=None)

    configurator._fill_from_command_line(arg_list=["--config", Path("abc")])
    assert configurator.config.get("config") == "abc"


def test_fill_from_config_dict(configurator, args_dict):
    # _fill_from_environmental_variables() is always called after _fill_from_command_line()
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)

    configurator._fill_from_config_dict({})
    assert args_dict == configurator.config

    _tmp_config = copy(dict(args_dict))
    _tmp_config["config"] = "my_file"
    _tmp_config["test"] = True
    configurator._fill_from_config_dict({"config": "my_file", "test": True})

    assert _tmp_config == configurator.config

    # No AttributeError is raised for non-dict inputs
    configurator._fill_from_config_dict("abc")

    # No error raise for None arguments
    configurator._fill_from_config_dict(None)


def test_fill_from_config_file_not_existing_file(configurator):
    # _fill_from_config_file() is always called after _fill_from_command_line()
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)

    # config_file not found raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        configurator._fill_from_config_file(config_file="this_file_does_not_exist")


def test_fill_from_config_file(configurator, args_dict, tmp_test_directory):
    _tmp_config = copy(dict(args_dict))
    _tmp_dict = {
        "output_path": "./abc/",
        "test": True,
    }
    _config_file = tmp_test_directory / "configuration-test.yml"
    with open(_config_file, "w") as output:
        yaml.safe_dump(_tmp_dict, output, sort_keys=False)

    configurator.config["config"] = str(_config_file)
    _tmp_config["config"] = str(_config_file)
    configurator.config["output_path"] = None
    configurator._fill_from_config_file(_config_file)
    for key, value in _tmp_dict.items():
        # none values are explicitly not set in Configurator._arglist_from_config()
        if value is not None:
            if "_path" in key:
                _tmp_config[key] = Path(value)
            else:
                _tmp_config[key] = value
    assert _tmp_config == configurator.config

    # test that no error is raised
    configurator._fill_from_config_file(config_file=None)
    with pytest.raises(FileNotFoundError):
        configurator._fill_from_config_file(config_file="abc")


def test_fill_from_workflow_config_file(configurator, args_dict, tmp_test_directory):
    _tmp_config = copy(dict(args_dict))
    _tmp_dict = {
        "output_path": "./abc/",
        "test": True,
    }
    _tmp_dict_workflow = {"applications": [{"application": "test", "configuration": _tmp_dict}]}
    _workflow_file = tmp_test_directory / "configuration-test.yml"
    with open(_workflow_file, "w") as output:
        yaml.safe_dump(_tmp_dict_workflow, output, sort_keys=False)
    configurator.config["config"] = str(_workflow_file)
    _tmp_config["config"] = str(_workflow_file)
    configurator.config["output_path"] = None
    configurator._fill_from_config_file(_workflow_file)
    for key, value in _tmp_dict.items():
        # none values are explicitly not set in Configurator._arglist_from_config()
        if value is not None:
            if "_path" in key:
                _tmp_config[key] = Path(value)
            else:
                _tmp_config[key] = value
    assert _tmp_config == configurator.config


def test_initialize_io_handler(configurator, tmp_test_directory):
    # io_handler is a Singleton, so configurator changes should
    # be reflected in the io_handler
    _io_handler = io_handler.IOHandler()

    configurator.config["output_path"] = tmp_test_directory
    configurator._initialize_io_handler()

    assert _io_handler.output_path == tmp_test_directory


def test_check_parameter_configuration_status(configurator, args_dict, tmp_test_directory):
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)
    configurator.config["output_path"] = Path(tmp_test_directory)

    # default value (no change)
    configurator._check_parameter_configuration_status("data_path", args_dict["data_path"])
    assert args_dict == configurator.config

    # None value
    configurator._check_parameter_configuration_status("config", None)
    assert args_dict == configurator.config

    # parameter changed; should raise Exception
    configurator.config["config"] = "non_default_config_file"

    with pytest.raises(InvalidConfigurationParameterError):
        configurator._check_parameter_configuration_status("config", "abc")


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
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)
    configurator._fill_from_environmental_variables()

    args_dict["db_api_user"] = "db_user"
    args_dict["db_api_pw"] = "12345"
    args_dict["db_api_port"] = 42
    args_dict["db_server"] = "abc@def.de"
    args_dict["db_simulation_model"] = "sim_model"
    args_dict["db_simulation_model_version"] = "v0.0.0"

    # remove user defined parameters from comparison (depends on environment)
    expected_config = {k: v for k, v in args_dict.items() if not k.startswith("user_")}
    actual_config = {k: v for k, v in configurator.config.items() if not k.startswith("user_")}
    actual_config.pop("db_api_authentication_database")  # depends on user setup; ignore here

    assert expected_config == actual_config


def test_initialize_output(configurator):
    configurator.parser.initialize_output_arguments()
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)

    # output file for testing
    configurator.config["test"] = True
    configurator._initialize_output()
    assert configurator.config["output_file"] == "TEST.ecsv"

    # output is not configured (and not activity_id)
    configurator.config["test"] = False
    configurator.config["output_file"] = None
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


def test_fill_from_environmental_variables(configurator):
    configurator.parser.initialize_output_arguments()
    configurator.parser.initialize_db_config_arguments()
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)

    _config_save = copy(configurator.config)

    # this is not a configuration parameter and therefore should not be set
    os.environ["SIMTOOLS_TEST_ENV_VARIABLE"] = "test_value"
    configurator._fill_from_environmental_variables()
    if "SIMTOOLS_TEST_ENV_VARIABLE" in os.environ:
        del os.environ["SIMTOOLS_TEST_ENV_VARIABLE"]
    assert "test_env_variable" not in configurator.config

    # this is a valid configuration parameter, but already configured
    # _fill_from_environmental_variables() should not change it
    os.environ["SIMTOOLS_LOG_LEVEL"] = "DEBUG"
    configurator._fill_from_environmental_variables()
    assert configurator.config["log_level"] == _config_save["log_level"] == "info"
    if "SIMTOOLS_LOG_LEVEL" in os.environ:
        del os.environ["SIMTOOLS_LOG_LEVEL"]

    # this is a valid configuration parameter, but not yet configured
    os.environ["SIMTOOLS_CONFIG"] = "test_config_file"
    configurator._fill_from_environmental_variables()
    assert configurator.config["config"] == "test_config_file"
    if "SIMTOOLS_CONFIG" in os.environ:
        del os.environ["SIMTOOLS_CONFIG"]

    # using .dotenv files with docker: comments are not removed by docker
    os.environ["SIMTOOLS_DB_API_PORT"] = "27017 #Port on the MongoDB server"
    os.environ["SIMTOOLS_DB_SERVER"] = "'abc@def.de' # MongoDB server"
    configurator.config["db_api_port"] = None
    configurator.config["db_server"] = None
    configurator._fill_from_environmental_variables()
    assert configurator.config["db_api_port"] == 27017
    assert configurator.config["db_server"] == "abc@def.de"
    if "SIMTOOLS_DB_API_PORT" in os.environ:
        del os.environ["SIMTOOLS_DB_API_PORT"]
    if "SIMTOOLS_DB_SERVER" in os.environ:
        del os.environ["SIMTOOLS_DB_SERVER"]

    # no config defined, should not raise key error
    configurator.config.pop("env_file")
    configurator._fill_from_environmental_variables()


def test_fill_from_environmental_variables_with_dotenv_file(configurator, tmp_test_directory):
    configurator.parser.initialize_output_arguments()
    configurator._fill_from_command_line(arg_list=[], require_command_line=False)

    # write a temporary file into tmp_test_directory with the environmental variables
    _env_file = tmp_test_directory / "test_env_file"
    with open(_env_file, "w") as output:
        output.write("SIMTOOLS_LABEL=test_label\n")
        output.write("SIMTOOLS_CONFIG=test_config_file_env\n")

    configurator.config["env_file"] = str(_env_file)
    configurator._fill_from_environmental_variables()

    assert configurator.config["label"] == "test_label"
    assert configurator.config["config"] == "test_config_file_env"


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
    configurator._fill_from_command_line = MagicMock()
    configurator._fill_from_config_file = MagicMock()
    configurator._fill_from_config_dict = MagicMock()
    configurator._fill_from_environmental_variables = MagicMock()
    configurator._initialize_model_versions = MagicMock()
    configurator._initialize_io_handler = MagicMock()
    configurator._initialize_output = MagicMock()
    configurator._get_db_parameters = MagicMock(return_value={"db_param": "test"})

    # Call initialize with default parameters
    config, db_dict = configurator.initialize()

    # Assert that the methods were called
    configurator.parser.initialize_default_arguments.assert_called_once()
    configurator._fill_from_command_line.assert_called_once_with(require_command_line=True)
    configurator._fill_from_config_file.assert_called_once_with(None)
    configurator._fill_from_config_dict.assert_called_once_with(None)
    configurator._fill_from_environmental_variables.assert_called_once()
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
    configurator._fill_from_command_line.assert_called()
    configurator._fill_from_config_file.assert_called()
    configurator._fill_from_config_dict.assert_called()
    configurator._fill_from_environmental_variables.assert_called()
    configurator._initialize_model_versions.assert_called()
    configurator._initialize_io_handler.assert_called()
    configurator._initialize_output.assert_called_once()
    configurator._get_db_parameters.assert_called_once()

    # test activity_id and label
    configurator.config["activity_id"] = "test_activity_id"
    configurator.config["label"] = "test_label"
    configurator.initialize()
    assert configurator.config["activity_id"] == "test_activity_id"
    assert configurator.config["label"] == "test_label"
