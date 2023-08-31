#!/usr/bin/python3

import logging
import os
from copy import copy
from pathlib import Path

import pytest
import yaml

from simtools.configuration.configurator import (
    Configurator,
    InvalidConfigurationParameter,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_fill_from_command_line(configurator, args_dict):
    configurator._fill_from_command_line(arg_list=[])
    assert args_dict == configurator.config

    configurator._fill_from_command_line(arg_list=["--data_path", Path("abc")])
    _tmp_config = copy(dict(args_dict))
    _tmp_config["data_path"] = Path("abc")
    assert _tmp_config == configurator.config

    with pytest.raises(SystemExit):
        configurator._fill_from_command_line(arg_list=["--data_pth", Path("abc")])


def test_fill_from_config_dict(configurator, args_dict):
    # _fill_from_environmental_variables() is always called after _fill_from_command_line()
    configurator._fill_from_command_line(arg_list=[])

    configurator._fill_from_config_dict({})
    assert args_dict == configurator.config

    _tmp_config = copy(dict(args_dict))
    _tmp_config["config"] = "my_file"
    _tmp_config["test"] = True
    configurator._fill_from_config_dict({"config": "my_file", "test": True})

    assert _tmp_config == configurator.config


def test_fill_from_config_file_not_existing_file(configurator):
    # _fill_from_config_file() is always called after _fill_from_command_line()
    configurator._fill_from_command_line(arg_list=[])

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
        # none values are explicitely not set in Configurator._arglist_from_config()
        if value is not None:
            if "_path" in key:
                _tmp_config[key] = Path(value)
            else:
                _tmp_config[key] = value
    assert _tmp_config == configurator.config


def test_fill_from_workflow_config_file(configurator, args_dict, tmp_test_directory):
    _tmp_config = copy(dict(args_dict))
    _tmp_dict = {
        "output_path": "./abc/",
        "test": True,
    }
    _tmp_dict_workflow = {"CTASIMPIPE": {"CONFIGURATION": _tmp_dict}}
    _workflow_file = tmp_test_directory / "configuration-test.yml"
    with open(_workflow_file, "w") as output:
        yaml.safe_dump(_tmp_dict_workflow, output, sort_keys=False)
    configurator.config["config"] = str(_workflow_file)
    _tmp_config["config"] = str(_workflow_file)
    configurator.config["output_path"] = None
    configurator._fill_from_config_file(_workflow_file)
    for key, value in _tmp_dict.items():
        # none values are explicitely not set in Configurator._arglist_from_config()
        if value is not None:
            if "_path" in key:
                _tmp_config[key] = Path(value)
            else:
                _tmp_config[key] = value
    assert _tmp_config == configurator.config


def test_check_parameter_configuration_status(configurator, args_dict, tmp_test_directory):
    configurator._fill_from_command_line(arg_list=[])
    configurator.config["output_path"] = Path(tmp_test_directory)

    # default value (no change)
    configurator._check_parameter_configuration_status("data_path", args_dict["data_path"])
    assert args_dict == configurator.config

    # None value
    configurator._check_parameter_configuration_status("config", None)
    assert args_dict == configurator.config

    # parameter changed; should raise Exception
    configurator.config["config"] = "non_default_config_file"

    with pytest.raises(InvalidConfigurationParameter):
        configurator._check_parameter_configuration_status("config", "abc")


def test_arglist_from_config():
    _tmp_dict = {"a": 1.0, "b": None, "c": True, "d": ["d1", "d2", "d3"]}

    assert ["--a", "1.0", "--c", "--d", "d1", "d2", "d3"] == Configurator._arglist_from_config(
        _tmp_dict
    )

    assert [] == Configurator._arglist_from_config({})

    assert [] == Configurator._arglist_from_config(None)
    assert [] == Configurator._arglist_from_config(5.0)

    assert ["--a", "1.0", "--b", "None", "--c"] == Configurator._arglist_from_config(
        ["--a", "1.0", "--b", None, "--c"]
    )


def test_convert_stringnone_to_none():
    assert {} == Configurator._convert_stringnone_to_none({})

    _tmp_dict = {
        "a": 1.0,
        "b": None,
        "c": True,
        "d": "None",
    }
    _tmp_none = copy(_tmp_dict)
    _tmp_none["d"] = None

    assert _tmp_none == Configurator._convert_stringnone_to_none(_tmp_dict)


def test_get_db_parameters(configurator, args_dict):
    configurator.parser.initialize_db_config_arguments()
    configurator._fill_from_command_line(arg_list=[])
    configurator._fill_from_environmental_variables()

    args_dict["db_api_user"] = "db_user"
    args_dict["db_api_pw"] = "12345"
    args_dict["db_api_port"] = 42
    args_dict["db_server"] = "abc@def.de"
    args_dict["db_api_authentication_database"] = "admin"

    assert configurator.config == args_dict


def test_initialize_output(configurator):
    configurator.parser.initialize_output_arguments()
    configurator._fill_from_command_line(arg_list=[])

    # outputfile for testing
    configurator.config["test"] = True
    configurator._initialize_output()
    assert configurator.config["output_file"] == "TEST.ecsv"

    # output file is configured
    configurator.config["test"] = False
    configurator.config["output_file"] = "unit_test.txt"
    configurator._initialize_output()
    assert configurator.config["output_file"] == "unit_test.txt"


def test_fill_from_environmental_variables(configurator):
    configurator.parser.initialize_output_arguments()
    configurator._fill_from_command_line(arg_list=[])

    _config_save = copy(configurator.config)

    # this is not a configuration parameter and therefore should not be set
    os.environ["TEST_ENV_VARIABLE"] = "test_value"
    configurator._fill_from_environmental_variables()
    if "TEST_ENV_VARIABLE" in os.environ:
        del os.environ["TEST_ENV_VARIABLE"]
    assert "test_env_variable" not in configurator.config

    # this is a valid configuration parameter, but already configured
    # _fill_from_environmental_variables() should not change it
    os.environ["LOG_LEVEL"] = "DEBUG"
    configurator._fill_from_environmental_variables()
    assert configurator.config["log_level"] == _config_save["log_level"] == "info"
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]

    # this is a valid configuration parameter, but not yet configured
    os.environ["CONFIG"] = "test_config_file"
    configurator._fill_from_environmental_variables()
    assert configurator.config["config"] == "test_config_file"
    if "CONFIG" in os.environ:
        del os.environ["CONFIG"]


def test_fill_from_environmental_variables_with_dotenv_file(configurator, tmp_test_directory):
    configurator.parser.initialize_output_arguments()
    configurator._fill_from_command_line(arg_list=[])

    # write a temporary file into tmp_test_directory with the environmental variables
    _env_file = tmp_test_directory / "test_env_file"
    with open(_env_file, "w") as output:
        output.write("SIMTOOLS_LABEL=test_label\n")
        output.write("SIMTOOLS_CONFIG=test_config_file_env\n")

    configurator.config["env_file"] = str(_env_file)
    configurator._fill_from_environmental_variables()

    assert configurator.config["label"] == "test_label"
    assert configurator.config["config"] == "test_config_file_env"
