#!/usr/bin/python3

import logging
from copy import copy
from pathlib import Path

import pytest
import yaml

from simtools.configuration import Configurator, InvalidConfigurationParameter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_fillFromCommandLine(configurator, args_dict):

    configurator._fillFromCommandLine(arg_list=[])
    assert args_dict == configurator.config

    configurator._fillFromCommandLine(arg_list=["--data_path", Path("abc")])
    _tmp_config = copy(dict(args_dict))
    _tmp_config["data_path"] = Path("abc")
    assert _tmp_config == configurator.config

    with pytest.raises(SystemExit):
        configurator._fillFromCommandLine(arg_list=["--data_pth", Path("abc")])


def test_fillFromConfigDict(configurator, args_dict):

    # _fillFromEnvironmentalVariables() is always called after _fillFromCommandLine()
    configurator._fillFromCommandLine(arg_list=[])

    configurator._fillFromConfigDict({})
    assert args_dict == configurator.config

    _tmp_config = copy(dict(args_dict))
    _tmp_config["config_file"] = "my_file"
    _tmp_config["test"] = True
    configurator._fillFromConfigDict({"config_file": "my_file", "test": True})

    assert _tmp_config == configurator.config


def test_fillFromEnvironmentalVariables(configurator, args_dict):

    # _fillFromEnvironmentalVariables() is always called after _fillFromCommandLine()
    configurator._fillFromCommandLine(arg_list=[])
    configurator._fillFromEnvironmentalVariables()
    assert args_dict == configurator.config


def test_fillFromConfigFile_not_existing_file(configurator):

    # _fillFromConfigFile() is always called after _fillFromCommandLine()
    configurator._fillFromCommandLine(arg_list=[])

    with pytest.raises(FileNotFoundError):
        configurator._fillFromConfigFile("this_file_does_not_exist")


def test_fillFromConfigFile(configurator, args_dict, tmp_test_directory):

    _tmp_config = copy(dict(args_dict))
    _tmp_dict = {
        "output_path": "./abc/",
        "test": True,
    }
    _config_file = tmp_test_directory / "configuration-test.yml"
    with open(_config_file, "w") as output:
        yaml.safe_dump(_tmp_dict, output, sort_keys=False)

    configurator.config["config_file"] = str(_config_file)
    _tmp_config["config_file"] = str(_config_file)
    configurator.config["output_path"] = None
    configurator._fillFromConfigFile(_config_file)
    for key, value in _tmp_dict.items():
        # none values are explicitely not set in Configurator._arglistFromConfig()
        if value is not None:
            if "_path" in key:
                _tmp_config[key] = Path(value)
            else:
                _tmp_config[key] = value
    assert _tmp_config == configurator.config


def test_fillFromWorkflowConfigFile(configurator, args_dict, tmp_test_directory):

    _tmp_config = copy(dict(args_dict))
    _tmp_dict = {
        "output_path": "./abc/",
        "test": True,
    }
    _tmp_dict_workflow = {"CTASIMPIPE": {"CONFIGURATION": _tmp_dict}}
    _workflow_file = tmp_test_directory / "configuration-test.yml"
    with open(_workflow_file, "w") as output:
        yaml.safe_dump(_tmp_dict_workflow, output, sort_keys=False)
    configurator.config["config_file"] = str(_workflow_file)
    _tmp_config["config_file"] = str(_workflow_file)
    configurator.config["output_path"] = None
    configurator._fillFromConfigFile(_workflow_file)
    for key, value in _tmp_dict.items():
        # none values are explicitely not set in Configurator._arglistFromConfig()
        if value is not None:
            if "_path" in key:
                _tmp_config[key] = Path(value)
            else:
                _tmp_config[key] = value
    assert _tmp_config == configurator.config


def test_check_parameter_configuration_status(configurator, args_dict, tmp_test_directory):

    configurator._fillFromCommandLine(arg_list=[])
    configurator.config["output_path"] = Path(tmp_test_directory)

    # default value (no change)
    configurator._check_parameter_configuration_status("data_path", args_dict["data_path"])
    assert args_dict == configurator.config

    # None value
    configurator._check_parameter_configuration_status("config_file", None)
    assert args_dict == configurator.config

    # parameter changed; should raise Exception
    configurator.config["config_file"] = "non_default_config_file"

    with pytest.raises(InvalidConfigurationParameter):
        configurator._check_parameter_configuration_status("config_file", "abc")


def test_arglistFromConfig():

    _tmp_dict = {
        "a": 1.0,
        "b": None,
        "c": True,
    }

    assert ["--a", "1.0", "--c"] == Configurator._arglistFromConfig(_tmp_dict)

    assert [] == Configurator._arglistFromConfig({})

    assert [] == Configurator._arglistFromConfig(None)
    assert [] == Configurator._arglistFromConfig(5.0)

    assert ["--a", "1.0", "--b", "None", "--c"] == Configurator._arglistFromConfig(
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


def test_getDBParameters(configurator, args_dict):

    configurator.parser.initialize_db_config_arguments()
    configurator._fillFromCommandLine(arg_list=[])
    configurator._fillFromEnvironmentalVariables()

    args_dict["db_api_user"] = "db_user"
    args_dict["db_api_pw"] = "12345"
    args_dict["db_api_port"] = 42
    args_dict["db_api_name"] = "abc@def.de"

    assert configurator.config == args_dict
