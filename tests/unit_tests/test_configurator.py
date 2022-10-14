#!/usr/bin/python3

import logging
import os
from copy import copy
from unittest import mock

import pytest
import yaml

from simtools.configuration import Configurator, InvalidConfigurationParameter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_fillFromCommandLine(configurator, args_dict):

    configurator._fillFromCommandLine(arg_list=[])
    assert args_dict == configurator.config

    configurator._fillFromCommandLine(arg_list=["--data_path", "abc"])
    _tmp_config = copy(dict(args_dict))
    _tmp_config["data_path"] = "abc"
    assert _tmp_config == configurator.config

    with pytest.raises(SystemExit):
        configurator._fillFromCommandLine(arg_list=["--data_pth", "abc"])


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


@mock.patch.dict(os.environ, {"SIMTELPATH": "./a_path/", "DATA_PATH": "./b_path"}, clear=True)
def test_fillFromEnvironmentalVariables(configurator, args_dict):

    # _fillFromEnvironmentalVariables() is always called after _fillFromCommandLine()
    configurator._fillFromCommandLine(arg_list=[])

    # expect here:
    # - simtelpath is None in default config; value is set from environmental variable
    # - data_path is set in default config; value is not set from environemtal variable
    configurator._fillFromEnvironmentalVariables()
    _tmp_config = copy(dict(args_dict))
    _tmp_config["simtelpath"] = "./a_path/"
    assert _tmp_config == configurator.config


def test_fillFromConfigFile_not_existing_file(configurator):

    # _fillFromConfigFile() is always called after _fillFromCommandLine()
    configurator._fillFromCommandLine(arg_list=[])
    configurator.config["config_file"] = "this_file_does_not_exist"

    with pytest.raises(FileNotFoundError):
        configurator._fillFromConfigFile()


def test_fillFromConfigFile(configurator, args_dict, tmp_test_directory):

    _tmp_config = copy(dict(args_dict))
    _tmp_dict = {
        "output_path": "./abc/",
        "mongodb_config_file": None,
        "test": True,
    }
    _config_file = tmp_test_directory / "configuration-test.yml"
    with open(_config_file, "w") as output:
        yaml.safe_dump(_tmp_dict, output, sort_keys=False)

    configurator.config["config_file"] = _config_file
    # need to reset output path to the default value
    # (otherwise _check_parameter_configuration_status will complain
    #  about a conflicting change)
    configurator.config["output_path"] = "./"
    _tmp_config["config_file"] = _config_file
    for key, value in _tmp_dict.items():
        # none values are explicitely not set in Configurator._arglistFromConfig()
        if value is not None:
            _tmp_config[key] = value

    configurator._fillFromConfigFile()
    assert _tmp_config == configurator.config


def test_check_parameter_configuration_status(configurator, args_dict, tmp_test_directory):

    configurator._fillFromCommandLine(arg_list=[])
    configurator.config["output_path"] = str(tmp_test_directory)

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
