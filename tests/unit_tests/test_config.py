#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

import simtools.config as cfg
from simtools.config import ParameterNotFoundInConfigFile

logging.getLogger().setLevel(logging.DEBUG)


def test_get_parameters(cfg_setup, configuration_parameters):

    for key, value in configuration_parameters.items():
        assert value == cfg.get(key, False)


def test_get_non_existing_parameter(cfg_setup):

    message = "Configuration file does not contain an entry for the parameter NonExistingEntry"
    with pytest.raises(ParameterNotFoundInConfigFile, match=message):
        cfg.get("NonExistingEntry")


def test_input_options(tmp_test_directory):
    configfile = str(tmp_test_directory) + "/config-test.yml"
    cfg.setConfigFileName(configfile)
    assert configfile == cfg.CONFIG_FILE_NAME


def test_find_file(cfg_setup, tmp_test_directory):

    tmp_mirror_file = tmp_test_directory / "resources/mirror_MST_D80.dat"
    Path(tmp_mirror_file).touch()
    tmp_par_values = tmp_test_directory / "resources/parValues-LST.yml"
    Path(tmp_par_values).touch()
    files = ("mirror_MST_D80.dat", "parValues-LST.yml")
    for file in files:
        cfg.findFile(file)
