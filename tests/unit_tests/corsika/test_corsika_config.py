#!/usr/bin/python3

import logging
from copy import copy

import pytest
from astropy import units as u

import simtools.util.general as gen
from simtools.corsika.corsika_config import (
    CorsikaConfig,
    InvalidCorsikaInput,
    MissingRequiredInputInCorsikaConfigData,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def corsikaConfigData():
    return {
        "nshow": 100,
        "nrun": 10,
        "zenith": 20 * u.deg,
        "viewcone": 5 * u.deg,
        "erange": [10 * u.GeV, 10 * u.TeV],
        "eslope": -2,
        "phi": 0 * u.deg,
        "cscat": [10, 1500 * u.m, 0],
        "primary": "proton",
    }


@pytest.fixture
def corsikaConfig(io_handler, corsikaConfigData):

    corsikaConfig = CorsikaConfig(
        site="Paranal",
        layoutName="4LST",
        label="test-corsika-config",
        corsikaConfigData=corsikaConfigData,
    )
    return corsikaConfig


def test_repr(corsikaConfig):

    logger.info("test_repr")
    text = repr(corsikaConfig)

    assert "site" in text


def test_user_parameters(corsikaConfig):

    logger.info("test_user_parameters")

    assert corsikaConfig.getUserParameter("nshow") == 100
    assert corsikaConfig.getUserParameter("thetap") == [20, 20]
    assert corsikaConfig.getUserParameter("erange") == [10.0, 10000.0]
    # Testing conversion between AZM (sim_telarray) and PHIP (corsika)
    assert corsikaConfig.getUserParameter("azm") == [0.0, 0.0]
    assert corsikaConfig.getUserParameter("phip") == [180.0, 180.0]

    with pytest.raises(KeyError):
        corsikaConfig.getUserParameter("inexistent_par")


def test_export_input_file(corsikaConfig):

    logger.info("test_export_input_file")
    corsikaConfig.exportInputFile()
    inputFile = corsikaConfig.getInputFile()
    assert inputFile.exists()


def test_wrong_par_in_config_data(corsikaConfigData):

    logger.info("test_wrong_primary_name")
    newConfigData = copy(corsikaConfigData)
    newConfigData["wrong_par"] = 20 * u.m
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            site="LaPalma",
            layoutName="1LST",
            label="test-corsika-config",
            corsikaConfigData=newConfigData,
        )
        corsika_test_Config.printUserParameters()


def test_units_of_config_data(corsikaConfigData):

    logger.info("test_units_of_config_data")
    newConfigData = copy(corsikaConfigData)
    newConfigData["zenith"] = 20 * u.m
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            site="LaPalma",
            layoutName="1LST",
            label="test-corsika-config",
            corsikaConfigData=newConfigData,
        )
        corsika_test_Config.printUserParameters()


def test_len_of_config_data(corsikaConfigData):

    logger.info("test_len_of_config_data")
    newConfigData = copy(corsikaConfigData)
    newConfigData["erange"] = [20 * u.TeV]
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            site="LaPalma",
            layoutName="1LST",
            label="test-corsika-config",
            corsikaConfigData=newConfigData,
        )
        corsika_test_Config.printUserParameters()


def test_wrong_primary_name(corsikaConfigData):

    logger.info("test_wrong_primary_name")
    newConfigData = copy(corsikaConfigData)
    newConfigData["primary"] = "rock"
    with pytest.raises(InvalidCorsikaInput):
        corsika_test_Config = CorsikaConfig(
            site="LaPalma",
            layoutName="1LST",
            label="test-corsika-config",
            corsikaConfigData=newConfigData,
        )
        corsika_test_Config.printUserParameters()


def test_missing_input(corsikaConfigData):

    logger.info("test_missing_input")
    newConfigData = copy(corsikaConfigData)
    newConfigData.pop("primary")
    with pytest.raises(MissingRequiredInputInCorsikaConfigData):
        corsika_test_Config = CorsikaConfig(
            site="LaPalma",
            layoutName="1LST",
            label="test-corsika-config",
            corsikaConfigData=newConfigData,
        )
        corsika_test_Config.printUserParameters()


def test_set_user_parameters(corsikaConfigData, corsikaConfig):
    logger.info("test_set_user_parameters")
    newConfigData = copy(corsikaConfigData)
    newConfigData["zenith"] = 0 * u.deg
    newCorsikaConfig = copy(corsikaConfig)
    newCorsikaConfig.setUserParameters(newConfigData)

    assert newCorsikaConfig.getUserParameter("thetap") == [0, 0]


def test_config_data_from_yaml_file(db, io_handler):

    logger.info("test_config_data_from_yaml_file")
    testFileName = "corsikaConfigTest.yml"
    db.exportFileDB(
        dbName="test-data",
        dest=io_handler.getOutputDirectory(dirType="model", test=True),
        fileName=testFileName,
    )

    corsikaConfigFile = gen.findFile(
        testFileName, io_handler.getOutputDirectory(dirType="model", test=True)
    )
    cc = CorsikaConfig(
        site="Paranal",
        layoutName="4LST",
        label="test-corsika-config",
        corsikaConfigFile=corsikaConfigFile,
    )
    cc.printUserParameters()

    test_dict = {
        "RUNNR": [10],
        "NSHOW": [100],
        "PRMPAR": [14],
        "ERANGE": [0.01, 10.0],
        "ESLOPE": [-2],
        "THETAP": [20.0, 20.0],
        "AZM": [0.0, 0.0],
        "CSCAT": [10, 150000.0, 0],
        "VIEWCONE": [0.0, 5.0],
        "EVTNR": [1],
        "PHIP": [180.0, 180.0],
    }
    assert test_dict == cc._userParameters
