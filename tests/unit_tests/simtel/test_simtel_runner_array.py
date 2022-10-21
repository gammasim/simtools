#!/usr/bin/python3

import logging
from pathlib import Path

import astropy.units as u
import pytest

from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_runner_array import SimtelRunnerArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def arrayConfigData():
    return {
        "site": "North",
        "layoutName": "1LST",
        "modelVersion": "Prod5",
        "default": {"LST": "1"},
    }


@pytest.fixture
def arrayModel(arrayConfigData, io_handler, db_connection):
    arrayModel = ArrayModel(
        label="test-lst-array",
        arrayConfigData=arrayConfigData,
        mongoDBConfigFile=str(db_connection),
    )
    return arrayModel


@pytest.fixture
def simtelRunner(arrayModel, simtelpath):
    simtelRunner = SimtelRunnerArray(
        arrayModel=arrayModel,
        simtelSourcePath=simtelpath,
        configData={
            "primary": "proton",
            "zenithAngle": 20 * u.deg,
            "azimuthAngle": 0 * u.deg,
        },
    )
    return simtelRunner


@pytest.fixture
def corsikaFile(io_handler):
    corsikaFile = io_handler.getInputDataFile(
        fileName="run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst", test=True
    )
    return corsikaFile


def test_run_script(simtelRunner, corsikaFile):
    script = simtelRunner.getRunScript(runNumber=1, inputFile=corsikaFile)
    assert Path(script).exists()
