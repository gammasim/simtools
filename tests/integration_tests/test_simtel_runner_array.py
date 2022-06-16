#!/usr/bin/python3

import pytest
import logging

import astropy.units as u

import simtools.io_handler as io
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.model.array_model import ArrayModel


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
def arrayModel(arrayConfigData, cfg_setup, set_db):
    arrayModel = ArrayModel(
        label="test-lst-array", arrayConfigData=arrayConfigData
    )
    return arrayModel


@pytest.fixture
def simtelRunner(arrayModel):
    simtelRunner = SimtelRunnerArray(
        arrayModel=arrayModel,
        configData={
            "primary": "proton",
            "zenithAngle": 20 * u.deg,
            "azimuthAngle": 0 * u.deg,
        },
    )
    return simtelRunner


@pytest.fixture
def corsikaFile():
    corsikaFile = io.getTestDataFile(
        "run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst"
    )
    return corsikaFile


def test_run(set_simtelarray, simtelRunner, corsikaFile):
    simtelRunner.run(test=False, force=True, inputFile=corsikaFile, run=1)
