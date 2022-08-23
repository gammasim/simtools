#!/usr/bin/python3

import logging
from copy import copy
from pathlib import Path

import astropy.units as u
import pytest

from simtools.array_simulator import ArraySimulator, MissingRequiredEntryInArrayConfig

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def label():
    return "test-array-simulator"


@pytest.fixture
def arrayConfigData(tmp_test_directory):
    return {
        "dataDirectory": str(tmp_test_directory) + "/test-output",
        "primary": "gamma",
        "zenith": 20 * u.deg,
        "azimuth": 0 * u.deg,
        "viewcone": [0 * u.deg, 0 * u.deg],
        # ArrayModel
        "site": "North",
        "layoutName": "1LST",
        "modelVersion": "Prod5",
        "default": {"LST": "1"},
        "M-01": "FlashCam-D",
    }


@pytest.fixture
def corsikaFile():
    return "run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst"


@pytest.fixture
def array_simulator(set_db, label, arrayConfigData):

    arraySimulator = ArraySimulator(label=label, configData=arrayConfigData)
    return arraySimulator


def test_guess_run(array_simulator):

    run = array_simulator._guessRunFromFile("run12345_bla_ble")
    assert run == 12345

    # Invalid run number - returns 1
    run = array_simulator._guessRunFromFile("run1test2_bla_ble")
    assert run == 1


def test_invalid_array_data(cfg_setup, arrayConfigData, label):

    newArrayConfigData = copy(arrayConfigData)
    newArrayConfigData.pop("site")

    with pytest.raises(MissingRequiredEntryInArrayConfig):
        ArraySimulator(label=label, configData=newArrayConfigData)


def test_run(array_simulator, corsikaFile):

    array_simulator.run(inputFileList=corsikaFile)

    assert len(array_simulator._results["output"])
    assert Path(array_simulator._results["output"][0]).parents[0].exists()

    assert len(array_simulator._results["log"])
    assert Path(array_simulator._results["log"][0]).parents[0].exists()


def test_submitting(array_simulator, corsikaFile):

    array_simulator.submit(inputFileList=corsikaFile, submitCommand="more ")
    # TODO - add a test here


def test_list_of_files(array_simulator, corsikaFile):

    array_simulator.submit(inputFileList=corsikaFile, submitCommand="more ", test=True)

    array_simulator.printListOfOutputFiles()
    array_simulator.printListOfLogFiles()
    array_simulator.printListOfInputFiles()

    inputFiles = array_simulator.getListOfInputFiles()
    assert str(inputFiles[0]) == str(corsikaFile)
