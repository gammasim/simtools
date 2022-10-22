#!/usr/bin/python3

import logging
import math
from copy import copy
from pathlib import Path

import astropy.units as u
import pytest

import simtools.util.general as gen
from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.simulator import InvalidRunsToSimulate, Simulator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def label():
    return "test-simulator"


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
        "MST-01": "FlashCam-D",
    }


@pytest.fixture
def input_file_list():
    return ["run1", "abc_run22", "def_run02_and"]


@pytest.fixture
def showerConfigData():
    return {
        "dataDirectory": ".",
        "site": "South",
        "layoutName": "test-layout",
        "runList": [3, 4],
        "runRange": [6, 10],
        "nshow": 10,
        "primary": "gamma",
        "erange": [100 * u.GeV, 1 * u.TeV],
        "eslope": -2,
        "zenith": 20 * u.deg,
        "azimuth": 0 * u.deg,
        "viewcone": 0 * u.deg,
        "cscat": [10, 1500 * u.m, 0],
    }


@pytest.fixture
def corsikaFile():
    return "run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst"


@pytest.fixture
def array_simulator(label, arrayConfigData, io_handler, db_config, simtelpath):

    arraySimulator = Simulator(
        label=label,
        simulator="simtel",
        simulatorSourcePath=simtelpath,
        configData=arrayConfigData,
        mongoDBConfig=db_config,
    )
    return arraySimulator


@pytest.fixture
def shower_simulator(label, showerConfigData, io_handler, simtelpath):

    showerSimulator = Simulator(
        label=label,
        simulator="corsika",
        simulatorSourcePath=simtelpath,
        configData=showerConfigData,
    )
    return showerSimulator


def test_guess_run_from_file(array_simulator):

    assert array_simulator._guessRunFromFile("run12345_bla_ble") == 12345

    # Invalid run number - returns 1
    assert array_simulator._guessRunFromFile("run1test2_bla_ble") == 1

    assert array_simulator._guessRunFromFile("abc-run12345_bla_ble") == 12345

    assert array_simulator._guessRunFromFile("run10") == 10

    assert array_simulator._guessRunFromFile("abc-ran12345_bla_ble") == 1

    # TODO add test for 'Error creating output directory'
    # (not sure how to test a failed mkdir


def test_setSimulator(array_simulator, shower_simulator):

    array_simulator._setSimulator("simtel")
    assert array_simulator.simulator == "simtel"

    shower_simulator._setSimulator("corsika")
    assert shower_simulator.simulator == "corsika"

    with pytest.raises(gen.InvalidConfigData):
        shower_simulator._setSimulator("this_simulator_is_not_there")


def test_loadConfigurationAndSimulationModel(array_simulator):

    with pytest.raises(gen.InvalidConfigData):
        array_simulator._loadConfigurationAndSimulationModel()


def test_loadCorsikaConfigAndModel(shower_simulator, showerConfigData):

    shower_simulator._loadCorsikaConfigAndModel(configData=showerConfigData)

    assert shower_simulator.site == "South"

    assert "site" not in shower_simulator._corsikaConfigData

    _temp_shower_data = copy(showerConfigData)
    _temp_shower_data.pop("site")
    with pytest.raises(KeyError):
        shower_simulator._loadCorsikaConfigAndModel(configData=_temp_shower_data)


def test_loadSimTelConfigAndModel(array_simulator, arrayConfigData):

    array_simulator._loadSimTelConfigAndModel(arrayConfigData)

    assert isinstance(array_simulator.arrayModel, ArrayModel)


def test_validateRunListAndRange(shower_simulator):

    assert not shower_simulator._validateRunListAndRange(None, None)

    run_list = [1, 24, 3]

    assert shower_simulator._validateRunListAndRange(runList=run_list, runRange=None) == [1, 3, 24]

    with pytest.raises(InvalidRunsToSimulate):
        shower_simulator._validateRunListAndRange(runList=[1, "a", 4], runRange=None)

    assert shower_simulator._validateRunListAndRange(runList=None, runRange=[3, 6]) == [3, 4, 5, 6]

    assert shower_simulator._validateRunListAndRange(runList=None, runRange=[6, 3]) == []

    with pytest.raises(InvalidRunsToSimulate):
        shower_simulator._validateRunListAndRange(runList=None, runRange=[3, "b"])

    with pytest.raises(InvalidRunsToSimulate):
        shower_simulator._validateRunListAndRange(runList=None, runRange=[3, 4, 5])


def test_collectArrayModelParameters(array_simulator, arrayConfigData):

    _arrayModelData, _restData = array_simulator._collectArrayModelParameters(
        configData=arrayConfigData
    )

    assert isinstance(_arrayModelData, dict)
    assert isinstance(_restData, dict)
    assert _arrayModelData["site"] == "North"
    assert _arrayModelData["MST-01"] == "FlashCam-D"
    newArrayConfigData = copy(arrayConfigData)
    newArrayConfigData.pop("site")

    with pytest.raises(KeyError):
        _, _ = array_simulator._collectArrayModelParameters(configData=newArrayConfigData)


def test_setSimulationRunner(array_simulator, shower_simulator):

    assert isinstance(array_simulator._simulationRunner, SimtelRunnerArray)

    assert isinstance(shower_simulator._simulationRunner, CorsikaRunner)


def test_fillResultsWithoutRun(array_simulator, input_file_list):

    array_simulator._fillResultsWithoutRun(inputFileList=[])
    assert array_simulator.runs == list()

    array_simulator._fillResultsWithoutRun(inputFileList=input_file_list)
    assert array_simulator.runs == [1, 22, 2]


def test_submitting(shower_simulator, array_simulator, corsikaFile):

    shower_simulator.test = True
    shower_simulator._submitCommand = "local"
    shower_simulator.simulate()

    run_script = shower_simulator._simulationRunner.getRunScript(runNumber=2)

    assert Path(run_script).exists()

    array_simulator._submitCommand = "local"
    array_simulator.simulate(inputFileList=corsikaFile)

    array_simulator.printListOfOutputFiles()
    array_simulator.printListOfLogFiles()
    array_simulator.printListOfInputFiles()

    inputFiles = array_simulator.getListOfInputFiles()
    assert str(inputFiles[0]) == str(corsikaFile)


def test_getRunsAndFilesToSubmit(array_simulator, shower_simulator, input_file_list):

    assert array_simulator._getRunsAndFilesToSubmit(inputFileList=None) == dict()

    assert array_simulator._getRunsAndFilesToSubmit(inputFileList=input_file_list) == {
        1: "run1",
        2: "def_run02_and",
        22: "abc_run22",
    }

    assert shower_simulator._getRunsAndFilesToSubmit(inputFileList=None) == {
        3: None,
        4: None,
        6: None,
        7: None,
        8: None,
        9: None,
        10: None,
    }


def test_enforceListType(array_simulator):

    assert array_simulator._enforceListType(None) == []

    assert array_simulator._enforceListType([1, 2, 3]) == [1, 2, 3]

    assert array_simulator._enforceListType(5) == [5]


def test_fillResults(array_simulator, shower_simulator, input_file_list):

    array_simulator._fillResultsWithoutRun(input_file_list)

    assert len(array_simulator._results["output"]) == 3
    assert len(array_simulator._results["sub_out"]) == 3
    assert len(array_simulator._results["log"]) == 3
    assert len(array_simulator._results["input"]) == 3
    assert len(array_simulator._results["hist"]) == 3
    assert array_simulator._results["input"][1] == "abc_run22"

    shower_simulator._fillResultsWithoutRun(input_file_list)
    assert len(shower_simulator._results["output"]) == 3
    assert shower_simulator._results["hist"][1] is None


def test_printHistograms(arrayConfigData, shower_simulator, input_file_list, db_config, simtelpath):

    _arraySimulator = Simulator(
        label="simtel_test",
        simulator="simtel",
        simulatorSourcePath=simtelpath,
        configData=arrayConfigData,
        mongoDBConfig=db_config,
    )

    assert len(str(_arraySimulator.printHistograms())) > 0

    _arraySimulator._results["hist"] = list()
    assert len(str(_arraySimulator.printHistograms(inputFileList=None))) > 0

    with pytest.raises(FileNotFoundError):
        _arraySimulator.printHistograms(inputFileList=input_file_list)

    assert shower_simulator.printHistograms() is None


def test_get_list_of_files(shower_simulator):

    assert len(shower_simulator.getListOfOutputFiles()) == len(shower_simulator.runs)
    assert len(shower_simulator.getListOfOutputFiles(runList=[2, 5, 7])) == 10
    assert len(shower_simulator.getListOfOutputFiles(runRange=[1, 4])) == 14


def test_no_corsika_data(showerConfigData, label, simtelpath, io_handler):

    newShowerConfigData = copy(showerConfigData)
    newShowerConfigData.pop("dataDirectory", None)
    newShowerSimulator = Simulator(
        label=label,
        simulator="corsika",
        configData=newShowerConfigData,
        simulatorSourcePath=simtelpath,
    )
    files = newShowerSimulator.getListOfOutputFiles(runList=[3])

    assert "/" + label + "/" in files[0]


def test_makeResourcesReport(array_simulator, input_file_list):

    _resources_1 = array_simulator._makeResourcesReport(inputFileList=None)
    assert math.isnan(_resources_1["Walltime/run [sec]"])

    with pytest.raises(FileNotFoundError):
        array_simulator._makeResourcesReport(input_file_list)


def test_getRunsToSimulate(showerConfigData, simtelpath, io_handler):

    showerSimulator = Simulator(
        label="corsika-test",
        simulator="corsika",
        configData=showerConfigData,
        simulatorSourcePath=simtelpath,
    )
    assert len(showerSimulator.runs) == len(
        showerSimulator._getRunsToSimulate(runList=None, runRange=None)
    )

    assert 3 == len(showerSimulator._getRunsToSimulate(runList=[2, 5, 7]))

    assert 4 == len(showerSimulator._getRunsToSimulate(runRange=[1, 4]))

    showerSimulator.runs = None
    assert showerSimulator._getRunsToSimulate() == list()


def test_printListOfFiles(array_simulator, input_file_list):

    array_simulator._fillResultsWithoutRun(input_file_list)
    with pytest.raises(KeyError):
        array_simulator._printListOfFiles("blabla")
    array_simulator._printListOfFiles("log")
