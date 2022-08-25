#!/usr/bin/python3

import logging
from copy import copy
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from simtools.shower_simulator import (
    InvalidRunsToSimulate,
    MissingRequiredEntryInShowerConfig,
    ShowerSimulator,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def label():
    return "test-shower-simulator"


@pytest.fixture
def showerConfigData():
    return {
        "dataDirectory": ".",
        "site": "South",
        "layoutName": "Prod5",
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
def showerSimulator(cfg_setup, label, showerConfigData):

    showerSimulator = ShowerSimulator(label=label, showerConfigData=showerConfigData)
    return showerSimulator


def test_invalid_shower_data(cfg_setup, showerConfigData, label):

    newShowerConfigData = copy(showerConfigData)
    newShowerConfigData.pop("site")

    with pytest.raises(MissingRequiredEntryInShowerConfig):
        newShowerSimulator = ShowerSimulator(label=label, showerConfigData=newShowerConfigData)
        newShowerSimulator.run()


def test_runs_invalid_input(cfg_setup, showerConfigData, label):

    newShowerConfigData = copy(showerConfigData)
    newShowerConfigData["runList"] = [1, 2.5, "bla"]  # Invalid run list

    with pytest.raises(InvalidRunsToSimulate):
        newShowerSimulator = ShowerSimulator(label=label, showerConfigData=newShowerConfigData)
        newShowerSimulator.run()


def test_runs_input(cfg_setup, showerConfigData, label):

    newShowerConfigData = copy(showerConfigData)
    newShowerConfigData["runList"] = [1, 2, 4]
    newShowerConfigData["runRange"] = [5, 8]
    newShowerSimulator = ShowerSimulator(label=label, showerConfigData=newShowerConfigData)

    assert newShowerSimulator.runs == [1, 2, 4, 5, 6, 7, 8]

    # With overlap
    newShowerConfigData["runList"] = [1, 3, 4]
    newShowerConfigData["runRange"] = [3, 7]
    newShowerSimulator = ShowerSimulator(label=label, showerConfigData=newShowerConfigData)

    assert newShowerSimulator.runs == [1, 3, 4, 5, 6, 7]


def test_no_corsika_data(cfg_setup, showerConfigData, label):

    newShowerConfigData = copy(showerConfigData)
    newShowerConfigData.pop("dataDirectory", None)
    newShowerSimulator = ShowerSimulator(label=label, showerConfigData=newShowerConfigData)
    files = newShowerSimulator.getListOfOutputFiles(runList=[3])
    print(files)

    assert "/" + label + "/" in files[0]


def test_submitting(showerSimulator):

    showerSimulator.submit(runList=[2], submitCommand="local")

    run_script = showerSimulator._corsikaRunner.getRunScriptFile(runNumber=2)

    assert Path(run_script).exists()


def test_runs_range(showerSimulator):

    showerSimulator.submit(runRange=[4, 8], submitCommand="local")

    run_range = np.arange(4, 8)
    for run in run_range:
        run_script = showerSimulator._corsikaRunner.getRunScriptFile(runNumber=run)

        assert Path(run_script).exists()


def test_get_list_of_files(showerSimulator):

    files = showerSimulator.getListOfOutputFiles()

    assert len(files) == len(showerSimulator.runs)

    files = showerSimulator.getListOfOutputFiles(runList=[2, 5, 7])

    assert len(files) == 3
