#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

import simtools.util.general as gen
from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def corsikaConfigData():
    return {
        "corsikaDataDirectory": "./corsika-data",
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
def corsikaRunner(corsikaConfigData, cfg_setup):

    corsikaRunner = CorsikaRunner(
        site="south",
        layoutName="Prod5",
        label="test-corsika-runner",
        corsikaConfigData=corsikaConfigData,
    )
    return corsikaRunner


def test_get_run_script(corsikaRunner):
    # No run number is given

    script = corsikaRunner.getRunScriptFile()

    assert script.exists()

    # Run number is given
    runNumber = 3
    script = corsikaRunner.getRunScriptFile(runNumber)

    assert script.exists()


def test_get_run_script_with_invalid_run(corsikaRunner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsikaRunner.getRunScriptFile(run)


def test_run_script_with_extra(corsikaRunner):

    extra = ["testing", "testing-extra-2"]
    script = corsikaRunner.getRunScriptFile(runNumber=3, extraCommands=extra)

    assert gen.fileHasText(script, "testing-extra-2")
