#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

import simtools.util.general as gen
from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def corsika_config_data():
    return {
        "data_directory": "./corsika-data",
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
def corsika_runner(corsika_config_data, io_handler, simtelpath):

    corsika_runner = CorsikaRunner(
        site="south",
        layout_name="test-layout",
        simtel_source_path=simtelpath,
        label="test-corsika-runner",
        corsika_config_data=corsika_config_data,
    )
    return corsika_runner


def test_get_run_script(corsika_runner):
    # No run number is given

    script = corsika_runner.get_run_script()

    assert script.exists()

    # Run number is given
    run_number = 3
    script = corsika_runner.get_run_script(run_number=run_number)

    assert script.exists()


def test_get_run_script_with_invalid_run(corsika_runner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_runner.get_run_script(run_number=run)


def test_run_script_with_extra(corsika_runner):

    extra = ["testing", "testing-extra-2"]
    script = corsika_runner.get_run_script(run_number=3, extra_commands=extra)

    assert gen.file_has_text(script, "testing-extra-2")
