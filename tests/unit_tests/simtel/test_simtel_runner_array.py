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
def array_config_data():
    return {
        "site": "North",
        "layout_name": "1LST",
        "model_version": "Prod5",
        "default": {"LST": "1"},
    }


@pytest.fixture
def array_model(array_config_data, io_handler, db_config):
    array_model = ArrayModel(
        label="test-lst-array", array_config_data=array_config_data, mongo_db_config=db_config
    )
    return array_model


@pytest.fixture
def simtel_runner(array_model, simtelpath):
    simtel_runner = SimtelRunnerArray(
        array_model=array_model,
        simtel_source_path=simtelpath,
        config_data={
            "primary": "proton",
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
        },
    )
    return simtel_runner


@pytest.fixture
def corsika_file(io_handler):
    corsika_file = io_handler.get_input_data_file(
        file_name="run1_proton_za20deg_azm0deg-North-1LST_trigger_rates.corsika.zst", test=True
    )
    return corsika_file


def test_run_script(simtel_runner, corsika_file):
    script = simtel_runner.get_run_script(run_number=1, input_file=corsika_file)
    assert Path(script).exists()
