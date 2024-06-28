#!/usr/bin/python3

import logging
from pathlib import Path

import astropy.units as u
import pytest

from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def simtel_runner(array_model_north, simtel_path):
    return SimulatorArray(
        array_model=array_model_north,
        simtel_path=simtel_path,
        config_data={
            "primary": "proton",
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
        },
    )


@pytest.fixture()
def corsika_file(io_handler):
    return io_handler.get_input_data_file(
        file_name="run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst", test=True
    )


def test_run_script(simtel_runner, corsika_file):
    script = simtel_runner.prepare_run_script(run_number=1, input_file=corsika_file)
    assert Path(script).exists()
