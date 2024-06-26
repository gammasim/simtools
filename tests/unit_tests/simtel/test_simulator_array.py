#!/usr/bin/python3

import logging
import shutil
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


def test_get_info_for_file_name(simtel_runner):
    info_for_file_name = simtel_runner.get_info_for_file_name(run_number=1)
    assert info_for_file_name["run"] == 1
    assert info_for_file_name["primary"] == "proton"
    assert info_for_file_name["array_name"] == "test_layout"
    assert info_for_file_name["site"] == "North"
    assert info_for_file_name["zenith"] == pytest.approx(20)
    assert info_for_file_name["azimuth"] == pytest.approx(0)
    assert info_for_file_name["label"] == "test-lst-array"


def test_get_file_name(simtel_runner):
    info_for_file_name = simtel_runner.get_info_for_file_name(run_number=1)
    file_name = "run000001_proton_za020deg_azm000deg_North_test_layout_test-lst-array"
    assert simtel_runner.get_file_name(
        "log", **info_for_file_name
    ) == simtel_runner._simtel_log_dir.joinpath(f"{file_name}.log.gz")
    assert simtel_runner.get_file_name(
        "histogram", **info_for_file_name
    ) == simtel_runner._simtel_log_dir.joinpath(f"{file_name}.hdata.zst")
    assert simtel_runner.get_file_name(
        "output", **info_for_file_name
    ) == simtel_runner._simtel_data_dir.joinpath(f"{file_name}.simtel.zst")
    assert simtel_runner.get_file_name(
        "sub_log", **info_for_file_name
    ) == simtel_runner._simtel_log_dir.joinpath(f"log_sub_{file_name}.log")
    assert simtel_runner.get_file_name(
        "sub_log", **info_for_file_name, mode="out"
    ) == simtel_runner._simtel_log_dir.joinpath(f"log_sub_{file_name}.out")
    with pytest.raises(ValueError):
        simtel_runner.get_file_name("foobar", **info_for_file_name, mode="out")


def test_has_file(simtel_runner, corsika_file):
    # Copying the corsika file to the expected location and
    # changing its name for the sake of this test.
    # This should not affect the efficiency of this test.
    shutil.copy(
        corsika_file,
        simtel_runner._simtel_data_dir.joinpath(
            "run000001_proton_za020deg_azm000deg_North_test_layout_test-lst-array.simtel.zst"
        ),
    )
    assert simtel_runner.has_file(file_type="output", run_number=1)
    assert not simtel_runner.has_file(file_type="log", run_number=1234)
