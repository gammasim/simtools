#!/usr/bin/python3

import logging
import shutil

import astropy.units as u
import pytest

from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def corsika_config_data(tmp_test_directory):
    return {
        "data_directory": f"{str(tmp_test_directory)}/test-output",
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
def corsika_runner(corsika_config_data, io_handler, simtel_path, db_config):
    corsika_runner = CorsikaRunner(
        mongo_db_config=db_config,
        site="south",
        layout_name="test-layout",
        simtel_source_path=simtel_path,
        label="test-corsika-runner",
        corsika_config_data=corsika_config_data,
    )
    return corsika_runner


@pytest.fixture
def corsika_file(io_handler):
    corsika_file = io_handler.get_input_data_file(
        file_name="run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst", test=True
    )
    return corsika_file


def test_prepare_run_script(corsika_runner):
    # No run number is given

    script = corsika_runner.prepare_run_script()

    assert script.exists()
    with open(script, "r") as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content

    # Run number is given
    run_number = 3
    script = corsika_runner.prepare_run_script(run_number=run_number)

    assert script.exists()
    with open(script, "r") as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content
        assert "-R 3" in script_content


def test_prepare_run_script_with_invalid_run(corsika_runner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_runner.prepare_run_script(run_number=run)


def test_prepare_run_script_with_extra(corsika_runner, file_has_text):
    extra = ["testing", "testing-extra-2"]
    script = corsika_runner.prepare_run_script(run_number=3, extra_commands=extra)

    assert file_has_text(script, "testing-extra-2")
    with open(script, "r") as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content


def test_prepare_run_script_without_pfp(corsika_runner):
    script = corsika_runner.prepare_run_script(use_pfp=False)

    assert script.exists()
    with open(script, "r") as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content


def test_get_info_for_file_name(corsika_runner):
    info_for_file_name = corsika_runner.get_info_for_file_name(run_number=1)
    assert info_for_file_name["run"] == 1
    assert info_for_file_name["primary"] == "gamma"
    assert info_for_file_name["array_name"] == "TestLayout"
    assert info_for_file_name["site"] == "South"
    assert info_for_file_name["label"] == "test-corsika-runner"


def test_get_file_name(corsika_runner, io_handler):
    info_for_file_name = corsika_runner.get_info_for_file_name(run_number=1)
    file_name = "corsika_run000001_gamma_South_TestLayout_test-corsika-runner"

    assert corsika_runner.get_file_name(
        "corsika_autoinputs_log", **info_for_file_name
    ) == corsika_runner._corsika_log_dir.joinpath(f"log_{file_name}.log.gz")

    assert corsika_runner.get_file_name(
        "corsika_log", **info_for_file_name
    ) == corsika_runner._corsika_data_dir.joinpath(corsika_runner._get_run_directory(1)).joinpath(
        "run1.log"
    )

    script_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "scripts"
    )
    assert corsika_runner.get_file_name("script", **info_for_file_name) == script_file_dir.joinpath(
        f"{file_name}.sh"
    )

    file_name_for_output = (
        "corsika_run000001_gamma_za020deg_azm000deg_South_TestLayout_test-corsika-runner.zst"
    )
    assert corsika_runner.get_file_name(
        "output", **info_for_file_name
    ) == corsika_runner._corsika_data_dir.joinpath(corsika_runner._get_run_directory(1)).joinpath(
        file_name_for_output
    )

    sub_log_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "logs"
    )
    assert corsika_runner.get_file_name(
        "sub_log", **info_for_file_name, mode="out"
    ) == sub_log_file_dir.joinpath(f"log_sub_{file_name}.out")
    with pytest.raises(ValueError):
        corsika_runner.get_file_name("foobar", **info_for_file_name, mode="out")
    assert corsika_runner.get_file_name(
        "sub_log", **info_for_file_name, mode=""
    ) == sub_log_file_dir.joinpath(f"log_sub_{file_name}.log")


def test_has_file(corsika_runner, corsika_file):
    # Copying the corsika file to the expected location and
    # changing its name for the sake of this test.
    # This should not affect the efficacy of this test.
    output_directory = corsika_runner._corsika_data_dir.joinpath(
        corsika_runner._get_run_directory(1)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        corsika_file,
        output_directory.joinpath(
            "corsika_run000001_gamma_za020deg_azm000deg_South_TestLayout_test-corsika-runner.zst"
        ),
    )
    assert corsika_runner.has_file(file_type="output", run_number=1)
    assert not corsika_runner.has_file(file_type="corsika_autoinputs_log", run_number=1234)
