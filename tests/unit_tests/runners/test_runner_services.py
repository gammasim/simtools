#!/usr/bin/python3

import logging
import pathlib
import shutil

import pytest

import simtools.runners.runner_services as runner_services

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def runner_service(corsika_runner):
    _runner_service = runner_services.RunnerServices(
        corsika_config=corsika_runner.corsika_config, label="test-corsika-runner"
    )
    _runner_service.load_data_directories("corsika")
    return _runner_service


@pytest.fixture()
def runner_service_config_only(corsika_config):
    return runner_services.RunnerServices(
        corsika_config=corsika_config,
        label="test-corsika-runner",
    )


def test_get_info_for_file_name(runner_service_config_only):
    info_for_file_name = runner_service_config_only._get_info_for_file_name(run_number=1)
    assert info_for_file_name["run"] == 1
    assert info_for_file_name["primary"] == "proton"
    assert info_for_file_name["array_name"] == "test_layout"
    assert info_for_file_name["site"] == "South"
    assert info_for_file_name["label"] == "test-corsika-runner"
    assert info_for_file_name["zenith"] == pytest.approx(20)
    assert info_for_file_name["azimuth"] == pytest.approx(0)


def test_load_corsika_data_directories(runner_service_config_only):
    assert isinstance(runner_service_config_only.directory, dict)

    # TODO - is dict actually empty as load_data_directories is not called?

    for item in runner_service_config_only.directory.values():
        assert isinstance(item, pathlib.Path)


def test_get_file_name(runner_service, corsika_runner, io_handler):
    file_name = "run000001_proton_za020deg_azm000deg_South_test_layout_test-corsika-runner"

    # log.gz file
    assert runner_service.get_file_name("log", run_number=1) == corsika_runner._directory[
        "logs"
    ].joinpath(f"{file_name}.log.gz")

    assert runner_service.get_file_name("corsika_log", run_number=1) == corsika_runner._directory[
        "data"
    ].joinpath(runner_service._get_run_number_string(1)).joinpath("run1.log")

    script_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "scripts"
    )
    assert runner_service.get_file_name("script", run_number=1) == script_file_dir.joinpath(
        f"{file_name}.sh"
    )

    file_name_for_output = (
        "run000001_proton_za020deg_azm000deg_South_test_layout_test-corsika-runner.zst"
    )
    assert runner_service.get_file_name(
        "corsika_output", run_number=1
    ) == corsika_runner._directory["data"].joinpath(
        runner_service._get_run_number_string(1)
    ).joinpath(
        file_name_for_output
    )

    sub_log_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "logs"
    )
    assert runner_service.get_file_name(
        "sub_log", run_number=1, mode="out"
    ) == sub_log_file_dir.joinpath(f"log_sub_{file_name}.out")
    with pytest.raises(ValueError):
        runner_service.get_file_name("foobar", run_number=1, mode="out")
    assert runner_service.get_file_name(
        "sub_log", run_number=1, mode=""
    ) == sub_log_file_dir.joinpath(f"log_sub_{file_name}.log")


def test_get_run_number_string(runner_service_config_only):
    run_directory = runner_service_config_only._get_run_number_string(1)
    assert run_directory == "run000001"
    run_directory = runner_service_config_only._get_run_number_string(123456)
    assert run_directory == "run123456"
    with pytest.raises(ValueError, match=r"^Run number cannot have more than 6 digits"):
        run_directory = runner_service_config_only._get_run_number_string(1234567)


def test_get_resources(runner_service, caplog):
    sub_log_file = runner_service.get_file_name(file_type="sub_log", run_number=None, mode="out")
    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["RUNTIME 500\n"]
        file.writelines(lines_to_write)
    resources = runner_service.get_resources()
    assert isinstance(resources, dict)
    assert "runtime" in resources
    assert resources["runtime"] == 500

    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["SOMETHING ELSE 500\n"]
        file.writelines(lines_to_write)

    with caplog.at_level(logging.DEBUG):
        resources = runner_service.get_resources()
    assert resources["runtime"] is None
    assert "RUNTIME was not found in run log file" in caplog.text


def test_has_file(io_handler, runner_service):
    corsika_file = io_handler.get_input_data_file(
        file_name="run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst", test=True
    )
    # Copying the corsika file to the expected location and
    # changing its name for the sake of this test.
    # This should not affect the efficiency of this test.
    output_directory = runner_service.directory["data"].joinpath(
        runner_service._get_run_number_string(1)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        corsika_file,
        output_directory.joinpath(
            "run000001_proton_za020deg_azm000deg_South_test_layout_test-corsika-runner.zst"
        ),
    )
    assert runner_service.has_file(file_type="corsika_output", run_number=1)
    assert not runner_service.has_file(file_type="log", run_number=1234)


def test_get_simulation_software_list(runner_service_config_only):
    runner_service_config_only._get_simulation_software_list("corsika") == ["corsika"]
    runner_service_config_only._get_simulation_software_list("CoRsIka") == ["corsika"]
    runner_service_config_only._get_simulation_software_list("simtel") == ["simtel"]
    runner_service_config_only._get_simulation_software_list("corsika_simtel") == [
        "corsika",
        "simtel",
    ]
    runner_service_config_only._get_simulation_software_list("something_else") == []
