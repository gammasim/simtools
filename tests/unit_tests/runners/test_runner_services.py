#!/usr/bin/python3

import copy
import logging
import pathlib
import shutil

import pytest

import simtools.runners.runner_services as runner_services

logger = logging.getLogger()


@pytest.fixture
def runner_service(corsika_runner_mock_array_model):
    """Runner services object for corsika."""
    _runner_service = runner_services.RunnerServices(
        corsika_config=corsika_runner_mock_array_model.corsika_config, label="test-corsika-runner"
    )
    _runner_service.load_data_directories("corsika")
    return _runner_service


@pytest.fixture
def runner_service_mock_array_model(corsika_runner_mock_array_model):
    """Runner services object for corsika."""
    _runner_service = runner_services.RunnerServices(
        corsika_config=corsika_runner_mock_array_model.corsika_config, label="test-corsika-runner"
    )
    _runner_service.load_data_directories("corsika")
    return _runner_service


@pytest.fixture
def runner_service_config_only(corsika_config_mock_array_model):
    """Runner services object with simplified config."""
    return runner_services.RunnerServices(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-runner",
    )


@pytest.fixture
def runner_service_config_only_diffuse_gamma(corsika_config_mock_array_model):
    """Runner services object with simplified config."""
    corsika_config_mock_array_model.primary_particle = {
        "primary_id_type": "common_name",
        "primary": "gamma",
    }

    return runner_services.RunnerServices(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-runner",
    )


@pytest.fixture
def file_base_name(model_version):
    """Base name for simulation test file."""
    return (
        f"proton_run000001_za20deg_azm000deg_South_test_layout_{model_version}_test-corsika-runner"
    )


def test_init_runner_services(runner_service_config_only):
    assert runner_service_config_only.label == "test-corsika-runner"
    assert runner_service_config_only.corsika_config.primary == "proton"
    assert runner_service_config_only.directory == {}


def test_get_info_for_file_name(runner_service_config_only, model_version):
    info_for_file_name = runner_service_config_only._get_info_for_file_name(run_number=1)
    assert info_for_file_name["run_number"] == 1
    assert info_for_file_name["primary"] == "proton"
    assert info_for_file_name["array_name"] == "test_layout"
    assert info_for_file_name["site"] == "South"
    assert info_for_file_name["label"] == "test-corsika-runner"
    assert info_for_file_name["model_version"] == model_version
    assert info_for_file_name["zenith"] == pytest.approx(20)
    assert info_for_file_name["azimuth"] == pytest.approx(0)


def test_get_info_for_file_name_diffuse_gamma(
    runner_service_config_only_diffuse_gamma, model_version
):
    info_for_file_name = runner_service_config_only_diffuse_gamma._get_info_for_file_name(
        run_number=1
    )
    assert info_for_file_name["run_number"] == 1
    assert info_for_file_name["primary"] == "gamma_diffuse"
    assert info_for_file_name["array_name"] == "test_layout"
    assert info_for_file_name["site"] == "South"
    assert info_for_file_name["label"] == "test-corsika-runner"
    assert info_for_file_name["model_version"] == model_version
    assert info_for_file_name["zenith"] == pytest.approx(20)
    assert info_for_file_name["azimuth"] == pytest.approx(0)


def test_get_simulation_software_list(runner_service_config_only):
    assert runner_service_config_only._get_simulation_software_list("corsika") == ["corsika"]
    assert runner_service_config_only._get_simulation_software_list("CoRsIka") == ["corsika"]
    assert runner_service_config_only._get_simulation_software_list("sim_telarray") == [
        "sim_telarray"
    ]
    assert runner_service_config_only._get_simulation_software_list("corsika_sim_telarray") == [
        "corsika",
        "sim_telarray",
    ]
    assert runner_service_config_only._get_simulation_software_list("something_else") == []


def test_load_corsika_data_directories(runner_service_config_only):
    runner_service_config_only.load_data_directories("corsika")
    assert isinstance(runner_service_config_only.directory, dict)

    for item in runner_service_config_only.directory.values():
        assert isinstance(item, pathlib.Path)


def test_has_file(io_handler, runner_service, file_base_name):
    corsika_file = io_handler.get_test_data_file(
        file_name="run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst"
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
        output_directory.joinpath(f"{file_base_name}.corsika.zst"),
    )
    assert runner_service.has_file(file_type="corsika_output", run_number=1)
    assert not runner_service.has_file(file_type="log", run_number=1234)


def test_get_file_basename(runner_service, file_base_name, model_version):
    assert runner_service._get_file_basename(1) == file_base_name
    _runner_service_copy = copy.deepcopy(runner_service)
    _runner_service_copy.label = ""
    assert _runner_service_copy._get_file_basename(1) == (
        f"proton_run000001_za20deg_azm000deg_South_test_layout_{model_version}"
    )

    _runner_service_copy.corsika_config.primary_particle = None
    assert _runner_service_copy._get_file_basename(1) == (
        f"run000001_za20deg_azm000deg_South_test_layout_{model_version}"
    )


def test_get_log_file_path(runner_service, corsika_runner_mock_array_model, file_base_name):
    # log.gz
    assert runner_service._get_log_file_path(
        "log", file_base_name
    ) == corsika_runner_mock_array_model._directory["logs"].joinpath(f"{file_base_name}.log.gz")

    # hdata.zst
    assert runner_service._get_log_file_path(
        "histogram", file_base_name
    ) == corsika_runner_mock_array_model._directory["logs"].joinpath(f"{file_base_name}.hdata.zst")
    # corsika log
    assert runner_service._get_log_file_path(
        "corsika_log", file_base_name
    ) == corsika_runner_mock_array_model._directory["logs"].joinpath(
        f"{file_base_name}.corsika.log.gz"
    )


def test_get_data_file_path(runner_service, corsika_runner_mock_array_model, file_base_name):
    # corsika output
    assert runner_service._get_data_file_path(
        file_type="corsika_output", file_name=file_base_name, run_number=1
    ) == corsika_runner_mock_array_model._directory["data"].joinpath(
        runner_service._get_run_number_string(1)
    ).joinpath(f"{file_base_name}.corsika.zst")

    # simtel output
    assert runner_service._get_data_file_path(
        file_type="simtel_output", file_name=file_base_name, run_number=1
    ) == corsika_runner_mock_array_model._directory["data"].joinpath(
        runner_service._get_run_number_string(1)
    ).joinpath(f"{file_base_name}.simtel.zst")


def test_get_sub_file_path(runner_service, file_base_name, io_handler):
    script_file_dir = io_handler.get_output_directory("corsika").joinpath("scripts")
    assert runner_service._get_sub_file_path(
        file_type="script",
        file_name=file_base_name,
        mode=None,
    ) == script_file_dir.joinpath(f"sub_{file_base_name}.sh")
    assert runner_service._get_sub_file_path(
        file_type="script",
        file_name=file_base_name,
        mode="err",
    ) == script_file_dir.joinpath(f"sub_{file_base_name}.err")

    log_file_dir = io_handler.get_output_directory("corsika").joinpath("sub_logs")
    assert runner_service._get_sub_file_path(
        file_type="sub_log",
        file_name=file_base_name,
        mode=None,
    ) == log_file_dir.joinpath(f"sub_{file_base_name}.log")


def test_get_file_name(runner_service):
    assert isinstance(runner_service.get_file_name("log", run_number=1), pathlib.Path)
    assert isinstance(runner_service.get_file_name("output", run_number=1), pathlib.Path)
    assert isinstance(runner_service.get_file_name("sub_log", run_number=1), pathlib.Path)

    with pytest.raises(ValueError, match=r"^The requested file type"):
        runner_service.get_file_name("foobar", run_number=1, mode="out")


def test_get_run_number_string(runner_service_config_only):
    run_directory = runner_service_config_only._get_run_number_string(1)
    assert run_directory == "run000001"
    run_directory = runner_service_config_only._get_run_number_string(123456)
    assert run_directory == "run123456"
    with pytest.raises(ValueError, match=r"^Run number cannot have more than 6 digits"):
        runner_service_config_only._get_run_number_string(1234567)


def test_get_resources(runner_service_mock_array_model, caplog):
    sub_log_file = runner_service_mock_array_model.get_file_name(
        file_type="sub_log", run_number=None, mode="out"
    )
    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["RUNTIME 500\n"]
        file.writelines(lines_to_write)
    resources = runner_service_mock_array_model.get_resources()
    assert isinstance(resources, dict)
    assert "runtime" in resources
    assert resources["runtime"] == 500

    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["SOMETHING ELSE 500\n"]
        file.writelines(lines_to_write)

    with caplog.at_level(logging.DEBUG):
        resources = runner_service_mock_array_model.get_resources()
    assert resources["runtime"] is None
    assert "RUNTIME was not found in run log file" in caplog.text
