#!/usr/bin/python3

import copy
import logging
import pathlib

import pytest

import simtools.runners.runner_services as runner_services

logger = logging.getLogger()


@pytest.fixture
def runner_service(corsika_runner_mock_array_model):
    """Runner services object for corsika."""
    _runner_service = runner_services.RunnerServices(
        corsika_config=corsika_runner_mock_array_model.corsika_config,
        label="test-corsika-runner",
        run_type="corsika",
    )
    _runner_service.load_data_directory()
    return _runner_service


@pytest.fixture
def runner_service_mock_array_model(corsika_runner_mock_array_model):
    """Runner services object for corsika."""
    _runner_service = runner_services.RunnerServices(
        corsika_config=corsika_runner_mock_array_model.corsika_config,
        label="test-corsika-runner",
        run_type="corsika",
    )
    _runner_service.load_data_directory()
    return _runner_service


@pytest.fixture
def runner_service_config_only(corsika_config_mock_array_model):
    """Runner services object with simplified config."""
    return runner_services.RunnerServices(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-runner",
        run_type="corsika",
    )


@pytest.fixture
def runner_service_pedestals(corsika_config_mock_array_model):
    """Runner services object with simplified config."""
    corsika_config_pedestals = copy.deepcopy(corsika_config_mock_array_model)
    corsika_config_pedestals.run_mode = "pedestals"
    return runner_services.RunnerServices(
        corsika_config=corsika_config_pedestals,
        label="test-pedestals-runner",
        run_type="pedestals",
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
        run_type="corsika",
    )


@pytest.fixture
def file_base_name(model_version):
    """Base name for simulation test file."""
    return (
        f"proton_run000001_za20deg_azm000deg_South_test_layout_{model_version}_test-corsika-runner"
    )


def test_init_runner_services(runner_service_config_only):
    assert runner_service_config_only.label == "test-corsika-runner"
    assert runner_service_config_only.corsika_config.primary_particle.name == "proton"
    assert str(runner_service_config_only.directory).endswith("corsika")


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

    _runner_service_copy.corsika_config.primary_particle = {
        "primary_id_type": "common_name",
        "primary": "gamma",
    }
    _runner_service_copy.corsika_config.config["USER_INPUT"]["VIEWCONE"] = [0, 5]
    assert _runner_service_copy._get_file_basename(1) == (
        f"gamma_diffuse_run000001_za20deg_azm000deg_South_test_layout_{model_version}"
    )


def test_get_file_basename_calibration_mode(runner_service_pedestals, model_version):
    basename_pedestals = runner_service_pedestals._get_file_basename(1)
    expected_basename = (
        f"pedestals_run000001_za20deg_azm000deg_South_test_layout_"
        f"{model_version}_test-pedestals-runner"
    )
    assert basename_pedestals == expected_basename

    _runner_service_copy = copy.deepcopy(runner_service_pedestals)
    _runner_service_copy.label = ""
    basename_pedestals_no_label = _runner_service_copy._get_file_basename(1)
    expected_basename_no_label = (
        f"pedestals_run000001_za20deg_azm000deg_South_test_layout_{model_version}"
    )
    assert basename_pedestals_no_label == expected_basename_no_label


def test_get_file_name(runner_service):
    assert isinstance(runner_service.get_file_name("corsika_log", run_number=1), pathlib.Path)
    assert isinstance(runner_service.get_file_name("corsika_input", run_number=1), pathlib.Path)
    assert isinstance(runner_service.get_file_name("corsika_output", run_number=1), pathlib.Path)

    with pytest.raises(ValueError, match=r"Unknown file type: foobar"):
        runner_service.get_file_name("foobar", run_number=1)


def test_get_run_number_string(runner_service_config_only):
    run_directory = runner_service_config_only._get_run_number_string(1)
    assert run_directory == "run000001"
    run_directory = runner_service_config_only._get_run_number_string(123456)
    assert run_directory == "run123456"
    with pytest.raises(
        ValueError, match=r"Invalid type of run number \(1234567\) - it must be an uint < 1000000."
    ):
        runner_service_config_only._get_run_number_string(1234567)


def test_get_resources(runner_service_mock_array_model, caplog):
    sub_log_file = runner_service_mock_array_model.get_file_name(file_type="sub_log", run_number=5)
    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["RUNTIME 500\n"]
        file.writelines(lines_to_write)
    resources = runner_service_mock_array_model.get_resources(sub_log_file)
    assert isinstance(resources, dict)
    assert "runtime" in resources
    assert resources["runtime"] == 500
    # NSHOW from corsika_config_data fixture
    assert resources["n_events"] == 100

    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["SOMETHING ELSE 500\n"]
        file.writelines(lines_to_write)

    with caplog.at_level(logging.DEBUG):
        resources = runner_service_mock_array_model.get_resources(sub_log_file)
    assert resources["runtime"] is None
    assert "RUNTIME was not found in run log file" in caplog.text


def test_validate_corsika_run_number():
    assert runner_services.validate_corsika_run_number(1)
    assert runner_services.validate_corsika_run_number(123456)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        runner_services.validate_corsika_run_number("test")
    invalid_run_number = r"^Invalid type of run number"
    with pytest.raises(ValueError, match=invalid_run_number):
        runner_services.validate_corsika_run_number(1.5)
    with pytest.raises(ValueError, match=invalid_run_number):
        runner_services.validate_corsika_run_number(-1)
    with pytest.raises(ValueError, match=invalid_run_number):
        runner_services.validate_corsika_run_number(123456789)


def test_load_files(runner_service_config_only):
    run_files = runner_service_config_only.load_files(run_number=1)
    assert isinstance(run_files, dict)
    assert "corsika_input" in run_files
    assert "corsika_output" in run_files
    assert "corsika_log" in run_files
    for file_path in run_files.values():
        assert isinstance(file_path, pathlib.Path)


def test_get_sub_directory(runner_service_config_only):
    dir_path = runner_service_config_only._get_sub_directory(
        run_number=1, dir_path=pathlib.Path("/test/base/dir")
    )
    assert dir_path == pathlib.Path("/test/base/dir/run000001")
