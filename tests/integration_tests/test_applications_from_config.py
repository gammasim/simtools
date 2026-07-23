#!/usr/bin/python3
# Integration tests for applications from config file

import copy
import logging
import os
import subprocess
from pathlib import Path

import pytest

from simtools.testing import configuration, helpers, log_inspector, validate_output

logger = logging.getLogger()


def _get_simulation_models_path(config, request, simtools_root_path):
    """Return the configured model path or skip MongoDB-only applications."""
    simulation_models_path = request.config.getoption("simulation_models_path", default=None)
    if not simulation_models_path:
        return None
    if config.get("requires_mongodb"):
        pytest.skip(f"{config['application']} requires MongoDB")

    simulation_models_path = Path(simulation_models_path)
    if not simulation_models_path.is_absolute():
        simulation_models_path = Path(simtools_root_path) / simulation_models_path
    return simulation_models_path.resolve()


def pytest_generate_tests(metafunc):
    """Parametrize application tests using the configured test-resources path."""
    if "config" not in metafunc.fixturenames:
        return

    config_files = sorted(Path(__file__).parent.glob("config/*.yml"))
    test_configs, test_ids = configuration.get_list_of_test_configurations(
        config_files,
        test_resources_path=metafunc.config.getoption("test_resources_path", default=None),
    )
    test_parameters = []
    for config, test_id in zip(test_configs, test_ids):
        marks = []
        if config.get("test_requirement"):
            marks.append(pytest.mark.verifies_requirement(config["test_requirement"]))
        if config.get("test_use_case"):
            marks.append(pytest.mark.verifies_usecase(config["test_use_case"]))
        if config.get("xfail"):
            marks.append(pytest.mark.xfail(reason=config["xfail"]))
        test_parameters.append(pytest.param(config, id=test_id, marks=marks))

    metafunc.parametrize("config", test_parameters)


def test_applications_from_config(
    tmp_test_directory, config, request, simtools_root_path, monkeypatch
):
    """
    Test all applications from config files found in the config directory.

    Parameters
    ----------
    tmp_test_directory: str
        Temporary directory, into which test configuration and output is written.
    config: dict
        Dictionary with the configuration parameters for the test.

    """
    tmp_config = copy.deepcopy(config)
    skip_message = helpers.skip_camera_efficiency(tmp_config)
    if skip_message:
        pytest.skip(skip_message)

    model_version = request.config.getoption("--model_version", default=None)
    if model_version:
        model_version = model_version.split(",")
        model_version = model_version[0] if len(model_version) == 1 else model_version
    skip_message = helpers.skip_multiple_version_test(tmp_config, model_version)
    if skip_message:
        pytest.skip(skip_message)

    if tmp_config.get("skip_integration_test"):
        pytest.skip(tmp_config["skip_integration_test"])

    simulation_models_path = _get_simulation_models_path(tmp_config, request, simtools_root_path)
    if simulation_models_path:
        monkeypatch.setenv("SIMTOOLS_SIMULATION_MODELS_PATH", str(simulation_models_path))

    logger.info(f"Test configuration from config file: {tmp_config}")
    logger.info(f"Model version: {model_version}")
    logger.info(f"Application configuration: {tmp_config}")
    logger.info(f"Test requirement: {config.get('test_requirement')}")
    logger.info(f"Test use case: {config.get('test_use_case')}")
    try:
        cmd, config_file_model_version = configuration.configure(
            tmp_config, tmp_test_directory, request
        )
    except (configuration.ProductionDBError, configuration.VersionError) as exc:
        pytest.skip(str(exc))

    logger.info(f"Running application: {cmd}")
    env = os.environ.copy()
    env["SIMTOOLS_OFFLINE_IERS"] = "1"
    result = subprocess.run(
        cmd,
        shell=True,
        input="y\n",
        capture_output=True,
        text=True,
        env=env,
        cwd=simtools_root_path,
    )
    msg = f"Command {cmd!r} failed. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    if result.returncode != 0 and config.get("xfail_network_error"):
        combined_output = result.stdout + result.stderr
        network_error_patterns = (
            "URLError",
            "Network is unreachable",
            "ConnectionError",
            "TimeoutError",
            "gaierror",
        )
        if any(pattern in combined_output for pattern in network_error_patterns):
            pytest.xfail(f"Network error: {msg}")
    assert result.returncode == 0, msg

    assert log_inspector.inspect([result.stdout, result.stderr])

    validate_output.validate_application_output(
        tmp_config,
        model_version,
        config_file_model_version or model_version,
    )


def test_get_simulation_models_path(tmp_test_directory, mocker):
    """Resolve a relative simulation-model path against the simtools root."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = "../simulation-models"
    root_path = Path(tmp_test_directory) / "simtools"

    path = _get_simulation_models_path(
        {"application": "simtools-simulate-prod"}, request, root_path
    )

    assert path == (root_path / "../simulation-models").resolve()


def test_mongodb_only_application_is_skipped(tmp_test_directory, mocker):
    """Skip MongoDB-only applications when filesystem model access is selected."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = "../simulation-models"
    config = {"application": "simtools-mongodb-operation", "requires_mongodb": True}

    with pytest.raises(pytest.skip.Exception, match="simtools-mongodb-operation requires MongoDB"):
        _get_simulation_models_path(config, request, tmp_test_directory)


def test_get_simulation_models_path_is_optional(tmp_test_directory, mocker):
    """Leave integration tests unchanged when no filesystem path is configured."""
    request = mocker.MagicMock()
    request.config.getoption.return_value = None

    assert (
        _get_simulation_models_path(
            {"application": "simtools-simulate-prod"}, request, tmp_test_directory
        )
        is None
    )
