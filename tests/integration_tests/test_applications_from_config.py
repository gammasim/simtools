#!/usr/bin/python3
# Integration tests for applications from config file

import copy
import logging
import subprocess
from pathlib import Path

import pytest

from simtools.testing import configuration, helpers, log_inspector, validate_output

logger = logging.getLogger()

config_files = sorted(Path(__file__).parent.glob("config/*.yml"))
test_configs, test_ids = configuration.get_list_of_test_configurations(config_files)
test_parameters = []
for config, test_id in zip(test_configs, test_ids):
    marks = []
    if config.get("test_requirement"):
        marks.append(pytest.mark.verifies_requirement(config["test_requirement"]))

    if config.get("test_use_case"):
        marks.append(pytest.mark.verifies_usecase(config["test_use_case"]))

    param = pytest.param(config, id=test_id, marks=marks)
    test_parameters.append(param)


@pytest.mark.parametrize("config", test_parameters)
def test_applications_from_config(tmp_test_directory, config, request, db_config):
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
    result = subprocess.run(cmd, shell=True, input="y\n", capture_output=True, text=True)
    msg = f"Command {cmd!r} failed. stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, msg

    assert log_inspector.inspect([result.stdout, result.stderr])

    validate_output.validate_application_output(
        tmp_config,
        model_version,
        config_file_model_version or model_version,
        db_config=db_config,
    )
