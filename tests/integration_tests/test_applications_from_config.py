#!/usr/bin/python3
# Integration tests for applications from config file

import copy
import logging
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from dotenv import load_dotenv

from simtools.testing import configuration, helpers, validate_output

logger = logging.getLogger()
load_dotenv(".env")

config_files = sorted(Path(__file__).parent.glob("config/*.yml"))
test_configs, test_ids = configuration.get_list_of_test_configurations(config_files)


test_configs = [pytest.param(config, id=test_id) for config, test_id in zip(test_configs, test_ids)]


@pytest.fixture(scope="session", params=test_configs)
def test_config(request):
    """
    Fixture to parametrize the test with the configurations from the config files.

    Allows to add markers to very requirements and use cases for CTAO using the
    'pytest-requirements' package.
    """
    config = request.param

    if config.get("TEST_REQUIREMENT"):
        request.applymarker(pytest.mark.verifies_requirement(config["TEST_REQUIREMENT"]))

    if config.get("TEST_USE_CASE"):
        request.applymarker(pytest.mark.verifies_usecase(config["TEST_USE_CASE"]))

    return config


def test_applications_from_config(tmp_test_directory, test_config, monkeypatch, request):
    """
    Test all applications from config files found in the config directory.

    Parameters
    ----------
    tmp_test_directory: str
        Temporary directory, into which test configuration and output is written.
    config: dict
        Dictionary with the configuration parameters for the test.

    """
    config = test_config
    tmp_config = copy.deepcopy(config)
    skip_message = helpers.skip_camera_efficiency(tmp_config)
    if skip_message:
        pytest.skip(skip_message)

    # The db_add_file_to_db.py application requires a user confirmation.
    # With this line we mock the user confirmation to be y for the test
    # Notice this is done for all tests, so keep in mind if in the future we add tests with input.
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))

    logger.info(f"Test configuration from config file: {tmp_config}")
    logger.info(f"Model version: {request.config.getoption('--model_version')}")
    logger.info(f"Application configuration: {tmp_config}")
    logger.info(f"Test requirement: {config.get('TEST_REQUIREMENT')}")
    logger.info(f"Test use case: {config.get('TEST_USE_CASE')}")
    try:
        cmd, config_file_model_version = configuration.configure(
            tmp_config, tmp_test_directory, request
        )
    except configuration.VersionError as exc:
        pytest.skip(str(exc))

    logger.info(f"Running application: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert result.returncode == 0, f"Application failed: {cmd}"

    validate_output.validate_application_output(
        tmp_config,
        request.config.getoption("--model_version"),
        config_file_model_version or request.config.getoption("--model_version"),
    )
