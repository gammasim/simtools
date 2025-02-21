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


@pytest.mark.parametrize("config", test_configs, ids=test_ids)
def test_applications_from_config(tmp_test_directory, config, monkeypatch, request, model_version):
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

    # The db_add_file_to_db.py application requires a user confirmation.
    # With this line we mock the user confirmation to be y for the test
    # Notice this is done for all tests, so keep in mind if in the future we add tests with input.
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))

    logger.info(f"Test configuration from config file: {tmp_config}")
    logger.info(f"Model version: {request.config.getoption('--model_version')}")
    logger.info(f"Application configuration: {tmp_config}")
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
