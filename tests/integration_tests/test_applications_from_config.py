#!/usr/bin/python3
# Integration tests for applications from config file

import copy
import logging
import os
from io import StringIO
from pathlib import Path

import pytest
from dotenv import load_dotenv

from simtools.testing import configuration, validate_output

logger = logging.getLogger()
load_dotenv(".env")

config_files = sorted(Path(__file__).parent.glob("config/*.yml"))
test_configs = configuration.get_list_of_test_configurations(config_files)
test_ids = configuration.get_list_of_test_configurations(config_files, get_test_names=True)


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
    skip_camera_efficiency(tmp_config)

    # The db_add_file_to_db.py application requires a user confirmation.
    # With this line we mock the user confirmation to be y for the test
    # Notice this is done for all tests, so keep in mind if in the future we add tests with input.
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))

    logger.info(f"Test configuration from config file: {tmp_config}")
    logger.info(f"Model version: {request.config.getoption('--model_version')}")
    logger.info(f"Application configuration: {tmp_config}")
    cmd, config_file_model_version = configuration.configure(
        tmp_config, tmp_test_directory, request
    )

    logger.info(f"Running application: {cmd}")
    assert os.system(cmd) == 0, f"Application failed: {cmd}"

    validate_output.validate_all_tests(tmp_config, request, config_file_model_version)


def skip_camera_efficiency(config):
    """
    Skip camera efficiency tests if the old version of testeff is used.
    """
    if "camera-efficiency" in config["APPLICATION"]:
        if not new_testeff_version():
            pytest.skip(
                "Any applications calling the old version of testeff are skipped "
                "due to a limitation of the old testeff not allowing to specify "
                "the include directory. Please update your sim_telarray tarball."
            )
        full_test_name = f"{config['APPLICATION']}_{config['TEST_NAME']}"
        if "simtools-validate-camera-efficiency_SSTS" == full_test_name:
            pytest.skip(
                "The test simtools-validate-camera-efficiency_SSTS is skipped "
                "since the fake SST mirrors are not yet implemented (#1155)"
            )


def new_testeff_version():
    """
    Testeff has been updated to allow to specify the include directory.
    This test checks if the new version is used.
    """

    with open(os.path.join(os.getenv("SIMTOOLS_SIMTEL_PATH"), "sim_telarray/testeff.c")) as file:
        file_content = file.read()
        if "/* Combine the include paths such that those from '-I...' options */" in file_content:
            return True
        return False
