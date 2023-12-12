#!/usr/bin/python3
# Integration tests for applications from config file
#

import logging
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.table import Table

import simtools.utils.general as gen

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_application_command(app, config_file=None):
    """
    Return the command to run the application with the given config file.

    """
    if app.find("simtools-") < 0:
        cmd = "python simtools/applications/" + app + ".py"
    else:
        cmd = app
    if config_file is not None:
        cmd += f" --config {config_file}"
    return cmd


def get_list_of_test_configurations(get_test_names=False):
    """
    Read all config files in the config directory and return a list
    of configuration dicts (equivalent to list of application tests).

    Parameters:
    -----------
    get_test_names: bool
        If True, return a list of test names instead of a list of configuration dicts.

    """

    config_files = Path(__file__).parent.glob("config/*.yml")
    logger.debug(f"Configuration files: {config_files}")

    configs = []
    for config_file in config_files:
        # read config file
        # remove new line characters from config - otherwise issues
        # with especially long file names
        _dict = gen.remove_substring_recursively_from_dict(
            gen.collect_data_from_yaml_or_dict(in_yaml=config_file, in_dict=None), substring="\n"
        )
        print("AAAA", _dict)
        configs.append(_dict.get("CTA_SIMPIPE", None))

    # list of all applications
    _applications = list(set(item["APPLICATION"] for item in configs if "APPLICATION" in item))
    for _app in _applications:
        # add for all applications "--help" call
        # TODO - disabled, as not all application have help implemented
        # configs.append(
        # {"APPLICATION": _app, "TEST_NAME": "auto-help", "CONFIGURATION": {"HELP": True}})
        # add for all applications "--version" call
        # configs.append(
        #    {"APPLICATION": _app, "TEST_NAME": "auto-version", "CONFIGURATION": {"VERSION": True}}
        # )
        # TODO - disabled, as not all applications can be called without command line parameters
        # add for all applications call without config file
        # configs.append({"APPLICATION": _app, "TEST_NAME": "auto-no_config"})
        logger.info("Missing implementations of help, versions, no command line parameter")

    if get_test_names:
        return [
            f"{item.get('APPLICATION', 'no-app-name')}_{item.get('TEST_NAME', 'no-test-name')}"
            for item in configs
        ]

    return configs


def compare_ecsv_files(file1, file2, tolerance=1.0e-5):
    """
    Compare two ecsv files:
    - same column table names
    - numerical values in columns are close

    Parameters
    ----------
    file1: str
        First file to compare
    file2: str
        Second file to compare
    tolerance: float
        Tolerance for comparing numerical values.

    """

    logger.info(f"Comparing files: {file1} and {file2}")
    table1 = Table.read(file1, format="ascii.ecsv")
    table2 = Table.read(file2, format="ascii.ecsv")

    assert len(table1) == len(table2)

    assert table1.colnames == table2.colnames

    for col_name in table1.colnames:
        if np.issubdtype(table1[col_name].dtype, np.number):
            assert np.allclose(table1[col_name], table2[col_name], rtol=tolerance)


def validate_application_output(config):
    """
    Validate application output against expected output.

    Expected output is defined in configuration file.

    """

    if "INTEGRATION_TESTS" not in config:
        return 0

    for integration_test in config["INTEGRATION_TESTS"]:
        logger.info(f"Testing application output: {integration_test}")
        if "REFERENCE_OUTPUT_FILE" in integration_test:
            compare_ecsv_files(
                integration_test["REFERENCE_OUTPUT_FILE"],
                Path(config["CONFIGURATION"]["OUTPUT_PATH"]).joinpath(
                    config["CONFIGURATION"]["OUTPUT_FILE"]
                ),
                integration_test.get("TOLERANCE", 1.0e-5),
            )

    return 0


def get_tmp_config_file(config, output_path):
    """
    Write a temporary config file for the application to be tested.
    Change output path and file name to values suitable for tests.

    """

    tmp_config_file = output_path / "tmp_config.yml"
    if "OUTPUT_PATH" in config:
        config.update({"OUTPUT_PATH": str(output_path)})
        config.update({"USE_PLAIN_OUTPUT_PATH": True})
    if "DATA_DIRECTORY" in config:
        config["DATA_DIRECTORY"] = str(output_path) + "/data"

    # write config to a yaml file in tmp directory
    with open(tmp_config_file, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return tmp_config_file


@pytest.mark.parametrize(
    "config",
    get_list_of_test_configurations(),
    ids=get_list_of_test_configurations(get_test_names=True),
)
def test_applications_from_config(tmp_test_directory, config, monkeypatch):
    """
    Test all applications from config files found in the config directory.
    Test output is written to a temporary directory and tested.

    """

    # The add_file_to_db.py application requires a user confirmation.
    # With this line we mock the user confirmation to be y for the test
    # Notice this is done for all tests, so keep in mind if in the future we add tests with input.
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))

    try:
        tmp_output_path = Path(tmp_test_directory).joinpath(
            config["APPLICATION"] + "-" + config["TEST_NAME"]
        )
    except KeyError as exc:
        logger.error(f"No application defined in config file {config}.")
        raise exc

    tmp_output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Temporary output path: {tmp_output_path}")
    if "CONFIGURATION" in config:
        config_file = get_tmp_config_file(config["CONFIGURATION"], output_path=tmp_output_path)
    else:
        config_file = None

    cmd = get_application_command(
        app=config.get("APPLICATION", None),
        config_file=config_file,
    )
    logger.info(f"Application configuration: {config}")

    logger.info(f"Running application: {cmd}")
    assert os.system(cmd) == 0

    output_status = validate_application_output(config)
    assert output_status == 0
