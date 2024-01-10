#!/usr/bin/python3
# Integration tests for applications from config file
#

import json
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


def get_application_command(app, config_file=None, config_string=None):
    """
    Return the command to run the application with the given config file.

    Parameters
    ----------
    app: str
        Name of the application.
    config_file: str
        Configuration file.
    config_string: str
        Configuration string (e.g., '--version')

    Returns
    -------
    str: command to run the application test.

    """
    if app.find("simtools-") < 0:
        cmd = "python simtools/applications/" + app + ".py"
    else:
        cmd = app
    if config_string is not None:
        cmd += f" {config_string}"
    elif config_file is not None:
        cmd += f" --config {config_file}"
    return cmd


def get_list_of_test_configurations(get_test_names=False):
    """
    Read all config files in the config directory and return a list
    of configuration dicts (equivalent to list of application tests).

    Parameters:
    -----------
    get_test_names: bool
        If True, return a list of test names instead of a list of configuration dictionaries.

    Returns
    -------
    list:
        of test names or of configuration dictionaries.

    """

    # (needs to be sorted for pytest-xdist, see Known Limitations in their website)
    config_files = sorted(list(Path(__file__).parent.glob("config/*.yml")))
    logger.debug(f"Configuration files: {config_files}")

    configs = []
    for config_file in config_files:
        # read config file
        # remove new line characters from config - otherwise issues
        # with especially long file names
        _dict = gen.remove_substring_recursively_from_dict(
            gen.collect_data_from_file_or_dict(file_name=config_file, in_dict=None), substring="\n"
        )
        configs.append(_dict.get("CTA_SIMPIPE", None))

    # list of all applications
    # (needs to be sorted for pytest-xdist, see Known Limitations in their website)
    _applications = sorted(
        list(set(item["APPLICATION"] for item in configs if "APPLICATION" in item))
    )
    for _app in _applications:
        # add for all applications "--help" call
        configs.append(
            {"APPLICATION": _app, "TEST_NAME": "auto-help", "CONFIGURATION": {"HELP": True}}
        )
        # add for all applications "--version" call
        configs.append(
            {"APPLICATION": _app, "TEST_NAME": "auto-version", "CONFIGURATION": {"VERSION": True}}
        )
        # add for all applications call without config file
        configs.append({"APPLICATION": _app, "TEST_NAME": "auto-no_config"})

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

    #    assert table1.colnames == table2.colnames

    for col_name in table1.colnames:
        if np.issubdtype(table1[col_name].dtype, np.floating):
            assert np.allclose(table1[col_name], table2[col_name], rtol=tolerance)


def assert_file_type(file_type, file_name):
    """
    Assert that the file is of the given type.

    Parameters
    ----------
    file_type: str
        File type (json, yaml).
    file_name: str
        File name.

    """

    if file_type == "json":
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                json.load(file)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False
    if file_type == "yaml" or file_type == "yml":
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                yaml.safe_load(file)
            return True
        except (yaml.YAMLError, FileNotFoundError):
            return False

    # no dedicated tests for other file types, checking suffix only
    logger.info(f"File type test is checking suffix only for {file_name} (suffix: {file_type}))")
    if file_name.suffix[1:] == file_type:
        return True

    return False


def validate_application_output(config):
    """
    Validate application output against expected output.
    Expected output is defined in configuration file.

    Parameters
    ----------
    config: dict
        dictionary with the configuration for the application test.

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
        if "OUTPUT_FILE" in integration_test:
            # In this case the output file is the file generated by the application
            # First check if the output is in the data directory (simtel_array related),
            # Then check if the file is in the output directory (remaining tools).
            print("PATH", config["CONFIGURATION"]["OUTPUT_PATH"])
            print("FILE", integration_test["OUTPUT_FILE"])
            try:
                assert (
                    Path(config["CONFIGURATION"]["DATA_DIRECTORY"])
                    .joinpath(integration_test["OUTPUT_FILE"])
                    .exists()
                )
            except KeyError:
                assert (
                    Path(config["CONFIGURATION"]["OUTPUT_PATH"])
                    .joinpath(integration_test["OUTPUT_FILE"])
                    .exists()
                )
        if "FILE_TYPE" in integration_test:
            assert assert_file_type(
                integration_test["FILE_TYPE"],
                Path(config["CONFIGURATION"]["OUTPUT_PATH"]).joinpath(
                    config["CONFIGURATION"]["OUTPUT_FILE"]
                ),
            )

    return 0


def prepare_configuration(config, output_path):
    """
    Prepare configuration. This means either to write a temporary config file
    or to return a single string for single boolean options (e.g., --version).
    Change output path and file to paths provided with output_path.

    Parameters
    ----------
    config: dict
        Dictionary with the configuration for the application test.
    output_path: str
        Output path.

    Returns
    -------
    str: path to the temporary config file.
    str: configuration string.

    """

    if len(config) == 1 and next(iter(config.values())) is True:
        return None, "--" + list(config.keys())[0].lower()

    tmp_config_file = output_path / "tmp_config.yml"
    if "OUTPUT_PATH" in config:
        config.update({"OUTPUT_PATH": str(Path(output_path).joinpath(config["OUTPUT_PATH"]))})
        config.update({"USE_PLAIN_OUTPUT_PATH": True})
    if "DATA_DIRECTORY" in config:
        config.update({"DATA_DIRECTORY": str(Path(output_path).joinpath(config["DATA_DIRECTORY"]))})

    logger.info(f"Writing config file: {tmp_config_file}")
    with open(tmp_config_file, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return tmp_config_file, None


@pytest.mark.parametrize(
    "config",
    get_list_of_test_configurations(),
    ids=get_list_of_test_configurations(get_test_names=True),
)
def test_applications_from_config(tmp_test_directory, config, monkeypatch):
    """
    Test all applications from config files found in the config directory.

    Parameters
    ----------
    tmp_test_directory: str
        Temporary directory, into which test configuration and output is written.
    config: dict
        Dictionary with the configuration parameters for the test.

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
        config_file, config_string = prepare_configuration(
            config["CONFIGURATION"], output_path=tmp_output_path
        )
    else:
        config_file = None
        config_string = None

    cmd = get_application_command(
        app=config.get("APPLICATION", None),
        config_file=config_file,
        config_string=config_string,
    )
    logger.info(f"Application configuration: {config}")

    logger.info(f"Running application: {cmd}")
    assert os.system(cmd) == 0

    output_status = validate_application_output(config)
    assert output_status == 0
