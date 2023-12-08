#!/usr/bin/python3
# Integration tests for applications from config file
#


import glob
import logging
import os
import uuid
from pathlib import Path

import numpy as np
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


def get_list_of_test_configurations():
    """
    Read all config files in the config directory and return a list
    of configuration dicts (equivalent to list of application tests)

    """

    config_files = glob.glob(str(Path(__file__).parent / "config/*.yml"))

    configs = []
    for config_file in config_files:
        _dict = gen.collect_data_from_yaml_or_dict(in_yaml=config_file, in_dict=None)
        configs.append(_dict.get("CTA_SIMPIPE", None))

    return configs


def run_application(cmd):
    """
    Run the application with the given command.

    """
    logger.info(f"Running application: {cmd}")
    return os.system(cmd)


def compare_ecsv_files(file1, file2):
    """
    Compare two ecsv files:
    - same column table names
    - numerical values in columns are close

    """

    logger.info(f"Comparing files: {file1} and {file2}")
    table1 = Table.read(file1, format="ascii.ecsv")
    table2 = Table.read(file2, format="ascii.ecsv")

    assert len(table1) == len(table2)

    assert table1.colnames == table2.colnames

    # hardwired tolerance; not clear if this is good
    tolerance = 1e-5
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
            )

    return 0


def get_tmp_config_file(config, output_path):
    """
    Write a temporary config file for the application.
    Change output path and file name to values suitable for tests.

    """

    tmp_config_file = output_path / "tmp_config.yml"

    try:
        config["OUTPUT_PATH"] = str(output_path)
    except KeyError:
        pass

    # write config to a yaml file in tmp directory
    with open(tmp_config_file, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return tmp_config_file


def test_applications_from_config(tmp_test_directory):
    """
    Test all applications from config files found in the config directory.
    Test output is written to a temporary directory and tested.

    """

    configs = get_list_of_test_configurations()

    for config in configs:
        tmp_output_path = Path(tmp_test_directory).joinpath(str(uuid.uuid4()))
        tmp_output_path.mkdir(parents=True, exist_ok=True)
        config_file = get_tmp_config_file(config["CONFIGURATION"], output_path=tmp_output_path)

        cmd = get_application_command(
            app=config.get("APPLICATION", None),
            config_file=config_file,
        )
        logger.info(f"Application configuration: {config}")

        run_status = run_application(cmd)
        assert run_status == 0

        output_status = validate_application_output(config)
        assert output_status == 0
