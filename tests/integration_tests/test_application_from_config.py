#!/usr/bin/python3
# Integration tests for applications from config file
#


import glob
import logging
import os
from pathlib import Path

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
        cmd += " --config " + config_file
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
        try:
            configs[-1]["CONFIG_FILE"] = config_file
        except TypeError:
            pass

    return configs


def run_application(cmd):
    """
    Run the application with the given command.

    """
    logger.info(f"Running application: {cmd}")
    return os.system(cmd)


def validate_application_output(config):
    """
    Validate application output against expected output.

    Expected output is defined in configuration file.

    """

    if "REFERENCE_OUTPUT" not in config:
        return 0

    logger.info(f"Testing application output: {config['REFERENCE_OUTPUT']}")

    return 0


def test_applications_from_config():
    """
    Test all applications from config files in the config directory.

    """

    configs = get_list_of_test_configurations()

    for config in configs:
        cmd = get_application_command(
            config.get("APPLICATION", None), config.get("CONFIG_FILE", None)
        )
        logger.info(f"Application configuration: {config}")

        run_status = run_application(cmd)
        assert run_status == 0

        output_status = validate_application_output(config)
        assert output_status == 0
