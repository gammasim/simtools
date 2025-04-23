"""Integration test configuration."""

import logging
import os
from pathlib import Path

import yaml

import simtools.utils.general as gen

_logger = logging.getLogger(__name__)


class VersionError(Exception):
    """Raise if model version requested is not supported."""


class ProductionDBError(Exception):
    """Raise if production db is used."""


def get_list_of_test_configurations(config_files):
    """
    Return list of test configuration dictionaries or test names.

    Read all configuration files for testing.
    Add "--help" and "--version" calls for all applications.

    Parameters
    ----------
    config_files: list
        List of integration test configuration files.

    Returns
    -------
    list:
        list of test names or of configuration dictionaries.

    """
    _logger.debug(f"Configuration files: {config_files}")

    configs = _read_configs_from_files(config_files)

    # list of all applications
    # (needs to be sorted for pytest-xdist, see Known Limitations in their website)
    _applications = sorted({item["APPLICATION"] for item in configs if "APPLICATION" in item})
    for app in _applications:
        configs.extend(
            [
                {"APPLICATION": app, "TEST_NAME": "auto-help", "CONFIGURATION": {"HELP": True}},
                {
                    "APPLICATION": app,
                    "TEST_NAME": "auto-version",
                    "CONFIGURATION": {"VERSION": True},
                },
                {"APPLICATION": app, "TEST_NAME": "auto-no_config"},
            ]
        )

    return (
        configs,
        [
            f"{item.get('APPLICATION', 'no-app-name')}_{item.get('TEST_NAME', 'no-test-name')}"
            for item in configs
        ],
    )


def _read_configs_from_files(config_files):
    """Read test configuration from files."""
    configs = []
    for config_file in config_files:
        # read config file
        # remove new line characters from config - otherwise issues
        # with especially long file names
        _dict = gen.remove_substring_recursively_from_dict(
            gen.collect_data_from_file(file_name=config_file), substring="\n"
        )
        for application in _dict.get("CTA_SIMPIPE", {}).get("APPLICATIONS", []):
            configs.append(application)
    return configs


def configure(config, tmp_test_directory, request):
    """
    Prepare configuration and command for integration tests.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    tmp_test_directory: str
        Temporary test directory (from pytest fixture).
    request: request
        Request object.

    Returns
    -------
    str: command to run the application test.
    str: config file model version.

    """
    tmp_output_path = create_tmp_output_path(tmp_test_directory, config)
    model_version_requested = request.config.getoption("--model_version")
    model_version_requested = (
        model_version_requested.split(",") if model_version_requested else None
    )
    if isinstance(model_version_requested, list) and len(model_version_requested) == 1:
        model_version_requested = model_version_requested[0]

    if "CONFIGURATION" in config:
        _skip_test_for_model_version(config, model_version_requested)
        _skip_test_for_production_db(config)

        config_file, config_string, config_file_model_version = _prepare_test_options(
            config["CONFIGURATION"],
            output_path=tmp_output_path,
            model_version=model_version_requested,
        )
    else:
        config_file = None
        config_string = None
        config_file_model_version = None

    cmd = get_application_command(
        app=config.get("APPLICATION", None),
        config_file=config_file,
        config_string=config_string,
    )
    return cmd, config_file_model_version


def _skip_test_for_model_version(config, model_version_requested):
    """Skip test if model version requested is not supported."""
    if config.get("MODEL_VERSION_USE_CURRENT") is None or model_version_requested is None:
        return
    model_version_config = config["CONFIGURATION"]["MODEL_VERSION"]
    if model_version_requested != model_version_config:
        raise VersionError(
            f"Model version requested {model_version_requested} not supported for this test"
        )


def _skip_test_for_production_db(config):
    """Skip test if production db is used."""
    if not config.get("SKIP_FOR_PRODUCTION_DB"):
        return

    if "db.zeuthen.desy.de" in os.getenv("SIMTOOLS_DB_SERVER", ""):
        raise ProductionDBError("Production database used for this test")

    if "simpipe" in os.getenv("SIMTOOLS_DB_API_USER", ""):
        raise ProductionDBError("Production database used for this test")


def _prepare_test_options(config, output_path, model_version=None):
    """
    Prepare test configuration.

    This means either to write a temporary config file
    or to return a single string for single boolean options (e.g., --version).
    Change output path and file to paths provided with output_path.

    Parameters
    ----------
    config: dict
        Dictionary with the configuration for the application test.
    output_path: str
        Output path.
    model_version: str
        Model versions (default: use those given in config files)

    Returns
    -------
    config_file: str
        Path to the temporary configuration file.
    config_string: str
        Command line configuration as single string.
    config_file_model_version: str
        Configuration file model version

    """
    if len(config) == 1 and next(iter(config.values())) is True:
        return None, "--" + next(iter(config.keys())).lower(), None

    tmp_config_file = output_path / "tmp_config.yml"
    config_file_model_version = config.get("MODEL_VERSION")
    if model_version and "MODEL_VERSION" in config:
        config.update({"MODEL_VERSION": model_version})

    for key in ["OUTPUT_PATH", "DATA_DIRECTORY", "PACK_FOR_GRID_REGISTER"]:
        if key in config:
            config[key] = str(Path(output_path).joinpath(config[key]))
            if key == "OUTPUT_PATH":
                config["USE_PLAIN_OUTPUT_PATH"] = True

    _logger.info(f"Writing config file: {tmp_config_file}")
    with open(tmp_config_file, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return tmp_config_file, None, config_file_model_version


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
    cmd = app if "simtools-" in app else f"python simtools/applications/{app}.py"
    if config_string:
        return f"{cmd} {config_string}"
    if config_file is not None:
        return f"{cmd} --config {config_file}"
    return cmd


def create_tmp_output_path(tmp_test_directory, config):
    """
    Create temporary output path.

    Parameters
    ----------
    tmp_test_directory: str
        Temporary directory.
    config: dict
        Configuration dictionary.

    Returns
    -------
    str: path to the temporary output directory.

    """
    try:
        tmp_output_path = Path(tmp_test_directory).joinpath(
            config["APPLICATION"] + "-" + config["TEST_NAME"]
        )
    except KeyError as exc:
        raise KeyError(f"No application defined in configuration {config}.") from exc
    tmp_output_path.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Temporary output path: {tmp_output_path}")
    return tmp_output_path
