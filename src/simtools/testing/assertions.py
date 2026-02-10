"""Functions asserting certain conditions are met (used e.g., in integration tests)."""

import json
import logging
from pathlib import Path

import yaml

from simtools.simtel import simtel_output_validator
from simtools.testing.log_inspector import check_plain_logs, check_tar_logs

_logger = logging.getLogger(__name__)


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
            with open(file_name, encoding="utf-8") as file:
                json.load(file)
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False
    if file_type in ("yaml", "yml"):
        if Path(file_name).suffix[1:] not in ("yaml", "yml"):
            return False
        try:
            with open(file_name, encoding="utf-8") as file:
                yaml.safe_load(file)
            return True
        except (yaml.YAMLError, FileNotFoundError):
            return False

    # no dedicated tests for other file types, checking suffix only
    _logger.info(f"File type test is checking suffix only for {file_name} (suffix: {file_type}))")
    return Path(file_name).suffix[1:] == file_type


def check_output_from_sim_telarray(file, file_test):
    """
    Check that the sim_telarray simulation result is reasonable and matches the expected output.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    file_test: dict
        File test description including expected output and metadata.

    Raises
    ------
    ValueError
        If the file is not a zstd compressed file.
    """
    if (
        "expected_sim_telarray_output" not in file_test
        and "expected_sim_telarray_metadata" not in file_test
    ):
        _logger.debug(f"No expected output or metadata provided, skipping checks {file_test}")
        return True

    assert_sim_telarray = []

    expected_output_key_map = {
        "expected_sim_telarray_output": (
            "assert_expected_sim_telarray_output",
            "expected_sim_telarray_output",
        ),
        "expected_sim_telarray_metadata": (
            "assert_expected_sim_telarray_metadata",
            "expected_sim_telarray_metadata",
        ),
    }

    for file_key, (func_name, param_name) in expected_output_key_map.items():
        if file_key in file_test:
            func = getattr(simtel_output_validator, func_name)
            assert_sim_telarray.append(func(file=file, **{param_name: file_test[file_key]}))

    assert_sim_telarray.append(
        simtel_output_validator.assert_n_showers_and_energy_range(
            file,
            calibration_file=file_test.get("expected_sim_telarray_output", {}).get(
                "require_calibration_events", False
            ),
        )
    )

    return all(assert_sim_telarray)


def check_log_files(log_file, file_test):
    """
    Check log file (plain, tar) for wanted and forbidden patterns.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    file_test : dict
        Dictionary with the test configuration.

    Returns
    -------
    bool
        True if the logs are correct.
    """
    if str(log_file).endswith(".tar.gz"):
        return check_tar_logs(log_file, file_test)
    return check_plain_logs(log_file, file_test)
