"""Functions asserting certain conditions are met (used e.g., in integration tests)."""

import gzip
import json
import logging
import tarfile
from pathlib import Path

import yaml

from simtools.testing import sim_telarray_output

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
            func = getattr(sim_telarray_output, func_name)
            assert_sim_telarray.append(func(file=file, **{param_name: file_test[file_key]}))

    assert_sim_telarray.append(
        sim_telarray_output.assert_n_showers_and_energy_range(
            file,
            calibration_file=file_test.get("expected_sim_telarray_output", {}).get(
                "require_calibration_events", False
            ),
        )
    )

    return all(assert_sim_telarray)


def check_simulation_logs(tar_file, file_test):
    """
    Check simulation logs for wanted and forbidden patterns.

    Parameters
    ----------
    tar_file : str
        Path to the tar file.
    file_test : dict
        Dictionary with the test configuration.

    Returns
    -------
    bool
        True if the logs are correct.
    """
    wanted, forbidden = _get_expected_patterns(file_test)
    if wanted is None:
        return True

    if not tarfile.is_tarfile(tar_file):
        raise ValueError(f"{tar_file} is not a tar file")

    found_wanted = set()
    found_forbidden = set()
    with tarfile.open(tar_file, "r:*") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".log.gz"):
                continue
            _logger.info(f"Scanning {member.name}")
            text = _read_log(member, tar)
            found_wanted |= _find_patterns(text, wanted)
            found_forbidden |= _find_patterns(text, forbidden)

    return _validate_patterns(found_wanted, found_forbidden, wanted)


def check_plain_log(log_file, file_test):
    """
    Check plain log file for wanted and forbidden patterns.

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
    wanted, forbidden = _get_expected_patterns(file_test)
    if wanted is None:
        return True

    try:
        with open(log_file, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        _logger.error(f"Log file {log_file} not found")
        return False

    found = _find_patterns(text, wanted)
    bad = _find_patterns(text, forbidden)

    return _validate_patterns(found, bad, wanted)


def _get_expected_patterns(file_test):
    """Get wanted and forbidden patterns from file test configuration."""
    expected_log = file_test.get("expected_log_output")
    if isinstance(expected_log, dict):
        wanted = expected_log.get("pattern", [])
        forbidden = expected_log.get("forbidden_pattern", [])
    else:
        wanted = file_test.get("pattern", [])
        forbidden = file_test.get("forbidden_pattern", [])
    if not (wanted or forbidden):
        _logger.debug(f"No expected log output provided, skipping checks {file_test}")
        return None, None

    return wanted, forbidden


def _validate_patterns(found, bad, wanted):
    """Validate found patterns against wanted and forbidden ones."""
    if bad:
        _logger.error(f"Forbidden patterns found: {list(bad)}")
        return False
    missing = [p for p in wanted if p not in found]
    if missing:
        _logger.error(f"Missing expected patterns: {missing}")
        return False

    _logger.debug(f"All expected patterns found: {wanted}")
    return True


def _find_patterns(text, patterns):
    """Find patterns in text (case insensitive)."""
    text_lower = text.lower()
    return {p for p in patterns if p.lower() in text_lower}


def _read_log(member, tar):
    """Read and decode a gzipped log file from a tar archive."""
    with tar.extractfile(member) as gz, gzip.open(gz, "rb") as f:
        return f.read().decode("utf-8", "ignore")
