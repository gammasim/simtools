"""Functions asserting certain conditions are met (used e.g., in integration tests)."""

import gzip
import json
import logging
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from simtools.simtel.simtel_io_metadata import read_sim_telarray_metadata

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


def assert_n_showers_and_energy_range(file):
    """
    Assert the number of showers and the energy range.

    The number of showers should be consistent with the required one (up to 1% tolerance)
    and the energies simulated are required to be within the configured ones.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.

    """
    from eventio.simtel.simtelfile import SimTelFile  # pylint: disable=import-outside-toplevel

    simulated_energies = []
    simulation_config = {}
    with SimTelFile(file, skip_non_triggered=False) as f:
        simulation_config = f.mc_run_headers[0]
        for event in f:
            simulated_energies.append(event["mc_shower"]["energy"])

    # The relative tolerance is set to 1% because ~0.5% shower simulations do not
    # succeed, without resulting in an error. This tolerance therefore is not an issue.
    consistent_n_showers = np.isclose(
        len(np.unique(simulated_energies)), simulation_config["n_showers"], rtol=1e-2
    )
    consistent_energy_range = all(
        simulation_config["E_range"][0] <= energy <= simulation_config["E_range"][1]
        for energy in simulated_energies
    )

    return consistent_n_showers and consistent_energy_range


def assert_expected_output(file, expected_output):
    """
    Assert that the expected output is present in the sim_telarray file.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    expected_output: dict
        Expected output values.

    """
    from eventio.simtel.simtelfile import SimTelFile  # pylint: disable=import-outside-toplevel

    item_to_check = defaultdict(list)
    with SimTelFile(file) as f:
        for event in f:
            if "pe_sum" in expected_output:
                item_to_check["pe_sum"].extend(
                    event["photoelectron_sums"]["n_pe"][event["photoelectron_sums"]["n_pe"] > 0]
                )
            if "trigger_time" in expected_output:
                item_to_check["trigger_time"].extend(event["trigger_information"]["trigger_times"])
            if "photons" in expected_output:
                item_to_check["photons"].extend(
                    event["photoelectron_sums"]["photons_atm_qe"][
                        event["photoelectron_sums"]["photons"] > 0
                    ]
                )

    for key, value in expected_output.items():
        if len(item_to_check[key]) == 0:
            _logger.error(f"No data found for {key}")
            return False

        if not value[0] < np.mean(item_to_check[key]) < value[1]:
            _logger.error(
                f"Mean of {key} is not in the expected range, got {np.mean(item_to_check[key])}"
            )
            return False

    return True


def assert_expected_simtel_metadata(file, expected_metadata):
    """
    Assert that expected metadata is present in the sim_telarray file.

    Parameters
    ----------
    file: Path
        Path to the sim_telarray file.
    expected_metadata: dict
        Expected metadata values.

    """
    global_meta, telescope_meta = read_sim_telarray_metadata(file)

    for key, value in expected_metadata.items():
        if key not in global_meta and key not in telescope_meta:
            _logger.error(f"Metadata key {key} not found in sim_telarray file {file}")
            return False
        if key in global_meta and global_meta[key] != value:
            _logger.error(
                f"Metadata key {key} has value {global_meta[key]} instead of expected {value}"
            )
            return False
        _logger.debug(f"Metadata key {key} matches expected value {value}")

    return True


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
    if "expected_output" not in file_test and "expected_simtel_metadata" not in file_test:
        _logger.debug(f"No expected output or metadata provided, skipping checks {file_test}")
        return True

    assert_output = assert_metadata = True

    if "expected_output" in file_test:
        assert_output = assert_expected_output(
            file=file, expected_output=file_test["expected_output"]
        )

    if "expected_simtel_metadata" in file_test:
        assert_metadata = assert_expected_simtel_metadata(
            file=file, expected_metadata=file_test["expected_simtel_metadata"]
        )

    return assert_n_showers_and_energy_range(file=file) and assert_output and assert_metadata


def _find_patterns(text, patterns):
    """Find patterns in text."""
    return {p for p in patterns if p in text}


def _read_log(member, tar):
    """Read and decode a gzipped log file from a tar archive."""
    with tar.extractfile(member) as gz, gzip.open(gz, "rb") as f:
        return f.read().decode("utf-8", "ignore")


def check_simulation_logs(tar_file, file_test):
    """
    Check log files of CORSIKA and sim_telarray for expected output.

    Parameters
    ----------
    tar_file: Path
        Path to a log file tar package.
    file_test: dict
        File test description including expected log output.

    Raises
    ------
    ValueError
        If the file is not a tar file.
    """
    expected_log = file_test.get("expected_log_output", {})
    wanted = expected_log.get("pattern", [])
    forbidden = expected_log.get("forbidden_pattern", [])

    if not (wanted or forbidden):
        _logger.debug(f"No expected log output provided, skipping checks {file_test}")
        return True

    if not tarfile.is_tarfile(tar_file):
        raise ValueError(f"File {tar_file} is not a tar file.")

    found, bad = set(), set()
    with tarfile.open(tar_file, "r:*") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".log.gz"):
                continue
            _logger.info(f"Scanning {member.name}")
            text = _read_log(member, tar)
            found |= _find_patterns(text, wanted)
            bad |= _find_patterns(text, forbidden)

    if bad:
        _logger.error(f"Forbidden patterns found: {list(bad)}")
        return False
    missing = [p for p in wanted if p not in found]
    if missing:
        _logger.error(f"Missing expected patterns: {missing}")
        return False

    _logger.debug(f"All expected patterns found: {wanted}")
    return True


def check_plain_log(log_file, file_test):
    """Check a plain .log file for expected and forbidden patterns."""
    expected_log = file_test.get("expected_log_output", {})
    wanted = expected_log.get("pattern", [])
    forbidden = expected_log.get("forbidden_pattern", [])

    if not (wanted or forbidden):
        _logger.debug(f"No expected log output provided, skipping checks {file_test}")
        return True

    try:
        text = Path(log_file).read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        _logger.error(f"Log file {log_file} not found")
        return False

    found = _find_patterns(text, wanted)
    bad = _find_patterns(text, forbidden)

    if bad:
        _logger.error(f"Forbidden patterns found: {list(bad)}")
        return False
    missing = [p for p in wanted if p not in found]
    if missing:
        _logger.error(f"Missing expected patterns: {missing}")
        return False

    _logger.debug(f"All expected patterns found: {wanted}")
    return True
