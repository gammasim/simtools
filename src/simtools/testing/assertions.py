"""Functions asserting certain conditions are met (used e.g., in integration tests)."""

import json
import logging
from pathlib import Path

import h5py
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
        except json.JSONDecodeError, FileNotFoundError:
            return False
    if file_type in ("yaml", "yml"):
        if Path(file_name).suffix[1:] not in ("yaml", "yml"):
            return False
        try:
            with open(file_name, encoding="utf-8") as file:
                yaml.safe_load(file)
            return True
        except yaml.YAMLError, FileNotFoundError:
            return False

    # no dedicated tests for other file types, checking suffix only
    _logger.info(f"File type test is checking suffix only for {file_name} (suffix: {file_type}))")
    return Path(file_name).suffix[1:] == file_type


def assert_hdf5_datasets(file_name, expected_datasets):
    """Assert that an HDF5 file contains the requested root datasets.

    Parameters
    ----------
    file_name : str or pathlib.Path
        HDF5 file to inspect.
    expected_datasets : sequence of str
        Names of root-level HDF5 datasets that must be present.

    Returns
    -------
    bool
        ``True`` when all requested datasets are present.

    Raises
    ------
    AssertionError
        If a requested item is missing or is not an HDF5 dataset.
    OSError
        If ``file_name`` is not a readable HDF5 file.
    """
    with h5py.File(file_name, "r") as hdf5_file:
        missing = []
        invalid = []
        for dataset_name in expected_datasets:
            if dataset_name not in hdf5_file:
                missing.append(dataset_name)
            elif not isinstance(hdf5_file[dataset_name], h5py.Dataset):
                invalid.append(dataset_name)

        if missing or invalid:
            details = []
            if missing:
                details.append(f"missing {missing}")
            if invalid:
                details.append(f"not datasets {invalid}")
            raise AssertionError(
                f"HDF5 file {file_name} has invalid output structure: "
                f"{'; '.join(details)}. Available root items: {list(hdf5_file)}"
            )

    return True


def assert_hdf5_dataset_min_rows(file_name, expected_min_rows):
    """Assert that named HDF5 datasets contain at least the requested number of rows.

    Parameters
    ----------
    file_name : str or pathlib.Path
        HDF5 file to inspect.
    expected_min_rows : mapping
        Mapping from HDF5 dataset names to their minimum required row counts.

    Returns
    -------
    bool
        ``True`` when every requested dataset meets its minimum row count.

    Raises
    ------
    AssertionError
        If a requested item is missing, is not an HDF5 dataset, or contains too few rows.
    OSError
        If ``file_name`` is not a readable HDF5 file.
    """
    with h5py.File(file_name, "r") as hdf5_file:
        for dataset_name, minimum_rows in expected_min_rows.items():
            if dataset_name not in hdf5_file or not isinstance(
                hdf5_file[dataset_name], h5py.Dataset
            ):
                raise AssertionError(
                    f"HDF5 file {file_name} has no dataset named '{dataset_name}'."
                )
            actual_rows = len(hdf5_file[dataset_name])
            if actual_rows < minimum_rows:
                raise AssertionError(
                    f"HDF5 dataset '{dataset_name}' in {file_name} has {actual_rows} row(s), "
                    f"expected at least {minimum_rows}."
                )
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

    event_type = file_test.get("expected_sim_telarray_output", {}).get("event_type", "shower")
    if event_type == "shower":
        assert_sim_telarray.append(simtel_output_validator.assert_n_showers_and_energy_range(file))
    assert_sim_telarray.append(
        simtel_output_validator.assert_events_of_type(file, event_type=event_type)
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
