"""Validation of reduced event list files."""

import logging

from simtools.io import table_handler
from simtools.utils import general

_logger = logging.getLogger(__name__)


def validate_sim_events(data_files, expected_mc_events):
    """
    Validate reduced event lists files.

    Parameters
    ----------
    data_files: str, Path, list
        Path(s) to the reduced event list files to validate.
    expected_mc_events: int
        Expected number of simulated MC events.
    """
    data_files = general.ensure_list(data_files)
    validate_event_numbers(data_files, expected_mc_events)


def validate_event_numbers(data_files, expected_mc_events):
    """
    Validate that the number of simulated events in reduced event lists matches the expected number.

    Parameters
    ----------
    data_files: str, Path, list
        Path(s) to the reduced event list files to validate.
    expected_mc_events: int
        Expected number of simulated MC events.

    Raises
    ------
    ValueError
        If the number of simulated events does not match the expected number.
    """
    data_files = general.ensure_list(data_files)

    event_errors = []
    for data_file in data_files:
        if table_handler.read_table_file_type([data_file]) != "HDF5":
            raise ValueError(
                f"Unsupported reduced event data format for '{data_file}'. "
                "Only HDF5 files with suffix '.hdf5' or '.h5' are supported."
            )
        tables = table_handler.read_tables(data_file, ["SHOWERS"], file_type="HDF5")
        try:
            mc_events = len(tables["SHOWERS"])
        except KeyError as exc:
            raise ValueError(f"SHOWERS table not found in reduced event list {data_file}.") from exc

        if mc_events != expected_mc_events:
            event_errors.append(
                f"Number of simulated MC events ({mc_events}) does not match "
                f"the expected number ({expected_mc_events}) in reduced event list {data_file}."
            )
        else:
            _logger.info(
                f"Consistent number of events in reduced event list: {data_file}: MC events:"
                f" {mc_events} (expected: {expected_mc_events})"
            )

    if event_errors:
        _logger.error("Inconsistent event counts found in reduced event lists:")
        for error in event_errors:
            _logger.error(f" - {error}")
        error_message = "Inconsistent event counts found in reduced event lists:\n" + "\n".join(
            f" - {error}" for error in event_errors
        )
        raise ValueError(error_message)
