"""Validation of CORISKA data and log files."""

import logging

from simtools.sim_events import file_info
from simtools.utils import general

_logger = logging.getLogger(__name__)


def validate_corsika_output(data_files, log_files, expected_mc_events=None):
    """
    Validate CORSIKA output files.

    Parameters
    ----------
    data_files: list of Path
        List of paths to CORSIKA data files.
    log_files: list of Path
        List of paths to CORSIKA log files.
    expected_mc_events: int, optional
        Expected number of MC events.

    Raises
    ------
    ValueError
         If the number of events in the data files does not match the expected values.
     IOError
         If there is an issue reading the data or log files.
    """
    data_files = general.ensure_iterable(data_files) if data_files is not None else []
    log_files = general.ensure_iterable(log_files)

    validate_event_numbers(data_files, expected_mc_events)


def validate_event_numbers(data_files, expected_mc_events, tolerance=1.0e-3):
    """
    Verify the number of simulated events in CORSIKA output files.

    Allow for a small mismatch in the number of requested events.

    Parameters
    ----------
    expected_mc_events: int
        Expected number of simulated MC events.

    Raises
    ------
    ValueError
        If the number of simulated events does not match the expected number.
    """

    def consistent(a, b, tol):
        return abs(a - b) / max(a, b) <= tol

    event_errors = []

    for file in data_files:
        shower_events, _ = file_info.get_simulated_events(file)

        if shower_events != expected_mc_events:
            if consistent(shower_events, expected_mc_events, tol=tolerance):
                _logger.warning(
                    f"Small mismatch in number of events in: {file}: "
                    f"shower events: {shower_events} (expected: {expected_mc_events})"
                )
            else:
                event_errors.append(
                    f"Number of simulated MC events ({shower_events}) does not match "
                    f"the expected number ({expected_mc_events}) in CORSIKA {file}."
                )
        else:
            _logger.info(
                f"Consistent number of events in: {file}: shower events: {shower_events}"
                f" (expected: {expected_mc_events})"
            )

        if event_errors:
            _logger.error("Inconsistent event counts found in CORSIKA output:")
            for error in event_errors:
                _logger.error(f" - {error}")
            error_message = "Inconsistent event counts found in CORSIKA output:\n" + "\n".join(
                f" - {error}" for error in event_errors
            )
            raise ValueError(error_message)
