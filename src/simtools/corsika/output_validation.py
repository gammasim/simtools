"""Validation of CORSIKA output files and logs."""

import logging

from simtools.io.log_inspector import check_plain_logs
from simtools.sim_events import file_info
from simtools.utils import value_conversion

_logger = logging.getLogger(__name__)


def _raise_inconsistent(title, errors):
    """Raise a ValueError listing inconsistent entries under a title."""
    raise ValueError(f"{title}\n" + "\n".join(f" - {error}" for error in errors))


def validate_corsika_output(data_files, log_files, expected_shower_events=None, curved_atmo=False):
    """
    Validate CORSIKA output files.

    Parameters
    ----------
    data_files, log_files: list of Path
        CORSIKA data and log files.
    expected_shower_events: int, optional
        Expected number of shower events.
    curved_atmo: bool, optional
        Whether CORSIKA was run with the curved atmosphere option.
    """
    validate_corsika_event_numbers(data_files, expected_shower_events)
    validate_corsika_logs(
        log_files, expected_shower_events=expected_shower_events, curved_atmo=curved_atmo
    )


def validate_corsika_event_numbers(data_files, expected_shower_events=None, tolerance=1.0e-3):
    """
    Validate the number of simulated events in CORSIKA output files.

    A small relative mismatch (tolerance) is only logged as a warning.

    Parameters
    ----------
    data_files: list of Path
        CORSIKA data files.
    expected_shower_events: int
        Expected number of simulated shower events.
    tolerance: float
        Relative tolerance for a small mismatch.
    """
    if expected_shower_events is None:
        _logger.info("Skipping CORSIKA event-count validation: no expected count was supplied.")
        return

    event_errors = []
    for file in value_conversion.ensure_list(data_files):
        shower_events, _ = file_info.get_simulated_events(file)
        if shower_events == expected_shower_events:
            _logger.info(
                f"Consistent number of events in: {file}: shower events: {shower_events}"
                f" (expected: {expected_shower_events})"
            )
        elif (
            abs(shower_events - expected_shower_events) / max(shower_events, expected_shower_events)
            <= tolerance
        ):
            _logger.warning(
                f"Small mismatch in number of events in: {file}: "
                f"shower events: {shower_events} (expected: {expected_shower_events})"
            )
        else:
            event_errors.append(
                f"Number of simulated shower events ({shower_events}) does not match "
                f"the expected number ({expected_shower_events}) in CORSIKA {file}."
            )

    if event_errors:
        _raise_inconsistent("Inconsistent event counts found in CORSIKA output:", event_errors)


def validate_corsika_logs(log_files, expected_shower_events=None, curved_atmo=False):
    """
    Validate CORSIKA log files (raises ValueError if the expected patterns are missing).

    Parameters
    ----------
    log_files: list of Path
        CORSIKA log files.
    expected_shower_events: int, optional
        Expected number of shower events.
    curved_atmo: bool, optional
        Whether CORSIKA was run with the curved atmosphere option.
    """
    patterns = ["========== END OF RUN ======="]
    forbidden = []
    if expected_shower_events is not None:
        patterns.append(f"NUMBER OF GENERATED EVENTS =          {expected_shower_events}")
    if curved_atmo:
        patterns.append("CURVED VERSION WITH SLIDING PLANAR ATMOSPHERE")
    else:
        forbidden.append("CORSIKA was compiled without CURVED option.")

    if not check_plain_logs(log_files, {"pattern": patterns, "forbidden_pattern": forbidden}):
        raise ValueError(
            f"Log files {log_files} do not contain expected patterns "
            "indicating successful completion."
        )
