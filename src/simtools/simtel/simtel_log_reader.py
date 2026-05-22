"""Reader for SIMTEL log files to extract trigger statistics and event counts."""

import gzip
import logging
import re
from pathlib import Path

_logger = logging.getLogger(__name__)

# Regex patterns for extracting information from log files
TRIGGERED_PATTERN = re.compile(r"Tel\.\s+triggered:\s+(\d+)")
EVENT_COUNT_PATTERN = re.compile(r"Run\(s\) completed as expected after\s+(\d+)\s+events")
RUN_NUMBER_PATTERN = re.compile(r"run(\d+)", re.IGNORECASE)


def read_log_file(file_path):
    """
    Read a log file (plain text or gzipped).

    Parameters
    ----------
    file_path : Path or str
        Path to the log file.

    Returns
    -------
    str
        Contents of the log file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    if file_path.suffix == ".gz":
        with gzip.open(file_path, mode="rt", encoding="utf-8") as f:
            return f.read()
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def extract_trigger_count(log_text):
    """
    Extract telescope trigger count from log text.

    Searches for the pattern "Tel. triggered: N" in the log.

    Parameters
    ----------
    log_text : str
        Contents of the log file.

    Returns
    -------
    int or None
        Number of telescopes triggered, or None if not found.
    """
    match = TRIGGERED_PATTERN.search(log_text)
    if match:
        return int(match.group(1))
    _logger.warning("Could not find 'Tel. triggered' pattern in log")
    return None


def extract_event_count(log_text):
    """
    Extract total event count from log text.

    Searches for the pattern "Run(s) completed as expected after N events".

    Parameters
    ----------
    log_text : str
        Contents of the log file.

    Returns
    -------
    int or None
        Number of events, or None if not found.
    """
    match = EVENT_COUNT_PATTERN.search(log_text)
    if match:
        return int(match.group(1))
    _logger.warning("Could not find event count pattern in log")
    return None


def extract_run_number(file_path):
    """
    Extract run number from file path or filename.

    Looks for patterns like "run000015" or "run15".

    Parameters
    ----------
    file_path : Path or str
        Path to the log file.

    Returns
    -------
    int or None
        Run number, or None if not found.
    """
    file_path = Path(file_path)

    # Search in filename first
    match = RUN_NUMBER_PATTERN.search(file_path.name)
    if match:
        return int(match.group(1))

    # Search in parent directories
    for part in file_path.parts:
        match = RUN_NUMBER_PATTERN.search(part)
        if match:
            return int(match.group(1))

    _logger.warning(f"Could not extract run number from {file_path}")
    return None


def extract_threshold(file_path):
    """
    Extract threshold value from file path.

    Looks for threshold in parent directory names (e.g., "300/sim_telarray/...").

    Parameters
    ----------
    file_path : Path or str
        Path to the log file.

    Returns
    -------
    int or None
        Threshold value, or None if not found.
    """
    file_path = Path(file_path)

    # Look for a numeric directory name in the path
    # Start from the beginning of the path to find threshold directories
    for part in file_path.parts:
        # Check if this part is a number (threshold directory)
        if part.isdigit():
            threshold = int(part)
            if 10 <= threshold <= 1000:
                return threshold

    _logger.warning(f"Could not extract threshold from {file_path}")
    return None


def parse_log_file(file_path):
    """
    Parse a single log file and extract all relevant information.

    Parameters
    ----------
    file_path : Path or str
        Path to the log file.

    Returns
    -------
    dict or None
        Dictionary with keys: 'run', 'threshold', 'triggers', 'events', 'file_path'.
        Returns None if critical information cannot be extracted.
    """
    file_path = Path(file_path)

    try:
        log_text = read_log_file(file_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Catch all exceptions to continue processing remaining files
        _logger.error(f"Failed to read {file_path}: {e}")
        return None

    run_number = extract_run_number(file_path)
    threshold = extract_threshold(file_path)
    triggers = extract_trigger_count(log_text)
    events = extract_event_count(log_text)

    if run_number is None or threshold is None or triggers is None:
        _logger.warning(
            f"Skipping {file_path}: missing critical info "
            f"(run={run_number}, threshold={threshold}, triggers={triggers})"
        )
        return None

    return {
        "run": run_number,
        "threshold": threshold,
        "triggers": triggers,
        "events": events,
        "file_path": str(file_path),
    }


def crawl_log_files(root_dir, pattern="**/*.simtel.log.gz"):
    """
    Recursively find all log files matching the pattern.

    Parameters
    ----------
    root_dir : Path or str
        Root directory to search in.
    pattern : str, optional
        Glob pattern for log files. Default: "**/*.simtel.log.gz"

    Returns
    -------
    list of Path
        List of log file paths found.

    Raises
    ------
    FileNotFoundError
        If root_dir does not exist or no log files are found.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    log_files = sorted(root_dir.rglob(pattern))

    if not log_files:
        raise FileNotFoundError(f"No log files found in {root_dir} matching pattern '{pattern}'")

    _logger.info(f"Found {len(log_files)} log files in {root_dir}")
    return log_files


def parse_log_files(file_list):
    """
    Parse multiple log files and extract data.

    Parameters
    ----------
    file_list : list of Path
        List of log file paths to parse.

    Returns
    -------
    list of dict
        List of parsed data dictionaries. Failed parses are excluded.
    """
    data = []
    for file_path in file_list:
        _logger.debug(f"Parsing {file_path}")
        result = parse_log_file(file_path)
        if result is not None:
            data.append(result)

    _logger.info(f"Successfully parsed {len(data)} out of {len(file_list)} log files")

    if not data:
        raise ValueError("No log files could be parsed successfully")

    return data
