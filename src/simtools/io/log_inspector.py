"""Inspect logs and output for errors and warnings."""

import gzip
import logging
import os
import re
import tarfile
from pathlib import Path, PurePosixPath

from simtools.utils import value_conversion

_logger = logging.getLogger(__name__)

_GZIP_LOG_SUFFIX = ".log.gz"
_GZIP_STDERR_SUFFIX = ".err.gz"


ERROR_PATTERNS = [
    re.compile(r"error", re.IGNORECASE),
    re.compile(r"exception", re.IGNORECASE),
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"\b(failed to|has failed)\b", re.IGNORECASE),
    re.compile(r"runtime\s*warning", re.IGNORECASE),
    re.compile(r"segmentation fault", re.IGNORECASE),
]

IGNORE_PATTERNS = [
    re.compile(r"Falling back to 'utf-8' with errors='ignore'", re.IGNORECASE),
    re.compile(r"Failed to get user name[^\n]*setting it to UNKNOWN_USER", re.IGNORECASE),
    re.compile(r"adjust_text::Error", re.IGNORECASE),
    re.compile(r"Job output/error streams", re.IGNORECASE),
    re.compile(r"^\s*DEBUG(?:::|:)", re.IGNORECASE),
    # typically model parameters like 'transit_time_error'
    re.compile(r"'[A-Za-z0-9_]+_error'", re.IGNORECASE),  # NOSONAR
    re.compile(r"DEBUG: Setting environment variables for job execution:", re.IGNORECASE),
    re.compile(r"INFO:", re.IGNORECASE),
]


def inspect(log_text):
    """
    Inspect log text for errors and run-time warnings.

    Ignore any lines containing "INFO::" (to avoid false positives
    like "INFO:: Job error stream ").

    Parameters
    ----------
    log_text: str or list of str
        Text of the log to inspect.

    Returns
    -------
    bool
        True if no errors or warnings are found, False otherwise.
    """
    log_text = log_text if isinstance(log_text, list) else [log_text]

    issues = [
        (lineno, line)
        for txt in log_text
        for lineno, line in enumerate(txt.splitlines(), 1)
        if "INFO::" not in line
        and any(p.search(line) for p in ERROR_PATTERNS)
        and not any(p.search(line) for p in IGNORE_PATTERNS)
    ]

    for lineno, line in issues:
        _logger.error(f"Error or warning found in log at line {lineno}: {line.strip()}")

    return not issues


def check_tar_logs(tar_file, file_test):
    """
    Check log files in tar package for wanted and forbidden patterns.

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
    if wanted is None and forbidden is None:
        return True

    if not tarfile.is_tarfile(tar_file):
        raise ValueError(f"{tar_file} is not a tar file")

    found_wanted = set()
    found_forbidden = set()
    with tarfile.open(tar_file, "r:*") as tar:
        for member in tar.getmembers():
            if not _is_valid_log_member(member):
                continue
            if not is_safe_tar_member(member.name):
                _logger.warning(f"Skipping potentially unsafe tar member: {member.name}")
                continue
            _logger.info(f"Scanning {member.name}")
            text = _read_log(member, tar)
            found_wanted |= _find_patterns(text, wanted)
            found_forbidden |= _find_patterns(text, forbidden)

    return _validate_patterns(found_wanted, found_forbidden, wanted)


def _is_valid_log_member(member):
    """Validate member suffix for tar log scanning."""
    return member.name.endswith((_GZIP_LOG_SUFFIX, _GZIP_STDERR_SUFFIX))


def _read_log(member, tar):
    """Read and decode a gzipped tar member, or return None if extraction fails."""
    source = tar.extractfile(member)
    if source is None:
        return None

    with source, gzip.open(source, "rb") as f:
        return f.read().decode("utf-8", "ignore")


def check_plain_logs(log_files, file_test):
    """
    Check plain log file(s) for wanted and forbidden patterns.

    Log file can be plain or gzipped.

    Parameters
    ----------
    log_files : List, str
        Path to the log file.
    file_test : dict
        Dictionary with the test configuration.

    Returns
    -------
    bool
        True if the logs are correct.
    """
    wanted, forbidden = _get_expected_patterns(file_test)
    if wanted is None and forbidden is None:
        return True

    log_files = _include_stderr_logs(value_conversion.ensure_list(log_files))

    def file_open(file):
        if file.suffix == ".gz":
            return gzip.open(file, "rt", encoding="utf-8", errors="ignore")
        return open(file, encoding="utf-8", errors="ignore")

    found = set()
    bad = set()
    for log_file in log_files:
        try:
            with file_open(log_file) as f:
                text = f.read()
        except FileNotFoundError:
            _logger.error(f"Log file {log_file} not found")
            return False

        found |= _find_patterns(text, wanted)
        bad |= _find_patterns(text, forbidden)

    return _validate_patterns(found, bad, wanted)


def _include_stderr_logs(log_files):
    """Include existing stderr siblings without requiring them for older logs."""
    paths = []
    for log_file in log_files:
        path = Path(log_file)
        if path not in paths:
            paths.append(path)
        for stdout_suffix, stderr_suffix in (
            (_GZIP_LOG_SUFFIX, _GZIP_STDERR_SUFFIX),
            (".log", ".err"),
        ):
            if path.name.endswith(stdout_suffix):
                sibling = path.with_name(path.name.removesuffix(stdout_suffix) + stderr_suffix)
                if sibling.is_file() and sibling not in paths:
                    paths.append(sibling)
                break
    return paths


def _get_expected_patterns(file_test):
    """
    Get wanted and forbidden patterns from file test configuration.

    Parameters
    ----------
    file_test : dict
        Dictionary with expected / forbidden patterns
    """
    expected_log = file_test.get("expected_log_output", file_test)

    if not isinstance(expected_log, dict):
        _logger.debug("No expected log output provided, skipping checks %s", file_test)
        return None, None

    wanted = expected_log.get("pattern", [])
    forbidden = expected_log.get("forbidden_pattern", [])

    if not (wanted or forbidden):
        _logger.debug("No expected log output provided, skipping checks %s", file_test)
        return None, None

    return wanted, forbidden


def _validate_patterns(found, bad, wanted):
    """Validate found patterns against wanted and forbidden ones."""
    if bad:
        _logger.error(f"Forbidden patterns found: {list(bad)}")
        return False
    missing = [p for p in wanted if p and p not in found]
    if missing:
        _logger.error(f"Missing expected patterns: {missing}")
        return False

    _logger.debug(f"All expected patterns found: {wanted}")
    return True


def _find_patterns(text, patterns):
    """Find patterns in text (case and space insensitive)."""

    def _normalize(s):
        return re.sub(r"\s+", "", s.lower())

    text_n = _normalize(text)
    return {p for p in patterns if p and _normalize(p) in text_n}


def collect_final_lines(file, n_lines):
    """
    Collect final lines.

    Parameters
    ----------
    file: str or Path
        File to collect the lines from.
    n_lines: int
        Number of lines to be collected.

    Returns
    -------
    str
        Final lines collected.
    """
    list_of_lines = []

    if Path(file).suffix == ".gz":
        file_open_function = gzip.open
    else:
        file_open_function = open
    with file_open_function(file, "rb") as read_obj:
        # Move the cursor to the end of the file
        read_obj.seek(0, os.SEEK_END)
        # Create a buffer to keep the last read line
        buffer = bytearray()
        # Get the current position of pointer i.e eof
        pointer_location = read_obj.tell()
        # Loop till pointer reaches the top of the file
        while pointer_location >= 0:
            # Move the file pointer to the location pointed by pointer_location
            read_obj.seek(pointer_location)
            # Shift pointer location by -1
            pointer_location = pointer_location - 1
            # read that byte / character
            new_byte = read_obj.read(1)
            # If the read byte is new line character then it means one line is read
            if new_byte == b"\n":
                # Save the line in list of lines
                list_of_lines.append(buffer.decode()[::-1])
                # If the size of list reaches n_lines, then return the reversed list
                if len(list_of_lines) == n_lines:
                    return "".join(list(reversed(list_of_lines)))
                # Reinitialize the byte array to save next line
                buffer = bytearray()
            else:
                # If last read character is not eol then add it in buffer
                buffer.extend(new_byte)
        # As file is read completely, if there is still data in buffer, then its first line.
        if len(buffer) > 0:
            list_of_lines.append(buffer.decode()[::-1])

    return "".join(list(reversed(list_of_lines)))


def is_safe_tar_member(member_name):
    """
    Validate that a tar member path is safe and doesn't contain path traversal sequences.

    Parameters
    ----------
    member_name : str
        The path of the tar member to validate.

    Returns
    -------
    bool
        True if the path is safe, False otherwise.
    """
    # Check for null bytes
    if "\0" in member_name:
        return False

    # Check for absolute paths
    if Path(member_name).is_absolute():
        return False

    # Normalize path and validate components
    try:
        parts = PurePosixPath(member_name).parts
    except ValueError:
        return False

    # Reject if any component is ".." or if path becomes absolute
    if ".." in parts:
        return False

    # Verify the normalized path is still relative
    normalized = PurePosixPath(*parts) if parts else PurePosixPath()
    if normalized.is_absolute():
        return False

    return True


def get_log_excerpt(log_file, n_last_lines=30):
    """
    Get an excerpt from a log file, namely the n_last_lines of the file.

    Parameters
    ----------
    log_file: str or Path
        Log file to get the excerpt from.
    n_last_lines: int
        Number of last lines of the file to get.

    Returns
    -------
    str
        Excerpt from log file with header/footer
    """
    return (
        "\n\nRuntime error - See below the relevant part of the log/err file.\n\n"
        f"{log_file}\n"
        "====================================================================\n\n"
        f"{collect_final_lines(log_file, n_last_lines)}\n\n"
        "====================================================================\n"
    )
