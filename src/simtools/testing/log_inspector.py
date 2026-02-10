"""Inspect logs and output for errors and warnings."""

import gzip
import logging
import re
import tarfile

from simtools.utils import general as gen

_logger = logging.getLogger(__name__)


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
            if not member.name.endswith(".log.gz"):
                continue
            _logger.info(f"Scanning {member.name}")
            text = _read_log(member, tar)
            found_wanted |= _find_patterns(text, wanted)
            found_forbidden |= _find_patterns(text, forbidden)

    return _validate_patterns(found_wanted, found_forbidden, wanted)


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

    log_files = gen.ensure_iterable(log_files)

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


def _get_expected_patterns(file_test):
    """
    Get wanted and forbidden patterns from file test configuration.

    Parameters
    ----------
    file_test : dict
        Dictionary with expected / forbidden patterns
    """
    expected_log = file_test.get("expected_log_output", file_test)
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


def _read_log(member, tar):
    """Read and decode a gzipped log file from a tar archive."""
    with tar.extractfile(member) as gz, gzip.open(gz, "rb") as f:
        return f.read().decode("utf-8", "ignore")
