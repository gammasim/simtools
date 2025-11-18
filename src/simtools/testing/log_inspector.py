"""Inspect logs and output for errors and warnings."""

import logging
import re

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
    re.compile(r"Failed to get user name.*setting it to UNKNOWN_USER", re.IGNORECASE),
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
