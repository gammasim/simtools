"""Functions asserting certain conditions are met (used e.g., in integration tests)."""

import json
import logging

import yaml

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
        try:
            with open(file_name, encoding="utf-8") as file:
                yaml.safe_load(file)
            return True
        except (yaml.YAMLError, FileNotFoundError):
            return False

    # no dedicated tests for other file types, checking suffix only
    _logger.info(f"File type test is checking suffix only for {file_name} (suffix: {file_type}))")
    if file_name.suffix[1:] == file_type:
        return True

    return False
