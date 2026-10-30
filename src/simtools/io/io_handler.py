"""Handle input and output directories and file paths."""

import re
from pathlib import Path

from simtools import constants

_TEST_RESOURCE_PATTERN = re.compile(r"\$\{(static|generated|downloaded):([^}]+)\}")
_TEST_RESOURCE_PATH_PATTERN = re.compile(r"(?<![\w/])(?:\./)?tests/resources(?=/|$)")


def resolve_test_resource_paths(value, test_resources_path=None):
    """Resolve test-resource macros and canonical test-resource paths recursively.

    Parameters
    ----------
    value : object
        Configuration value, mapping, or sequence to resolve.
    test_resources_path : str or pathlib.Path, optional
        Base directory containing the ``static``, ``generated``, and ``downloaded``
        resource directories. The integration-test harness passes this argument explicitly.

    Returns
    -------
    object
        Configuration with absolute test-resource paths.
    """
    if not _contains_test_resource_reference(value):
        return value

    base_path = (
        Path(test_resources_path or constants.get_test_resources_root()).expanduser().resolve()
    )
    if isinstance(value, dict):
        return {
            key: resolve_test_resource_paths(item, test_resources_path=base_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_test_resource_paths(item, test_resources_path=base_path) for item in value]
    if isinstance(value, str):
        value = _TEST_RESOURCE_PATTERN.sub(
            lambda match: str(base_path / match.group(1) / match.group(2)), value
        )
        return _TEST_RESOURCE_PATH_PATTERN.sub(str(base_path), value)
    return value


def _contains_test_resource_reference(value):
    """Return whether a value contains a test-resource macro or legacy path."""
    if isinstance(value, dict):
        return any(_contains_test_resource_reference(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_test_resource_reference(item) for item in value)
    return isinstance(value, str) and (
        _TEST_RESOURCE_PATTERN.search(value) is not None
        or _TEST_RESOURCE_PATH_PATTERN.search(value) is not None
    )
