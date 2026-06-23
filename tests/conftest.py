"""Shared pytest configuration."""

from pathlib import Path

import pytest


def _configured_test_resources_path(config):
    """Return the absolute path to the configured test resources directory."""
    configured_path = config.getoption("test_resources_path")
    path = configured_path or config.rootpath / "tests" / "resources"
    return Path(path).expanduser().resolve()


def pytest_addoption(parser):
    """Register the test-resources directory command-line option."""
    parser.addoption(
        "--test_resources_path",
        "--test-resources-path",
        dest="test_resources_path",
        type=Path,
        default=None,
        help="Full path to the test resources directory (default: tests/resources).",
    )


def pytest_configure(config):
    """Configure test resource constants before test modules are imported."""
    import simtools.constants

    test_resources_path = _configured_test_resources_path(config)
    simtools.constants.TEST_RESOURCES_STATIC = str(test_resources_path / "static")
    simtools.constants.TEST_RESOURCES_GENERATED = str(test_resources_path / "generated")


@pytest.fixture(scope="session")
def test_resources_path(pytestconfig):
    """Return the absolute path to the test resources directory."""
    return _configured_test_resources_path(pytestconfig)
