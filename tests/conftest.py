"""Shared pytest configuration."""

import os
from pathlib import Path

import pytest

pytest_plugins = ("resource_benchmark",)

SIMTOOLS_ROOT_PATH = Path(__file__).resolve().parent.parent
SIMTOOLS_TEST_PATH = (
    Path(os.environ["SIMTOOLS_TEST_PATH"]).expanduser()
    if os.environ.get("SIMTOOLS_TEST_PATH")
    else None
)


def _versioned_test_resources_path(version):
    """Return the selected local version of the integration test resources."""
    if SIMTOOLS_TEST_PATH is None or not version:
        return None
    return SIMTOOLS_TEST_PATH / version / "integration_tests"


def _configured_test_resources_path(config):
    """Return the absolute path to the configured test resources directory."""
    configured_path = config.getoption("test_resources_path", default=None)
    path = configured_path or os.environ.get("SIMTOOLS_TEST_RESOURCES")
    version = config.getoption("simtools_tests_version", default=None) or os.environ.get(
        "SIMTOOLS_TESTS_VERSION"
    )
    path = path or _versioned_test_resources_path(version)
    path = path or SIMTOOLS_ROOT_PATH / "tests" / "unit_tests" / "resources"
    return Path(path).expanduser().resolve()


def pytest_addoption(parser):
    """Register test-resource configuration options."""
    parser.addoption(
        "--test_resources_path",
        dest="test_resources_path",
        type=Path,
        default=os.environ.get("SIMTOOLS_TEST_RESOURCES"),
        help="Full path to test resources (default: SIMTOOLS_TEST_RESOURCES).",
    )
    parser.addoption(
        "--simtools_tests_version",
        dest="simtools_tests_version",
        default=os.environ.get("SIMTOOLS_TESTS_VERSION"),
        help="Version of simtools-tests to use when no path is configured.",
    )


def pytest_configure(config):
    """Configure test resource constants before test modules are imported."""
    import simtools.constants

    test_resources_path = _configured_test_resources_path(config)
    config.option.test_resources_path = test_resources_path
    simtools.constants.TEST_RESOURCES_ROOT = test_resources_path
    simtools.constants.TEST_RESOURCES_STATIC = str(test_resources_path / "static")
    simtools.constants.TEST_RESOURCES_GENERATED = str(test_resources_path / "generated")
    simtools.constants.TEST_RESOURCES_DOWNLOADED = str(test_resources_path / "downloaded")


@pytest.fixture(scope="session")
def test_resources_path(pytestconfig):
    """Return the absolute path to the test resources directory."""
    return _configured_test_resources_path(pytestconfig)


@pytest.fixture(scope="session")
def simtools_root_path():
    """Return the path to the simtools repository root."""
    return SIMTOOLS_ROOT_PATH
