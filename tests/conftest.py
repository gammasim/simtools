"""Shared pytest configuration."""

from pathlib import Path

import pytest

SIMTOOLS_ROOT_PATH = Path(__file__).resolve().parent.parent


def _configured_test_resources_path(config):
    """Return the absolute path to the configured test resources directory."""
    configured_path = config.getoption("test_resources_path", default=None)
    path = configured_path or SIMTOOLS_ROOT_PATH / "tests" / "resources"
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


@pytest.fixture(scope="session")
def simtools_root_path():
    """Return the path to the simtools repository root."""
    return SIMTOOLS_ROOT_PATH
