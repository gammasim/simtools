"""Shared pytest configuration."""

from pathlib import Path

import pytest
from dotenv import load_dotenv

from simtools import dependency_versions
from simtools import version as versioning
from simtools.testing import options

pytest_plugins = ("resource_benchmark",)

SIMTOOLS_ROOT_PATH = Path(__file__).resolve().parent.parent


def _is_integration_test_argument(argument):
    """Return whether a pytest argument points into the integration tests."""
    argument_path = Path(str(argument).split("::", maxsplit=1)[0])
    if not argument_path.is_absolute():
        argument_path = Path.cwd() / argument_path
    try:
        argument_path.resolve().relative_to(SIMTOOLS_ROOT_PATH / "tests" / "integration_tests")
    except ValueError:
        return False
    return True


def _load_integration_environment(config):
    """Load the repository .env before integration test collection."""
    if any(_is_integration_test_argument(argument) for argument in config.args):
        load_dotenv(SIMTOOLS_ROOT_PATH / ".env")


def _versioned_test_resources_path(config, version, integration_test_run=False):
    """Return the selected local version for an integration-test run."""
    if not integration_test_run:
        return None
    test_path = options.get_mirrored_option(config, "simtools_tests_path")
    if not test_path or not version:
        return None
    return Path(test_path).expanduser() / version / "integration_tests"


def _catalog_test_resources_version():
    """Return the default integration-test resource version from the catalog."""
    catalog = dependency_versions.load_dependency_catalog(
        SIMTOOLS_ROOT_PATH / "dependency_versions.yml"
    )
    return catalog["simtools-tests"]["resource-version"]


def _configured_test_resources_path(config):
    """Return the absolute path to the configured test resources directory."""
    integration_test_run = any(_is_integration_test_argument(argument) for argument in config.args)
    resource_version = options.get_mirrored_option(config, "simtools_tests_resource_version")
    resource_version = resource_version or _catalog_test_resources_version()
    if resource_version:
        versioning.validate_release_tag(resource_version)
    path = _versioned_test_resources_path(config, resource_version, integration_test_run)
    path = path or SIMTOOLS_ROOT_PATH / "tests" / "unit_tests" / "resources"
    return Path(path).expanduser().resolve()


def pytest_addoption(parser):
    """Register test options mirrored by environment variables."""
    options.add_mirrored_options(parser)


def pytest_configure(config):
    """Configure test resource constants before test modules are imported."""
    import simtools.constants

    _load_integration_environment(config)
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
