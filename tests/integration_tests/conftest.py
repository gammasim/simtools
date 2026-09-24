"""Common fixtures for integration tests."""

import pytest

import simtools.io.io_handler
from simtools import settings


def pytest_addoption(parser):
    """Model version command line parameter."""
    parser.addoption("--model_version", action="store", default=None)
    parser.addoption(
        "--simulation_models_path",
        action="store",
        default=None,
        help="Read simulation models from files at this path.",
    )
    parser.addoption(
        "--simulation_models_git_path",
        action="store",
        default=None,
        help="Read simulation models from a local Git repository at this path.",
    )
    parser.addoption(
        "--simulation_models_git_revision",
        action="store",
        default=None,
        help="Git revision used for the simulation-model reader.",
    )


@pytest.fixture(autouse=True)
def simtools_settings():
    """Load simtools settings for a test."""
    settings.config.load()


@pytest.fixture
def tmp_test_directory(tmpdir_factory):
    """Sets temporary test directories. Some tests depend on this structure."""

    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ["resources", "output", "sim_telarray", "model", "application-plots"]
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir / sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir


@pytest.fixture(autouse=True)
def io_handler(tmp_test_directory):
    """Define io_handler fixture including output and model directories."""
    tmp_io_handler = simtools.io.io_handler.IOHandler()
    tmp_io_handler.set_paths(
        output_path=str(tmp_test_directory) + "/output",
        model_path=str(tmp_test_directory) + "/model",
    )
    return tmp_io_handler
