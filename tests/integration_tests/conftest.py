"""Common fixtures for integration tests."""

import pytest

MODEL_VERSION_INDEPENDENT = "model_version_independent"


def pytest_addoption(parser):
    """Integration-test command line parameters."""
    parser.addoption("--model_version", action="store", default=None)
    parser.addoption(
        "--skip_model_version_independent",
        action="store_true",
        default=False,
        help=(
            "Deselect integration tests marked as model_version_independent, whose commands "
            "do not depend on --model_version."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Deselect model-version independent tests when requested."""
    if not config.getoption("skip_model_version_independent", default=False):
        return
    selected = []
    deselected = []
    for item in items:
        if item.get_closest_marker(MODEL_VERSION_INDEPENDENT):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture
def tmp_test_directory(tmpdir_factory):
    """Sets temporary test directories. Some tests depend on this structure."""

    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ["resources", "output", "sim_telarray", "model", "application-plots"]
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir / sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir
