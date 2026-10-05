"""Tests for pytest options mirrored by environment variables."""

from simtools.testing import options


class _Config:
    def __init__(self, values):
        self.values = values

    def getoption(self, name, default=None):
        return self.values.get(name, default)


def test_get_mirrored_option_prefers_command_line(monkeypatch):
    """Use an explicit pytest option before the environment fallback."""
    monkeypatch.setenv("SIMTOOLS_TESTS_PATH", "/environment")

    assert options.get_mirrored_option(
        _Config({"simtools_tests_path": "/command-line"}), "simtools_tests_path"
    ) == "/command-line"


def test_get_mirrored_option_uses_environment(monkeypatch):
    """Use the environment value when the pytest option is absent."""
    monkeypatch.setenv("SIMTOOLS_TESTS_RESOURCE_VERSION", "v0.38.0")

    assert options.get_mirrored_option(
        _Config({}), "simtools_tests_resource_version"
    ) == "v0.38.0"
