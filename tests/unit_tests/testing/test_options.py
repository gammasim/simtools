"""Tests for pytest options mirrored by environment variables."""

import pytest

from simtools.testing import options


class _Config:
    def __init__(self, values):
        self.values = values

    def getoption(self, name, default=None):
        return self.values.get(name, default)


def test_get_mirrored_option_prefers_command_line(monkeypatch):
    """Use an explicit pytest option before the environment fallback."""
    monkeypatch.setenv("SIMTOOLS_TESTS_PATH", "/environment")

    assert (
        options.get_mirrored_option(
            _Config({"simtools_tests_path": "/command-line"}), "simtools_tests_path"
        )
        == "/command-line"
    )


def test_get_mirrored_option_uses_environment(monkeypatch):
    """Use the environment value when the pytest option is absent."""
    monkeypatch.setenv("SIMTOOLS_TESTS_RESOURCE_VERSION", "v0.38.0")

    assert options.get_mirrored_option(_Config({}), "simtools_tests_resource_version") == "v0.38.0"


def test_add_mirrored_options_registers_options():
    """Register every mirrored option with its name and a None default."""
    parser = pytest.Parser(_ispytest=True)
    options.add_mirrored_options(parser)

    defaults = parser.parse_known_args([])
    assert {name: getattr(defaults, name) for name in options.MIRRORED_OPTIONS} == dict.fromkeys(
        options.MIRRORED_OPTIONS
    )

    command_line = [
        argument for name in options.MIRRORED_OPTIONS for argument in (f"--{name}", f"value-{name}")
    ]
    parsed = parser.parse_known_args(command_line)
    assert {name: getattr(parsed, name) for name in options.MIRRORED_OPTIONS} == {
        name: f"value-{name}" for name in options.MIRRORED_OPTIONS
    }
