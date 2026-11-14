"""Tests for the Bypass_Optics capability gate of the installed sim_telarray."""

from types import SimpleNamespace

import pytest

from simtools.simtel import sim_telarray_capabilities as capabilities
from simtools.simtel.sim_telarray_capabilities import (
    BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE,
    bypass_optics_override,
    check_bypass_optics,
    installed_sim_telarray_version,
    supports_bypass_optics,
)


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.delenv(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, raising=False)


@pytest.fixture
def release_2026(monkeypatch):
    """Version table with mode 3 from release 2026.300.0 on."""
    monkeypatch.setattr(capabilities, "BYPASS_OPTICS_MODE_3_MIN_VERSION", "2026.300.0")


def _release_output(release):
    return SimpleNamespace(stdout=f"Release: {release} from 2026-10-01\n", stderr="")


def test_no_known_release_supports_mode_3_by_default():
    assert capabilities.BYPASS_OPTICS_MODE_3_MIN_VERSION is None
    assert supports_bypass_optics(3, "9999.1.1") is False
    assert supports_bypass_optics(3, None) is False


@pytest.mark.parametrize("level", [0, 1, 2])
def test_levels_0_to_2_always_supported(level):
    assert supports_bypass_optics(level, None) is True
    assert supports_bypass_optics(level, "2024.271.0") is True


@pytest.mark.parametrize("level", [-1, 4, "3"])
def test_unknown_levels_not_supported(level):
    assert supports_bypass_optics(level, "9999.1.1") is False


def test_version_table_mode_3(release_2026):
    assert supports_bypass_optics(3, "2026.300.0") is True
    assert supports_bypass_optics(3, "2026.301.0") is True
    assert supports_bypass_optics(3, "2026.299.9") is False
    assert supports_bypass_optics(3, "2024.271.0") is False
    assert supports_bypass_optics(3, None) is False


def test_version_table_invalid_version(release_2026):
    assert supports_bypass_optics(3, "not-a-version") is False


def test_override_takes_precedence_over_table(release_2026):
    assert supports_bypass_optics(3, "2024.271.0", max_level_override=3) is True
    assert supports_bypass_optics(3, "2026.301.0", max_level_override=2) is False
    assert supports_bypass_optics(2, None, max_level_override=1) is False


def test_override_variable_parsing(monkeypatch):
    monkeypatch.delenv(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, raising=False)
    assert bypass_optics_override() is None

    monkeypatch.setenv(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, "  ")
    assert bypass_optics_override() is None

    monkeypatch.setenv(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, "3")
    assert bypass_optics_override() == 3


def test_override_variable_must_be_integer(monkeypatch):
    monkeypatch.setenv(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, "three")

    with pytest.raises(ValueError, match="must be an integer, got 'three'"):
        bypass_optics_override()


def test_installed_version_read_from_sim_telarray(run_context, mocker, no_override):
    run = mocker.patch(
        "simtools.runtime.dependencies.subprocess.run", return_value=_release_output("2026.301.0")
    )

    assert installed_sim_telarray_version(run_context) == "2026.301.0"
    run.assert_called_once()
    assert run.call_args.args[0][-1] == "--version"


def test_installed_version_uses_container_prefix(run_context, mocker, no_override):
    run = mocker.patch(
        "simtools.runtime.dependencies.subprocess.run", return_value=_release_output("2026.301.0")
    )
    context = run_context.replace(run_time=["docker", "run", "image"])

    installed_sim_telarray_version(context)

    assert run.call_args.args[0][:3] == ["docker", "run", "image"]


def test_installed_version_unknown_when_no_release_line(run_context, mocker, no_override):
    mocker.patch(
        "simtools.runtime.dependencies.subprocess.run",
        return_value=SimpleNamespace(stdout="no release here", stderr=""),
    )

    assert installed_sim_telarray_version(run_context) is None


def test_check_levels_0_to_2_do_not_probe(run_context, mocker, no_override):
    run = mocker.patch("simtools.runtime.dependencies.subprocess.run")

    for level in (0, 1, 2):
        check_bypass_optics(level, run_context)

    run.assert_not_called()


def test_check_mode_3_refused_names_version_level_and_override(run_context, mocker, no_override):
    mocker.patch(
        "simtools.runtime.dependencies.subprocess.run", return_value=_release_output("2024.271.0")
    )

    with pytest.raises(ValueError, match=r"Bypass_Optics=3 is not supported") as exc:
        check_bypass_optics(3, run_context)

    message = str(exc.value)
    assert "2024.271.0" in message
    assert BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE in message


def test_check_mode_3_accepted_by_release_table(run_context, mocker, release_2026, no_override):
    mocker.patch(
        "simtools.runtime.dependencies.subprocess.run", return_value=_release_output("2026.301.0")
    )

    check_bypass_optics(3, run_context)


def test_check_mode_3_accepted_by_override_without_probe(run_context, mocker, monkeypatch):
    monkeypatch.setenv(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, "3")
    run = mocker.patch("simtools.runtime.dependencies.subprocess.run")

    check_bypass_optics(3, run_context)

    run.assert_not_called()


def test_check_mode_3_refused_by_too_low_override(run_context, mocker, monkeypatch):
    monkeypatch.setenv(BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE, "2")
    run = mocker.patch("simtools.runtime.dependencies.subprocess.run")

    with pytest.raises(ValueError, match=rf"{BYPASS_OPTICS_MAX_ENVIRONMENT_VARIABLE}=2"):
        check_bypass_optics(3, run_context)

    run.assert_not_called()
