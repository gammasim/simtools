#!/usr/bin/python3

import sys

import pytest

import simtools.version as version


@pytest.fixture
def simtools_version():
    return "simtools.version"


@pytest.fixture
def simtools_dev_version():
    return "simtools._dev_version"


@pytest.fixture
def simtools__version():
    return "simtools._version"


def test_import_dev_version(monkeypatch, mocker, simtools__version, simtools_version):
    """Test when _dev_version exists and is imported successfully."""

    # Ensure previous imports are cleared safely
    monkeypatch.delitem(sys.modules, simtools_version, raising=False)
    monkeypatch.delitem(sys.modules, simtools__version, raising=False)

    # Mock _dev_version before importing
    mock_dev_version = mocker.Mock()
    mock_dev_version.version = "1.2.3"

    monkeypatch.setitem(sys.modules, simtools__version, mock_dev_version)

    from simtools.version import __version__

    assert __version__ == "1.2.3"


def test_import_release_version(
    monkeypatch, mocker, simtools__version, simtools_dev_version, simtools_version
):
    """Test when _dev_version is missing, but _version is imported."""

    # Ensure previous imports are cleared safely
    monkeypatch.delitem(sys.modules, simtools_version, raising=False)
    monkeypatch.delitem(sys.modules, simtools_dev_version, raising=False)
    monkeypatch.delitem(sys.modules, simtools__version, raising=False)

    # Mock _version as the fallback
    mock_version = mocker.Mock()
    mock_version.version = "2.3.4"

    monkeypatch.setitem(sys.modules, simtools_dev_version, None)
    monkeypatch.setitem(sys.modules, simtools__version, mock_version)

    from simtools.version import __version__

    assert __version__ == "2.3.4"


def test_both_imports_fail(
    monkeypatch, mocker, simtools__version, simtools_dev_version, simtools_version
):
    """Test when both _dev_version and _version are missing."""
    monkeypatch.delitem(sys.modules, simtools_version, raising=False)
    monkeypatch.delitem(sys.modules, simtools_dev_version, raising=False)
    monkeypatch.delitem(sys.modules, simtools__version, raising=False)

    mocker.patch.dict("sys.modules", {simtools_dev_version: None, simtools__version: None})
    mock_warn = mocker.patch("warnings.warn")

    from simtools.version import __version__

    mock_warn.assert_called_once_with(
        "Could not determine simtools version; this indicates a broken installation."
    )
    assert __version__ == "0.0.0"


def test_resolve_version_to_latest_patch():
    available_versions = ["5.0.0", "5.0.1", "6.0.0", "6.0.1", "6.0.2", "6.1.0"]

    partial_version = "6.0.0"
    assert version.resolve_version_to_latest_patch(partial_version, available_versions) == "6.0.0"

    partial_version = "6.0"
    assert version.resolve_version_to_latest_patch(partial_version, available_versions) == "6.0.2"

    partial_version = "5.0"
    assert version.resolve_version_to_latest_patch(partial_version, available_versions) == "5.0.1"

    partial_version = "6.1"
    assert version.resolve_version_to_latest_patch(partial_version, available_versions) == "6.1.0"

    partial_version = "7.1"
    with pytest.raises(ValueError, match=r"^No versions found matching"):
        version.resolve_version_to_latest_patch(partial_version, available_versions)

    with pytest.raises(ValueError, match=r"^No versions found matching"):
        version.resolve_version_to_latest_patch(partial_version, [])

    partial_version = "not_a.version"
    with pytest.raises(ValueError, match=r"^Invalid version string"):
        version.resolve_version_to_latest_patch(partial_version, available_versions)

    partial_version = "6"
    with pytest.raises(ValueError, match=r"^Partial version must be major.minor"):
        version.resolve_version_to_latest_patch(partial_version, available_versions)
