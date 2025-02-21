#!/usr/bin/python3

import sys

import pytest


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
