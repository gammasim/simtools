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


def test_semver_to_int():
    assert version.semver_to_int("6.1.1") == 60101
    assert version.semver_to_int("6.1") == 60100

    with pytest.raises(ValueError, match=r"Invalid version: not_a.version"):
        version.semver_to_int("not_a.version")


def test_sort_versions():
    version_list = ["5.0.0", "6.0.2", "5.1.0", "6.0.0", "5.0.1"]

    # Test ascending order (default)
    result = version.sort_versions(version_list)
    expected = ["5.0.0", "5.0.1", "5.1.0", "6.0.0", "6.0.2"]
    assert result == expected

    # Test descending order
    result = version.sort_versions(version_list, reverse=True)
    expected = ["6.0.2", "6.0.0", "5.1.0", "5.0.1", "5.0.0"]
    assert result == expected

    # Test empty list
    assert version.sort_versions([]) == []

    # Test single version
    assert version.sort_versions(["1.0.0"]) == ["1.0.0"]

    # Test invalid version
    invalid_versions = ["1.0.0", "not_a_version", "2.0.0"]
    with pytest.raises(ValueError, match=r"Invalid version in list"):
        version.sort_versions(invalid_versions)


def test_version_kind():
    assert version.version_kind("6.0.0") == version.MAJOR_MINOR_PATCH
    assert version.version_kind("6.0") == version.MAJOR_MINOR
    assert version.version_kind("6") == "major"
    with pytest.raises(ValueError, match=r"Invalid version string"):
        version.version_kind("no_version")


def test_compare_versions():
    # Test exact equality
    assert version.compare_versions("1.0.0", "1.0.0") == 0
    assert version.compare_versions("2.5.3", "2.5.3") == 0

    # Test major version comparison
    assert version.compare_versions("2.0.0", "1.0.0") == 1
    assert version.compare_versions("1.0.0", "2.0.0") == -1

    # Test minor version comparison
    assert version.compare_versions("1.2.0", "1.1.0") == 1
    assert version.compare_versions("1.1.0", "1.2.0") == -1

    # Test patch version comparison
    assert version.compare_versions("1.0.2", "1.0.1") == 1
    assert version.compare_versions("1.0.1", "1.0.2") == -1

    # Test level parameter - major only
    assert version.compare_versions("1.9.9", "1.0.0", level="major") == 0
    assert version.compare_versions("2.0.0", "1.9.9", level="major") == 1
    assert version.compare_versions("1.0.0", "2.9.9", level="major") == -1

    # Test level parameter - major.minor
    assert version.compare_versions("1.2.9", "1.2.0", level=version.MAJOR_MINOR) == 0
    assert version.compare_versions("1.3.0", "1.2.9", level=version.MAJOR_MINOR) == 1
    assert version.compare_versions("1.2.0", "1.3.9", level=version.MAJOR_MINOR) == -1

    # Test level parameter - major.minor.patch (default)
    assert version.compare_versions("1.2.3", "1.2.3", level=version.MAJOR_MINOR_PATCH) == 0
    assert version.compare_versions("1.2.4", "1.2.3", level=version.MAJOR_MINOR_PATCH) == 1
    assert version.compare_versions("1.2.3", "1.2.4", level=version.MAJOR_MINOR_PATCH) == -1

    # Test invalid level
    with pytest.raises(ValueError, match=r"Unknown level"):
        version.compare_versions("1.0.0", "1.0.0", level="invalid")

    # CORSIKA style version (4-digit minor version)
    assert version.compare_versions("7.6900", "7.6400") == 1

    # sim_telarray style version (year, day of year, patch)
    assert version.compare_versions("2025.246.0", "2024.365.0") == 1
    assert version.compare_versions("2024.365.1", "2024.365.0") == 1
    assert version.compare_versions("2024.365.0", "2024.365.1") == -1

    # Invalid version strings
    with pytest.raises(ValueError, match=r"^Invalid version"):
        version.compare_versions("1.0.0", "not_a_version")
    with pytest.raises(ValueError, match=r"^Invalid version"):
        version.compare_versions("not_a_version", "1.0.0")


def test_check_version_constraint():
    assert version.check_version_constraint("6.0.2", ">=6.0.0")
    assert version.check_version_constraint("6.0.2", "<=6.0.2")
    assert version.check_version_constraint("6.0.2", "<6.0.2") is False
    assert version.check_version_constraint("7.550", ">7.500")
    assert version.check_version_constraint("2025.100.0", ">=2024.365.0")


def test_is_valid_semantic_version():
    assert version.is_valid_semantic_version("1.0.0")
    assert version.is_valid_semantic_version("6.0.2")
    assert version.is_valid_semantic_version("1.0.0a1")
    assert version.is_valid_semantic_version("2025.246.0")
    assert version.is_valid_semantic_version("1.0")
    assert version.is_valid_semantic_version("1")

    assert version.is_valid_semantic_version("invalid") is False
    assert version.is_valid_semantic_version("") is False

    assert version.is_valid_semantic_version("1.0.0-alpha", strict=False)
    assert version.is_valid_semantic_version("1.0.0-0.3.7", strict=False)
    assert version.is_valid_semantic_version("1.0.0+build123", strict=False)
