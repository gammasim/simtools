"""Tests for generic file-type helpers."""

import pytest

from simtools.io.file_type import (
    FILE_TYPE_SUFFIXES,
    is_path_type,
    matches_suffix,
    suffixes_for_file_type,
    validate_expected_suffixes,
    validate_file_type,
    validate_path_type,
)


def test_file_type_suffixes_registry_contains_supported_types():
    assert FILE_TYPE_SUFFIXES["hdf5"] == (".hdf5", ".h5")
    assert ".simtel.zst" in FILE_TYPE_SUFFIXES["sim_telarray"]


def test_is_path_type_supports_compound_suffixes(tmp_path):
    assert is_path_type(tmp_path / "table.fits.gz", "table") is True
    assert is_path_type(tmp_path / "run.simtel.zst", "sim_telarray") is True
    assert is_path_type(tmp_path / "run.simtel.zst", "table") is False


def test_suffixes_for_file_type_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported file type 'unknown'"):
        suffixes_for_file_type("unknown")


def test_validate_path_type_returns_path_for_registered_type(tmp_path):
    file_path = tmp_path / "table.fits.gz"

    assert validate_path_type(file_path, "table") == file_path


def test_validate_file_type_rejects_invalid_terminal_suffix(tmp_path):
    with pytest.raises(ValueError, match="expected one of"):
        validate_file_type(tmp_path / "file.txt", [".json"])


def test_validate_expected_suffixes_rejects_unsupported_suffix(tmp_path):
    with pytest.raises(ValueError, match="unsupported suffix"):
        validate_expected_suffixes(tmp_path / "file.txt", [".json", ".yaml"])


def test_matches_suffix_is_case_insensitive_and_supports_compound_suffixes():
    assert matches_suffix("DATA.FITS.GZ", [".fits.gz"]) is True
    assert matches_suffix("run.SIMTEL.ZST", [".simtel.zst"]) is True
    assert matches_suffix("file.txt", [".json"]) is False
