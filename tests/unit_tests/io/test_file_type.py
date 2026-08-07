"""Tests for generic file-type helpers."""

from pathlib import Path

import pytest

from simtools.io.file_type import (
    _suffixes_for_file_type,
    looks_like_text_file,
    validate_file_type,
)


def test_suffixes_for_file_type_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported file type 'unknown'"):
        _suffixes_for_file_type("unknown")


def test_validate_file_type_rejects_invalid_terminal_suffix(tmp_path):
    with pytest.raises(ValueError, match="expected one of"):
        validate_file_type(tmp_path / "file.txt", "json_or_yaml")


def test_looks_like_text_file_false_on_binary_or_invalid_bytes(tmp_path):
    binary_file = tmp_path / "binary.dat"
    binary_file.write_bytes(b"\x00binary")
    invalid_utf8_file = tmp_path / "invalid.dat"
    invalid_utf8_file.write_bytes(b"\xff\xfe")

    assert looks_like_text_file(binary_file) is False
    assert looks_like_text_file(invalid_utf8_file) is False


def test_looks_like_text_file_false_on_os_error(mocker):
    mocker.patch.object(Path, "read_bytes", side_effect=OSError("missing"))

    assert looks_like_text_file("missing.txt") is False
