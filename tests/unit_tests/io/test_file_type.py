"""Tests for generic file-type helpers."""

from pathlib import Path

import jsonschema
import pytest

from simtools.io import file_type
from simtools.io.file_type import (
    FILE_TYPE_SUFFIXES,
    _file_catalog,
    _get_product_filename,
    _get_product_suffixes,
    _get_product_value,
    _suffixes_for_file_type,
    looks_like_text_file,
    validate_file_type,
)


def test_product_suffixes_include_compressed_sim_telarray_outputs():
    product_id = "sim_telarray_output"
    assert _get_product_suffixes(product_id) == (
        ".simtel",
        ".simtel.gz",
        ".simtel.zst",
    )
    assert _get_product_value(product_id, "default_suffix") == ".simtel.zst"


def test_product_filename_and_unknown_product():
    assert _get_product_filename("submission_manifest") == "submission.json"

    with pytest.raises(ValueError, match="Unknown file product 'missing'"):
        _get_product_suffixes("missing")


def test_file_catalog_formats_match_legacy_file_type_names():
    catalog = _file_catalog()

    assert set(catalog["formats"]) == set(FILE_TYPE_SUFFIXES)
    assert catalog["products"]["iact_output"]["role"] == "intermediate"
    assert catalog["products"]["iact_output"]["retention"] == "temporary"
    for product_id, product in catalog["products"].items():
        assert product["role"] in {"input", "intermediate", "result", "diagnostic", "metadata"}
        assert product["retention"] in {"persistent", "temporary", "workflow_defined"}
        if "filename" in product:
            assert _get_product_filename(product_id) == product["filename"]
        else:
            suffixes = _get_product_suffixes(product_id)
            assert suffixes
            if "default_suffix" in product:
                assert product["default_suffix"] in suffixes


def test_file_catalog_schema_rejects_invalid_product_role():
    catalog = {
        "formats": {"text": [".txt"]},
        "products": {
            "bad_product": {
                "filename": "bad.txt",
                "role": "unknown",
                "retention": "persistent",
            }
        },
    }

    with pytest.raises(jsonschema.ValidationError, match="unknown"):
        file_type._validate_file_catalog_schema(catalog)


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
