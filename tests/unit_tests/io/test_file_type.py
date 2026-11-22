"""Tests for generic file-type helpers."""

from pathlib import Path

import jsonschema
import pytest

from simtools.io import file_type
from simtools.io.file_type import (
    FILE_TYPE_SUFFIXES,
    _file_catalog,
    _suffixes_for_file_type,
    looks_like_text_file,
    product_filename,
    product_suffixes,
    product_value,
    validate_file_type,
)


def test_product_suffixes_include_compressed_sim_telarray_outputs():
    product_id = "sim_telarray_output"
    assert product_suffixes(product_id) == (
        ".simtel",
        ".simtel.gz",
        ".simtel.zst",
    )
    assert product_value(product_id, "default_suffix") == ".simtel.zst"


def test_product_filename_and_unknown_product():
    assert product_filename("submission_manifest") == "submission.json"

    with pytest.raises(ValueError, match="does not define a fixed filename"):
        product_filename("sim_telarray_output")

    with pytest.raises(ValueError, match="Unknown file product 'missing'"):
        product_suffixes("missing")


def test_file_catalog_formats_match_legacy_file_type_names():
    catalog = _file_catalog()

    assert set(catalog["formats"]) == set(FILE_TYPE_SUFFIXES)
    assert catalog["products"]["iact_output"]["role"] == "intermediate"
    assert catalog["products"]["iact_output"]["retention"] == "temporary"
    for product_id, product in catalog["products"].items():
        assert product["role"] in {"input", "intermediate", "result", "diagnostic", "metadata"}
        assert product["retention"] in {"persistent", "temporary", "workflow_defined"}
        if "filename" in product:
            assert product_filename(product_id) == product["filename"]
        else:
            suffixes = product_suffixes(product_id)
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


def test_file_catalog_merges_the_files_of_the_simulation_software():
    # declared in resources/simulation_software.yml
    assert product_suffixes("corsika_output") == (".corsika", ".corsika.gz", ".corsika.zst")
    assert product_value("photon_stream", "default_suffix") == ".pst"
    assert product_value("stage_log", "directory") == "run"
    assert FILE_TYPE_SUFFIXES["sim_telarray"] == (".simtel", ".simtel.gz", ".simtel.zst")


def test_products_in_directory():
    assert {"corsika_output", "photon_stream", "stage_log"} <= set(
        file_type.products_in_directory("run")
    )
    assert set(file_type.products_in_directory("submission")) == {
        "sub_out",
        "sub_err",
        "sub_log",
        "sub_script",
    }


def test_merge_rejects_an_id_declared_twice():
    catalog = {"products": {"corsika_output": {}}}

    with pytest.raises(ValueError, match="File product 'corsika_output' is declared twice"):
        file_type._merge(catalog, "products", {"corsika_output": {}}, "corsika")  # pylint: disable=protected-access
