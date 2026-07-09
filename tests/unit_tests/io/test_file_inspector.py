"""Tests for generic simulation-file inspection helpers."""

from pathlib import Path
from unittest.mock import patch

import h5py
import pytest
from astropy.table import Table

from simtools.io import ascii_handler
from simtools.io.file_inspector import (
    _format_hdf5_report,
    _format_key_value_report,
    _format_simtel_report,
    _format_table_report,
    _format_text_report,
    _is_truncated,
    _looks_like_text_file,
    _normalize_max_entries,
    _select_inspector,
    inspect_file,
    inspect_hdf5_file,
    inspect_json_or_yaml_file,
    inspect_sim_telarray_file,
    inspect_table_file,
    inspect_text_file,
)
from simtools.io.file_type import is_path_type


def test_inspect_file_reports_generic_hdf5_structure(tmp_path):
    file_path = tmp_path / "generic.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_group("group_a")
        hdf5_file.create_dataset("group_a/dataset_b", data=[1, 2, 3])

    reports = inspect_file(file_path, max_entries=10, format_report=False)
    report = reports[0]

    assert len(reports) == 1
    assert report["file_type"] == "hdf5"
    assert "group_a" in report["root_entries"]
    assert report["dataset_count"] == 1
    assert report["group_count"] >= 1


def test_inspect_file_reports_json_structure(tmp_path):
    file_path = tmp_path / "data.json"
    ascii_handler.write_data_to_file({"a": 1, "b": 2}, file_path, sort_keys=False)

    reports = inspect_file(file_path, format_report=False)

    assert len(reports) == 1
    assert reports[0]["file_type"] == "json"
    assert reports[0]["top_level_type"] == "dict"
    assert reports[0]["top_level_keys"] == ["a", "b"]


def test_inspect_file_reports_yaml_structure(tmp_path):
    file_path = tmp_path / "data.yml"
    ascii_handler.write_data_to_file({"a": [1, 2, 3]}, file_path, sort_keys=False)

    reports = inspect_file(file_path, format_report=False)

    assert len(reports) == 1
    assert reports[0]["file_type"] == "yaml"
    assert reports[0]["top_level_type"] == "dict"


def test_inspect_file_reports_ecsv_structure(tmp_path):
    file_path = tmp_path / "table.ecsv"
    Table({"alpha": [1, 2], "beta": [3, 4]}).write(file_path, format="ascii.ecsv")

    reports = inspect_file(file_path, format_report=False)

    assert len(reports) == 1
    assert reports[0]["file_type"] == "ecsv"
    assert reports[0]["row_count"] == 2
    assert reports[0]["column_count"] == 2
    assert reports[0]["columns"] == ["alpha", "beta"]


def test_inspect_file_reports_fits_gz_structure(tmp_path):
    file_path = tmp_path / "table.fits.gz"
    Table({"alpha": [1, 2], "beta": [3, 4]}).write(file_path)

    reports = inspect_file(file_path, format_report=False)

    assert len(reports) == 1
    assert reports[0]["file_type"] == "fits"
    assert reports[0]["row_count"] == 2
    assert reports[0]["column_count"] == 2


def test_inspect_file_reports_simtel_file_size(tmp_path):
    file_path = tmp_path / "run.simtel.zst"
    file_path.write_bytes(b"simtel")

    with patch(
        "simtools.io.file_inspector.simtel_io_metadata.read_sim_telarray_metadata",
        return_value=(
            {"corsika_iact_options": "123", "site": "North"},
            {1: {"camera": "LST"}, 2: {"camera": "MST"}},
        ),
    ):
        reports = inspect_file(file_path, format_report=False)

    assert len(reports) == 1
    assert reports[0]["file_type"] == "sim_telarray"
    assert reports[0]["file_size_bytes"] == 6
    assert reports[0]["global_metadata"] == {
        "corsika_iact_options": "123",
        "site": "North",
    }
    assert reports[0]["telescope_count"] == 2
    assert reports[0]["telescope_ids"] == [1, 2]


def test_inspect_file_formats_simtel_metadata(tmp_path):
    file_path = tmp_path / "run.simtel.zst"
    file_path.write_bytes(b"simtel")

    with patch(
        "simtools.io.file_inspector.simtel_io_metadata.read_sim_telarray_metadata",
        return_value=({"site": "North"}, {7: {"camera": "LST"}}),
    ):
        reports = inspect_file(file_path, format_report=True)

    assert "global_metadata:" in reports[0]
    assert "- site: North" in reports[0]
    assert "telescope_count: 1" in reports[0]
    assert "telescope_ids: [7]" in reports[0]


def test_inspect_file_reports_text_preview(tmp_path):
    file_path = tmp_path / "run.cfg"
    file_path.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    reports = inspect_file(file_path, max_entries=2, format_report=False)

    assert len(reports) == 1
    assert reports[0]["file_type"] == "text"
    assert reports[0]["line_count"] == 3
    assert reports[0]["preview_lines"] == ["line-1", "line-2"]
    assert reports[0]["preview_truncated"] is True


def test_inspect_file_unlimits_entries_for_non_positive_max_entries(tmp_path):
    file_path = tmp_path / "run.cfg"
    file_path.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    reports = inspect_file(file_path, max_entries=0, format_report=False)

    assert reports[0]["preview_lines"] == ["line-1", "line-2", "line-3"]
    assert reports[0]["preview_truncated"] is False


def test_looks_like_text_file_false_on_binary_or_invalid_bytes(tmp_path):
    binary_file = tmp_path / "binary.dat"
    binary_file.write_bytes(b"\x00binary")
    invalid_utf8_file = tmp_path / "invalid.dat"
    invalid_utf8_file.write_bytes(b"\xff\xfe")

    assert _looks_like_text_file(binary_file) is False
    assert _looks_like_text_file(invalid_utf8_file) is False


def test_looks_like_text_file_false_on_os_error(mocker):
    mocker.patch.object(Path, "read_bytes", side_effect=OSError("missing"))

    assert _looks_like_text_file("missing.txt") is False


def test_normalize_and_truncation_helpers():
    assert _normalize_max_entries(None) is None
    assert _normalize_max_entries(0) is None
    assert _normalize_max_entries(-1) is None
    assert _normalize_max_entries(3) == 3
    assert _is_truncated([1, 2, 3], None) is False
    assert _is_truncated([1, 2, 3], 3) is False
    assert _is_truncated([1, 2, 3], 2) is True


def test_select_inspector_rejects_unknown_binary_file(tmp_path):
    file_path = tmp_path / "unknown.bin"
    file_path.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="Unsupported file type"):
        _select_inspector(file_path)


def test_inspect_file_appends_trigger_histogram_report(tmp_path):
    file_path = tmp_path / "generic.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_dataset("dataset", data=[1])

    with patch(
        "simtools.io.file_inspector._inspect_trigger_histogram_file",
        return_value={"kind": "trigger"},
    ):
        reports = inspect_file(file_path, format_report=False)

    assert reports[1] == {"kind": "trigger"}


def test_inspect_hdf5_file_rejects_non_hdf5_container(tmp_path, mocker):
    file_path = tmp_path / "invalid.hdf5"
    file_path.write_text("not hdf5", encoding="utf-8")
    mocker.patch("simtools.io.file_inspector.h5py.is_hdf5", return_value=False)

    with pytest.raises(ValueError, match="not a valid HDF5 container"):
        inspect_hdf5_file(file_path)


def test_inspect_json_or_yaml_file_formats_list_payload(tmp_path):
    file_path = tmp_path / "data.yaml"
    ascii_handler.write_data_to_file([1, 2, 3], file_path, sort_keys=False)

    report = inspect_json_or_yaml_file(file_path, format_report=True)

    assert "Detected file type: yaml" in report
    assert "top_level_type: list" in report
    assert "item_count: 3" in report
    assert "top_level_keys" not in report


def test_inspect_table_file_formats_truncated_columns(tmp_path):
    file_path = tmp_path / "table.ecsv"
    Table({"alpha": [1], "beta": [2], "gamma": [3]}).write(file_path, format="ascii.ecsv")

    report = inspect_table_file(file_path, max_entries=2, format_report=True)

    assert "Columns (3): alpha, beta" in report
    assert "Column list truncated." in report


def test_inspect_sim_telarray_file_rejects_when_suffix_check_is_overridden_false(tmp_path, mocker):
    file_path = tmp_path / "run.simtel.zst"
    file_path.write_bytes(b"simtel")
    mocker.patch("simtools.io.file_inspector.is_path_type", return_value=False)

    with pytest.raises(ValueError, match="unsupported suffix for sim_telarray inspection"):
        inspect_sim_telarray_file(file_path)


def test_select_inspector_uses_generic_file_type_helper(tmp_path, mocker):
    file_path = tmp_path / "run.hdf5"
    mocker.patch("simtools.io.file_inspector.is_path_type", side_effect=[True])

    inspector = _select_inspector(file_path)

    assert inspector is inspect_hdf5_file


def test_generic_file_type_helper_is_imported_for_path_checks(tmp_path):
    assert is_path_type(tmp_path / "file.h5", "hdf5") is True


def test_inspect_sim_telarray_file_formats_empty_metadata(tmp_path):
    file_path = tmp_path / "run.simtel.zst"
    file_path.write_bytes(b"simtel")

    with patch(
        "simtools.io.file_inspector.simtel_io_metadata.read_sim_telarray_metadata",
        return_value=(None, {}),
    ):
        report = inspect_sim_telarray_file(file_path, format_report=True)

    assert "global_metadata:" in report
    assert "- none" in report


def test_inspect_text_file_formats_truncated_preview(tmp_path):
    file_path = tmp_path / "run.cfg"
    file_path.write_text("line-1\nline-2\nline-3\n", encoding="utf-8")

    report = inspect_text_file(file_path, max_entries=2, format_report=True)

    assert "Preview (2 lines):" in report
    assert "- ... output truncated ..." in report


def test_format_helpers_cover_optional_branches(tmp_path):
    hdf5_report = _format_hdf5_report(
        {
            "file_path": tmp_path / "data.hdf5",
            "file_type": "hdf5",
            "root_entries": ["group_a"],
            "group_count": 1,
            "dataset_count": 1,
            "entries": [
                {"path": "/group_a", "kind": "group"},
                {"path": "/group_a/data", "kind": "dataset", "shape": (2,), "dtype": "int64"},
            ],
            "entries_truncated": True,
            "total_entries": 3,
        }
    )
    assert "shape=(2,) dtype=int64" in hdf5_report
    assert "- ... output truncated ..." in hdf5_report

    key_value_report = _format_key_value_report(
        {
            "file_path": tmp_path / "data.json",
            "file_type": "json",
            "top_level_type": "dict",
            "top_level_keys": None,
            "item_count": 2,
        }
    )
    assert "top_level_type: dict" in key_value_report
    assert "top_level_keys" not in key_value_report

    table_report = _format_table_report(
        {
            "file_path": tmp_path / "table.ecsv",
            "file_type": "ecsv",
            "row_count": 1,
            "column_count": 2,
            "columns": ["alpha", "beta"],
            "columns_truncated": True,
        }
    )
    assert "Column list truncated." in table_report

    text_report = _format_text_report(
        {
            "file_path": tmp_path / "run.cfg",
            "file_type": "text",
            "line_count": 2,
            "preview_lines": ["a", "b"],
            "preview_truncated": True,
        }
    )
    assert "- ... output truncated ..." in text_report

    simtel_report = _format_simtel_report(
        {
            "file_path": tmp_path / "run.simtel.zst",
            "file_type": "sim_telarray",
            "file_size_bytes": 6,
            "telescope_count": 3,
            "telescope_ids": [1, 2],
            "global_metadata": {"site": "North"},
            "global_metadata_truncated": True,
            "telescope_ids_truncated": True,
        }
    )
    assert "- ... metadata truncated ..." in simtel_report
    assert "telescope_ids_truncated: True" in simtel_report
