"""Tests for generic simulation-file inspection helpers."""

from unittest.mock import patch

import h5py
from astropy.table import Table

from simtools.io import ascii_handler
from simtools.io.file_inspector import inspect_file


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
