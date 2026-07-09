"""Tests for generic simulation-file inspection helpers."""

import h5py

from simtools.io.file_inspector import format_inspection_report, inspect_file


def test_inspect_file_reports_generic_hdf5_structure(tmp_path):
    file_path = tmp_path / "generic.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_group("group_a")
        hdf5_file.create_dataset("group_a/dataset_b", data=[1, 2, 3])

    report = inspect_file(file_path, max_entries=10)

    assert report["file_type"] == "hdf5"
    assert "group_a" in report["root_entries"]
    assert report["dataset_count"] == 1
    assert report["group_count"] >= 1
    assert "Trigger histogram consistency:" not in format_inspection_report(report)
