"""Tests for trigger-histogram inspection helpers."""

import h5py
from astropy.table import Table

from simtools.io import table_handler
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_DENSE_GROUP,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    _format_trigger_histogram_inspection,
    inspect_trigger_histogram_file,
)


def test_inspect_trigger_histogram_file_reports_reference_mismatches(tmp_path):
    file_path = tmp_path / "trigger_histograms.hdf5"
    metadata = Table(rows=[{"reference_id": "ref-1"}, {"reference_id": "ref-2"}])
    metadata.meta["EXTNAME"] = TRIGGER_HISTOGRAM_METADATA_TABLE
    table_handler.write_tables([metadata], file_path, file_type="HDF5")
    with h5py.File(file_path, "a") as hdf5_file:
        dense_group = hdf5_file.create_group(TRIGGER_HISTOGRAM_DENSE_GROUP)
        dense_group.create_group("ref-1")
        dense_group.create_group("ref-3")

    report = inspect_trigger_histogram_file(file_path, format_report=False)
    formatted = _format_trigger_histogram_inspection(report)

    assert report["missing_dense_reference_ids"] == ["ref-2"]
    assert report["orphan_dense_reference_ids"] == ["ref-3"]
    assert "missing dense payloads for metadata ids: ref-2" in formatted
    assert "orphan dense payload ids without metadata rows: ref-3" in formatted


def test_inspect_trigger_histogram_file_returns_none_for_generic_hdf5(tmp_path):
    file_path = tmp_path / "generic.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_dataset("values", data=[1, 2, 3])

    assert inspect_trigger_histogram_file(file_path) is None
