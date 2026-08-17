"""Tests for HDF5 output validators."""

from pathlib import Path

import pytest

from simtools.testing.output_validation import hdf5


def test_validate_datasets_dispatches_checks(mocker):
    """Dispatch required datasets and minimum-row checks."""
    path = Path("output.hdf5")
    datasets = mocker.patch.object(hdf5.assertions, "assert_hdf5_datasets")
    rows = mocker.patch.object(hdf5.assertions, "assert_hdf5_dataset_min_rows")

    hdf5.validate_datasets(path, required=["DATA"], minimum_rows={"DATA": 1})

    datasets.assert_called_once_with(path, ["DATA"])
    rows.assert_called_once_with(path, {"DATA": 1})


def test_validate_product_rejects_unknown_product():
    """Reject an unregistered HDF5 product validator."""
    with pytest.raises(ValueError, match="Unsupported HDF5 product"):
        hdf5.validate_product(Path("output.hdf5"), "missing")
