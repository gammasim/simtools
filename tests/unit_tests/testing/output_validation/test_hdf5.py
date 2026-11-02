"""Tests for HDF5 output validators."""

from pathlib import Path

import pytest

from simtools.testing.output_validation import hdf5


def test_validate_datasets_rejects_non_hdf5_output():
    """Reject dataset checks for a non-HDF5 output file."""
    with pytest.raises(AssertionError, match="require an HDF5 output file"):
        hdf5.validate_datasets(Path("output.txt"), required=["DATA"])


def test_validate_datasets_accepts_optional_checks():
    """Allow an HDF5 output without optional dataset checks."""
    hdf5.validate_datasets(Path("output.h5"))


def test_hdf5_validators_dispatch(mocker):
    """Dispatch HDF5 dataset and product checks."""
    path = Path("output.hdf5")
    required = mocker.patch.object(hdf5.assertions, "assert_hdf5_datasets")
    minimum = mocker.patch.object(hdf5.assertions, "assert_hdf5_dataset_min_rows")
    hdf5.validate_datasets(path, required=["DATA"], minimum_rows={"DATA": 1})
    required.assert_called_once_with(path, ["DATA"])
    minimum.assert_called_once_with(path, {"DATA": 1})
    product = mocker.patch.object(hdf5, "validate_reduced_event_data_file")
    hdf5.validate_product(path, "reduced_event_data")
    product.assert_called_once_with(path)
    with pytest.raises(ValueError, match="Unsupported HDF5 product"):
        hdf5.validate_product(path, "missing")
