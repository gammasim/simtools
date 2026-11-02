"""HDF5 integration-output validators."""

from simtools.sim_events.hdf5_output_validation import (
    validate_reduced_event_data_file,
    validate_trigger_histograms_file,
)
from simtools.testing import assertions


def validate_datasets(path, required=None, minimum_rows=None):
    """Validate configured HDF5 datasets and their minimum row counts."""
    if path.suffix.lower() not in (".hdf5", ".h5"):
        raise AssertionError(f"HDF5 dataset checks require an HDF5 output file, got {path}.")
    if required:
        assertions.assert_hdf5_datasets(path, required)
    if minimum_rows:
        assertions.assert_hdf5_dataset_min_rows(path, minimum_rows)


def validate_product(path, product):
    """Validate one structured HDF5 product using its registered validator."""
    validators = {
        "reduced_event_data": validate_reduced_event_data_file,
        "trigger_histograms": validate_trigger_histograms_file,
    }
    try:
        validator = validators[product]
    except KeyError as exc:
        raise ValueError(f"Unsupported HDF5 product '{product}'.") from exc
    validator(path)
