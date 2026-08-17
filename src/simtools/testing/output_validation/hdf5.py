"""HDF5 integration-output validators."""

from operator import methodcaller

from simtools.sim_events import output_validator
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
    validator_name = f"validate_{product}_file"
    try:
        methodcaller(validator_name, path)(output_validator)
    except AttributeError as exc:
        raise ValueError(f"Unsupported HDF5 product '{product}'.") from exc
    except TypeError as exc:
        raise ValueError(f"HDF5 product validator '{validator_name}' is not callable.") from exc
