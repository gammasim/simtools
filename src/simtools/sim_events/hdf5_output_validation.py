"""Validation of reduced event and trigger-histogram HDF5 products."""

import logging

import h5py
import numpy as np

from simtools.constants import METADATA_JSON_SCHEMA, SCHEMA_PATH
from simtools.data_model import schema, validate_data
from simtools.io import table_handler
from simtools.sim_events.metadata import validate_simulation_metadata
from simtools.sim_events.trigger_histogram_format import (
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_HISTOGRAM_DENSE_GROUP,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
)
from simtools.utils import value_conversion

_logger = logging.getLogger(__name__)

_REDUCED_EVENT_TABLE_SCHEMAS = {
    "SHOWERS": SCHEMA_PATH / "reduced_event_showers.schema.yml",
    "TRIGGERS": SCHEMA_PATH / "reduced_event_triggers.schema.yml",
    "FILE_INFO": SCHEMA_PATH / "reduced_event_file_info.schema.yml",
}
_TRIGGER_HISTOGRAM_TABLES = (
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
)
_TRIGGER_HISTOGRAM_TABLE_SCHEMAS = {
    TRIGGER_HISTOGRAM_METADATA_TABLE: SCHEMA_PATH
    / "trigger_histogram_reference_metadata.schema.yml",
    TRIGGER_HISTOGRAM_BINS_TABLE: SCHEMA_PATH / "trigger_histogram_reference_bins.schema.yml",
    TRIGGER_TOPOLOGY_COUNTS_TABLE: SCHEMA_PATH / "trigger_histogram_topology_counts.schema.yml",
    TRIGGER_SUBSET_HISTOGRAMS_TABLE: SCHEMA_PATH / "trigger_histogram_subset_histograms.schema.yml",
}


def _raise_inconsistent(title, errors):
    """Raise a ValueError listing inconsistent entries under a title."""
    raise ValueError(f"{title}\n" + "\n".join(f" - {error}" for error in errors))


def validate_sim_events(data_files, expected_mc_events):
    """
    Validate that the number of simulated events in reduced event lists matches the expectation.

    Parameters
    ----------
    data_files: str, Path, list
        Reduced event list files (HDF5).
    expected_mc_events: int
        Expected number of simulated MC events.
    """
    event_errors = []
    for data_file in value_conversion.ensure_list(data_files):
        if table_handler.read_table_file_type([data_file]) != "HDF5":
            raise ValueError(
                f"Unsupported reduced event data format for '{data_file}'. "
                "Only HDF5 files with suffix '.hdf5' or '.h5' are supported."
            )
        tables = table_handler.read_tables(data_file, ["SHOWERS"], file_type="HDF5")
        try:
            mc_events = len(tables["SHOWERS"])
        except KeyError as exc:
            raise ValueError(f"SHOWERS table not found in reduced event list {data_file}.") from exc

        if mc_events != expected_mc_events:
            event_errors.append(
                f"Number of simulated MC events ({mc_events}) does not match "
                f"the expected number ({expected_mc_events}) in reduced event list {data_file}."
            )
        else:
            _logger.info(
                f"Consistent number of events in reduced event list: {data_file}: MC events:"
                f" {mc_events} (expected: {expected_mc_events})"
            )

    if event_errors:
        _raise_inconsistent("Inconsistent event counts found in reduced event lists:", event_errors)


def validate_reduced_event_data_file(data_file):
    """
    Validate structure, tables, metadata, and references of one reduced-event file.

    Parameters
    ----------
    data_file: str or pathlib.Path
        HDF5 reduced-event file to validate.

    Returns
    -------
    bool
        True when the file passes all checks.
    """
    _check_hdf5_file(data_file, "Reduced event data")
    required_entries = [*_REDUCED_EVENT_TABLE_SCHEMAS, "METADATA", "SIMULATION_METADATA"]
    _check_entries_present(
        data_file, table_handler.read_table_list(data_file, required_entries), "Reduced event data"
    )

    tables = table_handler.read_tables(
        data_file, list(_REDUCED_EVENT_TABLE_SCHEMAS), file_type="HDF5"
    )
    _validate_tables_against_schemas(tables, _REDUCED_EVENT_TABLE_SCHEMAS)
    _validate_standard_metadata(data_file)
    validate_simulation_metadata(
        table_handler.read_metadata_document(data_file, "SIMULATION_METADATA")
    )
    _validate_reduced_event_table_references(tables, data_file)
    return True


def _check_hdf5_file(data_file, label):
    """Raise ValueError if a file is not an HDF5 file."""
    if table_handler.read_table_file_type([data_file]) != "HDF5":
        raise ValueError(f"{label} file '{data_file}' must be an HDF5 file.")


def _check_entries_present(data_file, available_entries, label):
    """Raise ValueError if any required entry is missing from a file."""
    missing_entries = [name for name, entries in available_entries.items() if not entries]
    if missing_entries:
        raise ValueError(
            f"{label} file '{data_file}' is missing required entries: {', '.join(missing_entries)}."
        )


def _validate_tables_against_schemas(tables, schema_files):
    """Validate each table against its schema file."""
    for table_name, schema_file in schema_files.items():
        validate_data.DataValidator(
            schema_file=schema_file,
            data_table=tables[table_name],
        ).validate_and_transform()


def _validate_standard_metadata(data_file):
    """Validate the standard METADATA document of an HDF5 file."""
    standard_metadata = table_handler.read_metadata_document(data_file, "METADATA")
    schema.validate_dict_using_schema(standard_metadata, schema_file=METADATA_JSON_SCHEMA)


def validate_trigger_histogram_file(data_file):
    """
    Validate structure, tables, metadata, and references of one trigger-histogram file.

    Parameters
    ----------
    data_file: str or pathlib.Path
        HDF5 trigger-histogram file to validate.

    Returns
    -------
    bool
        True when the file passes all checks.
    """
    _check_hdf5_file(data_file, "Trigger histogram")
    _check_entries_present(
        data_file,
        table_handler.read_table_list(data_file, list(_TRIGGER_HISTOGRAM_TABLES)),
        "Trigger histogram",
    )
    _validate_standard_metadata(data_file)
    tables = table_handler.read_tables(data_file, list(_TRIGGER_HISTOGRAM_TABLES), file_type="HDF5")
    _validate_tables_against_schemas(tables, _TRIGGER_HISTOGRAM_TABLE_SCHEMAS)
    reference_ids = _validate_trigger_histogram_table_references(tables, data_file)
    _validate_trigger_histogram_dense_payload(data_file, reference_ids)
    return True


def validate_trigger_histograms_file(data_file):
    """Validate a trigger-histogram product using its registered product name."""
    return validate_trigger_histogram_file(data_file)


def _validate_trigger_histogram_table_references(tables, data_file):
    """Validate reference IDs used by trigger-histogram tables."""
    reference_ids = {str(row["reference_id"]) for row in tables[TRIGGER_HISTOGRAM_METADATA_TABLE]}
    if not reference_ids:
        raise ValueError(f"Trigger histogram file '{data_file}' has no reference metadata rows.")
    if len(reference_ids) != len(tables[TRIGGER_HISTOGRAM_METADATA_TABLE]):
        raise ValueError(f"Trigger histogram file '{data_file}' has duplicate reference IDs.")

    for table_name in _TRIGGER_HISTOGRAM_TABLES[1:]:
        table_reference_ids = {str(row["reference_id"]) for row in tables[table_name]}
        unknown_ids = table_reference_ids.difference(reference_ids)
        if unknown_ids:
            raise ValueError(
                f"Trigger histogram file '{data_file}' table '{table_name}' references unknown "
                f"reference IDs: {sorted(unknown_ids)}."
            )

    bin_reference_ids = {str(row["reference_id"]) for row in tables[TRIGGER_HISTOGRAM_BINS_TABLE]}
    missing_bin_ids = reference_ids.difference(bin_reference_ids)
    if missing_bin_ids:
        raise ValueError(
            f"Trigger histogram file '{data_file}' has references without histogram bins: "
            f"{sorted(missing_bin_ids)}."
        )
    return reference_ids


def _validate_trigger_histogram_dense_payload(data_file, reference_ids):
    """Validate dense histogram groups and their links to reference metadata."""
    prefix = f"Trigger histogram file '{data_file}'"
    with h5py.File(data_file, "r") as hdf5_file:
        if TRIGGER_HISTOGRAM_DENSE_GROUP not in hdf5_file:
            raise ValueError(f"{prefix} has no dense payload group.")
        dense_group = hdf5_file[TRIGGER_HISTOGRAM_DENSE_GROUP]
        if not isinstance(dense_group, h5py.Group):
            raise ValueError(f"{prefix} dense payload is not a group.")
        dense_reference_ids = set(dense_group.keys())
        if dense_reference_ids != reference_ids:
            raise ValueError(
                f"{prefix} dense reference IDs do not match metadata: "
                f"metadata={sorted(reference_ids)}, dense={sorted(dense_reference_ids)}."
            )
        for reference_id, reference_group in dense_group.items():
            if not isinstance(reference_group, h5py.Group) or not reference_group:
                raise ValueError(f"{prefix} reference '{reference_id}' has no payload.")
            for histogram_name, histogram_group in reference_group.items():
                _validate_dense_histogram_payload(histogram_name, histogram_group, data_file)


def _validate_dense_histogram_payload(histogram_name, histogram_group, data_file):
    """Validate one dense histogram's values and bin-edge datasets."""
    prefix = f"Trigger histogram file '{data_file}' payload '{histogram_name}'"
    if not isinstance(histogram_group, h5py.Group):
        raise ValueError(f"{prefix} is not a group.")
    values = histogram_group.get("values")
    if not isinstance(values, h5py.Dataset):
        raise ValueError(f"{prefix} has no values dataset.")
    if values.ndim == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{prefix} has invalid values data.")

    edge_names = [name for name in histogram_group if name != "values"]
    expected_edge_names = [f"edges_{axis_index}" for axis_index in range(values.ndim)]
    if set(edge_names) != set(expected_edge_names):
        raise ValueError(
            f"{prefix} has invalid bin-edge datasets: expected {expected_edge_names}, "
            f"found {sorted(edge_names)}."
        )
    for axis_index, edge_name in enumerate(expected_edge_names):
        edges = histogram_group[edge_name]
        if not isinstance(edges, h5py.Dataset) or edges.ndim != 1:
            raise ValueError(
                f"{prefix} edge dataset '{edge_name}' is not a one-dimensional dataset."
            )
        if not np.issubdtype(edges.dtype, np.number):
            raise ValueError(f"{prefix} edge dataset '{edge_name}' is not numeric.")
        expected_length = values.shape[axis_index] + 1
        if len(edges) != expected_length:
            raise ValueError(
                f"{prefix} edge dataset '{edge_name}' has {len(edges)} values, "
                f"expected {expected_length}."
            )


def _validate_reduced_event_table_references(tables, data_file):
    """Validate file and event references between reduced-event tables."""
    file_ids = set(np.asarray(tables["FILE_INFO"]["file_id"], dtype=int))
    for table_name in ("SHOWERS", "TRIGGERS"):
        unknown_file_ids = set(np.asarray(tables[table_name]["file_id"], dtype=int)).difference(
            file_ids
        )
        if unknown_file_ids:
            raise ValueError(
                f"Reduced event data file '{data_file}' table '{table_name}' references unknown "
                f"file_id values: {sorted(map(int, unknown_file_ids))}."
            )

    def _keys(table):
        return {
            tuple(int(row[name]) for name in ("file_id", "event_id", "shower_id")) for row in table
        }

    shower_ids = _keys(tables["SHOWERS"])
    if len(shower_ids) != len(tables["SHOWERS"]):
        raise ValueError(
            f"Reduced event data file '{data_file}' contains duplicate shower composite keys."
        )
    unknown_showers = _keys(tables["TRIGGERS"]).difference(shower_ids)
    if unknown_showers:
        raise ValueError(
            f"Reduced event data file '{data_file}' contains triggers without matching showers: "
            f"{sorted(unknown_showers)}."
        )
