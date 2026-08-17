"""Validation of reduced event list files."""

import logging

import h5py
import numpy as np

from simtools.constants import METADATA_JSON_SCHEMA, SCHEMA_PATH
from simtools.data_model import schema, validate_data
from simtools.io import table_handler
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_HISTOGRAM_DENSE_GROUP,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
)
from simtools.sim_events.metadata import validate_simulation_metadata
from simtools.utils import general

_logger = logging.getLogger(__name__)

_REDUCED_EVENT_TABLE_SCHEMAS = {
    "SHOWERS": SCHEMA_PATH / "reduced_event_showers.schema.yml",
    "TRIGGERS": SCHEMA_PATH / "reduced_event_triggers.schema.yml",
    "FILE_INFO": SCHEMA_PATH / "reduced_event_file_info.schema.yml",
}
_REDUCED_EVENT_METADATA_DOCUMENTS = ("METADATA", "SIMULATION_METADATA")
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


def validate_sim_events(data_files, expected_mc_events):
    """
    Validate reduced event lists files.

    Parameters
    ----------
    data_files: str, Path, list
        Path(s) to the reduced event list files to validate.
    expected_mc_events: int
        Expected number of simulated MC events.
    """
    data_files = general.ensure_list(data_files)
    validate_event_numbers(data_files, expected_mc_events)


def validate_event_numbers(data_files, expected_mc_events):
    """
    Validate that the number of simulated events in reduced event lists matches the expected number.

    Parameters
    ----------
    data_files: str, Path, list
        Path(s) to the reduced event list files to validate.
    expected_mc_events: int
        Expected number of simulated MC events.

    Raises
    ------
    ValueError
        If the number of simulated events does not match the expected number.
    """
    data_files = general.ensure_list(data_files)

    event_errors = []
    for data_file in data_files:
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
        _logger.error("Inconsistent event counts found in reduced event lists:")
        for error in event_errors:
            _logger.error(f" - {error}")
        error_message = "Inconsistent event counts found in reduced event lists:\n" + "\n".join(
            f" - {error}" for error in event_errors
        )
        raise ValueError(error_message)


def validate_reduced_event_data_file(data_file):
    """Validate the structure, tables, metadata, and references of one reduced-event file.

    Parameters
    ----------
    data_file : str or pathlib.Path
        HDF5 reduced-event file to validate.

    Returns
    -------
    bool
        ``True`` when the file passes all structural and semantic checks.

    Raises
    ------
    ValueError
        If the file format, required entries, metadata, or references are invalid.
    KeyError
        If a required table column is missing.
    OSError
        If ``data_file`` is not a readable HDF5 file.
    """
    if table_handler.read_table_file_type([data_file]) != "HDF5":
        raise ValueError(f"Reduced event data file '{data_file}' must be an HDF5 file.")

    required_entries = [*_REDUCED_EVENT_TABLE_SCHEMAS, *_REDUCED_EVENT_METADATA_DOCUMENTS]
    available_entries = table_handler.read_table_list(data_file, required_entries)
    missing_entries = [name for name, entries in available_entries.items() if not entries]
    if missing_entries:
        raise ValueError(
            f"Reduced event data file '{data_file}' is missing required entries: "
            f"{', '.join(missing_entries)}."
        )

    tables = table_handler.read_tables(
        data_file, list(_REDUCED_EVENT_TABLE_SCHEMAS), file_type="HDF5"
    )
    for table_name, schema_file in _REDUCED_EVENT_TABLE_SCHEMAS.items():
        validate_data.DataValidator(
            schema_file=schema_file,
            data_table=tables[table_name],
        ).validate_and_transform()

    standard_metadata = table_handler.read_metadata_document(data_file, "METADATA")
    schema.validate_dict_using_schema(standard_metadata, schema_file=METADATA_JSON_SCHEMA)
    validate_simulation_metadata(
        table_handler.read_metadata_document(data_file, "SIMULATION_METADATA")
    )
    _validate_reduced_event_table_references(tables, data_file)
    return True


def validate_trigger_histogram_file(data_file):
    """Validate the structure, tables, metadata, and references of one trigger-histogram file.

    Parameters
    ----------
    data_file : str or pathlib.Path
        HDF5 trigger-histogram file to validate.

    Returns
    -------
    bool
        ``True`` when the file passes all structural and semantic checks.

    Raises
    ------
    ValueError
        If the file format, required entries, metadata, table contents, or references are invalid.
    KeyError
        If a required table column is missing.
    OSError
        If ``data_file`` is not a readable HDF5 file.
    """
    if table_handler.read_table_file_type([data_file]) != "HDF5":
        raise ValueError(f"Trigger histogram file '{data_file}' must be an HDF5 file.")

    available_entries = table_handler.read_table_list(data_file, list(_TRIGGER_HISTOGRAM_TABLES))
    missing_entries = [name for name, entries in available_entries.items() if not entries]
    if missing_entries:
        raise ValueError(
            f"Trigger histogram file '{data_file}' is missing required entries: "
            f"{', '.join(missing_entries)}."
        )

    standard_metadata = table_handler.read_metadata_document(data_file, "METADATA")
    schema.validate_dict_using_schema(standard_metadata, schema_file=METADATA_JSON_SCHEMA)
    tables = table_handler.read_tables(data_file, list(_TRIGGER_HISTOGRAM_TABLES), file_type="HDF5")
    for table_name, schema_file in _TRIGGER_HISTOGRAM_TABLE_SCHEMAS.items():
        validate_data.DataValidator(
            schema_file=schema_file,
            data_table=tables[table_name],
        ).validate_and_transform()
    reference_ids = _validate_trigger_histogram_table_references(tables, data_file)
    _validate_trigger_histogram_dense_payload(data_file, reference_ids)
    return True


def validate_trigger_histograms_file(data_file):
    """Validate a trigger-histogram product using its registered product name."""
    return validate_trigger_histogram_file(data_file)


def _validate_trigger_histogram_table_references(tables, data_file):
    """Validate reference IDs used by trigger-histogram tables."""
    reference_ids = {str(row["reference_id"]) for row in tables[TRIGGER_HISTOGRAM_METADATA_TABLE]}
    metadata_row_count = len(tables[TRIGGER_HISTOGRAM_METADATA_TABLE])
    if not reference_ids:
        raise ValueError(f"Trigger histogram file '{data_file}' has no reference metadata rows.")
    if len(reference_ids) != metadata_row_count:
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
    with h5py.File(data_file, "r") as hdf5_file:
        dense_group = _get_dense_histogram_group(hdf5_file, data_file)
        _validate_dense_reference_ids(dense_group, reference_ids, data_file)
        for reference_id, reference_group in dense_group.items():
            _validate_dense_reference_payload(reference_id, reference_group, data_file)


def _get_dense_histogram_group(hdf5_file, data_file):
    """Return the dense payload group from a trigger-histogram file."""
    if TRIGGER_HISTOGRAM_DENSE_GROUP not in hdf5_file:
        raise ValueError(f"Trigger histogram file '{data_file}' has no dense payload group.")
    dense_group = hdf5_file[TRIGGER_HISTOGRAM_DENSE_GROUP]
    if not isinstance(dense_group, h5py.Group):
        raise ValueError(f"Trigger histogram file '{data_file}' dense payload is not a group.")
    return dense_group


def _validate_dense_reference_ids(dense_group, reference_ids, data_file):
    """Validate that dense payload groups correspond to reference metadata."""
    dense_reference_ids = set(dense_group.keys())
    if dense_reference_ids != reference_ids:
        raise ValueError(
            f"Trigger histogram file '{data_file}' dense reference IDs do not match metadata: "
            f"metadata={sorted(reference_ids)}, dense={sorted(dense_reference_ids)}."
        )


def _validate_dense_reference_payload(reference_id, reference_group, data_file):
    """Validate all dense histograms for one reference ID."""
    if not isinstance(reference_group, h5py.Group) or not reference_group:
        raise ValueError(
            f"Trigger histogram file '{data_file}' reference '{reference_id}' has no payload."
        )
    for histogram_name, histogram_group in reference_group.items():
        _validate_dense_histogram_payload(histogram_name, histogram_group, data_file)


def _validate_dense_histogram_payload(histogram_name, histogram_group, data_file):
    """Validate one dense histogram's values and bin-edge datasets."""
    if not isinstance(histogram_group, h5py.Group):
        raise ValueError(
            f"Trigger histogram file '{data_file}' payload '{histogram_name}' is not a group."
        )
    values = histogram_group.get("values")
    if not isinstance(values, h5py.Dataset):
        raise ValueError(
            f"Trigger histogram file '{data_file}' payload '{histogram_name}' "
            "has no values dataset."
        )
    if values.ndim == 0 or not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"Trigger histogram file '{data_file}' payload '{histogram_name}' "
            "has invalid values data."
        )

    edge_names = [name for name in histogram_group if name != "values"]
    expected_edge_names = [f"edges_{axis_index}" for axis_index in range(values.ndim)]
    if set(edge_names) != set(expected_edge_names):
        raise ValueError(
            f"Trigger histogram file '{data_file}' payload '{histogram_name}' "
            f"has invalid bin-edge datasets: expected {expected_edge_names}, "
            f"found {sorted(edge_names)}."
        )
    for axis_index, edge_name in enumerate(expected_edge_names):
        edges = histogram_group[edge_name]
        if not isinstance(edges, h5py.Dataset) or edges.ndim != 1:
            raise ValueError(
                f"Trigger histogram file '{data_file}' payload '{histogram_name}' "
                f"edge dataset '{edge_name}' is not a one-dimensional dataset."
            )
        if not np.issubdtype(edges.dtype, np.number):
            raise ValueError(
                f"Trigger histogram file '{data_file}' payload '{histogram_name}' "
                f"edge dataset '{edge_name}' is not numeric."
            )
        expected_length = values.shape[axis_index] + 1
        if len(edges) != expected_length:
            raise ValueError(
                f"Trigger histogram file '{data_file}' payload '{histogram_name}' "
                f"edge dataset '{edge_name}' has {len(edges)} values, expected {expected_length}."
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

    shower_ids = {
        tuple(int(row[name]) for name in ("file_id", "event_id", "shower_id"))
        for row in tables["SHOWERS"]
    }
    if len(shower_ids) != len(tables["SHOWERS"]):
        raise ValueError(
            f"Reduced event data file '{data_file}' contains duplicate shower composite keys."
        )
    trigger_ids = {
        tuple(int(row[name]) for name in ("file_id", "event_id", "shower_id"))
        for row in tables["TRIGGERS"]
    }
    unknown_showers = trigger_ids.difference(shower_ids)
    if unknown_showers:
        raise ValueError(
            f"Reduced event data file '{data_file}' contains triggers without matching showers: "
            f"{sorted(unknown_showers)}."
        )
