"""Read and write executable job grids for production preparation."""

import logging
from io import StringIO
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.production_configuration.job_grid_summary import build_job_grid_summary

logger = logging.getLogger(__name__)

_ECSV_SUFFIX = ".ecsv"
_ECSV_FORMAT = "ascii.ecsv"
_STREAM_CHUNK_SIZE = 10_000

JOB_GRID_COLUMNS = [
    "primary",
    "azimuth_angle_value",
    "azimuth_angle_unit",
    "zenith_angle_value",
    "zenith_angle_unit",
    "energy_min_value",
    "energy_min_unit",
    "energy_max_value",
    "energy_max_unit",
    "core_scatter_number",
    "core_scatter_max_value",
    "core_scatter_max_unit",
    "view_cone_min_value",
    "view_cone_min_unit",
    "view_cone_max_value",
    "view_cone_max_unit",
    "showers_per_run",
    "nsb_rate",
    "model_version",
    "array_layout_name",
    "corsika_le_interaction",
    "corsika_he_interaction",
    "run_number",
]

_QUANTITY_FIELDS = {
    "azimuth_angle": ("azimuth_angle_value", "azimuth_angle_unit"),
    "zenith_angle": ("zenith_angle_value", "zenith_angle_unit"),
    "energy_min": ("energy_min_value", "energy_min_unit"),
    "energy_max": ("energy_max_value", "energy_max_unit"),
    "core_scatter_max": ("core_scatter_max_value", "core_scatter_max_unit"),
    "view_cone_min": ("view_cone_min_value", "view_cone_min_unit"),
    "view_cone_max": ("view_cone_max_value", "view_cone_max_unit"),
}
JOB_GRID_QUANTITY_FIELDS = dict(_QUANTITY_FIELDS)

_OPTIONAL_ANGLE_FIELDS = ("ra", "dec")

_JOB_GRID_COLUMN_DTYPES = {
    "primary": str,
    "azimuth_angle_value": float,
    "azimuth_angle_unit": str,
    "zenith_angle_value": float,
    "zenith_angle_unit": str,
    "energy_min_value": float,
    "energy_min_unit": str,
    "energy_max_value": float,
    "energy_max_unit": str,
    "core_scatter_number": int,
    "core_scatter_max_value": float,
    "core_scatter_max_unit": str,
    "view_cone_min_value": float,
    "view_cone_min_unit": str,
    "view_cone_max_value": float,
    "view_cone_max_unit": str,
    "showers_per_run": int,
    "nsb_rate": float,
    "model_version": str,
    "array_layout_name": str,
    "corsika_le_interaction": str,
    "corsika_he_interaction": str,
    "run_number": int,
    "ra": float,
    "dec": float,
}


def _serialize_quantity(value):
    """Serialize a Quantity to value/unit columns."""
    return float(value.value), str(value.unit)


def _deserialize_quantity(value, unit):
    """Deserialize value/unit columns to a Quantity."""
    return float(value) * u.Unit(unit)


def _serialize_job_row(job_row):
    """Serialize one job row to the on-disk schema."""
    serialized_row = {
        "primary": job_row["primary"],
        "core_scatter_number": int(job_row["core_scatter_number"]),
        "showers_per_run": int(job_row["showers_per_run"]),
        "nsb_rate": float(job_row["nsb_rate"]),
        "model_version": job_row["model_version"],
        "array_layout_name": job_row["array_layout_name"],
        "corsika_le_interaction": job_row["corsika_le_interaction"],
        "corsika_he_interaction": job_row["corsika_he_interaction"],
        "run_number": int(job_row["run_number"]),
    }

    for quantity_name, (value_key, unit_key) in _QUANTITY_FIELDS.items():
        serialized_row[value_key], serialized_row[unit_key] = _serialize_quantity(
            job_row[quantity_name]
        )

    for angle_name in _OPTIONAL_ANGLE_FIELDS:
        angle_value = job_row.get(angle_name)
        if angle_value is None:
            continue
        if isinstance(angle_value, u.Quantity):
            serialized_row[angle_name] = float(angle_value.to_value(u.deg))
        else:
            serialized_row[angle_name] = float(angle_value)

    return serialized_row


def _deserialize_job_row(serialized_row):
    """Deserialize one stored row to the in-memory job-row schema."""
    job_row = {
        "primary": serialized_row["primary"],
        "core_scatter_number": int(serialized_row["core_scatter_number"]),
        "showers_per_run": int(serialized_row["showers_per_run"]),
        "nsb_rate": float(serialized_row.get("nsb_rate", 1.0)),
        "model_version": serialized_row["model_version"],
        "array_layout_name": serialized_row["array_layout_name"],
        "corsika_le_interaction": serialized_row["corsika_le_interaction"],
        "corsika_he_interaction": serialized_row["corsika_he_interaction"],
        "run_number": int(serialized_row["run_number"]),
    }

    for quantity_name, (value_key, unit_key) in _QUANTITY_FIELDS.items():
        job_row[quantity_name] = _deserialize_quantity(
            serialized_row[value_key],
            serialized_row[unit_key],
        )

    for angle_name in _OPTIONAL_ANGLE_FIELDS:
        if angle_name not in serialized_row:
            continue
        angle_value = serialized_row[angle_name]
        if np.ma.is_masked(angle_value) or angle_value is None:
            continue
        job_row[angle_name] = float(angle_value) * u.deg

    return job_row


def serialize_job_grid(job_rows, output_file, metadata=None):
    """
    Serialize executable job rows to ECSV output.

    Parameters
    ----------
    job_rows : list[dict]
        Job rows in the in-memory schema.
    output_file : str or Path
        Output file path. Must use the ``.ecsv`` suffix.
    metadata : dict, optional
        Metadata to store alongside the rows.
    """
    output_path = Path(output_file)
    serialized_rows = [_serialize_job_row(job_row) for job_row in job_rows]
    metadata = metadata.copy() if metadata else {}
    metadata["job_grid_summary"] = build_job_grid_summary(job_rows)

    if output_path.suffix.lower() != _ECSV_SUFFIX:
        raise ValueError("Job grid output file must use the '.ecsv' extension.")

    optional_columns = [
        angle_name
        for angle_name in _OPTIONAL_ANGLE_FIELDS
        if any(angle_name in row for row in serialized_rows)
    ]
    output_columns = [*JOB_GRID_COLUMNS, *optional_columns]
    output_rows = [
        {column: row.get(column) for column in output_columns} for row in serialized_rows
    ]

    output_table = Table(rows=output_rows, names=output_columns)
    output_table.meta = metadata
    logger.info(f"Writing job grid with {len(job_rows)} rows to '{output_path}'.")
    output_table.write(output_path, format=_ECSV_FORMAT, overwrite=True)


def _build_output_table(output_rows, output_columns, metadata=None):
    """Build an Astropy table for serialized output rows."""
    output_table = Table(rows=output_rows, names=output_columns)
    output_table.meta = metadata or {}
    return output_table


def _write_empty_ecsv_header(output_path, output_columns, metadata):
    """Write an ECSV header using Astropy's schema/metadata handling."""
    empty_table = Table(
        names=output_columns,
        dtype=[_JOB_GRID_COLUMN_DTYPES[column] for column in output_columns],
    )
    empty_table.meta = metadata
    empty_table.write(output_path, format=_ECSV_FORMAT, overwrite=True)


def _extract_ecsv_data_rows(table):
    """Return Astropy-formatted ECSV data rows without metadata or column header."""
    buffer = StringIO()
    table.write(buffer, format=_ECSV_FORMAT)
    data_rows = []
    column_header_seen = False
    for line in buffer.getvalue().splitlines():
        if line.startswith("#"):
            continue
        if not column_header_seen:
            column_header_seen = True
            continue
        data_rows.append(line)
    return data_rows


def _write_ecsv_data_rows(output_path, output_rows, output_columns):
    """Append Astropy-formatted ECSV data rows to an existing ECSV file."""
    output_table = _build_output_table(output_rows, output_columns)
    data_rows = _extract_ecsv_data_rows(output_table)
    with output_path.open("a", encoding="utf-8") as output:
        for row in data_rows:
            output.write(f"{row}\n")


def _serialize_output_row(job_row, output_columns):
    """Serialize one job row and restrict it to output columns."""
    serialized_row = _serialize_job_row(job_row)
    return {column: serialized_row.get(column) for column in output_columns}


def _flush_stream_chunk(output_path, output_rows, output_columns, metadata, write_header):
    """Write or append one chunk of serialized output rows."""
    if write_header:
        output_table = _build_output_table(output_rows, output_columns, metadata)
        output_table.write(output_path, format=_ECSV_FORMAT, overwrite=True)
    else:
        _write_ecsv_data_rows(output_path, output_rows, output_columns)


def _iter_with_first(first_row, row_iterator):
    """Iterate over the first row followed by the remaining row iterator."""
    yield first_row
    yield from row_iterator


def serialize_job_grid_stream(job_rows, output_file, metadata=None):
    """
    Stream executable job rows to ECSV output.

    This avoids materializing serialized rows and the full Astropy table in memory.
    Optional RA/Dec columns are determined from the first row, which matches the
    homogeneous output produced by the production grid generator.

    Parameters
    ----------
    job_rows : iterable of dict
        Job rows in the in-memory schema.
    output_file : str or pathlib.Path
        Output file path. Must use the ``.ecsv`` suffix.
    metadata : dict, optional
        Metadata to store alongside the rows.

    Returns
    -------
    int
        Number of rows written to the output file.
    """
    output_path = Path(output_file)
    metadata = metadata or {}

    if output_path.suffix.lower() != _ECSV_SUFFIX:
        raise ValueError("Job grid output file must use the '.ecsv' extension.")

    row_iterator = iter(job_rows)
    try:
        first_row = next(row_iterator)
    except StopIteration:
        _write_empty_ecsv_header(output_path, JOB_GRID_COLUMNS, metadata)
        logger.info(f"Writing job grid with 0 rows to '{output_path}'.")
        return 0

    serialized_first_row = _serialize_job_row(first_row)
    optional_columns = [
        angle_name for angle_name in _OPTIONAL_ANGLE_FIELDS if angle_name in serialized_first_row
    ]
    output_columns = [*JOB_GRID_COLUMNS, *optional_columns]

    output_rows = []
    row_count = 0
    write_header = True
    serialized_rows = _iter_with_first(serialized_first_row, map(_serialize_job_row, row_iterator))
    for serialized_row in serialized_rows:
        output_rows.append({column: serialized_row.get(column) for column in output_columns})
        row_count += 1
        if len(output_rows) >= _STREAM_CHUNK_SIZE:
            _flush_stream_chunk(
                output_path,
                output_rows,
                output_columns,
                metadata,
                write_header=write_header,
            )
            output_rows = []
            write_header = False

    if output_rows:
        _flush_stream_chunk(
            output_path,
            output_rows,
            output_columns,
            metadata,
            write_header=write_header,
        )

    logger.info(f"Writing job grid with {row_count} rows to '{output_path}'.")
    return row_count


def read_job_grid(input_file):
    """
    Read executable job rows from ECSV input.

    Parameters
    ----------
    input_file : str or Path
        Input file path.

    Returns
    -------
    tuple[list[dict], dict]
        Deserialized job rows and metadata.
    """
    input_path = Path(input_file)

    if input_path.suffix.lower() != _ECSV_SUFFIX:
        raise ValueError("Job grid input file must use the '.ecsv' extension.")

    table = Table.read(input_path, format=_ECSV_FORMAT)
    rows = [{column_name: row[column_name] for column_name in table.colnames} for row in table]
    return [_deserialize_job_row(row) for row in rows], dict(table.meta)
