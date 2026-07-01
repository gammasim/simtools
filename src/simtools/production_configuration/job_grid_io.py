"""Read and write executable job grids for production preparation."""

import logging
from io import StringIO
from pathlib import Path

from astropy import units as u
from astropy.table import Table

from simtools.production_configuration.job_grid_summary import build_job_grid_summary

logger = logging.getLogger(__name__)

_ECSV_SUFFIX = ".ecsv"
_ECSV_FORMAT = "ascii.ecsv"
_STREAM_CHUNK_SIZE = 10_000
JOB_GRID_FORMAT_VERSION = "0.2.0"

JOB_GRID_COLUMNS = [
    "run_number",
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "ha",
    "dec",
    "energy_min",
    "energy_max",
    "cores_per_shower",
    "core_scatter_max",
    "view_cone_min",
    "view_cone_max",
    "showers_per_run",
    "nsb_rate",
    "model_version",
    "array_layout_name",
    "corsika_le_interaction",
    "corsika_he_interaction",
]

JOB_GRID_QUANTITY_UNITS = {
    "azimuth_angle": u.deg,
    "zenith_angle": u.deg,
    "ha": u.deg,
    "dec": u.deg,
    "energy_min": u.GeV,
    "energy_max": u.GeV,
    "core_scatter_max": u.m,
    "view_cone_min": u.deg,
    "view_cone_max": u.deg,
}
JOB_GRID_COLUMN_UNITS = {
    **JOB_GRID_QUANTITY_UNITS,
    "nsb_rate": 1 / (u.cm**2 * u.ns * u.sr),
}

JOB_GRID_QUANTITY_FIELDS = tuple(JOB_GRID_QUANTITY_UNITS)

_JOB_GRID_COLUMN_DTYPES = {
    "primary": str,
    "azimuth_angle": float,
    "zenith_angle": float,
    "energy_min": float,
    "energy_max": float,
    "cores_per_shower": int,
    "core_scatter_max": float,
    "view_cone_min": float,
    "view_cone_max": float,
    "showers_per_run": int,
    "nsb_rate": float,
    "model_version": str,
    "array_layout_name": str,
    "corsika_le_interaction": str,
    "corsika_he_interaction": str,
    "run_number": int,
    "ha": float,
    "dec": float,
}


def _serialize_quantity(value, unit):
    """Serialize a quantity as a scalar in its canonical grid unit."""
    return float(u.Quantity(value).to_value(unit))


def _deserialize_quantity(value, unit):
    """Deserialize a scalar with its ECSV column unit."""
    return float(value) * u.Unit(unit)


def _serialize_job_row(job_row):
    """Serialize one job row to the on-disk schema."""
    serialized_row = {
        "primary": job_row["primary"],
        "cores_per_shower": int(job_row["cores_per_shower"]),
        "showers_per_run": int(job_row["showers_per_run"]),
        "nsb_rate": float(job_row["nsb_rate"]),
        "model_version": job_row["model_version"],
        "array_layout_name": job_row["array_layout_name"],
        "corsika_le_interaction": job_row["corsika_le_interaction"],
        "corsika_he_interaction": job_row["corsika_he_interaction"],
        "run_number": int(job_row["run_number"]),
    }

    for quantity_name, unit in JOB_GRID_QUANTITY_UNITS.items():
        serialized_row[quantity_name] = _serialize_quantity(job_row[quantity_name], unit)

    return serialized_row


def _deserialize_job_row(serialized_row, column_units=None):
    """Deserialize one stored row to the in-memory job-row schema."""
    job_row = {
        "primary": serialized_row["primary"],
        "cores_per_shower": int(serialized_row["cores_per_shower"]),
        "showers_per_run": int(serialized_row["showers_per_run"]),
        "nsb_rate": float(serialized_row.get("nsb_rate", 1.0)),
        "model_version": serialized_row["model_version"],
        "array_layout_name": serialized_row["array_layout_name"],
        "corsika_le_interaction": serialized_row["corsika_le_interaction"],
        "corsika_he_interaction": serialized_row["corsika_he_interaction"],
        "run_number": int(serialized_row["run_number"]),
    }

    column_units = column_units or {}
    for quantity_name, canonical_unit in JOB_GRID_QUANTITY_UNITS.items():
        if quantity_name in serialized_row:
            unit = column_units.get(quantity_name) or canonical_unit
            job_row[quantity_name] = _deserialize_quantity(serialized_row[quantity_name], unit).to(
                canonical_unit
            )
            continue

        if quantity_name not in ("ha", "dec"):
            raise KeyError(f"Missing required job-grid column '{quantity_name}'.")

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
    metadata["job_grid_format_version"] = JOB_GRID_FORMAT_VERSION
    metadata["job_grid_summary"] = build_job_grid_summary(job_rows)

    if output_path.suffix.lower() != _ECSV_SUFFIX:
        raise ValueError("Job grid output file must use the '.ecsv' extension.")

    output_columns = JOB_GRID_COLUMNS
    output_rows = [
        {column: row.get(column) for column in output_columns} for row in serialized_rows
    ]

    output_table = _build_output_table(output_rows, output_columns, metadata)
    logger.info(f"Writing job grid with {len(job_rows)} rows to '{output_path}'.")
    output_table.write(output_path, format=_ECSV_FORMAT, overwrite=True)


def _build_output_table(output_rows, output_columns, metadata=None):
    """Build an Astropy table for serialized output rows."""
    output_table = Table(rows=output_rows, names=output_columns)
    for column_name, unit in JOB_GRID_COLUMN_UNITS.items():
        if column_name in output_table.colnames:
            output_table[column_name].unit = unit
    output_table.meta = metadata or {}
    return output_table


def _write_empty_ecsv_header(output_path, output_columns, metadata):
    """Write an ECSV header using Astropy's schema/metadata handling."""
    empty_table = Table(
        names=output_columns,
        dtype=[_JOB_GRID_COLUMN_DTYPES[column] for column in output_columns],
    )
    for column_name, unit in JOB_GRID_COLUMN_UNITS.items():
        empty_table[column_name].unit = unit
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
    All coordinate columns are required so every row carries HA/Dec and Az/Ze.

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
    metadata = metadata.copy() if metadata else {}
    metadata["job_grid_format_version"] = JOB_GRID_FORMAT_VERSION

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
    output_columns = JOB_GRID_COLUMNS

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
    column_units = {column_name: table[column_name].unit for column_name in table.colnames}
    return [_deserialize_job_row(row, column_units) for row in rows], dict(table.meta)


def read_job_grid_row(input_file, row_index):
    """
    Read a single row from an ECSV job grid by its 1-based index.

    Parameters
    ----------
    input_file : str or Path
        Input file path.
    row_index : int
        1-based index of the row to read (first row is 1).

    Returns
    -------
    tuple[dict, dict]
        Deserialized job row and metadata.

    Raises
    ------
    IndexError
        If ``row_index`` is outside the valid range.
    """
    rows, metadata = read_job_grid(input_file)
    if row_index < 1 or row_index > len(rows):
        raise IndexError(
            f"Row index {row_index} is out of range for a grid with {len(rows)} row(s)."
        )
    return rows[row_index - 1], metadata


def job_grid_row_to_simulate_prod_args(job_row, metadata=None):
    """
    Convert an in-memory job grid row to simulate_prod argument format.

    The returned dictionary can be merged into a simulate_prod ``args_dict`` so that
    values from the job grid row take precedence over previously parsed arguments.

    Parameters
    ----------
    job_row : dict
        A single deserialized job row as returned by :func:`read_job_grid` or
        :func:`read_job_grid_row`.
    metadata : dict, optional
        Job grid metadata as returned alongside the rows.  When provided,
        ``site`` and ``simulation_software`` are included in the result.

    Returns
    -------
    dict
        Argument dictionary compatible with ``simulate_prod`` ``args_dict`` keys.
    """
    args = {
        "primary": job_row["primary"],
        "azimuth_angle": job_row["azimuth_angle"],
        "zenith_angle": job_row["zenith_angle"],
        "energy_range": (job_row["energy_min"], job_row["energy_max"]),
        "core_scatter": (int(job_row["cores_per_shower"]), job_row["core_scatter_max"]),
        "view_cone": (job_row["view_cone_min"], job_row["view_cone_max"]),
        "showers_per_run": int(job_row["showers_per_run"]),
        "model_version": job_row["model_version"],
        "array_layout_name": job_row["array_layout_name"],
        "corsika_le_interaction": job_row["corsika_le_interaction"],
        "corsika_he_interaction": job_row["corsika_he_interaction"],
        "run_number": int(job_row["run_number"]),
        # Force the run number offset to zero,
        # since the job grid row already specifies the run number.
        "run_number_offset": 0,
    }
    if metadata:
        for key in ("site", "simulation_software"):
            if metadata.get(key):
                args[key] = metadata[key]
    return args
