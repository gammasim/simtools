"""Read and write executable job grids for production preparation."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.constants import SCHEMA_PATH, SCHEMA_URL
from simtools.data_model import validate_data
from simtools.io.ascii_handler import collect_data_from_file
from simtools.production_configuration.job_grid_summary import build_job_grid_summary
from simtools.utils.value_conversion import get_value_as_quantity, get_value_in_unit

logger = logging.getLogger(__name__)

_ECSV_SUFFIX = ".ecsv"
_ECSV_FORMAT = "ascii.ecsv"
_JOB_GRID_SCHEMA_FILE = SCHEMA_PATH / "job_grid_density.schema.yml"
_JOB_GRID_SCHEMA_URL = SCHEMA_URL + "job_grid_density.schema.yml"


@dataclass(frozen=True)
class JobGridSchema:
    """Runtime representation of the job-grid YAML schema."""

    version: str
    columns: tuple[str, ...]
    column_units: dict[str, u.Unit]
    column_dtypes: dict[str, object]


def _load_job_grid_schema():
    """Load the job-grid format definition from its YAML schema."""
    schema = collect_data_from_file(_JOB_GRID_SCHEMA_FILE)
    table_definition = next(item for item in schema["data"] if item["type"] == "data_table")
    column_definitions = table_definition["table_columns"]
    column_units = {
        column["name"]: u.Unit(column["unit"])
        for column in column_definitions
        if column.get("unit")
    }
    return JobGridSchema(
        version=schema["schema_version"],
        columns=tuple(column["name"] for column in column_definitions),
        column_units=column_units,
        column_dtypes={
            column["name"]: str if column["type"] == "string" else np.dtype(column["type"])
            for column in column_definitions
        },
    )


JOB_GRID_SCHEMA = _load_job_grid_schema()


def _cast_scalar(value, dtype):
    """Cast a scalar using its schema dtype and return a Python value."""
    if dtype is str:
        return str(value)
    return dtype.type(value).item()


def _serialize_job_row(job_row):
    """Serialize one job row to the on-disk schema."""
    return {
        name: float(get_value_in_unit(job_row[name], JOB_GRID_SCHEMA.column_units[name]))
        if name in JOB_GRID_SCHEMA.column_units
        else _cast_scalar(job_row[name], JOB_GRID_SCHEMA.column_dtypes[name])
        for name in JOB_GRID_SCHEMA.columns
    }


def _deserialize_job_row(serialized_row, column_units=None):
    """Deserialize one stored row to the in-memory job-row schema."""
    column_units = column_units or {}
    return {
        name: get_value_as_quantity(
            float(serialized_row[name]),
            column_units.get(name) or JOB_GRID_SCHEMA.column_units[name],
        ).to(JOB_GRID_SCHEMA.column_units[name])
        if name in JOB_GRID_SCHEMA.column_units
        else _cast_scalar(serialized_row[name], JOB_GRID_SCHEMA.column_dtypes[name])
        for name in JOB_GRID_SCHEMA.columns
    }


def _add_job_grid_schema_metadata(metadata):
    """Add schema reference metadata used for validation."""
    metadata.setdefault("cta", {}).setdefault("product", {}).setdefault("data", {}).setdefault(
        "model", {}
    )["url"] = _JOB_GRID_SCHEMA_URL
    return metadata


def _validate_job_grid_table(table):
    """Validate a job-grid table against the job-grid schema."""
    if len(table) == 0:
        return table
    return validate_data.DataValidator(
        schema_file=_JOB_GRID_SCHEMA_FILE,
        data_table=table.copy(copy_data=True),
    ).validate_and_transform()


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
    _add_job_grid_schema_metadata(metadata)
    metadata["job_grid_format_version"] = JOB_GRID_SCHEMA.version
    metadata["job_grid_summary"] = build_job_grid_summary(job_rows)

    if output_path.suffix.lower() != _ECSV_SUFFIX:
        raise ValueError("Job grid output file must use the '.ecsv' extension.")

    output_columns = JOB_GRID_SCHEMA.columns
    output_rows = [
        {column: row.get(column) for column in output_columns} for row in serialized_rows
    ]
    output_table = _build_output_table(output_rows, output_columns, metadata)
    output_table = _validate_job_grid_table(output_table)
    logger.info(f"Writing job grid with {len(job_rows)} rows to '{output_path}'.")
    output_table.write(output_path, format=_ECSV_FORMAT, overwrite=True)


def _build_output_table(output_rows, output_columns, metadata=None):
    """Build an Astropy table for serialized output rows."""
    output_table = Table(
        rows=[tuple(row[column] for column in output_columns) for row in output_rows],
        names=output_columns,
        dtype=[JOB_GRID_SCHEMA.column_dtypes[column] for column in output_columns],
    )
    for column_name, unit in JOB_GRID_SCHEMA.column_units.items():
        if column_name in output_table.colnames:
            output_table[column_name].unit = unit
    output_table.meta = metadata or {}
    return output_table


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

    table = _validate_job_grid_table(Table.read(input_path, format=_ECSV_FORMAT))
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
