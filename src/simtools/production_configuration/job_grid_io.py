"""Read and write executable job grids for production preparation."""

from pathlib import Path

from astropy import units as u
from astropy.table import Table

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
    "nshow",
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

_INTEGER_FIELDS = ["core_scatter_number", "nshow", "run_number"]


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
        "nshow": int(job_row["nshow"]),
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

    return serialized_row


def _deserialize_job_row(serialized_row):
    """Deserialize one stored row to the in-memory job-row schema."""
    job_row = {
        "primary": serialized_row["primary"],
        "core_scatter_number": int(serialized_row["core_scatter_number"]),
        "nshow": int(serialized_row["nshow"]),
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
    metadata = metadata or {}

    if output_path.suffix.lower() != ".ecsv":
        raise ValueError("Job grid output file must use the '.ecsv' extension.")

    output_table = Table(rows=serialized_rows, names=JOB_GRID_COLUMNS)
    output_table.meta = metadata
    output_table.write(output_path, format="ascii.ecsv", overwrite=True)


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

    if input_path.suffix.lower() != ".ecsv":
        raise ValueError("Job grid input file must use the '.ecsv' extension.")

    table = Table.read(input_path, format="ascii.ecsv")
    rows = [{column_name: row[column_name] for column_name in table.colnames} for row in table]
    return [_deserialize_job_row(row) for row in rows], dict(table.meta)
