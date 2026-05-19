"""Serialization helpers for production-grid rows."""

from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

DEFAULT_SERIALIZATION_ROUND_DECIMALS = 6


def collect_point_keys(grid_points):
    """Collect all grid-point keys while preserving first-seen order."""
    all_keys = []
    for point in grid_points:
        for key in point:
            if key not in all_keys:
                all_keys.append(key)
    return all_keys


def serialize_grid_value(value, serialization_round_decimals=DEFAULT_SERIALIZATION_ROUND_DECIMALS):
    """Serialize one grid value and return ``(value, unit)``."""
    if isinstance(value, u.Quantity):
        serialized = round(float(value.value), serialization_round_decimals)
        return serialized, str(value.unit)

    if isinstance(value, dict) and "value" in value:
        return value["value"], value.get("unit")

    if value is None:
        return np.nan, None

    if isinstance(value, (np.floating, float)):
        return round(float(value), serialization_round_decimals), None

    if isinstance(value, (np.integer, int)):
        return int(value), None

    return value, None


def build_serialized_rows(
    grid_points, all_keys, serialization_round_decimals=DEFAULT_SERIALIZATION_ROUND_DECIMALS
):
    """Build serialized row dictionaries and collect units."""
    rows = []
    units = {}

    for point in grid_points:
        row = {}
        for key in all_keys:
            serialized_value, unit = serialize_grid_value(
                point.get(key),
                serialization_round_decimals=serialization_round_decimals,
            )
            row[key] = serialized_value
            if unit is not None:
                units.setdefault(key, unit)
        rows.append(row)

    return rows, units


def build_grid_metadata(
    coordinate_system, observing_time=None, telescope_ids=None, lookup_table=None
):
    """Build metadata for a serialized production grid."""
    return {
        "coordinate_system": coordinate_system,
        "reference_frame": "ICRS (J2000)",
        "observing_time_utc": observing_time.isot if observing_time else None,
        "observing_time_scale": observing_time.scale if observing_time else None,
        "telescope_ids": telescope_ids,
        "lookup_table": str(Path(lookup_table)) if lookup_table else None,
    }


def serialize_grid_points(
    grid_points,
    output_file,
    coordinate_system,
    observing_time=None,
    telescope_ids=None,
    lookup_table=None,
    serialization_round_decimals=DEFAULT_SERIALIZATION_ROUND_DECIMALS,
):
    """
    Serialize grid points to an ECSV table file.

    Parameters
    ----------
    grid_points : list[dict]
        Grid rows to serialize.
    output_file : str or Path
        Output ECSV file path.
    coordinate_system : str
        Coordinate-system label stored in metadata.
    observing_time : Time, optional
        Observing time stored in metadata.
    telescope_ids : list, optional
        Telescope selection stored in metadata.
    lookup_table : str or Path, optional
        Lookup-table path stored in metadata.
    serialization_round_decimals : int, optional
        Number of decimal places used for scalar serialization.

    Returns
    -------
    astropy.table.Table
        Serialized output table.
    """
    if Path(output_file).suffix.lower() != ".ecsv":
        raise ValueError("Grid output file must use '.ecsv' extension.")

    all_keys = collect_point_keys(grid_points)
    rows, units = build_serialized_rows(
        grid_points,
        all_keys,
        serialization_round_decimals=serialization_round_decimals,
    )

    output_table = Table(rows=rows, names=all_keys)
    for column_name, unit in units.items():
        output_table[column_name].unit = u.Unit(unit)

    output_table.meta = build_grid_metadata(
        coordinate_system=coordinate_system,
        observing_time=observing_time,
        telescope_ids=telescope_ids,
        lookup_table=lookup_table,
    )
    output_table.write(output_file, format="ascii.ecsv", overwrite=True)
    return output_table
