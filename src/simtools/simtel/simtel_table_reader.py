#!/usr/bin/python3
"""Read tabular data in sim_telarray format and return as astropy table."""

import logging

from astropy.table import Table

logger = logging.getLogger(__name__)


def _data_columns_pm_photoelectron_spectrum():
    """
    Column description for single p.e. data (parameter pm_photoelectron_spectrum).

    Returns
    -------
    list, str
        List of dictionaries with column name, description and unit and a description of the table.
    """
    return (
        [
            {"name": "amplitude", "description": "Signal amplitude", "unit": None},
            {
                "name": "response",
                "description": "response without afterpulsing component",
                "unit": None,
            },
            {
                "name": "response_with_ap",
                "description": "response including afterpulsing component",
                "unit": None,
            },
        ],
        "Photoelectron spectrum",
    )


def _data_columns_quantum_efficiency():
    """
    Column description for quantum efficiency data (parameter quantum_efficiency).

    Returns
    -------
    list, str
        List of dictionaries with column name, description and unit and a description of the table.
    """
    return (
        [
            {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
            {
                "name": "efficiency",
                "description": "Quantum efficiency",
                "unit": None,
            },
            {
                "name": "efficiency_rms",
                "description": "Quantum efficiency (standard deviation)",
                "unit": None,
            },
        ],
        "Quantum efficiency",
    )


def _data_columns_camera_filter():
    """
    Column description for camera filter data (parameter camera_filter).

    Returns
    -------
    list, str
        List of dictionaries with column name, description and unit and a description of the table.
    """
    return (
        [
            {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
            {
                "name": "transmission",
                "description": "Average transmission",
                "unit": None,
            },
        ],
        "Camera window transmission",
    )


def read_simtel_table(parameter_name, file_path):
    """
    Read sim_telarray table file for a given parameter.

    Parameters
    ----------
    parameter_name: str
        Model parameter name.
    file_path: Path
        Name (full path) of the sim_telarray table file.

    Returns
    -------
    Table
        Astropy table.
    """
    try:
        columns_info, description = globals()[f"_data_columns_{parameter_name}"]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported parameter for sim_telarray table reading: {parameter_name}"
        ) from exc

    data, meta_from_simtel = _read_simtel_data(file_path)

    metadata = {
        "Name": parameter_name,
        "File": str(file_path),
        "Description:": description,
        "Context_from_sim_telarray": meta_from_simtel,
    }

    rows = []
    for line in data.splitlines():
        parts = line.split()
        if len(parts) < len(columns_info):  # Fill missing columns with zeros
            parts += [0.0] * (len(columns_info) - len(parts))
        rows.append([float(part) for part in parts])

    table = Table(rows=rows, names=[col["name"] for col in columns_info])

    for col, info in zip(table.colnames, columns_info):
        table[col].unit = info.get("unit")
        table[col].description = info.get("description")

    table.meta.update(metadata)

    return table


def _read_simtel_data(file_path):
    """
    Read data and comments from sim_telarray table.

    Parameters
    ----------
    file_path: Path
        Path to the sim_telarray table file.

    Returns
    -------
    str, str
        data, metadata (comments)
    """
    logger.debug(f"Reading sim_telarray table from {file_path}")
    meta_lines = []
    data_lines = []

    with open(file_path, encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith("#"):
                meta_lines.append(stripped.lstrip("#").strip())
            elif stripped:
                data_lines.append(stripped.split("%%%")[0].split("#")[0].strip())  # Remove comments

    return "\n".join(data_lines), "\n".join(meta_lines)
