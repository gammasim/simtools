#!/usr/bin/python3
"""Read tabular data in sim_telarray format and return as astropy table."""

import logging
import re
from pathlib import Path

import astropy.units as u
from astropy.table import Table

from simtools.utils import general as gen

logger = logging.getLogger(__name__)


def _data_columns_nsb_altitude_correction():
    """Column description for NSB spectrum altitude correction."""
    return [
        {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
        {"name": "transmission", "description": "Atmospheric transmission", "unit": None},
    ], "Night Sky Background spectrum correction for telescope altitude"


def _data_columns(parameter_name, n_columns, n_dim):
    """Get column definitions for parameter type."""
    parameter_handlers = {
        "primary_mirror_segmentation": _data_columns_primary_mirror_segmentation,
        "secondary_mirror_segmentation": _data_columns_secondary_mirror_segmentation,
        "fake_mirror_list": _data_columns_fake_mirror_list,
        "lightguide_efficiency_vs_wavelength": _data_columns_lightguide_efficiency_vs_wavelength,
        "mirror_list": _data_columns_mirror_list,
        "camera_filter": _data_columns_camera_filter,
        "quantum_efficiency": _data_columns_quantum_efficiency,
        "discriminator_pulse_shape": _data_columns_pulse_shape,
        "fadc_pulse_shape": _data_columns_pulse_shape,
        "dsum_shaping": _data_columns_pulse_shape,
        "pm_photoelectron_spectrum": _data_columns_pm_photoelectron_spectrum,
        "atmospheric_profile": _data_columns_atmospheric_profile,
        "nsb_reference_spectrum": _data_columns_nsb_reference_spectrum,
        "mirror_segmentation": _data_columns_mirror_segmentation,
        "correct_nsb_spectrum_to_telescope_altitude": _data_columns_nsb_altitude_correction,
        "lightguide_efficiency_vs_incidence_angle": (
            lambda: _data_columns_lightguide_efficiency_vs_incidence_angle(n_columns)
        ),
        "mirror_reflectivity": lambda: _data_columns_mirror_reflectivity(n_dim),
        "secondary_mirror_reflectivity": (
            lambda: _data_columns_secondary_mirror_reflectivity(n_dim)
        ),
    }

    if parameter_name in parameter_handlers:
        handler = parameter_handlers[parameter_name]
        return handler() if callable(handler) else handler

    raise ValueError(f"Unsupported parameter for sim_telarray table reading: {parameter_name}")


def _data_columns_atmospheric_profile():
    """Column representation for parameter atmospheric_profile."""
    return (
        [
            {"name": "altitude", "description": "Altitude", "unit": "km"},
            {"name": "density", "description": "Density", "unit": "g/cm^3"},
            {"name": "thickness", "description": "Thickness", "unit": "g/cm^2"},
            {
                "name": "refractive_index",
                "description": "Refractive index (n-1)",
                "unit": None,
            },
            {
                "name": "temperature",
                "description": "Temperature",
                "unit": "K",
            },
            {
                "name": "pressure",
                "description": "Pressure",
                "unit": "mbar",
            },
            {
                "name": "pw/w",
                "description": "Partial pressure of water vapor",
                "unit": None,
            },
        ],
        "Atmospheric profile",
    )


def _data_columns_pm_photoelectron_spectrum():
    """Column description for PM photoelectron spectrum."""
    return [
        {"name": "amplitude", "description": "Signal amplitude", "unit": "pe"},
        {"name": "response", "description": "Response", "unit": None},
        {"name": "response_with_ap", "description": "Response with afterpulsing", "unit": None},
    ], "Photoelectron spectrum"


def _data_columns_quantum_efficiency():
    """Column description for parameter quantum_efficiency."""
    return [
        {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
        {"name": "efficiency", "description": "Quantum Efficiency", "unit": None},
        {"name": "efficiency_rms", "description": "Quantum Efficiency RMS", "unit": None},
    ], "Quantum efficiency vs wavelength"


def _data_columns_camera_filter():
    """Column description for parameter camera_filter."""
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


def _data_columns_lightguide_efficiency_vs_wavelength():
    """Column description for parameter lightguide_efficiency_vs_wavelength."""
    return [
        {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
        {"name": "efficiency", "description": "Light collection efficiency", "unit": None},
    ], "Light guide efficiency vs wavelength"


def _data_columns_lightguide_efficiency_vs_incidence_angle(n_columns=2):
    """Column description for lightguide efficiency vs incidence angle parameters."""
    _columns = [
        {"name": "angle", "description": "Incidence angle", "unit": "deg"},
        {"name": "efficiency", "description": "Light collection efficiency", "unit": None},
    ]
    if n_columns == 3:
        _columns.append(
            {
                "name": "efficiency_rms",
                "description": "Light guide efficiency RMS",
                "unit": None,
            }
        )
    return _columns, "Light guide efficiency vs incidence angle"


def _data_columns_mirror_reflectivity(n_dim=None):
    """Column description for mirror reflectivity parameters.

    Parameters
    ----------
    n_columns : int
        Number of columns in the data (2 or 3 for RMS)
    n_dim : list, optional
        List of angles for multi-dimensional reflectivity data

    Returns
    -------
    tuple
        (column definitions, description string)
    """
    _columns = [{"name": "wavelength", "description": "Wavelength", "unit": "nm"}]

    if n_dim:
        # Multi-dimensional data with angles
        for angle in [0, 10, 20, 30, 40, 50, 60, 70, 80]:
            _columns.append(
                {
                    "name": f"reflectivity_{angle}deg",
                    "description": f"Mirror reflectivity at {angle} deg",
                    "unit": None,
                }
            )
    else:
        # Standard reflectivity data
        _columns.append(
            {
                "name": "reflectivity",
                "description": "Mirror reflectivity",
                "unit": None,
            }
        )

    return _columns, "Mirror reflectivity"


def _data_columns_primary_mirror_segmentation():
    """Column description for primary mirror segmentation parameters."""
    return [
        {"name": "segment_id", "description": "Mirror segment ID", "unit": None},
        {"name": "x_pos", "description": "X position", "unit": "m"},
        {"name": "y_pos", "description": "Y Position", "unit": "m"},
        {"name": "z_pos", "description": "Z Position", "unit": "m"},
    ], "Primary mirror segmentation layout"


def _data_columns_mirror_list():
    """Column description for mirror list parameters."""
    return [
        {"name": "x_pos", "description": "X position", "unit": "m"},
        {"name": "y_pos", "description": "Y position", "unit": "m"},
        {"name": "z_pos", "description": "Z position", "unit": "m"},
        {"name": "size", "description": "Mirror size", "unit": "m"},
    ], "Mirror positions and sizes"


def _data_columns_pulse_shape():
    """Column description for pulse shape parameters.

    Parameters
    ----------
    n_columns : int, optional
        Number of columns in the data file (default: 2)

    Returns
    -------
    tuple
        (column definitions, description string)
    """
    columns = [
        {"name": "time", "description": "Time", "unit": "ns"},
        {"name": "amplitude", "description": "Amplitude", "unit": None},
    ]
    return columns, "Pulse shape vs time"


def _data_columns_nsb_reference_spectrum():
    """Column description for parameter nsb_reference_spectrum."""
    return (
        [
            {"name": "wavelength", "description": "Wavelength", "unit": "nm"},
            {
                "name": "differential photon rate",
                "description": "Differential photon rate",
                "unit": "1.e9 / (nm s m^2 sr)",
            },
        ],
        "NSB reference spectrum",
    )


def _data_columns_secondary_mirror_segmentation():
    """Column description for secondary mirror segmentation parameters."""
    return [
        {"name": "segment_id", "description": "Mirror segment ID", "unit": None},
        {"name": "x_pos", "description": "X Position", "unit": "m"},
        {"name": "y_pos", "description": "Y Position", "unit": "m"},
        {"name": "z_pos", "description": "Z Position", "unit": "m"},
        {"name": "size", "description": "Segment size", "unit": "m"},
    ], "Secondary mirror segmentation layout"


def _data_columns_ray_tracing():
    """Column description for ray tracing parameters."""
    return [
        {"name": "Off-axis angle", "unit": "deg"},
        {"name": "d80_cm", "unit": "cm"},
        {"name": "d80_deg", "unit": "deg"},
        {"name": "eff_area", "unit": "m2"},
        {"name": "eff_flen", "unit": "cm"},
    ], "Ray Tracing Data"


def _data_columns_dsum_shaping():
    """Column description for digital sum shaping parameters."""
    return [
        {"name": "time", "description": "Time", "unit": "ns"},
        {"name": "amplitude", "description": "Digital sum shaping amplitude", "unit": None},
    ], "Digital sum shaping function"


def _data_columns_fake_mirror_list():
    """Column description for fake mirror list parameters."""
    return [
        {"name": "x_pos", "description": "X Position", "unit": "m"},
        {"name": "y_pos", "description": "Y Position", "unit": "m"},
        {"name": "z_pos", "description": "Z Position", "unit": "m"},
        {"name": "size", "description": "Mirror size", "unit": "m"},
    ], "Fake mirror positions and sizes"


def _data_columns_secondary_mirror_reflectivity(n_dim=None):
    """Column description for secondary mirror reflectivity."""
    columns = [{"name": "wavelength", "description": "Wavelength", "unit": "nm"}]

    if n_dim:
        for angle in range(0, 90, 10):
            columns.append(
                {
                    "name": f"reflectivity_{angle}deg",
                    "description": f"Reflectivity at {angle} deg",
                    "unit": None,
                }
            )
    else:
        columns.append({"name": "reflectivity", "description": "Reflectivity", "unit": None})
    return columns, "Secondary mirror reflectivity vs wavelength"


def _data_columns_mirror_segmentation():
    """Column description for primary mirror segmentation parameters."""
    return [
        {"name": "segment_id", "description": "Mirror segment ID", "unit": None},
        {"name": "x_pos", "description": "X position", "unit": "m"},
        {"name": "y_pos", "description": "Y position", "unit": "m"},
        {"name": "z_pos", "description": "Z position", "unit": "m"},
    ], "Primary mirror segmentation layout"


def read_simtel_table(parameter_name, file_path):
    """Read sim_telarray table file for a given parameter."""
    if Path(file_path).suffix == ".ecsv":  # table is already in correct format
        return Table.read(file_path, format="ascii.ecsv")

    if parameter_name == "atmospheric_transmission":
        return _read_simtel_data_for_atmospheric_transmission(file_path)

    if parameter_name == "dsum_shaping":
        columns, description = _data_columns_dsum_shaping()
        rows, meta_from_simtel, n_columns, _ = _read_simtel_data(file_path)

        # Ensure rows match column count
        rows = _adjust_columns_length(rows, len(columns))

        # Create table with metadata
        table = Table(rows=rows, names=[col["name"] for col in columns])

        # Set units and descriptions
        for col, info in zip(table.colnames, columns):
            table[col].unit = info.get("unit")
            table[col].description = info.get("description")

        # Add metadata
        table.meta.update(
            {
                "Name": parameter_name,
                "File": str(file_path),
                "Description": description,
                "Context_from_sim_telarray": meta_from_simtel,
            }
        )

        return table

    rows, meta_from_simtel, n_columns, n_dim = _read_simtel_data(file_path)
    columns_info, description = _data_columns(parameter_name, n_columns, n_dim)

    rows = _adjust_columns_length(rows, len(columns_info))

    metadata = {
        "Name": parameter_name,
        "File": str(file_path),
        "Description:": description,
        "Context_from_sim_telarray": meta_from_simtel,
    }

    table = Table(rows=rows, names=[col["name"] for col in columns_info])
    for col, info in zip(table.colnames, columns_info):
        table[col].unit = info.get("unit")
        table[col].description = info.get("description")
    table.meta.update(metadata)

    return table


def get_column_definitions(parameter_name, n_columns=None, n_dim=None):
    """Get column definitions for a parameter type.

    Parameters
    ----------
    parameter_name : str
        Name of the parameter to get column definitions for
    n_columns : int, optional
        Number of columns in the data
    n_dim : list, optional
        List of dimensions for multi-dimensional data

    Returns
    -------
    tuple
        (columns, description)
    """
    if parameter_name == "ray-tracing":
        return _data_columns_ray_tracing()
    return _data_columns(parameter_name, n_columns, n_dim)


def read_data(file_path):
    """Read data from sim_telarray format file.

    Parameters
    ----------
    file_path : Path
        Path to the file to read

    Returns
    -------
    tuple
        (data_lines, metadata, n_columns, n_dim)
    """
    return _read_simtel_data(file_path)


def _adjust_columns_length(rows, n_columns):
    """
    Adjust row lengths to match the specified column count.

    - Truncate rows with extra columns beyond the specified count 'n_columns'.
    - Pad shorter rows with zeros.
    """
    return [row[:n_columns] + [0.0] * max(0, n_columns - len(row)) for row in rows]


def _read_simtel_data(file_path):
    """Read data from sim_telarray table file.

    Parameters
    ----------
    file_path : Path or str
        Path to the file to read

    Returns
    -------
    tuple
        (data_lines, meta_lines, n_columns, n_dim_axis)
    """

    def _convert_value(value: str):
        """Convert string value to numeric type, handling special cases."""
        value = value.strip()

        # Skip comment markers and metadata markers without warning
        if any(marker in value for marker in ["%%%", "original:", "#"]):
            return None

        # Handle special markers
        if value.lower() in ("hex", "+-", "na", "null"):
            return 0.0

        try:
            return float(value)
        except ValueError:
            logger.warning(f"Could not convert '{value}' to float")
            return None

    logger.debug(f"Reading sim_telarray table from {file_path}")
    meta_lines = []
    data_lines = []
    n_dim_axis = None

    with Path(file_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                meta_lines.append(line.lstrip("#").strip())
            else:
                data = line.split("#")[0].strip()
                if data:
                    values = [_convert_value(v) for v in data.split()]
                    if values:
                        data_lines.append(values)

    max_cols = max(len(row) for row in data_lines) if data_lines else 0
    return data_lines, "\n".join(meta_lines), max_cols, n_dim_axis


def _read_simtel_data_for_atmospheric_transmission(file_path):
    """
    Read data and comments from sim_telarray table for atmospheric_transmission.

    Parameters
    ----------
    file_path: Path
        Path to the sim_telarray table file.

    Returns
    -------
    astropy table
        Table with atmospheric transmission.
    """
    lines = lines = gen.read_file_encoded_in_utf_or_latin(file_path)

    observatory_level, height_bins = _read_header_line_for_atmospheric_transmission(
        lines, file_path
    )

    wavelengths = []
    heights = []
    extinctions = []
    meta_lines = []

    for line in lines:
        if line.startswith("#") or not line.strip():
            meta_lines.append(line.lstrip("#").strip())
            continue
        parts = line.split()
        try:
            wl = float(parts[0])
            for i, height in enumerate(height_bins):
                extinction_value = float(parts[i + 1])
                if extinction_value == 99999.0:
                    continue
                wavelengths.append(wl)
                heights.append(height)
                extinctions.append(extinction_value)
        except (ValueError, IndexError):
            logger.debug(f"Skipping malformed line: {line.strip()}")

    table = Table()
    table["wavelength"] = wavelengths
    table["altitude"] = heights
    table["extinction"] = extinctions

    table.meta.update(
        {
            "Name": "atmospheric_transmission",
            "File": str(file_path),
            "Description": "Atmospheric transmission",
            "Context_from_sim_telarray": "\n".join(meta_lines),
            "observatory_level": observatory_level,
        }
    )

    return table


def _read_header_line_for_atmospheric_transmission(lines, file_path):
    """Reader observatory level and height bins from header line for atmospheric transmission."""
    header_line = None
    observatory_level = None
    for line in lines:
        if "H2=" in line and "H1=" in line:
            match_h2 = re.search(r"H2=\s*([\d.]+)", line)
            if match_h2:
                observatory_level = float(match_h2.group(1)) * u.km

            if "H1=" in line:
                header_line = line.split("H1=")[-1].strip()
            break

    if header_line is None:
        raise ValueError(f"Header with 'H1=' not found file {file_path}")

    height_bins = [float(x.replace(",", "")) for x in header_line.split()]

    return observatory_level, height_bins
