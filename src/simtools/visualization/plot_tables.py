#!/usr/bin/python3
"""Plot tabular data."""

import logging

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.db import db_handler
from simtools.simtel import simtel_table_reader
from simtools.visualization import visualize

logger = logging.getLogger(__name__)


def _get_default_plot_options():
    """Get default plotting options."""
    return {
        "xscale": "linear",
        "yscale": "linear",
        "xlim": None,
        "ylim": None,
        "title": None,
        "palette": "default",
        "no_legend": False,
        "big_plot": False,
        "no_markers": False,
        "empty_markers": False,
        "plot_ratio": False,
        "plot_difference": False,
        "marker": None,
        "linestyle": "-",
    }


def plot(config, output_file, db_config=None):
    """Plot tabular data from file or from model parameter files."""
    # Get default plot options
    plot_config = _get_default_plot_options()

    # Read data and get updated config with axis labels
    data, updated_config = read_table_data(config, db_config)

    # Update plot config with original config and any updates from data reading
    plot_config.update(config)
    plot_config.update(updated_config)

    fig = visualize.plot_1d(data, **plot_config)
    visualize.save_figure(fig, output_file)


def _get_column_definitions(parameter, n_cols=None):
    """Get column definitions for parameter type."""
    try:
        return simtel_table_reader.get_column_definitions(parameter, n_cols)
    except (ValueError, AttributeError) as e:
        logger.debug(f"Could not get column definitions: {e}")
        return None, None


def _update_config_from_columns(columns):
    """Update plot configuration with column information."""
    if not columns:
        return {}

    updated = {}
    updated["xtitle"] = f"{columns[0]['description']}"
    if columns[0].get("unit"):
        updated["xtitle"] += f" [{columns[0]['unit']}]"

    updated["ytitle"] = f"{columns[1]['description']}"
    if columns[1].get("unit"):
        updated["ytitle"] += f" [{columns[1]['unit']}]"

    return updated


def _handle_reflectivity_data(table, config):
    """Handle multi-angle reflectivity data."""
    data = {}
    angle_columns = [c for c in table.colnames if c.startswith("reflectivity_")]
    for col in angle_columns:
        angle = col.replace("reflectivity_", "").replace("deg", "")
        label = f"{config.get('label', '')} {angle}°"
        data[label] = gen.get_structure_array_from_table(table, ["wavelength", col])
    return data


def _infer_parameter_type(filename):
    """Infer parameter type from filename."""
    filename = filename.lower()
    if "spe" in filename:
        return "pm_photoelectron_spectrum"
    if "pulse" in filename:
        return "pulse_shape"
    if "qe" in filename:
        return "quantum_efficiency"
    if "ref" in filename:
        return "mirror_reflectivity"
    return None


def read_table_data(config, db_config):
    """Read and prepare table data for plotting."""
    data = {}
    updated_config = {}

    for table_config in config["tables"]:
        try:
            parameter = table_config.get("parameter", "") or _infer_parameter_type(
                table_config.get("file_name", "")
            )
            if parameter:
                parameter = parameter.lower()

            # Read table data
            table = _read_table_from_model_database(table_config, db_config)

            # Handle multi-angle reflectivity data
            if "reflectivity_" in str(table.colnames):
                data.update(_handle_reflectivity_data(table, table_config))
                updated_config.update(
                    {
                        "xtitle": "Wavelength [nm]",
                        "ytitle": "Reflectivity",
                        "title": table_config.get("parameter", "Reflectivity vs Wavelength"),
                    }
                )
                continue

            # Get column definitions and update configuration
            columns, description = _get_column_definitions(parameter, len(table.colnames))

            # Set default columns if no definitions found
            if not columns and table.colnames:
                columns = [
                    {"name": table.colnames[0], "description": table.colnames[0], "unit": None},
                    {"name": table.colnames[1], "description": table.colnames[1], "unit": None},
                ]

            # Update table configuration with column information
            if columns:
                table_config.update(
                    {
                        "column_x": columns[0]["name"],
                        "column_y": columns[1]["name"],
                        "title": description or parameter,
                    }
                )
                updated_config.update(_update_config_from_columns(columns))

            # Set label from parameter version if available
            if table_config.get("parameter_version"):
                table_config["label"] = table_config["parameter_version"]
            elif not table_config.get("label"):
                table_config["label"] = parameter

            # For mirror list plots, set linestyle to empty string (dots only)
            if parameter and "mirror_list" in parameter:
                updated_config["linestyle"] = ""
            # Prepare and store data
            data[table_config["label"]] = _prepare_table_data(table, table_config)

        except Exception as e:
            logger.error(f"Error reading table data: {e!s}")
            raise

    return data, updated_config


def _read_table_from_model_database(table_config, db_config):
    """
    Read table data from model parameter database.

    Parameters
    ----------
    table_config: dict
        Configuration dictionary for table data.

    Returns
    -------
    Table
        Astropy table.
    """
    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    return db.export_model_file(
        parameter=table_config["parameter"],
        site=table_config["site"],
        array_element_name=table_config.get("telescope"),
        parameter_version=table_config.get("parameter_version"),
        model_version=table_config.get("model_version"),
        export_file_as_table=True,
    )


def _select_values_from_table(table, column_name, value):
    """Return a table with only the rows where column_name == value."""
    return table[np.isclose(table[column_name], value)]


def read_simtel_table(parameter_name, file_path):
    """Read tabular data in sim_telarray format."""
    try:
        # Get column definitions
        columns, description = simtel_table_reader.get_column_definitions(parameter_name)

        # Read data using the public API
        data = simtel_table_reader.read_data(file_path)

        # Create table with proper column names
        table = Table(rows=data[0], names=[col["name"] for col in columns])

        # Add metadata
        table.meta.update(
            {"Name": parameter_name, "Description": description, "File": str(file_path)}
        )

        return table

    except Exception as e:
        logger.error(f"Error reading {parameter_name} from {file_path}: {e!s}")
        raise


def _prepare_table_data(table, config):
    """Prepare table data for plotting."""
    # Validate columns
    for col_key in ["column_x", "column_y"]:
        if col_key not in config:
            raise ValueError(f"Missing required {col_key} in configuration")
        if config[col_key] not in table.colnames:
            cols = ", ".join(table.colnames)
            raise ValueError(f"Column '{config[col_key]}' not found in table. Available: {cols}")

    # Handle multi-column tables
    if len(table.colnames) > 2:
        table = Table(
            [table[config["column_x"]], table[config["column_y"]]],
            names=[config["column_x"], config["column_y"]],
            copy=True,
        )

    # Apply transformations
    if config.get("normalize_y"):
        table[config["column_y"]] /= table[config["column_y"]].max()
    if config.get("select_values"):
        table = _select_values_from_table(
            table, config["select_values"]["column_name"], config["select_values"]["value"]
        )

    return gen.get_structure_array_from_table(
        table,
        [
            config["column_x"],
            config["column_y"],
            config.get("column_x_err"),
            config.get("column_y_err"),
        ],
    )
