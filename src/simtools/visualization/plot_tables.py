#!/usr/bin/python3
"""Plot tabular data."""

from pathlib import Path

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.constants import SCHEMA_PATH
from simtools.db import db_handler
from simtools.io_operations import legacy_data_handler
from simtools.visualization import visualize


def plot(config, output_file, db_config=None):
    """
    Plot tabular data from data or from model parameter files.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.
    output_file: str
        Output file.
    db_config: dict, optional
        Database configuration dictionary for accessing the model parameter database.
    """
    data = read_table_data(config, db_config)

    fig = visualize.plot_1d(
        data,
        **config,
    )
    visualize.save_figure(fig, output_file)

    return output_file


def read_table_data(config, db_config):
    """
    Read table data from file or parameter database.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.

    Returns
    -------
    Dict
        Dict with table data (astropy tables).
    """
    data = {}

    for _config in config["tables"]:
        if "parameter" in _config:
            table = _read_table_from_model_database(_config, db_config)
        elif "file_name" in _config:
            if "legacy" in _config.get("type", ""):
                table = legacy_data_handler.read_legacy_data_as_table(
                    _config["file_name"], _config["type"]
                )
            else:
                table = Table.read(_config["file_name"], format="ascii.ecsv")
        else:
            raise ValueError("No table data defined in configuration.")

        if _config.get("normalize_y"):
            table[_config["column_y"]] = (
                table[_config["column_y"]] / table[_config["column_y"]].max()
            )
        if _config.get("select_values"):
            table = _select_values_from_table(
                table,
                _config["select_values"]["column_name"],
                _config["select_values"]["value"],
            )
        label = _config.get("label", f"{_config.get('column_x')} vs {_config.get('column_y')}")
        data[label] = gen.get_structure_array_from_table(
            table,
            [
                _config["column_x"],
                _config["column_y"],
                _config.get("column_x_err"),
                _config.get("column_y_err"),
            ],
        )
    return data


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
        Astropy table
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


def generate_plot_configurations(
    parameter, parameter_version, site, telescope, output_path, plot_type
):
    """
    Generate plot configurations for a model parameter from schema files.

    Parameters
    ----------
    parameter: str
        Model parameter name.

    Returns
    -------
    tuple
        Tuple containing a list of plot configurations and a list of output file names.
        Return None, if no plot configurations are found.
    """
    schema = gen.change_dict_keys_case(
        gen.collect_data_from_file(
            file_name=SCHEMA_PATH / "model_parameters" / f"{parameter}.schema.yml"
        )
    )
    configs = schema.get("plot_configuration")
    if not configs:
        return None
    if plot_type != "all":
        configs = [config for config in configs if config.get("type") == plot_type]
        if not configs:
            raise ValueError(
                f"No plot configuration found for type '{plot_type}' in parameter '{parameter}'."
            )

    output_files = []
    for _config in configs:
        for _table in _config.get("tables", []):
            _table["parameter_version"] = parameter_version
            _table["site"] = site
        output_files.append(
            _generate_output_file_name(
                parameter=parameter,
                parameter_version=parameter_version,
                site=site,
                telescope=telescope,
                plot_type=_config.get("type"),
                output_path=output_path,
            )
        )

    return configs, output_files


def _generate_output_file_name(
    parameter,
    parameter_version,
    site,
    telescope,
    plot_type,
    output_path=None,
    file_extension=".pdf",
):
    """Generate output file name based on table file and appendix."""
    parts = [parameter, parameter_version, site]
    if telescope:
        parts.append(telescope)
    if plot_type != parameter:
        parts.append(plot_type)
    filename = "_".join(parts) + file_extension

    return Path(output_path) / filename if output_path else Path(filename)
