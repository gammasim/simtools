#!/usr/bin/python3
"""Plot tabular data."""

import logging
from pathlib import Path

import numpy as np
import packaging.version
from astropy.table import Table

from simtools.constants import SCHEMA_PATH
from simtools.io import ascii_handler, legacy_data_handler
from simtools.model_repository.reader import resolve_model_reader
from simtools.runtime.context import RunContext
from simtools.utils import value_conversion
from simtools.visualization import visualize

_logger = logging.getLogger(__name__)


def plot(config, output_file, data_path=None, model_reader=None, *, context=None):
    """
    Plot tabular data from data or from model parameter files.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.
    output_file: str
        Output file.
    data_path: Path or str, optional
        Path to the data files. All files are expected in the same directory.
    """
    context = context or RunContext.from_environment()
    data = read_table_data(config, data_path, model_reader=model_reader, context=context)
    return _plot_data(data, config, output_file, context=context)


def _plot_data(data, config, output_file, *, context=None):
    """Render pre-read table data with the configured table-plot renderer."""
    visualize.save_figure(
        visualize.plot_1d(data, **config), output_file, close=True, context=context
    )
    return output_file


def read_table_data(config, data_path=None, model_reader=None, *, context=None):
    """
    Read table data from file or parameter repository.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.
    data_path: Path or str, optional
        Path to the data files. All files are expected in the same directory.

    Returns
    -------
    Dict
        Dict with table data (numpy structured arrays).
    """
    context = context or RunContext.from_environment()
    model_reader = resolve_model_reader(model_reader, context)
    data = {}
    for table_config in config["tables"]:
        if "file_name" in table_config:
            table = _read_table_file(table_config, data_path)
        elif "parameter" in table_config:
            table = _read_table_from_model_repository(table_config, model_reader, context=context)
        else:
            raise ValueError("No table data defined in configuration.")
        data[_get_plotting_label(table_config, data)] = _process_table_data(table, table_config)
    return data


def _read_table_file(table_config, data_path):
    """Read one configured legacy or ECSV table file."""
    file_name = table_config["file_name"]
    if data_path is not None and not table_config.get("ignore_table_data_path", False):
        file_name = Path(data_path) / file_name
    _logger.info(f"Reading tabular data from {file_name}")
    if "legacy" in table_config.get("type", ""):
        return legacy_data_handler.read_legacy_data_as_table(file_name, table_config["type"])
    if Path(file_name).suffix.lower() != ".ecsv":
        raise ValueError(f"Model table files must use ECSV format: {file_name}")
    return Table.read(file_name, format="ascii.ecsv")


def _get_plotting_label(config, data):
    """Get a label for plotting based on the configuration, made unique within data."""
    label = config.get("label", f"{config.get('column_x')} vs {config.get('column_y')}")
    if label in data:
        index = 1
        while f"{label} ({index})" in data:
            index += 1
        label = f"{label} ({index})"
    return label


def _process_table_data(table, _config):
    """Normalise, select and convert table columns to a structured array."""
    if _config.get("normalize_y"):
        table[_config["column_y"]] = table[_config["column_y"]] / table[_config["column_y"]].max()
    if _config.get("select_values"):
        table = _select_values_from_table(
            table,
            _config["select_values"]["column_name"],
            _config["select_values"]["value"],
        )

    columns = [
        _config["column_x"],
        _config["column_y"],
        _config.get("column_x_err"),
        _config.get("column_y_err"),
    ]
    columns = [column for column in columns if column in table.colnames]
    return np.array(
        list(zip(*(np.array(table[column]) for column in columns))),
        dtype=[(column, np.array(table[column]).dtype) for column in columns],
    )


def _read_table_from_model_repository(table_config, model_reader=None, *, context=None):
    """Export a table from the simulation-model repository (returns an astropy Table)."""
    context = context or RunContext.from_environment()
    model_reader = resolve_model_reader(model_reader, context)
    if model_reader is None:
        raise RuntimeError("No simulation model reader is configured.")
    destination = table_config.get("model_export_path") or context.paths.get_output_directory()
    return model_reader.export_model_file(
        parameter=table_config["parameter"],
        site=table_config["site"],
        array_element_name=table_config.get("telescope"),
        parameter_version=table_config.get("parameter_version"),
        model_version=table_config.get("model_version"),
        export_file_as_table=True,
        dest=destination,
    )


def _read_parameter_dict_from_model_repository(table_config, model_reader=None, *, context=None):
    """Read a model parameter dictionary from the simulation-model repository."""
    context = context or RunContext.from_environment()
    model_reader = resolve_model_reader(model_reader, context)
    if model_reader is None:
        raise RuntimeError("No simulation model reader is configured.")
    parameter_dict = model_reader.get_model_parameter(
        parameter=table_config["parameter"],
        site=table_config["site"],
        array_element_name=table_config.get("telescope"),
        parameter_version=table_config.get("parameter_version"),
        model_version=table_config.get("model_version"),
    )
    return parameter_dict[table_config["parameter"]]


def _select_values_from_table(table, column_name, value):
    """Return a table with only the rows where column_name == value."""
    column = table[column_name]
    if getattr(column, "unit", None) is not None:
        if hasattr(value, "to_value"):
            value = value.to_value(column.unit)
        return table[np.isclose(column.value, value)]
    return table[np.isclose(column, value)]


def _validate_config_columns(config, valid_columns, logger):
    """Return True if the x and y columns of all tables of a config have valid data."""
    for table_config in config.get("tables", []):
        required_cols = [table_config.get("column_x"), table_config.get("column_y")]
        if not all(col in valid_columns for col in required_cols if col):
            missing_cols = [col for col in required_cols if col not in valid_columns]
            logger.info(
                f"Skipping plot config {config.get('type')}: "
                f"Missing valid data in columns: {missing_cols}"
            )
            return False
    return True


def _select_schema_entry(schema_data, schema_version=None):
    """Return the schema dict matching schema_version or the newest available one."""
    if isinstance(schema_data, dict):
        return schema_data

    if isinstance(schema_data, list) and schema_data:
        if schema_version is not None:
            for entry in schema_data:
                if entry.get("schema_version") == schema_version:
                    return entry

        return max(
            schema_data,
            key=lambda entry: packaging.version.Version(entry.get("schema_version", "0.0.0")),
        )

    return {}


def generate_plot_configurations(
    parameter,
    parameter_version,
    site,
    telescope,
    output_path,
    plot_type,
    model_reader=None,
    model_version=None,
    *,
    context=None,
):
    """
    Generate plot configurations for a model parameter from schema files.

    Parameters
    ----------
    parameter: str
        Model parameter name.
    parameter_version: str
        Parameter version.
    site: str
        Site name.
    telescope: str
        Telescope name.
    output_path: str or Path
        Output path for the plots.
    plot_type: str
        Plot type or "all" for all plots.
    model_version: str, optional
        Exact model version used to read the table data.

    Returns
    -------
    tuple or None
        List of plot configurations and list of output file names, or None if no
        plot configurations are found.
    """
    logger = logging.getLogger(__name__)
    table_config = {
        "parameter": parameter,
        "site": site,
        "telescope": telescope,
        "parameter_version": parameter_version,
    }
    if model_version:
        table_config["model_version"] = model_version
    parameter_dict = _read_parameter_dict_from_model_repository(
        table_config, model_reader, context=context
    )

    schema = value_conversion.change_dict_keys_case(
        _select_schema_entry(
            ascii_handler.collect_data_from_file(
                file_name=SCHEMA_PATH / "model_parameters" / f"{parameter}.schema.yml"
            ),
            parameter_dict.get("model_parameter_schema_version"),
        )
    )
    configs = schema.get("plot_configuration")
    if not configs:
        return None

    table = _read_table_from_model_repository(table_config, model_reader, context=context)
    valid_columns = [
        col
        for col in table.colnames
        if not np.issubdtype(table[col].dtype, np.number) or not np.all(np.isnan(table[col]))
    ]
    valid_configs = [
        config
        for config in configs
        if (plot_type == "all" or config.get("type") == plot_type)
        and _validate_config_columns(config, valid_columns, logger)
    ]
    if not valid_configs:
        if plot_type != "all":
            logger.warning("No valid plot config found.")
        return None

    output_files = []
    for _config in valid_configs:
        for _table in _config.get("tables", []):
            _table["parameter_version"] = parameter_version
            _table["site"] = site
            _table["telescope"] = telescope
            if model_version:
                _table["model_version"] = model_version

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

    return valid_configs, output_files


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
