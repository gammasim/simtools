#!/usr/bin/python3
"""Plot tabular data."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import packaging.version

import simtools.utils.general as gen
from simtools.constants import SCHEMA_PATH
from simtools.db import db_handler
from simtools.io import ascii_handler, legacy_data_handler
from simtools.simtel.simtel_table_reader import read_simtel_table
from simtools.visualization import visualize

_logger = logging.getLogger(__name__)


def plot(config, output_file, data_path=None):
    """
    Plot tabular data from data or from model parameter files.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.
    output_file: str
        Output file.
    data_path: Path or str, optional
        Path to the data files (optional). Expect all files to be in the same directory.
    """
    data = read_table_data(config, data_path)

    fig = visualize.plot_1d(
        data,
        **config,
    )
    visualize.save_figure(fig, output_file)

    return output_file


def read_table_data(config, data_path=None):
    """
    Read table data from file or parameter database.

    Parameters
    ----------
    config: dict
        Configuration dictionary for plotting.
    data_path: Path or str, optional
        Path to the data files (optional). Expect all files to be in the same directory.

    Returns
    -------
    Dict
        Dict with table data (astropy tables).
    """
    data = {}

    for _config in config["tables"]:
        if "file_name" in _config:
            file_name = (
                _config["file_name"]
                if data_path is None or _config.get("ignore_table_data_path", False)
                else Path(data_path) / _config["file_name"]
            )
            _logger.info(f"Reading tabular data from {file_name}")

            if "legacy" in _config.get("type", ""):
                table = legacy_data_handler.read_legacy_data_as_table(file_name, _config["type"])
            else:
                table = read_simtel_table(_config.get("parameter"), file_name)
        elif "parameter" in _config:
            table = _read_table_from_model_database(_config)
        else:
            raise ValueError("No table data defined in configuration.")

        data[_get_plotting_label(_config, data)] = _process_table_data(table, _config)

    return data


def _get_plotting_label(config, data):
    """Get a label for plotting based on the configuration."""
    label = config.get("label", f"{config.get('column_x')} vs {config.get('column_y')}")
    if label in data:
        index = 1
        while f"{label} ({index})" in data:
            index += 1
        label = f"{label} ({index})"
    return label


def _process_table_data(table, _config):
    """Process table data based on configuration."""
    if _config.get("normalize_y"):
        table[_config["column_y"]] = table[_config["column_y"]] / table[_config["column_y"]].max()
    if _config.get("select_values"):
        table = _select_values_from_table(
            table,
            _config["select_values"]["column_name"],
            _config["select_values"]["value"],
        )

    return gen.get_structure_array_from_table(
        table,
        [
            _config["column_x"],
            _config["column_y"],
            _config.get("column_x_err"),
            _config.get("column_y_err"),
        ],
    )


def _read_table_from_model_database(table_config):
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
    db = db_handler.DatabaseHandler()
    db_export_path = table_config.get("db_export_path")
    original_output_path = db.io_handler.output_path.get("default")

    if db_export_path:
        db.io_handler.set_paths(output_path=db_export_path)

    try:
        return db.export_model_file(
            parameter=table_config["parameter"],
            site=table_config["site"],
            array_element_name=table_config.get("telescope"),
            parameter_version=table_config.get("parameter_version"),
            model_version=table_config.get("model_version"),
            export_file_as_table=True,
        )
    finally:
        if db_export_path:
            db.io_handler.set_paths(output_path=original_output_path)


def _read_parameter_dict_from_model_database(table_config):
    """Read a model parameter dictionary from the model parameter database."""
    db = db_handler.DatabaseHandler()
    parameter_dict = db.get_model_parameter(
        parameter=table_config["parameter"],
        site=table_config["site"],
        array_element_name=table_config.get("telescope"),
        parameter_version=table_config.get("parameter_version"),
        model_version=table_config.get("model_version"),
    )
    return parameter_dict[table_config["parameter"]]


def _select_values_from_table(table, column_name, value):
    """Return a table with only the rows where column_name == value."""
    return table[np.isclose(table[column_name], value)]


def _filter_config_by_plot_type(config, plot_type):
    """Filter a configuration based on plot type."""
    if plot_type != "all" and config.get("type") != plot_type:
        return False
    return True


def _validate_config_columns(config, valid_columns, logger):
    """Validate that all required columns in a config exist and have valid data."""
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


def _get_valid_columns(table):
    """Return columns that exist and have valid data (not all NaN)."""
    return [col for col in table.colnames if not all(np.isnan(table[col]))]


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
    parameter, parameter_version, site, telescope, output_path, plot_type
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

    Returns
    -------
    tuple
        Tuple containing a list of plot configurations and a list of output file names.
        Return None, if no plot configurations are found.
    """
    logger = logging.getLogger(__name__)
    table_config = {
        "parameter": parameter,
        "site": site,
        "telescope": telescope,
        "parameter_version": parameter_version,
    }
    parameter_dict = _read_parameter_dict_from_model_database(table_config)

    schema = gen.change_dict_keys_case(
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

    table = _read_table_from_model_database(table_config)
    valid_columns = _get_valid_columns(table)

    valid_configs = []
    for config in configs:
        if not _filter_config_by_plot_type(config, plot_type):
            continue

        if _validate_config_columns(config, valid_columns, logger):
            valid_configs.append(config)

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


def resolve_plot_output_path(output, file_name="bias_curve.png"):
    """Resolve output as either an explicit plot file or an output directory."""
    output_path = Path(output)

    if output_path.suffix:
        return output_path

    return output_path / file_name


def plot_bias_curves(nsb_stats, proton_stats, config, output_path):
    """
    Plot NSB and proton bias curves.

    Parameters
    ----------
    nsb_stats : dict
        NSB statistics by threshold.
    proton_stats : dict
        Proton statistics by threshold.
    config : dict
        Plot configuration with title, ymin, and ymax.
    output_path : Path or str
        Output path for plot image.
    """
    fig, axis = plt.subplots(figsize=(10, 7))

    _plot_nsb_curve(axis, nsb_stats)
    _plot_proton_curve(axis, proton_stats)
    _configure_bias_curve_axis(axis, config)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_nsb_curve(axis, nsb_stats):
    """Plot NSB trigger rates."""
    if not nsb_stats:
        return

    nsb_thresholds = sorted(nsb_stats.keys())
    nsb_rates = [nsb_stats[t]["rate_hz"] for t in nsb_thresholds]
    nsb_errors = [nsb_stats[t]["error_hz"] for t in nsb_thresholds]

    axis.errorbar(
        nsb_thresholds,
        nsb_rates,
        yerr=nsb_errors,
        fmt="o",
        label="NSB",
        color="tab:blue",
        capsize=3,
    )

    _plot_log_linear_trend(axis, nsb_thresholds, nsb_rates, color="tab:blue")


def _plot_proton_curve(axis, proton_stats):
    """Plot proton trigger rates."""
    if not proton_stats:
        return

    proton_thresholds = sorted(proton_stats.keys())
    proton_rates = [proton_stats[t]["rate_hz"] for t in proton_thresholds]

    axis.plot(
        proton_thresholds,
        proton_rates,
        "s",
        label="Proton",
        color="tab:orange",
        markersize=8,
    )

    _plot_log_linear_trend(axis, proton_thresholds, proton_rates, color="tab:orange")


def _plot_log_linear_trend(axis, thresholds, rates, color):
    """Plot a log-linear trend line when at least two positive rates are available."""
    if len(thresholds) < 2:
        return

    valid_mask = np.array(rates) > 0
    if np.sum(valid_mask) < 2:
        return

    fit_thresholds = np.array(thresholds)[valid_mask]
    fit_rates = np.array(rates)[valid_mask]

    coeffs = np.polyfit(fit_thresholds, np.log10(fit_rates), 1)
    x_fit = np.linspace(min(fit_thresholds), max(fit_thresholds), 100)
    y_fit = 10 ** (coeffs[0] * x_fit + coeffs[1])

    axis.plot(x_fit, y_fit, "--", color=color, alpha=0.5, linewidth=1)


def _configure_bias_curve_axis(axis, config):
    """Configure bias-curve axis labels, scaling, and legend."""
    axis.set_title(config["title"], fontsize=14, fontweight="bold")
    axis.set_xlabel("Threshold", fontsize=12)
    axis.set_ylabel("Trigger Rate [Hz]", fontsize=12)
    axis.set_yscale("log")
    axis.set_ylim(config["ymin"], config["ymax"])
    axis.grid(which="both", alpha=0.3, linestyle=":")

    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=11, loc="best")
    else:
        _logger.warning("No NSB or proton rates found; writing empty bias-curve plot")
