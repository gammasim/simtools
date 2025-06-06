"""Derive CORSIKA limits for a grid of parameters."""

import datetime
import logging

import numpy as np
from astropy.table import Column, Table

import simtools.utils.general as gen
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler
from simtools.production_configuration.derive_corsika_limits import LimitCalculator

_logger = logging.getLogger(__name__)


def generate_corsika_limits_grid(args_dict):
    """
    Generate CORSIKA limits for a grid of parameters.

    Requires at least one event data file per parameter set.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
    """
    event_data_files = gen.collect_data_from_file(args_dict["event_data_files"])["files"]
    telescope_configs = gen.collect_data_from_file(args_dict["telescope_ids"])["telescope_configs"]

    results = []
    for file_path in event_data_files:
        for array_name, telescope_ids in telescope_configs.items():
            _logger.info(f"Processing file: {file_path} with telescope config: {array_name}")
            result = _process_file(
                file_path,
                array_name,
                telescope_ids,
                args_dict["loss_fraction"],
                args_dict["plot_histograms"],
            )
            result["layout"] = array_name
            results.append(result)

    write_results(results, args_dict)


def _process_file(file_path, array_name, telescope_ids, loss_fraction, plot_histograms):
    """
    Compute limits for a single file.

    Parameters
    ----------
    file_path : str
        Path to the event data file.
    array_name : str
        Name of the telescope array configuration.
    telescope_ids : list[int]
        List of telescope IDs to filter the events.
    loss_fraction : float
        Fraction of events to be lost.
    plot_histograms : bool
        Whether to plot histograms.

    Returns
    -------
    dict
        Dictionary containing the computed limits and metadata.
    """
    calculator = LimitCalculator(file_path, array_name=array_name, telescope_list=telescope_ids)
    limits = calculator.compute_limits(loss_fraction)

    if plot_histograms:
        calculator.plot_data(io_handler.IOHandler().get_output_directory())

    return limits


def write_results(results, args_dict):
    """
    Write the computed limits as astropy table to file.

    Parameters
    ----------
    results : list[dict]
        List of computed limits.
    args_dict : dict
        Dictionary containing command line arguments.
    """
    table = _create_results_table(results, args_dict["loss_fraction"])

    output_dir = io_handler.IOHandler().get_output_directory("corsika_limits")
    output_file = output_dir / args_dict["output_file"]

    table.write(output_file, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Results saved to {output_file}")

    MetadataCollector.dump(args_dict, output_file)


def _create_results_table(results, loss_fraction):
    """
    Convert list of simulation results to an astropy Table with metadata.

    Round values to appropriate precision and add metadata.

    Parameters
    ----------
    results : list[dict]
        Computed limits per file and telescope configuration.
    loss_fraction : float
        Fraction of lost events (added to metadata).

    Returns
    -------
    astropy.table.Table
        Table with computed limits.
    """
    cols = [
        "primary_particle",
        "array_name",
        "telescope_ids",
        "zenith",
        "azimuth",
        "nsb_level",
        "lower_energy_limit",
        "upper_radius_limit",
        "viewcone_radius",
    ]

    columns = {name: [] for name in cols}
    units = {}

    for res in results:
        _process_result_row(res, cols, columns, units)

    table_cols = _create_table_columns(cols, columns, units)
    table = Table(table_cols)

    table.meta.update(
        {
            "created": datetime.datetime.now().isoformat(),
            "description": "Lookup table for CORSIKA limits computed from simulations.",
            "loss_fraction": loss_fraction,
        }
    )

    return table


def _process_result_row(res, cols, columns, units):
    """Process a single result row and add values to columns."""
    for k in cols:
        val = res.get(k, None)
        if val is not None:
            val = _round_value(k, val)
            _logger.debug(f"Adding {k}: {val} to column data")

        if hasattr(val, "unit"):
            columns[k].append(val.value)
            units[k] = val.unit
        else:
            columns[k].append(val)
            if k not in units:
                units[k] = None


def _round_value(key, val):
    """Round value based on key type."""
    if key == "lower_energy_limit":
        return np.floor(val * 1e3) / 1e3
    if key == "upper_radius_limit":
        return np.ceil(val / 25) * 25
    if key == "viewcone_radius":
        return np.ceil(val / 0.25) * 0.25
    return val


def _create_table_columns(cols, columns, units):
    """Create table columns with appropriate data types."""
    table_cols = []
    for k in cols:
        col_data = columns[k]
        if any(isinstance(v, list | tuple) for v in col_data):
            col = Column(data=col_data, name=k, unit=units.get(k), dtype=object)
        else:
            col = Column(data=col_data, name=k, unit=units.get(k))
        table_cols.append(col)
    return table_cols
