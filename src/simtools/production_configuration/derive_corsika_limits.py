"""Derive CORSIKA limits from a reduced event data file."""

import datetime
import logging

import astropy.units as u
import numpy as np
from astropy.table import Column, Table

from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler, io_handler
from simtools.layout.array_layout_utils import get_array_elements_from_db_for_layouts
from simtools.simtel.simtel_io_event_histograms import SimtelIOEventHistograms
from simtools.visualization import plot_simtel_event_histograms

_logger = logging.getLogger(__name__)


def generate_corsika_limits_grid(args_dict):
    """
    Generate CORSIKA limits.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
    """
    if args_dict.get("array_layout_name"):
        telescope_configs = get_array_elements_from_db_for_layouts(
            args_dict["array_layout_name"],
            args_dict.get("site"),
            args_dict.get("model_version"),
        )
    else:
        telescope_configs = ascii_handler.collect_data_from_file(args_dict["telescope_ids"])[
            "telescope_configs"
        ]

    results = []
    for array_name, telescope_ids in telescope_configs.items():
        _logger.info(
            f"Processing file: {args_dict['event_data_file']} with telescope config: {array_name}"
        )
        result = _process_file(
            args_dict["event_data_file"],
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
    Compute limits for a given event data file and telescope configuration.

    Compute limits for energy, radial distance, and viewcone.

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
    histograms = SimtelIOEventHistograms(
        file_path, array_name=array_name, telescope_list=telescope_ids
    )
    histograms.fill()

    limits = {
        "lower_energy_limit": compute_lower_energy_limit(histograms, loss_fraction),
        "upper_radius_limit": compute_upper_radius_limit(histograms, loss_fraction),
        "viewcone_radius": compute_viewcone(histograms, loss_fraction),
    }

    if plot_histograms:
        plot_simtel_event_histograms.plot(
            histograms.histograms,
            output_path=io_handler.IOHandler().get_output_directory(),
            limits=limits,
            array_name=array_name,
        )

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

    output_dir = io_handler.IOHandler().get_output_directory()
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


def _compute_limits(hist, bin_edges, loss_fraction, limit_type="lower"):
    """
    Compute the limits based on the loss fraction.

    Add or subtract one bin to be on the safe side of the limit.

    Parameters
    ----------
    hist : np.ndarray
        1D histogram array.
    bin_edges : np.ndarray
        Array of bin edges.
    loss_fraction : float
        Fraction of events to be lost.
    limit_type : str, optional
        Type of limit ('lower' or 'upper'). Default is 'lower'.

    Returns
    -------
    float
        Bin edge value corresponding to the threshold.
    """
    total_events = np.sum(hist)
    threshold = (1 - loss_fraction) * total_events
    if limit_type == "upper":
        cum = np.cumsum(hist)
        idx = np.searchsorted(cum, threshold) + 1
        return bin_edges[min(idx, len(bin_edges) - 1)]
    if limit_type == "lower":
        cum = np.cumsum(hist[::-1])
        idx = np.searchsorted(cum, threshold) + 1
        return bin_edges[max(len(bin_edges) - 1 - idx, 0)]
    raise ValueError("limit_type must be 'lower' or 'upper'")


def compute_lower_energy_limit(histograms, loss_fraction):
    """
    Compute the lower energy limit in TeV based on the event loss fraction.

    Parameters
    ----------
    histograms : SimtelIOEventHistograms
        Histograms.
    loss_fraction : float
        Fraction of events to be lost.

    Returns
    -------
    astropy.units.Quantity
        Lower energy limit.
    """
    energy_min = (
        _compute_limits(
            histograms.histograms["energy"]["histogram"],
            histograms.energy_bins,
            loss_fraction,
            limit_type="lower",
        )
        * u.TeV
    )

    return _is_close(
        energy_min,
        histograms.file_info["energy_min"].to("TeV")
        if "energy_min" in histograms.file_info
        else None,
        "Lower energy limit is equal to the minimum energy of",
    )


def _is_close(value, reference, warning_text):
    """Check if the value is close to the reference value and log a warning if so."""
    if reference is not None and np.isclose(value.value, reference.value, rtol=1.0e-2):
        _logger.warning(f"{warning_text} {value}.")
    return value


def compute_upper_radius_limit(histograms, loss_fraction):
    """
    Compute the upper radial distance based on the event loss fraction.

    Parameters
    ----------
    histograms : SimtelIOEventHistograms
        Histograms.
    loss_fraction : float
        Fraction of events to be lost.

    Returns
    -------
    astropy.units.Quantity
        Upper radial distance in m.
    """
    radius_limit = (
        _compute_limits(
            histograms.histograms["core_distance"]["histogram"],
            histograms.core_distance_bins,
            loss_fraction,
            limit_type="upper",
        )
        * u.m
    )
    return _is_close(
        radius_limit,
        histograms.file_info["core_scatter_max"].to("m")
        if "core_scatter_max" in histograms.file_info
        else None,
        "Upper radius limit is equal to the maximum core scatter distance of",
    )


def compute_viewcone(histograms, loss_fraction):
    """
    Compute the viewcone based on the event loss fraction.

    The shower IDs of triggered events are used to create a mask for the
    azimuth and altitude of the triggered events. A mapping is created
    between the triggered events and the simulated events using the shower IDs.

    Parameters
    ----------
    histograms : SimtelIOEventHistograms
        Histograms.
    loss_fraction : float
        Fraction of events to be lost.

    Returns
    -------
    astropy.units.Quantity
        Viewcone radius in degrees.
    """
    viewcone_limit = (
        _compute_limits(
            histograms.histograms["angular_distance"]["histogram"],
            histograms.view_cone_bins,
            loss_fraction,
            limit_type="upper",
        )
        * u.deg
    )
    return _is_close(
        viewcone_limit,
        histograms.file_info["viewcone_max"].to("deg")
        if "viewcone_max" in histograms.file_info
        else None,
        "Upper viewcone limit is equal to the maximum viewcone distance of",
    )
