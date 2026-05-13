"""Derive CORSIKA limits from a reduced event data file."""

import datetime
import logging
import re
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Column, Table

from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler, io_handler
from simtools.job_execution.process_pool import process_pool_map_ordered
from simtools.layout.array_layout_utils import get_array_elements_from_db_for_layouts
from simtools.sim_events.histograms import EventDataHistograms
from simtools.utils.general import get_uuid
from simtools.utils.names import normalize_array_element_identifier_container
from simtools.visualization import plot_simtel_event_histograms

_logger = logging.getLogger(__name__)

FILE_INFO_KEYS = ("primary_particle", "zenith", "azimuth", "nsb_level")
RESULT_COLUMNS = [
    "production_index",
    "event_data_file",
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


def _normalize_event_data_file(event_data_file):
    """
    Normalize event_data_file to an ordered list of production patterns.

    Parameters
    ----------
    event_data_file : str or list
        A single pattern string or list of pattern strings.

    Returns
    -------
    list
        Ordered list of event data file patterns.
    """
    if isinstance(event_data_file, str):
        return [event_data_file]
    if isinstance(event_data_file, list):
        return list(event_data_file)
    raise TypeError(f"event_data_file must be str or list, got {type(event_data_file)}")


def _get_production_directory_name(production_pattern, existing_names=None):
    """
    Generate a readable, filesystem-safe production directory name.

    The name is derived from the glob pattern and kept human-readable.
    If a collision is detected with an existing directory name, append
    a UUID7 suffix.

    Parameters
    ----------
    production_pattern : str
        The glob pattern for this production.
    existing_names : set[str] or None
        Existing directory names used in this run.

    Returns
    -------
    str
        Safe directory name (e.g., "production_prod_a_events").
    """
    pattern_path = Path(production_pattern)
    parts = []

    if pattern_path.parent.name and pattern_path.parent.name != ".":
        parts.append(pattern_path.parent.name)
    if pattern_path.stem:
        parts.append(pattern_path.stem)

    readable_name = "_".join(parts) if parts else "production"
    readable_name = re.sub(r"[^A-Za-z0-9]+", "_", readable_name)
    readable_name = readable_name.strip("_")
    readable_name = re.sub(r"_+", "_", readable_name)

    if not readable_name:
        readable_name = "production"

    base_name = f"production_{readable_name}"

    if existing_names is None or base_name not in existing_names:
        return base_name

    return f"{base_name}_{get_uuid()}"


def _execute_production_job(job_spec):
    """
    Execute a single production job (top-level picklable worker).

    This function is called by process_pool_map_ordered and must be
    picklable (top-level defined).

    Parameters
    ----------
    job_spec : dict
        Dictionary containing with job specifications

    Returns
    -------
    dict
        Result dictionary with limits and production metadata.
    """
    production_index = job_spec["production_index"]
    production_pattern = job_spec["production_pattern"]
    array_name = job_spec["array_name"]
    telescope_ids = job_spec["telescope_ids"]
    loss_fraction = job_spec["loss_fraction"]
    plot_histograms = job_spec["plot_histograms"]
    output_subdir = job_spec.get("output_subdir")
    differential_loss_bins_per_decade = job_spec.get("differential_loss_bins_per_decade", 0)

    _logger.info(
        f"Processing production {production_index}: pattern={production_pattern}, "
        f"array={array_name}"
    )

    result = _process_file(
        production_pattern,
        array_name,
        telescope_ids,
        loss_fraction,
        plot_histograms,
        output_subdir=output_subdir,
        differential_loss_bins_per_decade=differential_loss_bins_per_decade,
    )

    result.update(
        {
            "production_index": production_index,
            "event_data_file": production_pattern,
            "array_name": array_name,
            "telescope_ids": telescope_ids,
        }
    )

    return result


def _resolve_telescope_configs(args_dict):
    """Resolve telescope configurations from one of the supported input options."""
    if args_dict.get("array_layout_name"):
        return get_array_elements_from_db_for_layouts(
            args_dict["array_layout_name"],
            args_dict.get("site"),
            args_dict.get("model_version"),
        )
    if args_dict.get("array_element_list"):
        return {"array_element_list": args_dict["array_element_list"]}
    if args_dict.get("telescope_ids"):
        return ascii_handler.collect_data_from_file(args_dict["telescope_ids"])["telescope_configs"]

    raise ValueError(
        "No telescope configuration provided. Use one of --array_layout_name, "
        "--array_element_list, or --telescope_ids."
    )


def _build_production_subdirectories(production_patterns, output_dir, is_multi_production):
    """Build and create per-production output subdirectories when needed."""
    if not is_multi_production:
        return {}

    production_subdirs = {}
    used_subdir_names = set()
    for production_pattern in production_patterns:
        subdir_name = _get_production_directory_name(production_pattern, used_subdir_names)
        used_subdir_names.add(subdir_name)
        output_subdir = output_dir / subdir_name
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        production_subdirs[production_pattern] = output_subdir

    return production_subdirs


def generate_corsika_limits_grid(args_dict):
    """
    Generate CORSIKA limits for one or more production patterns.

    Single- and multi-production runs share the same dispatch path using
    process_pool_map_ordered.

    Parameters
    ----------
    args_dict : dict
        Dictionary containing command line arguments.
    """
    production_patterns = _normalize_event_data_file(args_dict["event_data_file"])
    differential_loss_bins_per_decade = int(args_dict.get("differential_loss_bins_per_decade", 0))
    n_productions = len(production_patterns)
    is_multi_production = n_productions > 1

    _logger.info(f"Processing {n_productions} production(s)")

    telescope_configs = _resolve_telescope_configs(args_dict)

    # Build deterministic job specs: Cartesian product of productions and telescope configs
    job_specs = []
    output_dir = io_handler.IOHandler().get_output_directory()

    production_subdirs = {}
    if is_multi_production and args_dict["plot_histograms"]:
        production_subdirs = _build_production_subdirectories(
            production_patterns,
            output_dir,
            is_multi_production,
        )

    for prod_idx, production_pattern in enumerate(production_patterns):
        for array_name, telescope_ids_raw in telescope_configs.items():
            telescope_ids = normalize_array_element_identifier_container(telescope_ids_raw)

            output_subdir = production_subdirs.get(production_pattern)

            job_spec = {
                "production_index": prod_idx,
                "production_pattern": production_pattern,
                "array_name": array_name,
                "telescope_ids": telescope_ids,
                "loss_fraction": args_dict["loss_fraction"],
                "plot_histograms": args_dict["plot_histograms"],
                "output_subdir": output_subdir,
                "differential_loss_bins_per_decade": differential_loss_bins_per_decade,
            }
            job_specs.append(job_spec)

    n_workers = int(args_dict.get("n_workers", 1))
    _logger.info(f"Executing {len(job_specs)} jobs with {n_workers or 'auto'} workers")
    results = process_pool_map_ordered(
        _execute_production_job,
        job_specs,
        max_workers=n_workers,
    )

    write_results(results, args_dict)


def _process_file(
    file_path,
    array_name,
    telescope_ids,
    loss_fraction,
    plot_histograms,
    output_subdir=None,
    differential_loss_bins_per_decade=0,
):
    """
    Compute limits for a given event data file and telescope configuration.

    Compute limits for energy, radial distance, and viewcone.

    Parameters
    ----------
    file_path : str or list
        Path or glob pattern to the event data file, or a list of files.
    array_name : str
        Name of the telescope array configuration.
    telescope_ids : list[str]
        List of telescope IDs (array-element names) to filter the events.
    loss_fraction : float
        Fraction of events to be lost.
    plot_histograms : bool
        Whether to plot histograms.
    output_subdir : Path or None, optional
        Output subdirectory for plots. If None, uses default output directory.
    differential_loss_bins_per_decade : int, optional
        Number of energy bins per decade for differential per-bin limits.
        Set to 0 (default) to use integrated limits.

    Returns
    -------
    dict
        Dictionary containing the computed limits and metadata.
    """
    histograms = EventDataHistograms(
        file_path,
        array_name=array_name,
        telescope_list=telescope_ids,
        energy_bins_per_decade=differential_loss_bins_per_decade or 10,
    )
    histograms.fill()

    limits = {
        "lower_energy_limit": compute_lower_energy_limit(histograms, loss_fraction),
    }
    if differential_loss_bins_per_decade > 0:
        limits.update(
            _compute_differential_limits(
                histograms, loss_fraction, differential_loss_bins_per_decade
            )
        )
    else:
        limits["upper_radius_limit"] = compute_upper_radius_limit(histograms, loss_fraction)
        limits["viewcone_radius"] = compute_viewcone(histograms, loss_fraction)

    limits.update({key: histograms.file_info.get(key) for key in FILE_INFO_KEYS})

    if plot_histograms:
        plot_output_path = output_subdir or io_handler.IOHandler().get_output_directory()
        plot_simtel_event_histograms.plot(
            histograms.histograms,
            output_path=plot_output_path,
            limits=limits,
            array_name=array_name,
        )

    return limits


def _compute_differential_limits(histograms, loss_fraction, bins_per_decade):
    """Compute core and viewcone limits per energy bin and return max limits."""
    low = int(np.floor(np.log10(np.min(histograms.energy_bins))))
    high = int(np.ceil(np.log10(np.max(histograms.energy_bins))))
    diff_e_bins = np.logspace(low, high, (high - low) * bins_per_decade + 1)

    core_max, core_x, core_y = _differential_upper_limits(
        histograms.histograms["core_vs_energy"]["histogram"],
        histograms.core_distance_bins,
        histograms.energy_bins,
        diff_e_bins,
        loss_fraction,
        "core_scatter",
        "m",
    )
    vc_max, vc_x, vc_y = _differential_upper_limits(
        histograms.histograms["angular_distance_vs_energy"]["histogram"],
        histograms.view_cone_bins,
        histograms.energy_bins,
        diff_e_bins,
        loss_fraction,
        "viewcone",
        "deg",
    )

    upper_radius_limit = _is_close(
        core_max * u.m,
        histograms.file_info["core_scatter_max"].to("m")
        if "core_scatter_max" in histograms.file_info
        else None,
        "Upper radius limit is equal to the maximum core scatter distance of",
    )
    viewcone_radius = _is_close(
        vc_max * u.deg,
        histograms.file_info["viewcone_max"].to("deg")
        if "viewcone_max" in histograms.file_info
        else None,
        "Upper viewcone limit is equal to the maximum viewcone distance of",
    )
    _logger.info(f"Differential upper_radius_limit (max over bins): {upper_radius_limit}")
    _logger.info(f"Differential viewcone_radius (max over bins): {viewcone_radius}")
    return {
        "upper_radius_limit": upper_radius_limit,
        "viewcone_radius": viewcone_radius,
        "core_vs_energy_curve": {"x": core_x, "y": core_y},
        "angular_distance_vs_energy_curve": {"x": vc_x, "y": vc_y},
    }


def _differential_upper_limits(
    histogram2d,
    x_bins,
    y_bins,
    diff_e_bins,
    loss_fraction,
    name,
    unit,
):
    """Compute upper limits per energy slice of a 2D (x, energy) histogram."""
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    limits, energy_centers = [], []
    n = len(diff_e_bins) - 1
    for i in range(n):
        e_low, e_high = diff_e_bins[i], diff_e_bins[i + 1]
        hi_op = np.less_equal if i == n - 1 else np.less
        projected = np.sum(histogram2d[:, (y_centers >= e_low) & hi_op(y_centers, e_high)], axis=1)
        total = float(np.sum(projected))
        if total <= 0:
            continue
        limit = _compute_limits(projected, x_bins, loss_fraction, limit_type="upper")
        keep = np.searchsorted(x_bins, limit, side="left")
        loss = (total - float(np.sum(projected[:keep]))) / total
        limits.append(limit)
        energy_centers.append(float(np.sqrt(e_low * e_high)))
        _logger.info(
            f"Differential {name}: E=[{e_low:.4g}, {e_high:.4g}] TeV, "
            f"N={int(total)}, limit={limit:.4g} {unit}, loss={loss:.5f}"
        )
    return (
        (float(np.max(limits)), limits, energy_centers) if limits else (float(x_bins[-1]), [], [])
    )


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
        Table with computed limits and production-origin columns.
    """
    cols = list(RESULT_COLUMNS)

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
    histograms : EventDataHistograms
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
    histograms : EventDataHistograms
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
    histograms : EventDataHistograms
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
