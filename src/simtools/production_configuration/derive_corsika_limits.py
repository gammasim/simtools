"""Derive CORSIKA limits from a reduced event data file."""

import datetime
import logging

import astropy.units as u
import numpy as np
from astropy.table import Column, Table

from simtools.constants import SCHEMA_PATH
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler, io_handler
from simtools.job_execution.process_pool import process_pool_map_ordered
from simtools.production_configuration.production_event_data_helpers import (
    accumulate_histograms_by_telescope_config,
    build_production_subdirectories,
    normalize_event_data_file,
    normalize_telescope_configs,
    resolve_telescope_configs,
)
from simtools.visualization import plot_simtel_event_histograms

_logger = logging.getLogger(__name__)

CORSIKA_LIMITS_TABLE_SCHEMA_FILE = SCHEMA_PATH / "corsika_limits_table.schema.yml"


def _load_output_table_configuration_from_schema(schema_file):
    """Load output table columns, descriptions, and file-info mappings from schema."""
    schema_data = ascii_handler.collect_data_from_file(file_name=schema_file)
    data_entries = schema_data.get("data", [])
    if not data_entries:
        raise KeyError(f"No 'data' entry found in schema {schema_file}")

    table_columns = data_entries[0].get("table_columns", [])
    if not table_columns:
        raise KeyError(f"No 'table_columns' entry found in schema {schema_file}")

    result_columns = [entry["name"] for entry in table_columns]
    column_descriptions = {
        entry["name"]: entry.get("description")
        for entry in table_columns
        if entry.get("description") is not None
    }
    file_info_columns = {
        entry["name"]: entry["file_info_key"]
        for entry in table_columns
        if entry.get("file_info_key") is not None
    }
    return result_columns, column_descriptions, file_info_columns


RESULT_COLUMNS, COLUMN_DESCRIPTIONS, FILE_INFO_COLUMNS = (
    _load_output_table_configuration_from_schema(CORSIKA_LIMITS_TABLE_SCHEMA_FILE)
)
LOSS_AXES = ("core_distance", "angular_distance")


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
    list[dict]
        Result dictionaries with limits and production metadata.
    """
    production_index = job_spec["production_index"]
    production_pattern = job_spec["production_pattern"]
    allowed_losses = job_spec["allowed_losses"]
    energy_threshold_fraction = job_spec["energy_threshold_fraction"]
    plot_histograms = job_spec["plot_histograms"]
    output_subdir = job_spec.get("output_subdir")
    differential_loss_bins_per_decade = job_spec.get("differential_loss_bins_per_decade", 0)
    skip_invalid_event_data_files = job_spec.get("skip_invalid_event_data_files", False)

    telescope_configs = job_spec["telescope_configs"]

    _logger.info(
        f"Processing production {production_index}: pattern={production_pattern}, "
        f"arrays={len(telescope_configs)}"
    )

    results = _process_production(
        production_pattern,
        telescope_configs,
        allowed_losses,
        energy_threshold_fraction,
        plot_histograms,
        output_subdir=output_subdir,
        differential_loss_bins_per_decade=differential_loss_bins_per_decade,
        skip_invalid_event_data_files=skip_invalid_event_data_files,
    )

    for result, telescope_config in zip(results, telescope_configs):
        result.update(
            {
                "production_index": production_index,
                "event_data_file": production_pattern,
                "array_name": telescope_config["array_name"],
                "telescope_ids": telescope_config["telescope_ids"],
            }
        )

    return results


def _parse_allowed_losses(allowed_losses_args):
    """
    Parse repeatable --allowed_losses values for core/viewcone axes.

    Parameters
    ----------
    allowed_losses_args : list[str]
        List of values in the form "axis,fraction,min_events".

    Returns
    -------
    dict
        Mapping of axis name to dict with keys "loss_fraction" and "loss_min_events".
    """
    if not allowed_losses_args:
        raise ValueError(
            "No allowed-loss configuration provided. Use --allowed_losses axis,fraction,min_events"
        )

    parsed = {}
    for raw_value in allowed_losses_args:
        parts = [part.strip() for part in raw_value.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --allowed_losses value '{raw_value}'. "
                "Expected format: axis,fraction,min_events"
            )

        axis_raw, fraction_raw, min_events_raw = parts
        try:
            fraction = float(fraction_raw)
            min_events = int(min_events_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid --allowed_losses value '{raw_value}': "
                "fraction must be float and min_events must be int"
            ) from exc

        axis_name = axis_raw.strip().lower()
        if axis_name == "all":
            for axis_name in LOSS_AXES:
                parsed[axis_name] = {
                    "loss_fraction": fraction,
                    "loss_min_events": min_events,
                }
            continue

        if axis_name not in LOSS_AXES:
            raise ValueError(
                f"Invalid axis for --allowed_losses. Allowed axes: {', '.join(LOSS_AXES)}, all."
            )
        parsed[axis_name] = {
            "loss_fraction": fraction,
            "loss_min_events": min_events,
        }

    missing_axes = [axis_name for axis_name in LOSS_AXES if axis_name not in parsed]
    if missing_axes:
        raise ValueError(
            f"Missing --allowed_losses entries for axis/axes: {', '.join(missing_axes)}"
        )

    return parsed


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
    production_patterns = normalize_event_data_file(args_dict["event_data_file"])
    allowed_losses = _parse_allowed_losses(args_dict.get("allowed_losses"))
    energy_threshold_fraction = float(args_dict.get("energy_threshold_fraction", 0.01))
    differential_loss_bins_per_decade = int(args_dict.get("differential_loss_bins_per_decade", 0))
    n_productions = len(production_patterns)
    is_multi_production = n_productions > 1

    _logger.info(f"Processing {n_productions} production(s)")

    telescope_configs = resolve_telescope_configs(args_dict)

    # Build deterministic production jobs. Each worker reads a production once and
    # accumulates the same complete histogram set for every telescope configuration.
    job_specs = []
    output_dir = io_handler.IOHandler().get_output_directory()

    production_subdirs = {}
    if is_multi_production and args_dict["plot_histograms"]:
        production_subdirs = build_production_subdirectories(
            production_patterns,
            output_dir,
            is_multi_production,
        )

    for prod_idx, production_pattern in enumerate(production_patterns):
        normalized_telescope_configs = normalize_telescope_configs(telescope_configs)
        job_specs.append(
            {
                "production_index": prod_idx,
                "production_pattern": production_pattern,
                "telescope_configs": normalized_telescope_configs,
                "allowed_losses": allowed_losses,
                "energy_threshold_fraction": energy_threshold_fraction,
                "plot_histograms": args_dict["plot_histograms"],
                "output_subdir": production_subdirs.get(production_pattern),
                "differential_loss_bins_per_decade": differential_loss_bins_per_decade,
                "skip_invalid_event_data_files": args_dict.get(
                    "skip_invalid_event_data_files", False
                ),
            }
        )

    max_workers = int(args_dict.get("max_workers", 1))
    _logger.info(f"Executing {len(job_specs)} jobs with {max_workers or 'auto'} workers")
    production_results = process_pool_map_ordered(
        _execute_production_job,
        job_specs,
        max_workers=max_workers,
    )
    results = [result for production_result in production_results for result in production_result]

    write_results(results, args_dict, allowed_losses, energy_threshold_fraction)


def _process_production(
    file_path,
    telescope_configs,
    allowed_losses,
    energy_threshold_fraction=0.01,
    plot_histograms=False,
    output_subdir=None,
    differential_loss_bins_per_decade=0,
    skip_invalid_event_data_files=False,
):
    """Read one production once and derive limits for all telescope configurations."""
    energy_bins_per_decade = differential_loss_bins_per_decade or 10
    finalized_histograms = accumulate_histograms_by_telescope_config(
        file_path,
        telescope_configs,
        energy_bins_per_decade=energy_bins_per_decade,
        skip_invalid_event_data_files=skip_invalid_event_data_files,
        fill_efficiency_histogram=False,
    )

    results = []
    for config, histograms in finalized_histograms:
        results.append(
            _derive_limits_from_histograms(
                histograms,
                config["array_name"],
                allowed_losses,
                energy_threshold_fraction,
                plot_histograms,
                output_subdir,
                differential_loss_bins_per_decade,
            )
        )
    return results


def _derive_limits_from_histograms(
    histograms,
    array_name,
    allowed_losses,
    energy_threshold_fraction,
    plot_histograms,
    output_subdir,
    differential_loss_bins_per_decade,
):
    """Compute, optionally plot, and return limits from finalized histograms."""
    limits = {
        "lower_energy_limit": compute_lower_energy_limit(
            histograms,
            energy_threshold_fraction,
        ),
    }
    limits.update(
        _compute_limits(
            histograms,
            allowed_losses,
            differential_loss_bins_per_decade,
        )
    )
    limits.update(
        {
            column_name: histograms.file_info.get(file_info_key)
            for column_name, file_info_key in FILE_INFO_COLUMNS.items()
        }
    )

    if plot_histograms:
        plot_output_path = output_subdir or io_handler.IOHandler().get_output_directory()
        histograms_to_plot = {
            name: histogram for name, histogram in histograms.histograms.items() if name != "energy"
        }
        if limits.get("angular_distance_is_constant", False):
            histograms_to_plot = {
                name: histogram
                for name, histogram in histograms_to_plot.items()
                if not name.startswith("angular_distance_vs_")
            }
        plot_simtel_event_histograms.plot(
            histograms_to_plot,
            output_path=plot_output_path,
            limits=limits,
            array_name=array_name,
            add_distance_projections=True,
            use_broad_range_limits=True,
        )

    return limits


def _compute_limits(histograms, allowed_losses, bins_per_decade):
    """
    Compute core and viewcone limits per energy bin and return max limits.

    Apply the allowed loss criteria per energy bin and return the maximum limit
    """
    energy_range = [float(histograms.energy_bins[0]), float(histograms.energy_bins[-1])]
    axis_configs = {
        "core_distance": {
            "x_bins": histograms.core_distance_bins,
            "name": "core_scatter",
            "unit": "m",
        },
        "angular_distance": {
            "x_bins": histograms.view_cone_bins,
            "name": "viewcone",
            "unit": "deg",
        },
    }

    per_axis_limits = {}
    constant_angular_distance = _get_constant_data_value(histograms, "angular_distance")
    differential_energy_bins = None
    if bins_per_decade > 0:
        low = int(np.floor(np.log10(np.min(histograms.energy_bins))))
        high = int(np.ceil(np.log10(np.max(histograms.energy_bins))))
        differential_energy_bins = np.logspace(low, high, (high - low) * bins_per_decade + 1)

    for axis_name, config in axis_configs.items():
        if axis_name == "angular_distance" and constant_angular_distance is not None:
            axis_max = constant_angular_distance
            curve_x = [axis_max, axis_max]
            curve_y = energy_range
            _logger.info(
                f"All simulated events have angular distance {axis_max} deg; "
                "using this exact value as the viewcone limit."
            )
        elif bins_per_decade > 0:
            axis_max, curve_x, curve_y = _differential_upper_limits(
                histograms.histograms[f"{axis_name}_vs_energy"]["histogram"],
                config["x_bins"],
                histograms.energy_bins,
                differential_energy_bins,
                allowed_losses[axis_name],
                config["name"],
                config["unit"],
            )
        else:
            axis_max = _integral_limits(
                histograms.histograms[axis_name]["histogram"],
                config["x_bins"],
                allowed_losses[axis_name]["loss_fraction"],
                allowed_losses[axis_name]["loss_min_events"],
                limit_type="upper",
            )
            curve_x = [axis_max, axis_max]
            curve_y = energy_range

        per_axis_limits[axis_name] = {
            "max": axis_max,
            "curve": {"x": curve_x, "y": curve_y},
            "curve_key": f"{axis_name}_vs_energy_curve",
        }

    core_max = per_axis_limits["core_distance"]["max"]
    vc_max = per_axis_limits["angular_distance"]["max"]

    upper_radius_limit = core_max * u.m
    upper_radius_limit = _is_close(
        upper_radius_limit,
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
        "angular_distance_is_constant": constant_angular_distance is not None,
        per_axis_limits["core_distance"]["curve_key"]: per_axis_limits["core_distance"]["curve"],
        per_axis_limits["angular_distance"]["curve_key"]: per_axis_limits["angular_distance"][
            "curve"
        ],
    }


def _get_constant_data_value(histograms, name, abs_tol=0.01):
    """Return an event-data value when its accumulated range is constant."""
    data_ranges = getattr(histograms, "data_ranges", None)
    if not isinstance(data_ranges, dict) or name not in data_ranges:
        return None

    minimum, maximum = data_ranges[name]
    if np.isclose(minimum, maximum, rtol=0, atol=abs_tol):
        avg_value = (minimum + maximum) / 2.0
        return float(avg_value) if avg_value > abs_tol else 0.0
    return None


def _differential_upper_limits(
    histogram2d,
    x_bins,
    y_bins,
    diff_e_bins,
    allowed_loss,
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
        limit = _integral_limits(
            projected,
            x_bins,
            allowed_loss["loss_fraction"],
            allowed_loss["loss_min_events"],
            limit_type="upper",
        )
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


def write_results(results, args_dict, allowed_losses, energy_threshold_fraction):
    """
    Write the computed limits as astropy table to file.

    Parameters
    ----------
    results : list[dict]
        List of computed limits.
    args_dict : dict
        Dictionary containing command line arguments.
    allowed_losses : dict
        Per-axis loss settings for core_distance/angular_distance.
    energy_threshold_fraction : float
        Fraction used for deriving the lower energy threshold.
    """
    table = _create_results_table(
        results,
        allowed_losses,
        energy_threshold_fraction,
    )

    output_dir = io_handler.IOHandler().get_output_directory()
    output_file = output_dir / args_dict["output_file"]

    table.write(output_file, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Results saved to {output_file}")

    MetadataCollector.dump(args_dict, output_file)


def _create_results_table(results, allowed_losses, energy_threshold_fraction):
    """
    Convert list of simulation results to an astropy Table with metadata.

    Round values to appropriate precision and add metadata.

    Parameters
    ----------
    results : list[dict]
        Computed limits per file and telescope configuration.
    allowed_losses : dict
        Per-axis loss settings added to metadata.
    energy_threshold_fraction : float
        Fraction used for deriving the lower energy threshold.

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
        }
    )
    for axis_name in LOSS_AXES:
        table.meta[f"loss_fraction_{axis_name}"] = allowed_losses[axis_name]["loss_fraction"]
        table.meta[f"loss_min_events_{axis_name}"] = int(
            allowed_losses[axis_name]["loss_min_events"]
        )
    table.meta["energy_threshold_fraction"] = energy_threshold_fraction

    return table


def _process_result_row(res, cols, columns, units):
    """Process a single result row and add values to columns."""
    for k in cols:
        val = res.get(k, None)
        if val is not None:
            val = _round_value(k, val, res)
            _logger.debug(f"Adding {k}: {val} to column data")

        if hasattr(val, "unit"):
            columns[k].append(val.value)
            units[k] = val.unit
        else:
            columns[k].append(val)
            if k not in units:
                units[k] = None


def _enforce_minimum_value(candidate, minimum):
    """Ensure candidate is not below minimum while preserving units when present."""
    if minimum is None:
        return candidate

    if isinstance(candidate, u.Quantity) and isinstance(minimum, u.Quantity):
        minimum = minimum.to(candidate.unit)
    elif isinstance(candidate, u.Quantity) and not isinstance(minimum, u.Quantity):
        minimum = minimum * candidate.unit
    elif not isinstance(candidate, u.Quantity) and isinstance(minimum, u.Quantity):
        minimum = minimum.value

    return candidate if candidate >= minimum else minimum


def _round_value(key, val, row=None):
    """Round value based on key type."""
    if key == "lower_energy_limit":
        rounded = np.floor(val * 1e3) / 1e3
        if row is not None:
            rounded = _enforce_minimum_value(rounded, row.get("br_energy_min"))
        return rounded
    if key == "upper_radius_limit":
        return np.ceil(val / 25) * 25
    if key == "viewcone_radius":
        if row is not None and row.get("angular_distance_is_constant", False):
            return val
        return np.ceil(val / 0.25) * 0.25
    return val


def _create_table_columns(cols, columns, units):
    """Create table columns with appropriate data types."""
    table_cols = []
    for k in cols:
        col_data = columns[k]
        col_description = COLUMN_DESCRIPTIONS.get(k)
        if any(isinstance(v, list | tuple) for v in col_data):
            col = Column(
                data=col_data,
                name=k,
                unit=units.get(k),
                dtype=object,
                description=col_description,
            )
        else:
            col = Column(data=col_data, name=k, unit=units.get(k), description=col_description)
        table_cols.append(col)
    return table_cols


def _find_bin_index_for_value(bin_edges, value):
    """Return index of the histogram bin containing value."""
    edges = np.asarray(bin_edges, dtype=float)
    idx = int(np.searchsorted(edges, float(value), side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


def _apply_broad_range_lower_energy_floor(derived_limit, broad_range_min, energy_bins):
    """Apply physical and bin-consistent floor to the derived lower energy limit."""
    if broad_range_min is None:
        return derived_limit

    derived_tev = derived_limit.to("TeV")
    broad_range_tev = broad_range_min.to("TeV")

    derived_idx = _find_bin_index_for_value(energy_bins, derived_tev.value)
    broad_range_idx = _find_bin_index_for_value(energy_bins, broad_range_tev.value)
    if derived_idx == broad_range_idx:
        return broad_range_tev.to(derived_limit.unit)

    return _enforce_minimum_value(derived_limit, broad_range_min)


def _integral_limits(hist, bin_edges, loss_fraction, loss_min_events=10, limit_type="lower"):
    """
    Compute integral limits based on the loss fraction and minimal required lost events.

    Add or subtract one bin to be on the safe side of the limit.

    Parameters
    ----------
    hist : np.ndarray
        1D histogram array.
    bin_edges : np.ndarray
        Array of bin edges.
    loss_fraction : float
        Fraction of events to be lost.
    loss_min_events : int, optional
        Minimum number of events to be lost after applying a limit.
    limit_type : str, optional
        Type of limit ('lower' or 'upper'). Default is 'lower'.

    Returns
    -------
    float
        Bin edge value corresponding to the threshold.
    """
    total_events = np.sum(hist)
    # Keep-threshold corresponding to a strictly greater-than requested absolute event loss.
    max_kept_for_min_loss = np.nextafter(total_events - float(loss_min_events), -np.inf)
    threshold = min(
        (1 - loss_fraction) * total_events,
        max_kept_for_min_loss,
    )
    threshold = np.clip(threshold, 0.0, total_events)
    if limit_type == "upper":
        cum = np.cumsum(hist)
        idx = np.searchsorted(cum, threshold) + 1
        return bin_edges[min(idx, len(bin_edges) - 1)]
    if limit_type == "lower":
        cum = np.cumsum(hist[::-1])
        idx = np.searchsorted(cum, threshold)
        return bin_edges[max(len(bin_edges) - 1 - idx, 0)]
    raise ValueError("limit_type must be 'lower' or 'upper'")


def _find_low_energy_threshold_from_histogram(counts, bin_edges, threshold_fraction=0.1):
    """Find low-energy threshold from a 1D histogram using a peak-relative criterion.

    The threshold is defined as the first bin (walking to lower energies from the
    histogram maximum) where the count drops below ``threshold_fraction`` times a
    stable peak estimate. The stable peak estimate is the mean of the maximum bin
    and its immediate neighbors (neighbors included only when available).

    Parameters
    ----------
    counts : np.ndarray
        Histogram bin counts.
    bin_edges : np.ndarray
        Histogram bin edges (length must be ``len(counts) + 1``).
    threshold_fraction : float, optional
        Fraction of the stable peak used as threshold.

    Returns
    -------
    float
        Derived energy threshold.
    """
    counts = np.asarray(counts, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)

    if counts.ndim != 1 or bin_edges.ndim != 1:
        raise ValueError("counts and bin_edges must be one-dimensional arrays")
    if counts.size == 0:
        raise ValueError("counts must not be empty")
    if not np.any(counts > 0):
        raise ValueError("counts must contain at least one positive entry")
    if bin_edges.size != counts.size + 1:
        raise ValueError("bin_edges length must be len(counts) + 1")
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in the interval (0, 1]")

    peak_idx = int(np.argmax(counts))

    left = max(peak_idx - 1, 0)
    right = min(peak_idx + 1, counts.size - 1)
    n_peak = float(np.mean(counts[left : right + 1]))
    threshold = threshold_fraction * n_peak

    for idx in range(peak_idx, -1, -1):
        if counts[idx] < threshold:
            return float(bin_edges[idx])

    return float(bin_edges[0])


def compute_lower_energy_limit(histograms, threshold_fraction):
    """
    Compute the lower energy limit in TeV based on the threshold fraction.

    Parameters
    ----------
    histograms : EventDataHistograms
        Histograms.
    threshold_fraction : float
        Fraction of the stable peak used as threshold.

    Returns
    -------
    astropy.units.Quantity
        Lower energy limit.
    """
    energy_min = (
        _find_low_energy_threshold_from_histogram(
            histograms.histograms["energy"]["histogram"],
            histograms.energy_bins,
            threshold_fraction=threshold_fraction,
        )
        * u.TeV
    )

    broad_range_energy_min = (
        histograms.file_info["energy_min"].to("TeV")
        if "energy_min" in histograms.file_info
        else None
    )
    energy_min = _apply_broad_range_lower_energy_floor(
        energy_min,
        broad_range_energy_min,
        histograms.energy_bins,
    )

    return _is_close(
        energy_min,
        broad_range_energy_min,
        "Lower energy limit is equal to the minimum energy of",
    )


def _is_close(value, reference, warning_text):
    """Check if the value is close to the reference value and log a warning if so."""
    if reference is not None and np.isclose(value.value, reference.value, rtol=1.0e-2):
        _logger.warning(f"{warning_text} {value}.")
    return value
