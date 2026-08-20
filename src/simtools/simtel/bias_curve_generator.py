"""Generate bias curves from NSB and proton trigger rates."""

import logging
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.data_model import model_data_writer
from simtools.io import io_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.nsb_trigger_calculator import (
    derive_nsb_triggers,
    extract_run_number,
    extract_threshold,
)
from simtools.telescope_trigger_rates import telescope_trigger_rates
from simtools.visualization import plot_tables

_logger = logging.getLogger(__name__)

_REDUCED_EVENT_DATA_SUFFIX = ".reduced_event_data.hdf5"


def generate_bias_curves(args):
    """Generate bias curves from NSB/proton reduced event-data HDF5 files."""
    time_window = _calculate_time_window(args)
    _logger.info(f"Calculated time window: {time_window * 1e9:.2f} ns")

    _logger.info("Extracting NSB trigger rates from HDF5 files...")
    nsb_stats = _extract_nsb_rates(args, time_window)
    if not nsb_stats:
        raise FileNotFoundError(f"No NSB input files found in {args['data_dir']}")

    _logger.info("Calculating proton trigger rates...")
    proton_stats = _extract_proton_rates(args)
    if not proton_stats:
        raise FileNotFoundError(f"No proton input files found in {args['data_dir']}")

    if args.get("proton_table_file"):
        _write_proton_ecsv(proton_stats, args["proton_table_file"])
        _logger.info(f"Proton table written to {args['proton_table_file']}")

    plot_output_path = plot_tables.resolve_plot_output_path(args["figure_file"])
    bias_curve_table_output = plot_output_path.with_suffix(".ecsv")
    trigger_threshold = _calculate_trigger_threshold(nsb_stats, proton_stats)

    _logger.info("Plotting bias curves...")
    plot_tables.plot_bias_curves(nsb_stats, proton_stats, args, plot_output_path, trigger_threshold)
    _write_bias_curve_ecsv(nsb_stats, proton_stats, bias_curve_table_output)
    _export_trigger_threshold_as_model_parameter(args, trigger_threshold)

    _logger.info(f"Bias curve plot written to {plot_output_path}")
    _logger.info(f"Bias curve table written to {bias_curve_table_output}")
    nsb_table_file = args.get("nsb_table_file")
    if nsb_table_file and Path(nsb_table_file).exists():
        _logger.info(f"NSB table written to {nsb_table_file}")


def _calculate_time_window(args):
    """
    Calculate time window from telescope parameters.

    Gets telescope from args and retrieves its parameters.
    time_window = disc_bins / (fadc_mhz x 1e6)

    Parameters
    ----------
    args : dict
        Arguments with telescope configuration.

    Returns
    -------
    float
        Time window in seconds.

    Raises
    ------
    ValueError
        If telescope name cannot be determined or parameters cannot be retrieved.
    """
    telescope_name = args.get("telescope")
    if not telescope_name:
        raise ValueError("telescope must be provided for telescope configuration")

    telescope_model = TelescopeModel(
        site=args["site"],
        telescope_name=telescope_name,
        model_version=args["model_version"],
    )

    disc_bins = telescope_model.get_parameter_value("disc_bins")
    fadc_mhz = telescope_model.get_parameter_value("fadc_mhz")
    time_window = disc_bins / (fadc_mhz * 1e6)

    _logger.info(
        f"Telescope {telescope_name}: disc_bins={disc_bins}, "
        f"fadc_mhz={fadc_mhz} MHz, time_window={time_window:.2f} s"
    )

    return time_window


def _extract_nsb_rates(args, time_window):
    """Extract NSB trigger rates from gamma reduced event-data HDF5 files."""
    data_dir = Path(args["data_dir"])
    gamma_hdf5_files = list(data_dir.rglob(f"gamma*{_REDUCED_EVENT_DATA_SUFFIX}"))
    if gamma_hdf5_files:
        _logger.info(f"Found {len(gamma_hdf5_files)} gamma HDF5 file(s)")
        return _run_nsb_trigger_derivation(data_dir, args, time_window)

    raise FileNotFoundError(f"No gamma*{_REDUCED_EVENT_DATA_SUFFIX} files found in {data_dir}")


def _run_nsb_trigger_derivation(root_dir, args, time_window):
    """Run NSB trigger derivation on gamma reduced event-data HDF5 files."""
    nsb_args = {
        "root_dir": root_dir,
        "pattern": f"gamma*{_REDUCED_EVENT_DATA_SUFFIX}",
        "output": args.get("nsb_table_file"),
        "time_window": time_window,
        "verbose": False,
    }
    return derive_nsb_triggers(nsb_args)


def _extract_proton_rates(args):
    """
    Extract proton trigger rates from HDF5 files.

    Thresholds and run numbers are extracted from file names.
    """
    data_dir = Path(args["data_dir"])
    proton_files = _group_hdf5_files_by_threshold_and_run(data_dir)

    if not proton_files:
        raise FileNotFoundError(f"No proton HDF5 files with threshold labels found in {data_dir}")

    _logger.info(f"Found proton HDF5 files for {len(proton_files)} thresholds")

    proton_stats = {}
    for threshold, run_files in sorted(proton_files.items()):
        _logger.info(f"Processing threshold {threshold}: {len(run_files)} HDF5 file(s)")
        proton_stats[threshold] = _calculate_proton_statistics_for_threshold(run_files, args)

    return proton_stats


def _group_hdf5_files_by_threshold_and_run(proton_dir):
    """Group proton HDF5 files by threshold and run extracted from file names."""
    threshold_files = {}

    for hdf5_file in proton_dir.rglob(f"*{_REDUCED_EVENT_DATA_SUFFIX}"):
        if "proton" not in hdf5_file.name.lower():
            continue

        threshold = extract_threshold(hdf5_file)
        run = extract_run_number(hdf5_file)

        if threshold is None or run is None:
            _logger.warning(
                f"Skipping proton file with missing threshold or run: "
                f"{hdf5_file} (threshold={threshold}, run={run})"
            )
            continue

        threshold_files.setdefault(threshold, {})[run] = hdf5_file

    return threshold_files


def _calculate_proton_statistics_for_threshold(run_files, args):
    """Calculate mean proton trigger rate and error for one threshold."""
    run_rates = {}

    for run, hdf5_file in sorted(run_files.items()):
        rate = _calculate_proton_rate_for_file(hdf5_file, args)
        if rate is not None:
            run_rates[run] = rate

    rates = list(run_rates.values())
    if not rates:
        return {
            "runs": {},
            "rate_hz": np.nan,
            "error_hz": np.nan,
            "num_runs": 0,
        }

    error_hz = 0
    if len(rates) > 1:
        error_hz = np.std(rates, ddof=1) / np.sqrt(len(rates))

    return {
        "runs": run_rates,
        "rate_hz": float(np.mean(rates)),
        "error_hz": float(error_hz),
        "num_runs": len(rates),
    }


def _calculate_proton_rate_for_file(hdf5_file, args):
    """
    Calculate proton trigger rate for a single HDF5 file.

    Parameters
    ----------
    hdf5_file : Path
        Path to HDF5 file.
    args : dict
        Arguments with telescope configuration.

    Returns
    -------
    float
        Trigger rate in Hz, or None if calculation fails.
    """
    if not args.get("telescope"):
        _logger.warning("No telescope configuration provided")
        return None

    trigger_args = {key: args[key] for key in ("telescope", "site", "model_version")}
    trigger_args.update({"event_data_file": str(hdf5_file), "plot_histograms": False})

    try:
        results = telescope_trigger_rates(trigger_args)

        if results:
            array_name = next(iter(results.keys()))
            rate_with_units = results[array_name]
            return rate_with_units.to(u.Hz).value

        return None

    except TypeError as e:
        if "NoneType" in str(e):
            _logger.info(f"No triggered event data found in {hdf5_file}; using 0 Hz")
            return 0.0

        _logger.debug(f"Error calculating rate for {hdf5_file}: {e}")
        return None

    except (OSError, KeyError, ValueError, AttributeError) as e:
        _logger.debug(f"Error calculating rate for {hdf5_file}: {e}")
        return None


def _write_proton_ecsv(proton_stats, output_file):
    """Write runwise proton trigger rates to an ECSV table."""
    output_file = Path(output_file)
    error_hz_column = "Error (Hz)"
    rate_hz_column = "Rate (Hz)"

    if not proton_stats:
        raise ValueError("No proton statistics to write")

    all_runs = sorted(
        {run for threshold_stats in proton_stats.values() for run in threshold_stats["runs"].keys()}
    )

    threshold_col = []
    run_cols = {run: [] for run in all_runs}
    rate_hz_col = []
    error_hz_col = []
    num_runs_col = []

    for threshold in sorted(proton_stats.keys()):
        stats = proton_stats[threshold]
        threshold_col.append(threshold)

        for run in all_runs:
            run_cols[run].append(stats["runs"].get(run, np.nan))

        rate_hz_col.append(stats["rate_hz"])
        error_hz_col.append(stats["error_hz"])
        num_runs_col.append(stats["num_runs"])

    table_data = {"threshold": threshold_col}
    for run in all_runs:
        table_data[f"run{run}"] = run_cols[run]

    table_data[rate_hz_column] = rate_hz_col
    table_data[error_hz_column] = error_hz_col
    table_data["Num runs"] = num_runs_col

    table = Table(table_data)
    table.meta["comments"] = ["Run columns contain proton trigger rates in Hz."]

    table[rate_hz_column] = np.round(table[rate_hz_column], 2)
    table[error_hz_column] = np.round(table[error_hz_column], 2)
    table[rate_hz_column].format = ".2f"
    table[error_hz_column].format = ".2f"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_file, format="ascii.ecsv", overwrite=True)


def _write_bias_curve_ecsv(nsb_stats, proton_stats, output_file):
    """Write final plotted bias-curve values to an ECSV table."""
    output_file = Path(output_file)
    thresholds = sorted(set(nsb_stats.keys()) | set(proton_stats.keys()))
    nsb_error_column = "NSB error (Hz)"
    nsb_rate_column = "NSB rate (Hz)"
    proton_error_column = "Proton error (Hz)"
    proton_rate_column = "Proton rate (Hz)"

    table = Table(
        {
            "threshold": thresholds,
            nsb_rate_column: [
                nsb_stats[threshold]["rate_hz"] if threshold in nsb_stats else np.nan
                for threshold in thresholds
            ],
            nsb_error_column: [
                nsb_stats[threshold]["error_hz"] if threshold in nsb_stats else np.nan
                for threshold in thresholds
            ],
            proton_rate_column: [
                proton_stats[threshold]["rate_hz"] if threshold in proton_stats else np.nan
                for threshold in thresholds
            ],
            proton_error_column: [
                proton_stats[threshold]["error_hz"] if threshold in proton_stats else np.nan
                for threshold in thresholds
            ],
        }
    )

    for column_name in (
        nsb_rate_column,
        nsb_error_column,
        proton_rate_column,
        proton_error_column,
    ):
        table[column_name] = np.round(table[column_name], 2)
        table[column_name].format = ".2f"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_file, format="ascii.ecsv", overwrite=True)


def _calculate_trigger_threshold(nsb_stats, proton_stats):
    """
    Calculate trigger threshold from bias curve intersection.

    Trigger threshold is calculated as the intersection between NSB curve and 1.35*proton curve.

    Parameters
    ----------
    args : dict
        Dictionary with configuration parameters.
    nsb_stats : dict
        NSB statistics by threshold.
    proton_stats : dict
        Proton statistics by threshold.

    Returns
    -------
    float
        The calculated trigger threshold.

    Raises
    ------
    ValueError
        If no valid threshold points exist or intersection cannot be found.
    """
    # Get all unique thresholds from both NSB and proton stats
    thresholds = sorted(set(nsb_stats.keys()) | set(proton_stats.keys()))
    # Extract rates for each threshold
    nsb_rates = []
    proton_rates = []
    for threshold in thresholds:
        nsb_rate = nsb_stats[threshold]["rate_hz"] if threshold in nsb_stats else np.nan
        proton_rate = proton_stats[threshold]["rate_hz"] if threshold in proton_stats else np.nan
        nsb_rates.append(nsb_rate)
        proton_rates.append(proton_rate)
    nsb_rates = np.array(nsb_rates)
    proton_rates = np.array(proton_rates)
    thresholds = np.array(thresholds)

    # Remove NaN values (keep only thresholds where both NSB and proton data exist)
    valid_mask = ~(np.isnan(nsb_rates) | np.isnan(proton_rates))
    nsb_rates = nsb_rates[valid_mask]
    proton_rates = proton_rates[valid_mask]
    thresholds = thresholds[valid_mask]

    if len(thresholds) == 0:
        raise ValueError(
            "No valid threshold points with both NSB and proton data. "
            "Cannot calculate trigger threshold."
        )
    # Scale proton rates by 1.35 to account for ions we didn't simulate
    scaled_proton_rates = 1.35 * proton_rates
    trigger_threshold = _find_intersection_point(thresholds, nsb_rates, scaled_proton_rates)
    if trigger_threshold is not None:
        _logger.info(f"Calculated trigger threshold: {trigger_threshold}")
        return trigger_threshold
    raise ValueError("Could not find intersection point between NSB and 1.35*proton curves.")


def _find_intersection_point(thresholds, nsb_rates, scaled_proton_rates):
    """
    Find the threshold value where NSB trigger rate intersects with 1.35 * proton trigger rate.

    Parameters
    ----------
    thresholds : numpy.ndarray
        Threshold values from bias curve.
    nsb_rates : numpy.ndarray
        NSB trigger rates at each threshold.
    scaled_proton_rates : numpy.ndarray
        Scaled (1.35x) proton trigger rates at each threshold.

    Returns
    -------
    float or None
        Threshold value at intersection point, or None if no intersection found.
    """
    differences = nsb_rates - scaled_proton_rates
    sign_diff = np.diff(np.sign(differences))
    sign_changes = np.nonzero(sign_diff)[0]
    if len(sign_changes) == 0:
        _logger.debug("No intersection found between NSB and scaled proton curves")
        return None

    # Take the first intersection point
    crossing_idx = sign_changes[0]
    # Interpolate to find the exact intersection point
    x1, x2 = thresholds[crossing_idx], thresholds[crossing_idx + 1]
    y1_nsb, y2_nsb = nsb_rates[crossing_idx], nsb_rates[crossing_idx + 1]
    y1_proton, y2_proton = scaled_proton_rates[crossing_idx], scaled_proton_rates[crossing_idx + 1]

    # Linear interpolation
    denominator = x2 - x1
    if abs(denominator) < 1e-10:
        # Use midpoint as approximation if thresholds are too close
        return float((x1 + x2) / 2.0)
    slope_nsb = (y2_nsb - y1_nsb) / denominator
    slope_proton = (y2_proton - y1_proton) / denominator
    if abs(slope_nsb - slope_proton) < 1e-10:  # Parallel lines
        # Use midpoint as approximation
        return float((x1 + x2) / 2.0)
    delta_y = y1_proton - y1_nsb
    delta_slope = slope_nsb - slope_proton
    if abs(delta_slope) < 1e-10:
        # Lines are too close to parallel, use midpoint
        return float((x1 + x2) / 2.0)
    delta_x = delta_y / delta_slope
    return float(x1 + delta_x)


def _export_trigger_threshold_as_model_parameter(args, trigger_threshold):
    """
    Export trigger threshold as a model parameter.

    Parameters
    ----------
    args : dict
        Dictionary with configuration parameters.
    trigger_threshold : float
        The calculated trigger threshold value.
    """
    try:
        # Get telescope name from args
        telescope_name = args.get("telescope")
        if not telescope_name:
            _logger.warning("No telescope name provided. Using 'unknown' as telescope name.")
            telescope_name = "unknown"
        parameter_version = args.get("parameter_version")
        output_path = io_handler.IOHandler().get_output_directory()
        output_file = f"trigger_threshold-{parameter_version}.json"
        model_data_writer.ModelDataWriter.write_model_parameter(
            parameter_name="trigger_threshold",
            value=round(trigger_threshold),
            instrument=telescope_name,
            parameter_version=parameter_version,
            output_file=output_file,
            output_path=output_path / telescope_name / "trigger_threshold",
            metadata_input_dict={"source": "bias_curve_analysis"},
            check_db_for_existing_parameter=False,
        )

        _logger.info(
            f"Exported trigger threshold as model parameter for {telescope_name}: "
            f"{trigger_threshold}"
        )

    except (OSError, ValueError, KeyError) as exc:
        _logger.warning(f"Failed to export trigger threshold as model parameter: {exc}")
