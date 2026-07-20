"""Generate bias curves from NSB and proton trigger rates."""

import logging
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.nsb_trigger_calculator import (
    derive_nsb_triggers,
    extract_threshold_from_file_name,
)
from simtools.simtel.simtel_log_reader import extract_run_number
from simtools.telescope_trigger_rates import telescope_trigger_rates
from simtools.visualization import plot_tables

_logger = logging.getLogger(__name__)

_SIMTEL_LOG_SUFFIX = ".simtel.log.gz"
_REDUCED_EVENT_DATA_SUFFIX = ".reduced_event_data.hdf5"


def generate_bias_curves(args):
    """Generate bias curves from NSB logs and proton simulations."""
    time_window = _calculate_time_window(args)
    _logger.info(f"Calculated time window: {time_window * 1e9:.2f} ns")

    _logger.info("Extracting NSB trigger rates from log files...")
    nsb_stats = _extract_nsb_rates(args, time_window)
    if not nsb_stats:
        raise FileNotFoundError(f"No NSB input files found in {args['data_dir']}")

    _logger.info("Calculating proton trigger rates...")
    proton_stats = _extract_proton_rates(args)
    if not proton_stats:
        raise FileNotFoundError(f"No proton input files found in {args['data_dir']}")

    if args.get("proton_output"):
        _write_proton_ecsv(proton_stats, args["proton_output"])
        _logger.info(f"Proton table written to {args['proton_output']}")

    plot_output_path = plot_tables.resolve_plot_output_path(args["output"])
    bias_curve_table_output = plot_output_path.with_suffix(".ecsv")

    _logger.info("Plotting bias curves...")
    plot_tables.plot_bias_curves(nsb_stats, proton_stats, args, plot_output_path)

    _write_bias_curve_ecsv(nsb_stats, proton_stats, bias_curve_table_output)

    _logger.info(f"Bias curve plot written to {plot_output_path}")
    _logger.info(f"Bias curve table written to {bias_curve_table_output}")
    nsb_output = args.get("nsb_output")
    if nsb_output and Path(nsb_output).exists():
        _logger.info(f"NSB table written to {nsb_output}")


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
    """Extract NSB trigger rates from direct sim_telarray logs."""
    data_dir = Path(args["data_dir"])
    direct_logs = list(data_dir.rglob(f"*{_SIMTEL_LOG_SUFFIX}"))
    if direct_logs:
        _logger.info(f"Found {len(direct_logs)} direct sim_telarray log file(s)")
        return _run_nsb_trigger_derivation(data_dir, args, time_window)

    raise FileNotFoundError(f"No *{_SIMTEL_LOG_SUFFIX} files found in {data_dir}")


def _run_nsb_trigger_derivation(root_dir, args, time_window):
    """Run NSB trigger derivation on a directory containing ``gamma*.simtel.log.gz`` files."""
    nsb_args = {
        "root_dir": root_dir,
        "pattern": f"gamma*{_SIMTEL_LOG_SUFFIX}",
        "output": args.get("nsb_output"),
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

        threshold = extract_threshold_from_file_name(hdf5_file)
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

    table_data["Rate (Hz)"] = rate_hz_col
    table_data["Error (Hz)"] = error_hz_col
    table_data["Num runs"] = num_runs_col

    table = Table(table_data)
    table.meta["comments"] = ["Run columns contain proton trigger rates in Hz."]

    table["Rate (Hz)"] = np.round(table["Rate (Hz)"], 2)
    table["Error (Hz)"] = np.round(table["Error (Hz)"], 2)
    table["Rate (Hz)"].format = ".2f"
    table["Error (Hz)"].format = ".2f"

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
