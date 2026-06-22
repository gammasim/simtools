"""Generate bias curves from NSB and proton trigger rates."""

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.layout.array_layout_utils import get_array_elements_from_db_for_layouts
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.nsb_trigger_calculator import (
    derive_nsb_triggers,
    extract_threshold_from_file_name,
)
from simtools.simtel.simtel_log_reader import extract_run_number
from simtools.telescope_trigger_rates import telescope_trigger_rates

_logger = logging.getLogger(__name__)

_SIMTEL_LOG_SUFFIX = ".simtel.log.gz"
_REDUCED_EVENT_DATA_SUFFIX = ".reduced_event_data.hdf5"
_COPY_CHUNK_SIZE = 1024 * 1024
_DEFAULT_PLOT_FILE_NAME = "bias_curve.png"


def generate_bias_curves(args):
    """
    Generate bias curves from NSB logs and proton simulations.

    Parameters
    ----------
    args : dict
        Configuration parameters including:
        - data_dir: Directory containing both NSB log_hist archives/logs and proton simulation files
        - output: Output plot file path or output directory
        - nsb_output: Optional ECSV table output for NSB rates
        - proton_output: Optional ECSV table output for proton rates
        - site, model_version, array_layout_name or telescope_ids: For telescope config
        - title, ymin, ymax: Plot parameters
    """
    time_window = _calculate_time_window(args)
    _logger.info(f"Calculated time window: {time_window * 1e9:.2f} ns")

    _logger.info("Extracting NSB trigger rates from log files...")
    nsb_stats = _extract_nsb_rates(args, time_window)

    _logger.info("Calculating proton trigger rates...")
    proton_stats = _extract_proton_rates(args)

    if args.get("proton_output"):
        _write_proton_ecsv(proton_stats, args["proton_output"])
        _logger.info(f"Proton table written to {args['proton_output']}")

    plot_output_path = _resolve_plot_output_path(args["output"])
    bias_curve_table_output = plot_output_path.with_suffix(".ecsv")

    _logger.info("Plotting bias curves...")
    _plot_bias_curves(nsb_stats, proton_stats, args, plot_output_path)

    _write_bias_curve_ecsv(nsb_stats, proton_stats, bias_curve_table_output)

    _logger.info(f"Bias curve plot written to {plot_output_path}")
    _logger.info(f"Bias curve table written to {bias_curve_table_output}")
    if args.get("nsb_output"):
        _logger.info(f"NSB table written to {args['nsb_output']}")


def _calculate_time_window(args):
    """
    Calculate time window from telescope parameters.

    Gets telescope from array_layout_name and retrieves its parameters.
    time_window = disc_bins / (fadc_mhz * 1e6)

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
    telescope_name = _get_telescope_name_from_layout(args)

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


def _get_telescope_name_from_layout(args):
    """
    Get telescope name from array layout.

    For NSB analysis, expects a single telescope in the layout.

    Parameters
    ----------
    args : dict
        Arguments with telescope configuration.

    Returns
    -------
    str
        Telescope name.

    Raises
    ------
    ValueError
        If array_layout_name is not provided or no telescopes found in layout.
    """
    if not args.get("array_layout_name"):
        raise ValueError("array_layout_name must be provided for telescope configuration")

    telescope_configs = get_array_elements_from_db_for_layouts(
        args["array_layout_name"],
        args["site"],
        args["model_version"],
    )

    if not telescope_configs:
        raise ValueError(
            f"No telescopes found in layout '{args['array_layout_name']}' "
            f"for site={args['site']}, model_version={args['model_version']}"
        )

    array_name = next(iter(telescope_configs.keys()))
    telescope_ids = telescope_configs[array_name]

    if not telescope_ids:
        raise ValueError(f"No telescope IDs found in array '{array_name}'")

    telescope_name = telescope_ids[0]
    _logger.info(f"Using telescope {telescope_name} from layout {args['array_layout_name']}")
    return telescope_name


def _extract_nsb_rates(args, time_window):
    """Extract NSB trigger rates from direct sim_telarray logs or log_hist archives."""
    data_dir = Path(args["data_dir"])

    try:
        direct_logs = list(data_dir.rglob(f"*{_SIMTEL_LOG_SUFFIX}"))
        if direct_logs:
            _logger.info(f"Found {len(direct_logs)} direct sim_telarray log file(s)")
            return _run_nsb_trigger_derivation(data_dir, args, time_window)

        log_hist_archives = list(data_dir.rglob("*.log_hist.tar.gz"))
        if not log_hist_archives:
            raise FileNotFoundError(
                f"No *{_SIMTEL_LOG_SUFFIX} files or *.log_hist.tar.gz archives found in {data_dir}"
            )

        return _extract_archived_nsb_rates(args, data_dir, log_hist_archives, time_window)

    except (FileNotFoundError, ValueError) as e:
        _logger.warning(f"Could not extract NSB rates: {e}")
        return {}


def _extract_archived_nsb_rates(args, data_dir, log_hist_archives, time_window):
    """Extract archived sim_telarray logs to /tmp and derive NSB rates from them."""
    with tempfile.TemporaryDirectory(
        prefix="simtools-nsb-logs-",
        dir="/tmp",  # NOSONAR: temporary extracted logs are short-lived.
    ) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        n_extracted = _extract_simtel_logs_from_archives(log_hist_archives, tmp_dir)

        if n_extracted == 0:
            raise FileNotFoundError(
                f"No *{_SIMTEL_LOG_SUFFIX} files found inside *.log_hist.tar.gz "
                f"archives in {data_dir}"
            )

        _logger.info(f"Extracted {n_extracted} sim_telarray log file(s) to {tmp_dir}")

        nsb_stats = _run_nsb_trigger_derivation(tmp_dir, args, time_window)
        _logger.info(f"Found NSB rates for {len(nsb_stats)} thresholds")
        return nsb_stats


def _run_nsb_trigger_derivation(root_dir, args, time_window):
    """Run NSB trigger derivation on a directory containing *.simtel.log.gz files."""
    nsb_args = {
        "root_dir": root_dir,
        "pattern": f"**/gamma*{_SIMTEL_LOG_SUFFIX}",
        "output": args.get("nsb_output"),
        "time_window": time_window,
        "verbose": False,
    }
    return derive_nsb_triggers(nsb_args)


def _extract_simtel_logs_from_archives(archives, output_dir):
    """Extract *.simtel.log.gz files from *.log_hist.tar.gz archives."""
    n_extracted = 0

    for archive in sorted(archives):
        n_extracted += _extract_simtel_logs_from_archive(archive, output_dir)

    return n_extracted


def _extract_simtel_logs_from_archive(archive, output_dir):
    """Extract *.simtel.log.gz files from one *.log_hist.tar.gz archive."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            return sum(
                _extract_simtel_log_member(tar, member, output_dir) for member in tar.getmembers()
            )

    except tarfile.TarError as e:
        _logger.warning(f"Could not read archive {archive}: {e}")
        return 0


def _extract_simtel_log_member(tar, member, output_dir):
    """Extract one tar member if it is a sim_telarray log file."""
    member_name = Path(member.name).name

    if not (member.isfile() and member_name.endswith(_SIMTEL_LOG_SUFFIX)):
        return 0

    source = tar.extractfile(member)
    if source is None:
        return 0

    _write_tar_member_to_file(source, output_dir / member_name)
    return 1


def _write_tar_member_to_file(source, target):
    """Write a tar member stream to disk."""
    with source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=_COPY_CHUNK_SIZE)


def _extract_proton_rates(args):
    """
    Extract proton trigger rates from HDF5 files.

    Thresholds and run numbers are extracted from file names.
    """
    data_dir = Path(args["data_dir"])
    proton_files = _group_hdf5_files_by_threshold_and_run(data_dir)

    if not proton_files:
        _logger.warning(f"No proton HDF5 files with threshold labels found in {data_dir}")
        return {}

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
    trigger_args = {
        "event_data_file": str(hdf5_file),
        "plot_histograms": False,
    }

    if args.get("array_layout_name"):
        trigger_args["array_layout_name"] = args["array_layout_name"]
        trigger_args["site"] = args["site"]
        trigger_args["model_version"] = args["model_version"]
    elif args.get("telescope_ids"):
        trigger_args["telescope_ids"] = args["telescope_ids"]
    else:
        _logger.warning("No telescope configuration provided")
        return None

    try:
        results = telescope_trigger_rates(trigger_args)

        if results:
            array_name = next(iter(results.keys()))
            rate_with_units = results[array_name]
            return rate_with_units.to(u.Hz).value

        return None

    except (OSError, KeyError, ValueError, AttributeError) as e:
        _logger.debug(f"Error calculating rate for {hdf5_file}: {e}")
        return None


def _resolve_plot_output_path(output):
    """Resolve output as either an explicit plot file or an output directory."""
    output_path = Path(output)

    if output_path.suffix:
        return output_path

    return output_path / _DEFAULT_PLOT_FILE_NAME


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

    for col in ["Rate (Hz)", "Error (Hz)"]:
        table[col].format = ".2f"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_file, format="ascii.ecsv", overwrite=True)


def _write_bias_curve_ecsv(nsb_stats, proton_stats, output_file):
    """Write final plotted bias-curve values to an ECSV table."""
    output_file = Path(output_file)
    thresholds = sorted(set(nsb_stats.keys()) | set(proton_stats.keys()))

    table = Table(
        {
            "threshold": thresholds,
            "NSB rate (Hz)": [
                nsb_stats[threshold]["rate_hz"] if threshold in nsb_stats else np.nan
                for threshold in thresholds
            ],
            "Proton rate (Hz)": [
                proton_stats[threshold]["rate_hz"] if threshold in proton_stats else np.nan
                for threshold in thresholds
            ],
        }
    )

    table["NSB rate (Hz)"].format = ".2f"
    table["Proton rate (Hz)"].format = ".2f"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    table.write(output_file, format="ascii.ecsv", overwrite=True)


def _plot_bias_curves(nsb_stats, proton_stats, args, output_path):
    """
    Plot NSB and proton bias curves.

    Parameters
    ----------
    nsb_stats : dict
        NSB statistics by threshold.
    proton_stats : dict
        Proton statistics by threshold.
    args : dict
        Plot parameters.
    output_path : Path
        Output path for plot image.
    """
    fig, axis = plt.subplots(figsize=(10, 7))

    _plot_nsb_curve(axis, nsb_stats)
    _plot_proton_curve(axis, proton_stats)
    _configure_bias_curve_axis(axis, args)

    fig.tight_layout()
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


def _configure_bias_curve_axis(axis, args):
    """Configure bias-curve axis labels, scaling, and legend."""
    axis.set_title(args["title"], fontsize=14, fontweight="bold")
    axis.set_xlabel("Threshold", fontsize=12)
    axis.set_ylabel("Trigger Rate [Hz]", fontsize=12)
    axis.set_yscale("log")
    axis.set_ylim(args["ymin"], args["ymax"])
    axis.grid(which="both", alpha=0.3, linestyle=":")

    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=11, loc="best")
    else:
        _logger.warning("No NSB or proton rates found; writing empty bias-curve plot")
