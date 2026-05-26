"""Generate bias curves from NSB and proton trigger rates."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from simtools.layout.array_layout_utils import get_array_elements_from_db_for_layouts
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.nsb_trigger_calculator import derive_nsb_triggers
from simtools.telescope_trigger_rates import telescope_trigger_rates

_logger = logging.getLogger(__name__)


def generate_bias_curves(args):
    """
    Generate bias curves from NSB logs and proton simulations.

    Parameters
    ----------
    args : dict
        Configuration parameters including:
        - nsb_dir: Directory for NSB log files
        - proton_dir: Directory for proton simulation files
        - output: Output plot file path
        - nsb_table_output: Optional ECSV table output for NSB rates
        - nsb_log_pattern: Pattern for NSB log files
        - proton_file_pattern: Pattern for proton HDF5 files
        - site, model_version, array_layout_name or telescope_ids: For telescope config
        - title, ymin, ymax: Plot parameters
    """
    time_window = _calculate_time_window(args)
    _logger.info(f"Calculated time window: {time_window * 1e9:.2f} ns")

    _logger.info("Extracting NSB trigger rates from log files...")
    nsb_stats = _extract_nsb_rates(args, time_window)

    _logger.info("Calculating proton trigger rates...")
    proton_stats = _extract_proton_rates(args)

    _logger.info("Plotting bias curves...")
    _plot_bias_curves(nsb_stats, proton_stats, args)

    _logger.info(f"Bias curves written to {args['output']}")
    if args.get("nsb_table_output"):
        _logger.info(f"NSB table written to {args['nsb_table_output']}")


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

    # Load telescope model
    telescope_model = TelescopeModel(
        site=args["site"],
        telescope_name=telescope_name,
        model_version=args["model_version"],
    )

    # Get parameters
    disc_bins = telescope_model.get_parameter_value("disc_bins")
    fadc_mhz = telescope_model.get_parameter_value("fadc_mhz")

    # Calculate time window
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

    # Get telescope elements from array layout
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

    # Currently set up for single telescope layouts
    telescope_name = telescope_ids[0]
    _logger.info(f"Using telescope {telescope_name} from layout {args['array_layout_name']}")
    return telescope_name


def _extract_nsb_rates(args, time_window):
    """
    Extract NSB trigger rates from log files.

    Parameters
    ----------
    args : dict
        Arguments including nsb_dir, nsb_log_pattern, and optional nsb_table_output.
    time_window : float
        Time window in seconds.

    Returns
    -------
    dict
        NSB statistics by threshold.
    """
    nsb_dir = Path(args["nsb_dir"])
    nsb_args = {
        "root_dir": nsb_dir,
        "pattern": args.get("nsb_log_pattern", "**/*.simtel.log.gz"),
        "output": args.get("nsb_table_output"),  # Write ECSV if specified
        "time_window": time_window,
        "verbose": False,
    }

    try:
        nsb_stats = derive_nsb_triggers(nsb_args)
        _logger.info(f"Found NSB rates for {len(nsb_stats)} thresholds")
        return nsb_stats
    except (FileNotFoundError, ValueError) as e:
        # FileNotFoundError: no log files found
        # ValueError: no log files could be parsed successfully
        _logger.warning(f"Could not extract NSB rates: {e}")
        return {}


def _extract_proton_rates(args):
    """
    Extract proton trigger rates from HDF5 files organized by threshold.

    Searches for threshold directories (numeric subdirectories) and processes
    HDF5 files within each.

    Parameters
    ----------
    args : dict
        Arguments including proton_dir, site, model_version, telescope config.

    Returns
    -------
    dict
        Proton trigger rates by threshold: {threshold: rate_in_hz}
    """
    proton_dir = Path(args["proton_dir"])
    proton_stats = {}

    # Find threshold directories (numeric subdirectories)
    threshold_dirs = []
    for item in proton_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            threshold = int(item.name)
            # Check if reasonable threshold range
            if 50 <= threshold <= 10000:
                threshold_dirs.append((threshold, item))

    if not threshold_dirs:
        _logger.warning("No threshold directories found")
        return proton_stats

    _logger.info(f"Found {len(threshold_dirs)} threshold directories")

    # Process each threshold directory
    for threshold, threshold_dir in sorted(threshold_dirs):
        _logger.info(f"Processing threshold {threshold} in {threshold_dir}")

        # Find HDF5 files in this threshold directory
        hdf5_files = list(threshold_dir.rglob(args.get("proton_file_pattern", "*.hdf5")))

        if not hdf5_files:
            _logger.warning(f"No HDF5 files found in {threshold_dir}")
            continue

        _logger.info(f"Found {len(hdf5_files)} HDF5 file(s) for threshold {threshold}")

        # Process each HDF5 file and average the rates
        threshold_rates = []
        for hdf5_file in hdf5_files:
            rate = _calculate_proton_rate_for_file(hdf5_file, args)
            if rate is not None:
                threshold_rates.append(rate)

        if threshold_rates:
            # Average rates for this threshold
            avg_rate = np.mean(threshold_rates)
            proton_stats[threshold] = avg_rate
            _logger.info(
                f"Threshold {threshold}: {avg_rate:.2f} Hz (from {len(threshold_rates)} files)"
            )

    return proton_stats


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
    # Build arguments for telescope_trigger_rates
    trigger_args = {
        "event_data_file": str(hdf5_file),
        "plot_histograms": False,
    }

    # Add telescope configuration
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
        # Call telescope_trigger_rates and get result
        results = telescope_trigger_rates(trigger_args)

        # Extract rate (take first array if multiple)
        if results:
            array_name = next(iter(results.keys()))
            rate_with_units = results[array_name]
            # Convert to Hz value
            return rate_with_units.to(u.Hz).value
        return None

    except (OSError, KeyError, ValueError, AttributeError) as e:
        _logger.debug(f"Error calculating rate for {hdf5_file}: {e}")
        return None


def _plot_bias_curves(nsb_stats, proton_stats, args):
    """
    Plot NSB and proton bias curves.

    Parameters
    ----------
    nsb_stats : dict
        NSB statistics by threshold.
    proton_stats : dict
        Proton rates by threshold.
    args : dict
        Plot parameters.
    """
    fig, axis = plt.subplots(figsize=(10, 7))

    # Plot NSB curve
    if nsb_stats:
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

        # Fit and plot trend line if we have enough points
        if len(nsb_thresholds) >= 2:
            valid_mask = np.array(nsb_rates) > 0
            if np.sum(valid_mask) >= 2:
                fit_thresh = np.array(nsb_thresholds)[valid_mask]
                fit_rates = np.array(nsb_rates)[valid_mask]
                coeffs = np.polyfit(fit_thresh, np.log10(fit_rates), 1)
                x_fit = np.linspace(min(fit_thresh), max(fit_thresh), 100)
                y_fit = 10 ** (coeffs[0] * x_fit + coeffs[1])
                axis.plot(x_fit, y_fit, "--", color="tab:blue", alpha=0.5, linewidth=1)

    # Plot proton curve
    if proton_stats:
        proton_thresholds = sorted(proton_stats.keys())
        proton_rates = [proton_stats[t] for t in proton_thresholds]

        axis.plot(
            proton_thresholds,
            proton_rates,
            "s",
            label="Proton",
            color="tab:orange",
            markersize=8,
        )

        # Fit and plot trend line
        if len(proton_thresholds) >= 2:
            valid_mask = np.array(proton_rates) > 0
            if np.sum(valid_mask) >= 2:
                fit_thresh = np.array(proton_thresholds)[valid_mask]
                fit_rates = np.array(proton_rates)[valid_mask]
                coeffs = np.polyfit(fit_thresh, np.log10(fit_rates), 1)
                x_fit = np.linspace(min(fit_thresh), max(fit_thresh), 100)
                y_fit = 10 ** (coeffs[0] * x_fit + coeffs[1])
                axis.plot(x_fit, y_fit, "--", color="tab:orange", alpha=0.5, linewidth=1)

    # Configure plot
    axis.set_title(args["title"], fontsize=14, fontweight="bold")
    axis.set_xlabel("Threshold", fontsize=12)
    axis.set_ylabel("Trigger Rate [Hz]", fontsize=12)
    axis.set_yscale("log")
    axis.set_ylim(args["ymin"], args["ymax"])
    axis.grid(which="both", alpha=0.3, linestyle=":")
    axis.legend(fontsize=11, loc="best")

    # Save figure
    fig.tight_layout()
    output_path = Path(args["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
