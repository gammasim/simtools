"""Plot simtel event histograms filled with SimtelIOEventHistograms."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from simtools.simtel.simtel_io_event_histograms import SimtelIOEventHistograms

_logger = logging.getLogger(__name__)


def plot(histograms, output_path=None, limits=None, rebin_factor=2, array_name=None):
    """
    Plot simtel event histograms.

    Parameters
    ----------
    histograms: SimtelIOEventHistograms
        Instance containing the histograms to plot.
    output_path: Path or str, optional
        Directory to save plots. If None, plots will be displayed.
    limits: dict, optional
        Dictionary containing limits for plotting. Keys can include:
        - "upper_radius_limit": Upper limit for core distance
        - "lower_energy_limit": Lower limit for energy
        - "viewcone_radius": Radius for the viewcone
    rebin_factor: int, optional
        Factor by which to reduce the number of bins in 2D histograms for re-binned plots.
        Default is 2 (merge every 2 bins). Set to 0 or 1 to disable re-binning.
    array_name: str, optional
        Name of the telescope array configuration.
    """
    _logger.info(f"Plotting histograms written to {output_path}")

    plots = _generate_plot_configurations(histograms, limits)
    _execute_plotting_loop(plots, output_path, rebin_factor, array_name)


def _get_limits(limits):
    """Extract limits from the provided dictionary for plotting."""
    if not limits:
        return None, None, None
    return tuple(
        limits.get(key).value if key in limits else None
        for key in ("upper_radius_limit", "lower_energy_limit", "viewcone_radius")
    )


def _generate_plot_configurations(histograms, limits):
    """Generate plot configurations for all histogram types."""
    plot_labels = {
        "core_distance": "Core Distance [m]",
        "energy": "Energy [TeV]",
        "pointing_direction": "Distance to pointing direction [deg]",
        "core_x": "Core X [m]",
        "core_y": "Core Y [m]",
    }

    plots = _generate_1d_plots(histograms, plot_labels, limits)
    plots.update(_generate_2d_plots(histograms, plot_labels, limits))
    return plots


def _generate_1d_plots(histograms, labels, limits):
    """Generate 1D histogram plot configurations."""
    hist_1d_params = {"color": "tab:green", "edgecolor": "tab:green", "lw": 1}
    upper_radius_limit, lower_energy_limit, viewcone_radius = _get_limits(limits)
    plot_config = {
        "energy": {
            "x_label": labels["energy"],
            "scales": {"x": "log", "y": "log"},
            "lines": {"x": lower_energy_limit},
        },
        "core_distance": {
            "x_label": labels["core_distance"],
            "scales": {},
            "lines": {"x": upper_radius_limit},
        },
        "angular_distance": {
            "x_label": labels["pointing_direction"],
            "scales": {},
            "lines": {"x": viewcone_radius},
        },
        # TODO should go to io_histograms? part of histo_type? (rename)
        "cr_rates": {
            "x_label": labels["energy"],
            "scales": {"x": "log", "y": "log"},
        },
        "trigger_rates": {
            "x_label": labels["energy"],
            "scales": {"x": "log", "y": "log"},
        },
    }

    plots = {}
    for name, config in plot_config.items():
        for histo_type in histograms.histogram_types().values():
            histogram_key = f"{name}{histo_type['suffix']}"
            if histograms.get(histogram_key) is not None:
                plots[histogram_key] = _create_1d_plot_config(
                    histograms,
                    histogram_key=histogram_key,
                    config=config,
                    plot_params=hist_1d_params,
                    y_label=histo_type["ordinate"],
                    title=histo_type["title"],
                )
            else:
                _logger.warning(f"Histogram {histogram_key} not found.")

    return plots


def _create_1d_plot_config(histograms, histogram_key, config, plot_params, y_label, title):
    """Create a 1D plot configuration."""
    print(f"Creating plot config for {histogram_key} with params: {plot_params}")
    return {
        "data": histograms.get(histogram_key),
        "bins": histograms.get(f"{histogram_key}_bin_edges"),
        "plot_type": "histogram",
        "plot_params": plot_params,
        "labels": {
            "x": config["x_label"],
            "y": y_label,
            "title": f"{title}: {histogram_key.replace('_', ' ')}",
        },
        "scales": config["scales"],
        "lines": config.get("lines"),
        "filename": histogram_key,
    }


def _generate_2d_plots(histograms, labels, limits):
    """Generate 2D histogram plot configurations."""
    hist_2d_params = {"norm": "log", "cmap": "viridis", "show_contour": False}
    hist_2d_equal_params = {
        "norm": "log",
        "cmap": "viridis",
        "aspect": "equal",
        "show_contour": False,
    }
    hist_2d_normalized_params = {"norm": "linear", "cmap": "viridis", "show_contour": True}
    upper_radius_limit, lower_energy_limit, viewcone_radius = _get_limits(limits)
    triggered_events_type = "Triggered events"
    plot_config = {
        "core_vs_energy": {
            "event_type": triggered_events_type,
            "x_label": labels["core_distance"],
            "y_label": labels["energy"],
            "plot_params": hist_2d_params,
            "plot_params_normalized": hist_2d_normalized_params,
            "lines": {"x": upper_radius_limit, "y": lower_energy_limit},
            "scales": {"y": "log"},
        },
        "angular_distance_vs_energy": {
            "event_type": triggered_events_type,
            "x_label": labels["pointing_direction"],
            "y_label": labels["energy"],
            "plot_params": hist_2d_params,
            "plot_params_normalized": hist_2d_normalized_params,
            "lines": {"x": viewcone_radius, "y": lower_energy_limit},
            "scales": {"y": "log"},
        },
        "x_core_shower_vs_y_core_shower": {
            "event_type": triggered_events_type,
            "x_label": labels["core_x"],
            "y_label": labels["core_y"],
            "plot_params": hist_2d_equal_params,
            "plot_params_normalized": hist_2d_normalized_params,
            "lines": {"r": upper_radius_limit},
            "scales": {},
        },
    }

    plots = {}
    for name, config in plot_config.items():
        for histo_type in histograms.histogram_types().values():
            histogram_key = f"{name}{histo_type['suffix']}"
            if histograms.get(histogram_key) is not None:
                plots[histogram_key] = _create_2d_plot_config(
                    histograms,
                    histogram_key=histogram_key,
                    config=config,
                    z_label=histo_type["ordinate"],
                    title=histo_type["title"],
                )

    return plots


def _create_2d_plot_config(histograms, histogram_key, config, z_label, title):
    """Create a 2D plot configuration."""
    if "cumulative" in histogram_key or "efficiency" in histogram_key:
        plot_params = config["plot_params_normalized"]
    else:
        plot_params = config["plot_params"]

    return {
        "data": histograms.get(histogram_key),
        "bins": [
            histograms.get(f"{histogram_key}_bin_x_edges"),
            histograms.get(f"{histogram_key}_bin_y_edges"),
        ],
        "plot_type": "histogram2d",
        "plot_params": plot_params,
        "labels": {
            "x": config["x_label"],
            "y": config["y_label"],
            "title": f"{title}: {histogram_key.replace('_', ' ')}",
        },
        "lines": config["lines"],
        "scales": config.get("scales", {}),
        "colorbar_label": z_label,
        "filename": histogram_key,
    }


def _execute_plotting_loop(plots, output_path, rebin_factor, array_name):
    """Execute the main plotting loop for all plot configurations."""
    for plot_key, plot_args in plots.items():
        plot_filename = plot_args.pop("filename")

        if plot_args.get("data") is None:
            _logger.warning(f"Skipping plot {plot_key} - no data available")
            continue

        if array_name and plot_args.get("labels", {}).get("title"):
            plot_args["labels"]["title"] += f" ({array_name} array)"

        filename = _build_plot_filename(plot_filename, array_name)
        output_file = output_path / filename if output_path else None
        result = _create_plot(**plot_args, output_file=output_file)

        # Skip re-binned plot if main plot failed
        if result is None:
            continue

        if _should_create_rebinned_plot(rebin_factor, plot_args, plot_key):
            _create_rebinned_plot(plot_args, filename, output_path, rebin_factor)


def _build_plot_filename(base_filename, array_name=None):
    """
    Build the full plot filename with appropriate extensions.

    Parameters
    ----------
    base_filename : str
        The base filename without extension
    array_name : str, optional
        Name of the array to append to filename

    Returns
    -------
    str
        Complete filename with extension
    """
    return f"{base_filename}_{array_name}.png" if array_name else f"{base_filename}.png"


def _should_create_rebinned_plot(rebin_factor, plot_args, plot_key):
    """
    Check if a re-binned version of the plot should be created.

    Parameters
    ----------
    rebin_factor : int
        Factor by which to rebin the energy axis
    plot_args : dict
        Plot arguments
    plot_key : str
        Key identifying the plot type

    Returns
    -------
    bool
        True if a re-binned plot should be created, False otherwise
    """
    return (
        rebin_factor > 1
        and plot_args["plot_type"] == "histogram2d"
        and plot_key.endswith("_cumulative")
        and plot_args.get("plot_params", {}).get("norm") == "linear"
    )


def _create_rebinned_plot(plot_args, filename, output_path, rebin_factor):
    """
    Create a re-binned version of a 2D histogram plot.

    Parameters
    ----------
    plot_args : dict
        Plot arguments for the original plot
    filename : str
        Filename of the original plot
    output_path : Path or None
        Path to save the plot to, or None
    rebin_factor : int
        Factor by which to rebin the energy axis
    """
    data = plot_args["data"]
    bins = plot_args["bins"]

    rebinned_data, rebinned_x_bins, rebinned_y_bins = SimtelIOEventHistograms.rebin_2d_histogram(
        data, bins[0], bins[1], rebin_factor
    )

    rebinned_plot_args = plot_args.copy()
    rebinned_plot_args["data"] = rebinned_data
    rebinned_plot_args["bins"] = [rebinned_x_bins, rebinned_y_bins]

    if rebinned_plot_args.get("labels", {}).get("title"):
        rebinned_plot_args["labels"]["title"] += f" (Energy rebinned {rebin_factor}x)"

    rebinned_filename = f"{filename.replace('.png', '')}_rebinned.png"
    rebinned_output_file = output_path / rebinned_filename if output_path else None
    _create_plot(**rebinned_plot_args, output_file=rebinned_output_file)


def _create_plot(
    data,
    bins=None,
    plot_type="histogram",
    plot_params=None,
    labels=None,
    scales=None,
    colorbar_label=None,
    output_file=None,
    lines=None,
):
    """Create and save a plot with the given parameters."""
    plot_params = plot_params or {}
    labels = labels or {}
    scales = scales or {}
    lines = lines or {}

    if not _has_data(data):
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_data(ax, data, bins, plot_type, plot_params, colorbar_label)
    _add_lines(ax, lines)
    ax.set(
        xlabel=labels.get("x", ""),
        ylabel=labels.get("y", ""),
        title=labels.get("title", ""),
        xscale=scales.get("x", "linear"),
        yscale=scales.get("y", "linear"),
    )
    if output_file:
        _logger.info(f"Saving plot to {output_file}")
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return fig


def _has_data(data):
    """Check that the data for plotting is not None or empty."""
    if data is None or (isinstance(data, np.ndarray) and data.size == 0):
        _logger.warning("No data available for plotting")
        return False
    return True


def _plot_data(ax, data, bins, plot_type, plot_params, colorbar_label):
    """Plot the data on the given axes."""
    if plot_type == "histogram":
        ax.bar(bins[:-1], data, width=np.diff(bins), **plot_params)
    elif plot_type == "histogram2d":
        pcm = _create_2d_histogram_plot(data, bins, plot_params)
        plt.colorbar(pcm, label=colorbar_label)


def _add_lines(ax, lines):
    """Add reference lines to the plot."""
    if lines.get("x") is not None:
        ax.axvline(lines["x"], color="r", linestyle="--", linewidth=0.5)
    if lines.get("y") is not None:
        ax.axhline(lines["y"], color="r", linestyle="--", linewidth=0.5)
    if lines.get("r") is not None:
        ax.add_artist(
            plt.Circle((0, 0), lines["r"], color="r", fill=False, linestyle="--", linewidth=0.5)
        )


def _create_2d_histogram_plot(data, bins, plot_params):
    """
    Create a 2D histogram plot with the given parameters.

    Parameters
    ----------
    data : np.ndarray
        2D histogram data
    bins : tuple of np.ndarray
        Bin edges for x and y axes
    plot_params : dict
        Plot parameters including norm, cmap, and show_contour

    Returns
    -------
    matplotlib.collections.QuadMesh
        The created pcolormesh object for colorbar attachment
    """
    if plot_params.get("norm") == "linear":
        pcm = plt.pcolormesh(
            bins[0],
            bins[1],
            data.T,
            vmin=0,
            vmax=1,
            cmap=plot_params.get("cmap", "viridis"),
        )
        # Add contour line at value=1.0 for normalized histograms
        if plot_params.get("show_contour", True):
            x_centers = (bins[0][1:] + bins[0][:-1]) / 2
            y_centers = (bins[1][1:] + bins[1][:-1]) / 2
            x_mesh, y_mesh = np.meshgrid(x_centers, y_centers)
            plt.contour(
                x_mesh,
                y_mesh,
                data.T,
                levels=[0.999999],  # very close to 1 for floating point precision
                colors=["tab:red"],
                linestyles=["--"],
                linewidths=[0.5],
            )
    else:
        # Handle empty or invalid data for logarithmic scaling
        data_max = data.max()
        if data_max <= 0:
            _logger.warning("No positive data found for logarithmic scaling, using linear scale")
            pcm = plt.pcolormesh(
                bins[0], bins[1], data.T, vmin=0, vmax=max(1, data_max), cmap="viridis"
            )
        else:
            # Ensure vmin is less than vmax for LogNorm
            vmin = max(1, data[data > 0].min()) if np.any(data > 0) else 1
            vmax = max(vmin + 1, data_max)
            pcm = plt.pcolormesh(
                bins[0], bins[1], data.T, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis"
            )

    return pcm
