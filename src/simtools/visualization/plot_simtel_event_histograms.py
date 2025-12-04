"""Plot simtel event histograms filled with EventDataHistograms."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from simtools.sim_events.histograms import EventDataHistograms

_logger = logging.getLogger(__name__)


def plot(histograms, output_path=None, limits=None, rebin_factor=2, array_name=None):
    """
    Plot simtel event histograms.

    Parameters
    ----------
    histograms: EventDataHistograms
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


def _get_limits(name, limits):
    """
    Extract limits from the provided dictionary for plotting.

    Fine tuned to expected histograms to be plotted.
    """

    def _safe_value(limits, key):
        val = limits.get(key)
        return getattr(val, "value", None)

    mapping = {
        "energy": {"x": _safe_value(limits, "lower_energy_limit")},
        "core_distance": {"x": _safe_value(limits, "upper_radius_limit")},
        "angular_distance": {"x": _safe_value(limits, "viewcone_radius")},
        "core_vs_energy": {
            "x": _safe_value(limits, "upper_radius_limit"),
            "y": _safe_value(limits, "lower_energy_limit"),
        },
        "angular_distance_vs_energy": {
            "x": _safe_value(limits, "viewcone_radius"),
            "y": _safe_value(limits, "lower_energy_limit"),
        },
        "x_core_shower_vs_y_core_shower": {"r": _safe_value(limits, "upper_radius_limit")},
    }
    return mapping.get(name)


def _generate_plot_configurations(histograms, limits):
    """Generate plot configurations for all histogram types."""
    hist_1d_params = {"color": "tab:green", "edgecolor": "tab:green", "lw": 1}
    hist_2d_params = {"norm": "log", "cmap": "viridis", "show_contour": False}
    hist_2d_normalized_params = {"norm": "linear", "cmap": "viridis", "show_contour": True}
    plots = {}
    for name, hist in histograms.items():
        if hist["histogram"] is None:
            continue
        if hist["1d"]:
            plots[name] = _create_1d_plot_config(
                hist, name=name, plot_params=hist_1d_params, limits=limits
            )
        else:
            if "cumulative" in name or "efficiency" in name:
                plot_params = hist_2d_normalized_params
            else:
                plot_params = hist_2d_params

            plots[name] = _create_2d_plot_config(
                hist, name=name, plot_params=plot_params, limits=limits
            )
    return plots


def _get_axis_title(axis_titles, axis):
    """Return axis title for given axis."""
    if axis_titles is None:
        return None
    if axis == "x" and len(axis_titles) > 0:
        return axis_titles[0]
    if axis == "y" and len(axis_titles) > 1:
        return axis_titles[1]
    if axis == "z" and len(axis_titles) > 2:
        return axis_titles[2]
    return None


def _create_1d_plot_config(histogram, name, plot_params, limits):
    """Create a 1D plot configuration."""
    _logger.debug(f"Creating plot config for {name} with params: {plot_params}")
    return {
        "data": histogram["histogram"],
        "bins": histogram["bin_edges"],
        "plot_type": "histogram",
        "plot_params": plot_params,
        "labels": {
            "x": _get_axis_title(histogram.get("axis_titles"), "x"),
            "y": _get_axis_title(histogram.get("axis_titles"), "y"),
            "title": f"{histogram['title']}: {name.replace('_', ' ')}",
        },
        "scales": histogram["plot_scales"],
        "lines": _get_limits(name, limits) if limits else {},
        "filename": name,
    }


def _create_2d_plot_config(histogram, name, plot_params, limits):
    """Create a 2D plot configuration."""
    _logger.debug(f"Creating plot config for {name} with params: {plot_params}")
    return {
        "data": histogram["histogram"],
        "bins": [histogram["bin_edges"][0], histogram["bin_edges"][1]],
        "plot_type": "histogram2d",
        "plot_params": plot_params,
        "labels": {
            "x": _get_axis_title(histogram.get("axis_titles"), "x"),
            "y": _get_axis_title(histogram.get("axis_titles"), "y"),
            "title": f"{histogram['title']}: {name.replace('_', ' ')}",
        },
        "lines": _get_limits(name, limits) if limits else {},
        "scales": histogram["plot_scales"],
        "colorbar_label": _get_axis_title(histogram.get("axis_titles"), "z"),
        "filename": name,
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

    rebinned_data, rebinned_x_bins, rebinned_y_bins = EventDataHistograms.rebin_2d_histogram(
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
