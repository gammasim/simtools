"""Plot simtel event histograms filled with EventDataHistograms."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_logger = logging.getLogger(__name__)

# Maps histogram dictionary keys to output plot filenames where the key alone
# is ambiguous (e.g. the triggered 2-D vs-energy histograms share a prefix
# with their _mc and _cumulative counterparts). Only the triggered variants are
# remapped; all other names fall through unchanged.
_PLOT_FILENAME_OVERRIDES = {
    "core_distance_vs_energy": "core_distance_vs_energy_triggered",
    "angular_distance_vs_energy": "angular_distance_vs_energy_triggered",
    "energy": "energy_triggered",
}


def plot(histograms, output_path=None, limits=None, array_name=None):
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
    array_name: str, optional
        Name of the telescope array configuration.
    """
    _logger.info(f"Plotting histograms written to {output_path}")

    file_info = _extract_file_info_from_limits(limits)
    plots = _generate_plot_configurations(histograms, limits)
    _execute_plotting_loop(plots, output_path, array_name, file_info)


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
        "core_distance_vs_energy": {
            "x": _safe_value(limits, "upper_radius_limit"),
            "y": _safe_value(limits, "lower_energy_limit"),
            "curve": limits.get("core_distance_vs_energy_curve"),
        },
        "core_distance_vs_energy_cumulative": {
            "x": _safe_value(limits, "upper_radius_limit"),
            "y": _safe_value(limits, "lower_energy_limit"),
            "curve": limits.get("core_distance_vs_energy_curve"),
        },
        "angular_distance_vs_energy": {
            "x": _safe_value(limits, "viewcone_radius"),
            "y": _safe_value(limits, "lower_energy_limit"),
            "curve": limits.get("angular_distance_vs_energy_curve"),
        },
        "angular_distance_vs_energy_cumulative": {
            "x": _safe_value(limits, "viewcone_radius"),
            "y": _safe_value(limits, "lower_energy_limit"),
            "curve": limits.get("angular_distance_vs_energy_curve"),
        },
        "x_core_shower_vs_y_core_shower": {"r": _safe_value(limits, "upper_radius_limit")},
    }
    return mapping.get(name)


def _generate_plot_configurations(histograms, limits):
    """Generate plot configurations for all histogram types."""
    hist_1d_params = {"color": "tab:green", "edgecolor": "tab:green", "lw": 1}
    hist_2d_params = {"norm": "log", "cmap": "viridis"}
    hist_2d_normalized_params = {"norm": "linear", "cmap": "viridis"}
    plots = {}
    for name, hist in histograms.items():
        if hist["histogram"] is None:
            continue
        if hist["1d"]:
            plots[name] = _create_1d_plot_config(
                hist, name=name, plot_params=hist_1d_params, limits=limits
            )
        else:
            histogram_name = name.lower()
            if (
                "cumulative" in histogram_name
                or "efficiency" in histogram_name
                or histogram_name.endswith("_eff")
            ):
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
        "filename": _PLOT_FILENAME_OVERRIDES.get(name, name),
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
        "filename": _PLOT_FILENAME_OVERRIDES.get(name, name),
    }


def _execute_plotting_loop(plots, output_path, array_name, file_info=None):
    """Execute the main plotting loop for all plot configurations."""
    for plot_key, plot_args in plots.items():
        plot_filename = plot_args.pop("filename")

        if plot_args.get("data") is None:
            _logger.warning(f"Skipping plot {plot_key} - no data available")
            continue

        if array_name and plot_args.get("labels", {}).get("title"):
            plot_args["labels"]["title"] += f" ({array_name} array)"

        filename = _build_plot_filename(plot_filename, array_name, file_info)
        output_file = output_path / filename if output_path else None
        _create_plot(**plot_args, output_file=output_file)


def _build_plot_filename(base_filename, array_name=None, file_info=None):
    """
    Build the full plot filename with appropriate extensions.

    Parameters
    ----------
    base_filename : str
        The base filename without extension.
    array_name : str, optional
        Name of the array to append to filename.
    file_info : dict, optional
        Dictionary with simulation metadata (zenith, azimuth, nsb_level) to
        include in the filename so each production's plots can be identified.

    Returns
    -------
    str
        Complete filename with extension.
    """
    parts = [base_filename]
    if array_name:
        parts.append(array_name)
    if file_info:
        suffix = _format_file_info_suffix(file_info)
        if suffix:
            parts.append(suffix)
    return "_".join(parts) + ".png"


def _format_file_info_suffix(file_info):
    """
    Build a filename-safe suffix string from file-info simulation metadata.

    Parameters
    ----------
    file_info : dict
        Dictionary with optional keys ``zenith``, ``azimuth`` (astropy
        quantities in degrees) and ``nsb_level`` (float).

    Returns
    -------
    str
        Underscore-joined suffix components, e.g. ``z20_az0_nsb0.3``.
        Returns an empty string if no metadata is present.
    """
    parts = []
    zenith = file_info.get("zenith")
    if zenith is not None:
        zenith_val = zenith.value if hasattr(zenith, "value") else float(zenith)
        parts.append(f"z{round(zenith_val)}")
    azimuth = file_info.get("azimuth")
    if azimuth is not None:
        azimuth_val = azimuth.value if hasattr(azimuth, "value") else float(azimuth)
        parts.append(f"az{round(azimuth_val)}")
    nsb = file_info.get("nsb_level")
    if nsb is not None:
        parts.append(f"nsb{nsb:g}")
    return "_".join(parts)


def _extract_file_info_from_limits(limits):
    """
    Extract simulation file-info metadata from the limits dictionary.

    Parameters
    ----------
    limits : dict or None
        Dictionary returned by the limit-computation step; may contain
        ``zenith``, ``azimuth``, and ``nsb_level`` keys.

    Returns
    -------
    dict
        Subset of *limits* with only the file-info metadata keys that are
        present and non-None.
    """
    if not limits:
        return {}
    keys = ("zenith", "azimuth", "nsb_level")
    return {k: limits[k] for k in keys if limits.get(k) is not None}


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
    if lines.get("r") is not None:
        ax.add_artist(
            plt.Circle((0, 0), lines["r"], color="r", fill=False, linestyle="--", linewidth=0.5)
        )

    for x_value in np.atleast_1d(lines.get("x", [])):
        ax.axvline(x_value, color="r", linestyle="--", linewidth=0.5)

    for y_value in np.atleast_1d(lines.get("y", [])):
        ax.axhline(y_value, color="r", linestyle="--", linewidth=0.5)

    curve = lines.get("curve")
    if curve and curve.get("x") and curve.get("y"):
        ax.plot(curve["x"], curve["y"], color="tab:orange", linestyle="-", linewidth=1.0)


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
        Plot parameters including norm, cmap

    Returns
    -------
    matplotlib.collections.QuadMesh
        The created pcolormesh object for colorbar attachment
    """
    cmap = plt.get_cmap(plot_params.get("cmap", "viridis")).copy()
    cmap.set_bad(color="white", alpha=0.0)

    if plot_params.get("norm") == "linear":
        masked_data = np.ma.masked_equal(data.T, 0)
        pcm = plt.pcolormesh(
            bins[0],
            bins[1],
            masked_data,
            vmin=0,
            vmax=1,
            cmap=cmap,
        )
    else:
        masked_data = np.ma.masked_less_equal(data.T, 0)
        # Handle empty or invalid data for logarithmic scaling
        data_max = data.max()
        if data_max <= 0:
            _logger.warning("No positive data found for logarithmic scaling, using linear scale")
            pcm = plt.pcolormesh(
                bins[0], bins[1], masked_data, vmin=0, vmax=max(1, data_max), cmap=cmap
            )
        else:
            # Ensure vmin is less than vmax for LogNorm
            vmin = max(1, data[data > 0].min()) if np.any(data > 0) else 1
            vmax = max(vmin + 1, data_max)
            pcm = plt.pcolormesh(
                bins[0], bins[1], masked_data, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap
            )

    return pcm
