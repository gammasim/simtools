"""Plot simtel event histograms filled with EventDataHistograms."""

import logging
from pathlib import Path

import numpy as np

from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

_logger = logging.getLogger(__name__)
colors = lazy_module("matplotlib.colors")

# Histograms whose output filename differs from their key (triggered variants).
_PLOT_FILENAME_OVERRIDES = {
    name: f"{name}_triggered"
    for name in ("core_distance_vs_energy", "angular_distance_vs_energy", "energy")
}

# Histograms with fixed-coordinate projection slices, and their projection kind.
_PROJECTION_KINDS = {
    "core_distance_vs_energy": "core_distance",
    "angular_distance_vs_energy": "angular_distance",
}

# Reference lines per histogram: axis ("x", "y", "r" or "curve") -> limits key.
_LINE_LIMITS = {
    "energy": {"x": "lower_energy_limit"},
    "core_distance": {"x": "upper_radius_limit"},
    "angular_distance": {"x": "viewcone_radius"},
    "x_core_shower_vs_y_core_shower": {"r": "upper_radius_limit"},
    **{
        f"{prefix}{suffix}": {"x": x_key, "y": "lower_energy_limit", "curve": f"{prefix}_curve"}
        for prefix, x_key in (
            ("core_distance_vs_energy", "upper_radius_limit"),
            ("angular_distance_vs_energy", "viewcone_radius"),
        )
        for suffix in ("", "_cumulative")
    },
}

_AXIS_LABELS = {
    "energy": "Energy (TeV)",
    "core_distance": "Core Distance (m)",
    "angular_distance": "Angular Distance (deg)",
    "core_x": "Core X (m)",
    "core_y": "Core Y (m)",
}

_LOG_SCALES = {"x": "log", "y": "log"}

_HISTOGRAM_STYLES = {
    "cr_rates_mc": {
        "title": "Cosmic Ray Rates (MC)",
        "value_label": "Cosmic Ray Rate (Hz)",
        "scales": _LOG_SCALES,
    },
    "trigger_rates": {
        "title": "Trigger Rates (MC)",
        "value_label": "Trigger Rate (Hz)",
        "scales": _LOG_SCALES,
    },
    "*_mc": {"title": "Simulated Events"},
    "*_eff": {"title": "Efficiency", "value_label": "Efficiency"},
    "*_cumulative": {"title": "Cumulative triggered events", "value_label": "Fraction of Events"},
    "reuse_*": {"title": "Triggered reuse"},
    "*": {"title": "Triggered Events"},
}

_REUSE_VALUE_LABELS = {"mean": "Mean Reuse", "max": "Max Reuse", "std": "Reuse Std Dev"}

_HIST_1D_PARAMS = {"color": "tab:green", "edgecolor": "tab:green", "lw": 1}
_HIST_2D_PARAMS = {"norm": "log", "cmap": "viridis"}
_HIST_2D_NORMALIZED_PARAMS = {"norm": "linear", "cmap": "viridis"}


def plot(
    histograms,
    output_path=None,
    limits=None,
    array_name=None,
    add_distance_projections=False,
    use_broad_range_limits=False,
):
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
    add_distance_projections: bool, optional
        Add overall and sliced x/y projections to raw distance-vs-energy plots.
    use_broad_range_limits: bool, optional
        Restrict plot axes to the broad-range simulation limits in ``limits``.
    """
    _logger.info(f"Plotting histograms written to {output_path}")
    for name, histogram in histograms.items():
        if histogram.ndim > 2:
            _logger.warning(f"Skipping plot {name} - plotting supports only 1D and 2D histograms")
            continue
        config, base_filename = _build_plot_config(name, histogram, limits)
        if use_broad_range_limits:
            config["axis_limits"] = _get_broad_range_axis_limits(name, limits)
        projection_kind = _PROJECTION_KINDS.get(name.removesuffix("_mc"))
        if add_distance_projections and projection_kind:
            config["projection_kind"] = projection_kind
        if config["data"] is None:
            _logger.warning(f"Skipping plot {name} - no data available")
            continue
        if array_name:
            config["labels"]["title"] += f" ({array_name})"

        filename = _build_plot_filename(base_filename, array_name, limits)
        _create_plot(**config, output_file=output_path / filename if output_path else None)


def plot_trigger_histograms(loaded_histograms, output_dir):
    """
    Plot loaded trigger histograms, writing one directory per array layout.

    Parameters
    ----------
    loaded_histograms : iterable
        Loaded histogram records, each paired with its array layout.
    output_dir : str or pathlib.Path
        Base directory where plot files will be written.
    """
    output_dir = Path(output_dir)
    for _, record in loaded_histograms:
        output_path = output_dir / record.array_name
        output_path.mkdir(parents=True, exist_ok=True)
        plot({n: h for n, h in record.histograms.items() if h.ndim <= 2}, output_path=output_path)


def plot_monte_carlo_statistics_diagnostics(
    output_dir,
    array_name,
    file_info,
    energy_edges,
    angular_edges,
    expected_counts,
    relative_uncertainty,
):
    """
    Write Monte Carlo statistics diagnostic plots.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory to write plot files.
    array_name : str
        Array or telescope-selection name used in plot titles and filenames.
    file_info : dict, optional
        Simulation metadata (zenith, azimuth, nsb_level) to make filenames unique per setup.
    energy_edges : array-like
        Energy bin edges in TeV.
    angular_edges : array-like
        Angular-distance bin edges in deg.
    expected_counts : numpy.ndarray
        Expected triggered events per angular-distance and energy bin.
    relative_uncertainty : numpy.ndarray
        Poisson relative uncertainty per angular-distance and energy bin.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(array_name))
    file_info_suffix = _format_file_info_suffix(file_info or {})
    if file_info_suffix:
        file_prefix = f"{file_prefix}_{file_info_suffix}"
    expected_title = f"Expected triggered events ({array_name})"
    relative_title = f"Expected relative uncertainty ({array_name})"
    specs = [
        (expected_counts, "expected_events", expected_title, "Expected events", True),
        (
            relative_uncertainty,
            "relative_uncertainty",
            relative_title,
            "Relative uncertainty",
            False,
        ),
    ]
    for matrix, suffix, title, colorbar_label, mask_zero in specs:
        plot_matrix = np.ma.masked_invalid(np.asarray(matrix, dtype=float))
        if mask_zero:
            plot_matrix = np.ma.masked_less_equal(plot_matrix, 0.0)
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap("viridis").with_extremes(bad="white")
        log_energy_edges = np.log10(np.asarray(energy_edges, dtype=float))
        mesh = ax.pcolormesh(
            log_energy_edges, angular_edges, plot_matrix, shading="auto", cmap=cmap
        )
        fig.colorbar(mesh, ax=ax, label=colorbar_label)
        ax.set(xlabel="log10(E / TeV)", ylabel="Angular distance (deg)", title=title)
        output_file = output_dir / f"{file_prefix}_{suffix}.png"
        save_figure(fig, output_file.with_suffix(""), figure_format=["png"], dpi=150, close=True)


def _get_limit_value(limits, key):
    """Return a plain scalar for one limits entry."""
    item = limits.get(key)
    return getattr(item, "value", item)


def _get_broad_range_axis_limits(name, limits):
    """Return axis limits from broad-range simulation metadata."""
    if not limits:
        return {}
    core_max = _get_limit_value(limits, "br_core_scatter_max")
    viewcone_max = _get_limit_value(limits, "br_viewcone_max")
    lower = _get_limit_value(limits, "br_energy_min")
    upper = _get_limit_value(limits, "br_energy_max")
    derived = _get_limit_value(limits, "lower_energy_limit")
    # Leave room below the broad-range minimum when it coincides with the derived limit.
    if None not in (lower, upper, derived) and lower > 0.0 and np.isclose(derived, lower):
        lower *= 0.9
    symmetric_core = (-core_max, core_max) if core_max is not None else (None, None)
    axis_limits_by_group = {
        "energy": {"x": (lower, upper)},
        "core_distance_vs_energy": {"x": (0.0, core_max), "y": (lower, upper)},
        "angular_distance_vs_energy": {"x": (0.0, viewcone_max), "y": (lower, upper)},
        "core_distance": {"x": (0.0, core_max)},
        "angular_distance": {"x": (0.0, viewcone_max)},
        "x_core_shower_vs_y_core_shower": {"x": symmetric_core, "y": symmetric_core},
    }
    # Groups are matched by name prefix in table order; the first match wins.
    group = next((g for g in axis_limits_by_group if name.startswith(g)), None)
    return axis_limits_by_group.get(group, {})


def _get_limits(name, limits):
    """Return the reference-line positions of a histogram, or None if it has none."""
    sources = _LINE_LIMITS.get(name)
    if sources is None:
        return None
    return {
        axis: limits.get(key) if axis == "curve" else getattr(limits.get(key), "value", None)
        for axis, key in sources.items()
    }


def _style_key(name):
    """Return the key of the histogram style table that applies to a histogram name."""
    if name in ("cr_rates_mc", "trigger_rates"):
        return name
    suffix = next((x for x in ("_mc", "_eff", "_cumulative") if name.endswith(x)), None)
    if suffix:
        return f"*{suffix}"
    return "reuse_*" if name.startswith("reuse_") else "*"


def _reuse_default_scales(axis_names):
    """Return default axis scales for reuse histograms (log for energy axes)."""
    names_by_axis = dict(zip("xy", axis_names))
    return {axis: "log" if names_by_axis.get(axis) == "energy" else "linear" for axis in "xy"}


def _default_scales(name, histogram):
    """Return default axis scales derived from the histogram name and dimensionality."""
    axis_names = histogram.axis_names or []
    base_name = name
    for x in ("_mc", "_eff", "_cumulative"):
        base_name = base_name.removesuffix(x)
    if histogram.ndim == 1 and base_name == "energy":
        return _LOG_SCALES
    if histogram.ndim == 1 and base_name in {"core_distance", "angular_distance"}:
        return {"y": "log"}
    if histogram.ndim == 2 and "energy" in axis_names:
        return {"y": "log"}
    return {}


def _histogram_style(name, histogram):
    """Return title, axis labels, scales and value label selected by histogram name."""
    style_key = _style_key(name)
    style = _HISTOGRAM_STYLES[style_key]
    axis_names = histogram.axis_names or []
    axis_labels = [_AXIS_LABELS.get(axis, axis) for axis in axis_names]
    title, value_label = style["title"], style.get("value_label", "Event Count")
    if style_key == "reuse_*":
        statistic = name.split("_")[1]
        title = f"{title} {statistic}"
        value_label = _REUSE_VALUE_LABELS.get(statistic, value_label)
        default_scales = _reuse_default_scales(axis_names)
    else:
        default_scales = _default_scales(name, histogram)
    scales = dict(style.get("scales", default_scales))
    return title, axis_labels, scales, value_label


def _build_plot_config(name, histogram, limits):
    """
    Return the keyword arguments of _create_plot for one histogram.

    Returns
    -------
    tuple
        ``(config, base_filename)``; the base filename is not part of the config dict.
    """
    title, axis_labels, scales, value_label = _histogram_style(name, histogram)
    is_1d = histogram.ndim == 1
    lowered = name.lower()
    normalized = (
        lowered.startswith("reuse_") or "cumulative" in lowered or "efficiency" in lowered
    ) or lowered.endswith("_eff")
    config = {
        "data": histogram.values,
        "bins": histogram.edges[0] if is_1d else [histogram.edges[0], histogram.edges[1]],
        "plot_type": "histogram" if is_1d else "histogram2d",
        "plot_params": (
            _HIST_1D_PARAMS
            if is_1d
            else _HIST_2D_NORMALIZED_PARAMS
            if normalized
            else _HIST_2D_PARAMS
        ),
        "labels": {
            "x": axis_labels[0] if axis_labels else None,
            "y": value_label if is_1d else (axis_labels[1] if len(axis_labels) > 1 else None),
            "title": f"{title}: {name.replace('_', ' ')}",
        },
        "scales": scales,
        "lines": _get_limits(name, limits) if limits else {},
    }
    if not is_1d:
        config["colorbar_label"] = value_label
    return config, _PLOT_FILENAME_OVERRIDES.get(name, name)


def _build_plot_filename(base_filename, array_name=None, file_info=None):
    """Return the PNG filename built from the base name, array name and file-info suffix."""
    parts = [base_filename, array_name, _format_file_info_suffix(file_info or {})]
    return "_".join(part for part in parts if part) + ".png"


def _format_file_info_suffix(file_info):
    """Return a filename-safe suffix such as ``z20_az0_nsb0.3`` from file-info metadata."""
    parts = []
    for key, prefix in (("zenith", "z"), ("azimuth", "az")):
        value = file_info.get(key)
        if value is not None:
            parts.append(
                f"{prefix}{round(value.value if hasattr(value, 'value') else float(value))}"
            )
    if file_info.get("nsb_level") is not None:
        parts.append(f"nsb{file_info['nsb_level']:g}")
    return "_".join(parts)


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
    axis_limits=None,
    projection_kind=None,
):
    """Create and save (or show) a plot with the given parameters."""
    if data is None or (isinstance(data, np.ndarray) and data.size == 0):
        _logger.warning("No data available for plotting")
        return None
    plot_params, labels = plot_params or {}, labels or {}
    scales, lines, axis_limits = scales or {}, lines or {}, axis_limits or {}

    if plot_type == "histogram2d" and projection_kind:
        fig = plt.figure(figsize=(10.4, 6), constrained_layout=True)
        grid = fig.add_gridspec(4, 2, width_ratios=(3.6, 2.1), wspace=0.08, hspace=0.08)
        ax = fig.add_subplot(grid[:, 0])
        ax_x, ax_y = fig.add_subplot(grid[:2, 1]), fig.add_subplot(grid[2:, 1])
        pcm = _create_2d_histogram_plot(data, bins, plot_params, ax=ax)
        fig.colorbar(pcm, ax=ax, label=colorbar_label, pad=0.01)
        _plot_distance_projections(
            ax_x, ax_y, data, bins, labels, axis_limits, lines, projection_kind
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        if plot_type == "histogram":
            ax.bar(bins[:-1], data, width=np.diff(bins), **plot_params)
        elif plot_type == "histogram2d":
            plt.colorbar(_create_2d_histogram_plot(data, bins, plot_params), label=colorbar_label)

    _add_lines(ax, lines)
    ax.set(
        xlabel=labels.get("x", ""),
        ylabel=labels.get("y", ""),
        title=labels.get("title", ""),
        xscale=scales.get("x", "linear"),
        yscale=scales.get("y", "linear"),
    )
    _apply_axis_limits(ax, axis_limits)

    if not output_file:
        plt.tight_layout()
        plt.show()
        return fig
    output_file = Path(output_file)
    _logger.info(f"Saving plot to {output_file}")
    file_format = output_file.suffix[1:]
    save_figure(fig, output_file.with_suffix(""), figure_format=[file_format], dpi=300, close=True)
    return fig


def _apply_axis_limits(ax, axis_limits):
    """Apply finite, non-degenerate axis limits."""
    for axis_name, setter in (("x", ax.set_xlim), ("y", ax.set_ylim)):
        bounds = axis_limits.get(axis_name)
        if bounds and None not in bounds and np.isfinite(bounds).all() and bounds[1] > bounds[0]:
            setter(*bounds)


def _plot_distance_projections(ax_x, ax_y, data, bins, labels, axis_limits, lines, projection_kind):
    """Plot overall projections and fixed-coordinate slices of a 2D histogram."""
    x_centers = 0.5 * (bins[0][:-1] + bins[0][1:])
    y_centers = 0.5 * (bins[1][:-1] + bins[1][1:])
    ax_x.step(x_centers, np.sum(data, axis=1), where="mid", color="black", label="overall")
    ax_y.step(y_centers, np.sum(data, axis=0), where="mid", color="black", label="overall")

    for value, label in _slice_values(bins[1], axis_limits.get("y")):
        index = np.clip(np.searchsorted(bins[1], value, side="right") - 1, 0, len(bins[1]) - 2)
        if np.any(data[:, index] > 0):
            ax_x.step(x_centers, data[:, index], where="mid", label=label)

    for value, label in _slice_values(bins[0], axis_limits.get("x"), projection_kind):
        index = np.clip(np.searchsorted(bins[0], value, side="right") - 1, 0, len(bins[0]) - 2)
        if np.any(data[index, :] > 0):
            ax_y.step(y_centers, data[index, :], where="mid", label=label)

    count = "Event count"
    x_label, y_label = labels.get("x", ""), labels.get("y", "")
    ax_x.set(xlabel=x_label, ylabel=count, title="Projection distance axis", yscale="log")
    ax_y.set(
        xlabel=y_label, ylabel=count, title="Projection energy axis", xscale="log", yscale="log"
    )
    _apply_axis_limits(ax_x, {"x": axis_limits.get("x")})
    _apply_axis_limits(ax_y, {"x": axis_limits.get("y")})
    _add_lines(ax_x, {"x": lines.get("x")})
    _add_lines(ax_y, {"x": lines.get("y")})
    for ax in (ax_x, ax_y):
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, legend_labels, fontsize="x-small", loc="upper right")


def _slice_values(bin_edges, plot_limits, projection_kind=None):
    """Return fixed slice coordinates and labels (energy when ``projection_kind`` is None)."""
    lower = plot_limits[0] if plot_limits and plot_limits[0] is not None else bin_edges[0]
    upper = plot_limits[1] if plot_limits and plot_limits[1] is not None else bin_edges[-1]
    if projection_kind is None:
        if lower <= 0 or upper <= 0:
            return []
        exponents = np.arange(-1.5, np.log10(upper) + 1.0e-12, 1.0)
        return [
            (10.0**exponent, f"log10(E/TeV)={exponent:g}")
            for exponent in exponents
            if lower <= 10.0**exponent <= upper
        ]
    step = 500.0 if projection_kind == "core_distance" else 1.0
    unit = "m" if projection_kind == "core_distance" else "deg"
    values = np.arange(0.0, upper + 1.0e-12, step)
    return [(value, f"{value:g} {unit}") for value in values if lower <= value <= upper]


def _add_lines(ax, lines):
    """Add reference lines (circle, vertical, horizontal and curve) to the plot."""
    if lines.get("r") is not None:
        style = {"color": "r", "fill": False, "linestyle": "--", "linewidth": 0.5}
        ax.add_artist(plt.Circle((0, 0), lines["r"], **style))
    for axis, add_line in (("x", ax.axvline), ("y", ax.axhline)):
        for value in np.atleast_1d(lines.get(axis, [])):
            if value is not None:
                add_line(value, color="r", linestyle="--", linewidth=0.5)
    curve = lines.get("curve")
    if curve:
        curve_x = np.asarray(curve.get("x", []), dtype=float).ravel()
        curve_y = np.asarray(curve.get("y", []), dtype=float).ravel()
        if curve_x.size and curve_x.size == curve_y.size:
            marker = {"marker": "o", "markersize": 3.0} if curve_x.size == 1 else {"marker": "None"}
            ax.plot(curve_x, curve_y, color="r", linestyle="--", linewidth=1.0, **marker)


def _create_2d_histogram_plot(data, bins, plot_params, ax=None):
    """Draw a 2D histogram with linear or logarithmic color scaling and return its mesh."""
    cmap = plt.get_cmap(plot_params.get("cmap", "viridis")).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    plotter = ax if ax is not None else plt
    if plot_params.get("norm") == "linear":
        finite_values = np.asarray(data, dtype=float)[np.isfinite(data)]
        data_min = float(np.min(finite_values)) if finite_values.size else 0.0
        data_max = float(np.max(finite_values)) if finite_values.size else 1.0
        data_max = data_min + 1.0 if data_max <= data_min else data_max
        masked_data, norm_kwargs = (
            np.ma.masked_invalid(data.T),
            {"vmin": data_min, "vmax": data_max},
        )
    else:
        masked_data = np.ma.masked_less_equal(data.T, 0)
        data_max = data.max()
        if data_max <= 0:
            _logger.warning("No positive data found for logarithmic scaling, using linear scale")
            norm_kwargs = {"vmin": 0, "vmax": max(1, data_max)}
        else:
            # Ensure vmin is less than vmax for LogNorm
            vmin = max(1, data[data > 0].min()) if np.any(data > 0) else 1
            norm_kwargs = {"norm": colors.LogNorm(vmin=vmin, vmax=max(vmin + 1, data_max))}
    return plotter.pcolormesh(bins[0], bins[1], masked_data, cmap=cmap, **norm_kwargs)
