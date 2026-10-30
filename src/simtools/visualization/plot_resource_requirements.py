"""Plot normalized CORSIKA and sim_telarray resource requirements."""

import logging
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import pdf_report, save_figure

_ROLE_STYLE = {
    "sim_telarray": {"marker": "s", "zorder": 3},
    "corsika": {"marker": "o", "zorder": 4},
}
_BYTES_PER_MEGABYTE = 1_000_000
_BYTES_PER_GIGABYTE = 1_000_000_000
_RESOURCE_CHANGE_WARNING_FACTOR = 1.25
_RESOURCE_CHANGE_MAJOR_FACTOR = 1.5
_RATIO_CONFIGURATION_COLUMNS = ("primary", "site", "array_layout_name", "model_version")
_STORAGE = "sim_telarray storage"
_SIMTEL = "sim_telarray"
_CORSIKA = "corsika"
_SIMTEL_OUTPUT = "sim_telarray output"
_REDUCED = "reduced event data"
_HISTOGRAM = "sim_telarray histogram"
# (column, label, file name stem, process role or None for both roles)
_DIAGNOSTIC_PLOTS = (
    ("wall_time_seconds", "Wall time (s/run)", "resource_wall_time_run", None),
    ("cpu_time_seconds", "CPU time (s/run)", "resource_cpu_time_run", None),
    ("wall_time_seconds_per_event", "Wall time (s/event)", "resource_wall_time", None),
    ("cpu_time_seconds_per_event", "CPU time (s/event)", "resource_cpu_time", None),
    ("peak_rss_bytes", "Peak RSS", "resource_peak_rss", None),
    ("sim_telarray_storage_bytes", _STORAGE, "resource_storage_run", None),
    ("sim_telarray_storage_bytes_per_event", _STORAGE, "resource_storage", None),
    ("corsika_output_bytes", "CORSIKA output", "resource_corsika_output_run", _CORSIKA),
    ("corsika_output_bytes_per_event", "CORSIKA output", "resource_corsika_output", _CORSIKA),
    ("sim_telarray_output_bytes", _SIMTEL_OUTPUT, "resource_sim_telarray_output_run", _SIMTEL),
    (
        "sim_telarray_output_bytes_per_event",
        _SIMTEL_OUTPUT,
        "resource_sim_telarray_output",
        _SIMTEL,
    ),
    ("reduced_event_data_bytes", _REDUCED, "resource_reduced_event_data_run", _SIMTEL),
    ("reduced_event_data_bytes_per_event", _REDUCED, "resource_reduced_event_data", _SIMTEL),
    ("sim_telarray_histogram_bytes", _HISTOGRAM, "resource_sim_telarray_histogram_run", _SIMTEL),
    (
        "sim_telarray_histogram_bytes_per_event",
        _HISTOGRAM,
        "resource_sim_telarray_histogram",
        _SIMTEL,
    ),
    (
        "wall_time_seconds_per_triggered_event",
        "Wall time (s/triggered event)",
        "resource_wall_time_triggered",
        _SIMTEL,
    ),
    (
        "cpu_time_seconds_per_triggered_event",
        "CPU time (s/triggered event)",
        "resource_cpu_time_triggered",
        _SIMTEL,
    ),
    (
        "sim_telarray_storage_bytes_per_triggered_event",
        _STORAGE,
        "resource_storage_triggered",
        _SIMTEL,
    ),
    (
        "sim_telarray_output_bytes_per_triggered_event",
        _SIMTEL_OUTPUT,
        "resource_sim_telarray_output_triggered",
        _SIMTEL,
    ),
    (
        "reduced_event_data_bytes_per_triggered_event",
        _REDUCED,
        "resource_reduced_event_data_triggered",
        _SIMTEL,
    ),
    (
        "sim_telarray_histogram_bytes_per_triggered_event",
        _HISTOGRAM,
        "resource_sim_telarray_histogram_triggered",
        _SIMTEL,
    ),
)
_COMPACT_PLOTS = (
    ("wall_time_seconds_per_event", "Wall time (s/event)"),
    ("cpu_time_seconds_per_event", "CPU time (s/event)"),
    ("peak_rss_bytes", "Peak RSS (bytes)"),
    ("sim_telarray_storage_bytes_per_event", "Storage (bytes/event)"),
)
_logger = logging.getLogger(__name__)


def plot(rows, output_path, figure_format=None, report_mode="diagnostic"):
    """Write resource plots versus energy.

    Parameters
    ----------
    rows : iterable[dict]
        Normalized process rows returned by the resource collector.
    output_path : str or pathlib.Path
        Destination directory for the figures.
    figure_format : iterable[str], optional
        File formats passed to :func:`visualization.visualize.save_figure`.
    report_mode : str, optional
        ``"diagnostic"`` writes one figure per metric and role, ``"compact"`` a
        multipage PDF, and ``"none"`` writes nothing.

    Returns
    -------
    list[pathlib.Path]
        Base paths of the written figures.

    Notes
    -----
    Byte-based quantities are plotted in MB or GB, selected from the largest
    value in each plot, using decimal units. Per-run plots use the ``*_bytes``
    and raw time columns; per-event and per-triggered-event plots use their
    normalized columns. When both baseline and candidate rows are present,
    additional plots show candidate-to-baseline ratios for matching primary,
    site, layout, and model-version configurations.
    """
    if report_mode not in {"compact", "diagnostic", "none"}:
        raise ValueError(f"Unsupported report mode: {report_mode}")
    rows = list(rows)
    output_path = Path(output_path)
    if report_mode == "none":
        return []
    if report_mode == "compact":
        return _write_compact_report(rows, output_path)
    return _write_diagnostic_plots(rows, output_path, figure_format)


def _write_diagnostic_plots(rows, output_path, figure_format):
    """Write absolute and ratio figures for the configured resource metrics."""
    output_files = []
    for column, label, filename, required_role in _DIAGNOSTIC_PLOTS:
        available = [
            row
            for row in rows
            if row.get(column) is not None
            and row.get("energy_midpoint_gev") is not None
            and row["energy_midpoint_gev"] > 0
            and (required_role is None or row["role"] == required_role)
            and (not column.startswith("sim_telarray_storage") or row["role"] == "sim_telarray")
        ]
        if not available:
            continue
        plot_label, value_scale = _byte_plot_label(column, label, available)
        zenith_colors = _zenith_colors(
            sorted({float(row.get("zenith_angle_deg", 0.0)) for row in available})
        )
        for role in sorted({row["role"] for row in available}):
            role_rows = [row for row in available if row["role"] == role]
            fig, axis = plt.subplots(figsize=(8, 5))
            _plot_role(axis, role_rows, column, zenith_colors, value_scale=value_scale)
            axis.set_yscale(_resource_y_scale(role_rows, column))
            _finish_axis(axis, plot_label, f"{plot_label}: {role}")
            output_file = output_path / f"{filename}_{role}"
            save_figure(fig, output_file, figure_format=figure_format, dpi=300, close=True)
            output_files.append(output_file)
            ratio_output = _write_ratio_plot(
                role_rows,
                output_path,
                column,
                filename,
                plot_label,
                zenith_colors,
                role,
                figure_format,
            )
            if ratio_output is not None:
                output_files.append(ratio_output)
    return output_files


def _write_ratio_plot(
    rows, output_path, column, filename, plot_label, zenith_colors, role, figure_format
):
    """Write a candidate-to-baseline ratio plot when both roles have valid data."""
    labels = _comparison_display_labels(rows)
    ratio_series = _ratio_series(rows, column) if {"baseline", "candidate"} <= set(labels) else {}
    if not ratio_series:
        return None
    fig, axis = plt.subplots(figsize=(8, 5))
    _plot_ratio(axis, ratio_series, role, zenith_colors)
    ratio_label = f"{labels['candidate']} / {labels['baseline']}"
    _warn_on_large_changes(ratio_series, plot_label, role, ratio_label)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    _finish_axis(axis, ratio_label, f"{plot_label} ratio: {ratio_label}: {role}")
    output_file = output_path / f"{filename}_ratio_{role}"
    save_figure(fig, output_file, figure_format=figure_format, dpi=300, close=True)
    return output_file


def _write_compact_report(rows, output_path):
    """Write a short multipage resource report without diagnostic image files."""
    output_file = output_path / "resources-report.pdf"
    figures = []
    for column, label in _COMPACT_PLOTS:
        available = [row for row in rows if row.get(column) is not None]
        if not available:
            continue
        roles = sorted({row["role"] for row in available})
        fig, axes = plt.subplots(1, len(roles), figsize=(9, 5), squeeze=False)
        for role, axis in zip(roles, axes[0]):
            role_rows = [row for row in available if row["role"] == role]
            zeniths = sorted({float(row.get("zenith_angle_deg", 0.0)) for row in role_rows})
            _plot_role(axis, role_rows, column, _zenith_colors(zeniths))
            axis.set_yscale(_resource_y_scale(role_rows, column))
            _finish_axis(axis, label, role, fontsize=6)
        fig.suptitle(label)
        fig.tight_layout()
        figures.append(fig)
    with pdf_report(output_file) as report:
        for figure in figures:
            report.add(figure)
    return [output_file]


def _finish_axis(axis, ylabel, title, **legend_kwargs):
    """Apply the shared log-energy axis scale, labels, grid, and legend."""
    axis.set_xscale("log")
    axis.set_xlabel("Energy midpoint (GeV)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(**legend_kwargs)


def _resource_y_scale(rows, column):
    """Return a y-axis scale that can represent all selected resource values."""
    return "linear" if any(float(row[column]) <= 0 for row in rows) else "log"


def _byte_plot_label(column, label, rows):
    """Return a human-readable label and scale factor; byte columns use MB or GB."""
    if "_bytes" not in column:
        return label, 1.0
    maximum = max(float(row[column]) for row in rows)
    if maximum >= _BYTES_PER_GIGABYTE:
        unit, scale = "GB", 1 / _BYTES_PER_GIGABYTE
    else:
        unit, scale = "MB", 1 / _BYTES_PER_MEGABYTE
    if column == "peak_rss_bytes":
        suffix = ""
    elif column.endswith("_per_triggered_event"):
        suffix = "/triggered event"
    elif column.endswith("_bytes"):
        suffix = "/run"
    else:
        suffix = "/event"
    return f"{label} ({unit}{suffix})", scale


def _ratio_series(rows, column):
    """Return candidate-to-baseline ratios grouped by configuration, zenith, and energy."""
    values = {}
    for row in rows:
        production = _comparison_role(row)
        if production not in {"baseline", "candidate"}:
            continue
        key = (
            production,
            _ratio_configuration_key(row),
            float(row.get("zenith_angle_deg", 0.0)),
            row["energy_midpoint_gev"],
        )
        values.setdefault(key, []).append(float(row[column]))

    series = {}
    for configuration, zenith in {(key[1], key[2]) for key in values}:
        baseline = _statistics_by_energy(values, "baseline", configuration, zenith)
        candidate = _statistics_by_energy(values, "candidate", configuration, zenith)
        points = []
        for energy in sorted(set(baseline) & set(candidate)):
            baseline_mean, baseline_rms, baseline_count = baseline[energy]
            candidate_mean, candidate_rms, candidate_count = candidate[energy]
            if baseline_mean <= 0 or candidate_mean <= 0:
                continue
            ratio = candidate_mean / baseline_mean
            baseline_error = baseline_rms / np.sqrt(baseline_count)
            candidate_error = candidate_rms / np.sqrt(candidate_count)
            ratio_error = ratio * np.sqrt(
                (candidate_error / candidate_mean) ** 2 + (baseline_error / baseline_mean) ** 2
            )
            points.append((energy, ratio, ratio_error))
        if points:
            series[(configuration, zenith)] = points
    return series


def _ratio_configuration_key(row):
    """Return configuration dimensions used to match ratio series."""
    return tuple(
        "unknown" if row.get(column) is None else str(row[column])
        for column in _RATIO_CONFIGURATION_COLUMNS
    )


def _comparison_role(row):
    """Return the stable baseline/candidate role for a resource row."""
    return row.get("comparison_role") or row.get("production_label")


def _comparison_display_labels(rows):
    """Return display labels for baseline and candidate rows."""
    labels = {}
    for row in rows:
        comparison_role = _comparison_role(row)
        if comparison_role in {"baseline", "candidate"}:
            labels.setdefault(comparison_role, row.get("production_label", comparison_role))
    return labels


def _warn_on_large_changes(series, plot_label, role, ratio_label):
    """Warn once for the largest substantial candidate-to-baseline change."""
    varying_dimensions = _ratio_varying_dimensions(series)
    changes = [
        (configuration, zenith, energy, ratio)
        for (configuration, zenith), points in series.items()
        for energy, ratio, _ in points
        if np.isfinite(ratio)
        and ratio > 0
        and max(ratio, 1 / ratio) >= _RESOURCE_CHANGE_WARNING_FACTOR
    ]
    if not changes:
        return
    configuration, zenith, energy, ratio = max(changes, key=lambda item: max(item[3], 1 / item[3]))
    factor = max(ratio, 1 / ratio)
    _logger.warning(
        "%s resource change in %s for %s at %s and energy %.6g GeV: %s shows a %.3g-fold %s.",
        "Major" if factor >= _RESOURCE_CHANGE_MAJOR_FACTOR else "Large",
        plot_label,
        role,
        _ratio_series_label(configuration, zenith, varying_dimensions),
        energy,
        ratio_label,
        factor,
        "increase" if ratio >= 1 else "decrease",
    )


def _statistics_by_energy(values, production, configuration, zenith):
    """Return means, RMS spreads, and sample counts keyed by energy."""
    return {
        energy: (
            float(np.mean(samples)),
            float(np.sqrt(np.mean((np.asarray(samples) - np.mean(samples)) ** 2))),
            len(samples),
        )
        for (label, sample_configuration, sample_zenith, energy), samples in values.items()
        if label == production and sample_configuration == configuration and sample_zenith == zenith
    }


def _plot_ratio(axis, series, role, zenith_colors):
    """Plot candidate-to-baseline ratios with propagated mean errors."""
    varying_dimensions = _ratio_varying_dimensions(series)
    style = _ROLE_STYLE.get(role, {})
    for series_index, ((configuration, zenith), points) in enumerate(sorted(series.items())):
        energies, ratios, errors = zip(*points, strict=True)
        axis.errorbar(
            energies,
            ratios,
            yerr=errors,
            fmt=style.get("marker", "o"),
            label=_ratio_series_label(configuration, zenith, varying_dimensions),
            color=zenith_colors[zenith],
            markerfacecolor=zenith_colors[zenith],
            markeredgecolor="black",
            linestyle=("-", "--", ":", "-.")[series_index % 4],
            markersize=8,
            capsize=3,
            linewidth=1.0,
            zorder=style.get("zorder", 3) + 1,
        )
    bounds = [
        bound
        for points in series.values()
        for _, ratio, error in points
        for bound in (ratio - error, ratio + error)
    ]
    lower, upper = min([*bounds, 1.0]), max([*bounds, 1.0])
    margin = max((upper - lower) * 0.1, 0.1)
    axis.set_ylim(max(0.0, lower - margin), upper + margin)
    axis.set_yscale("linear")


def _ratio_varying_dimensions(series):
    """Return configuration dimensions that differ between ratio series."""
    configurations = {configuration for configuration, _ in series}
    return tuple(
        index
        for index in range(len(_RATIO_CONFIGURATION_COLUMNS))
        if len({configuration[index] for configuration in configurations}) > 1
    )


def _ratio_series_label(configuration, zenith, varying_dimensions):
    """Return a compact label for a configuration-matched ratio series."""
    if not varying_dimensions:
        return f"za={zenith:g} deg"
    configuration_label = ", ".join(
        f"{_RATIO_CONFIGURATION_COLUMNS[index]}={configuration[index]}"
        for index in varying_dimensions
    )
    return f"{configuration_label}; za={zenith:g} deg"


def _series_key(row):
    """Return the identity of one plotted series (production, version, production id, zenith)."""
    return (
        row.get("production_label", "production"),
        row.get("model_version", row.get("simtools_version", "unknown")),
        row.get("production_id", "unknown"),
        float(row.get("zenith_angle_deg", 0.0)),
    )


def _plot_role(axis, rows, column, zenith_colors, value_scale=1.0):
    """Plot individual samples and averages for one process role."""
    keys = sorted({_series_key(row) for row in rows})
    role = rows[0]["role"]
    show_production_labels = any(row.get("production_label") for row in rows)
    comparison_roles = {_comparison_role(row) for row in rows}
    is_comparison = {"baseline", "candidate"} <= comparison_roles
    seen_series = set()
    for series_index, key in enumerate(keys):
        production, _, _, zenith = key
        series_rows = [row for row in rows if _series_key(row) == key]
        series_id = (production, zenith)
        if series_id in seen_series:
            legend_label = "_nolegend_"
        elif is_comparison or (show_production_labels and "baseline" not in comparison_roles):
            legend_label = f"{production}: za={zenith:g} deg"
        else:
            legend_label = f"za={zenith:g} deg"
        seen_series.add(series_id)
        _plot_averages(
            axis,
            series_rows,
            column,
            role,
            zenith_colors[zenith],
            legend_label,
            series_index,
            value_scale,
            _comparison_role(series_rows[0]),
        )


def _plot_averages(
    axis,
    rows,
    column,
    role,
    color,
    legend_label,
    series_index=0,
    value_scale=1.0,
    comparison_role=None,
):
    """Overlay one mean and RMS error bar for each energy in a series."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["energy_midpoint_gev"], []).append(row[column] * value_scale)
    style = _ROLE_STYLE.get(role, {})
    energies = sorted(grouped)
    means = [float(np.mean(grouped[energy])) for energy in energies]
    rms = [
        float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))
        for values in (np.asarray(grouped[energy], dtype=float) for energy in energies)
    ]
    axis.errorbar(
        energies,
        means,
        yerr=rms,
        fmt=style.get("marker", "o"),
        label=legend_label,
        color=color,
        markerfacecolor=color,
        markeredgecolor="black",
        linestyle={"baseline": "-", "candidate": "--"}.get(
            comparison_role, ("-", "--", ":", "-.")[series_index % 4]
        ),
        markersize=8,
        capsize=3,
        linewidth=1.0,
        zorder=style.get("zorder", 3) + 1,
    )


def _zenith_colors(zenith_values):
    """Return stable colors for the zenith values present in one plot."""
    colormap = colormaps["viridis"]
    denominator = max(len(zenith_values) - 1, 1)
    return {zenith: colormap(index / denominator) for index, zenith in enumerate(zenith_values)}
