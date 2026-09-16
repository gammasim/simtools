"""Plot normalized CORSIKA and sim_telarray resource requirements."""

import logging
from pathlib import Path

import numpy as np
from matplotlib import colormaps

from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

_ROLE_STYLE = {
    "sim_telarray": {"marker": "s", "zorder": 3},
    "corsika": {"marker": "o", "zorder": 4},
}
_BYTES_PER_MEGABYTE = 1_000_000
_BYTES_PER_GIGABYTE = 1_000_000_000
_RESOURCE_CHANGE_WARNING_FACTOR = 1.25
_RESOURCE_CHANGE_MAJOR_FACTOR = 1.5
_BYTE_PLOT_COLUMNS = frozenset(
    {
        "peak_rss_bytes",
        "sim_telarray_storage_bytes_per_event",
        "corsika_output_bytes_per_event",
        "sim_telarray_output_bytes_per_event",
        "reduced_event_data_bytes_per_event",
        "sim_telarray_histogram_bytes_per_event",
        "sim_telarray_storage_bytes_per_triggered_event",
        "sim_telarray_output_bytes_per_triggered_event",
        "reduced_event_data_bytes_per_triggered_event",
        "sim_telarray_histogram_bytes_per_triggered_event",
    }
)
_logger = logging.getLogger(__name__)


def plot(rows, output_path, figure_format=None):
    """Write resource plots versus energy.

    Parameters
    ----------
    rows : iterable[dict]
        Normalized process rows returned by the resource collector.
    output_path : str or pathlib.Path
        Destination directory for the figures.
    figure_format : iterable[str], optional
        File formats passed to :func:`visualization.visualize.save_figure`.

    Returns
    -------
    list[pathlib.Path]
        Base paths of the written figures.

    Notes
    -----
    Byte-based quantities are plotted in MB or GB, selected from the largest
    value in each plot, using decimal units. The input rows and resource table
    retain byte values. When both baseline and candidate rows are present,
    additional plots show candidate-to-baseline ratios.
    """
    rows = list(rows)
    output_path = Path(output_path)
    plots = (
        ("wall_time_seconds_per_event", "Wall time (s/event)", "resource_wall_time", None),
        ("cpu_time_seconds_per_event", "CPU time (s/event)", "resource_cpu_time", None),
        ("peak_rss_bytes", "Peak RSS", "resource_peak_rss", None),
        (
            "sim_telarray_storage_bytes_per_event",
            "sim_telarray storage",
            "resource_storage",
            None,
        ),
        (
            "corsika_output_bytes_per_event",
            "CORSIKA output",
            "resource_corsika_output",
            "corsika",
        ),
        (
            "sim_telarray_output_bytes_per_event",
            "sim_telarray output",
            "resource_sim_telarray_output",
            "sim_telarray",
        ),
        (
            "reduced_event_data_bytes_per_event",
            "reduced event data",
            "resource_reduced_event_data",
            "sim_telarray",
        ),
        (
            "sim_telarray_histogram_bytes_per_event",
            "sim_telarray histogram",
            "resource_sim_telarray_histogram",
            "sim_telarray",
        ),
        (
            "wall_time_seconds_per_triggered_event",
            "Wall time (s/triggered event)",
            "resource_wall_time_triggered",
            "sim_telarray",
        ),
        (
            "cpu_time_seconds_per_triggered_event",
            "CPU time (s/triggered event)",
            "resource_cpu_time_triggered",
            "sim_telarray",
        ),
        (
            "sim_telarray_storage_bytes_per_triggered_event",
            "sim_telarray storage",
            "resource_storage_triggered",
            "sim_telarray",
        ),
        (
            "sim_telarray_output_bytes_per_triggered_event",
            "sim_telarray output",
            "resource_sim_telarray_output_triggered",
            "sim_telarray",
        ),
        (
            "reduced_event_data_bytes_per_triggered_event",
            "reduced event data",
            "resource_reduced_event_data_triggered",
            "sim_telarray",
        ),
        (
            "sim_telarray_histogram_bytes_per_triggered_event",
            "sim_telarray histogram",
            "resource_sim_telarray_histogram_triggered",
            "sim_telarray",
        ),
    )
    output_files = []
    for column, label, filename, required_role in plots:
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
        zenith_values = sorted({float(row.get("zenith_angle_deg", 0.0)) for row in available})
        zenith_colors = _zenith_colors(zenith_values)
        for role in sorted({row["role"] for row in available}):
            role_rows = [row for row in available if row["role"] == role]
            fig, axis = plt.subplots(figsize=(8, 5))
            _plot_role(
                axis,
                role_rows,
                column,
                zenith_colors,
                value_scale=value_scale,
            )
            axis.set_xscale("log")
            axis.set_yscale(_resource_y_scale(role_rows, column))
            axis.set_xlabel("Energy midpoint (GeV)")
            axis.set_ylabel(plot_label)
            axis.set_title(f"{plot_label}: {role}")
            axis.grid(alpha=0.25)
            axis.legend()
            output_file = output_path / f"{filename}_{role}"
            save_figure(fig, output_file, figure_format=figure_format, dpi=300, close=True)
            output_files.append(output_file)
            comparison_labels = _comparison_display_labels(role_rows)
            if set(comparison_labels) >= {"baseline", "candidate"}:
                ratio_series = _ratio_series(role_rows, column)
                if ratio_series:
                    ratio_fig, ratio_axis = plt.subplots(figsize=(8, 5))
                    _plot_ratio(ratio_axis, ratio_series, role, zenith_colors)
                    ratio_axis.set_xscale("log")
                    ratio_axis.set_xlabel("Energy midpoint (GeV)")
                    baseline_label = comparison_labels["baseline"]
                    candidate_label = comparison_labels["candidate"]
                    ratio_label = f"{candidate_label} / {baseline_label}"
                    _warn_on_large_changes(ratio_series, plot_label, role, ratio_label)
                    ratio_axis.set_ylabel(ratio_label)
                    ratio_axis.set_title(f"{plot_label} ratio: {ratio_label}: {role}")
                    ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
                    ratio_axis.grid(alpha=0.25)
                    ratio_axis.legend()
                    ratio_output_file = output_path / f"{filename}_ratio_{role}"
                    save_figure(
                        ratio_fig,
                        ratio_output_file,
                        figure_format=figure_format,
                        dpi=300,
                        close=True,
                    )
                    output_files.append(ratio_output_file)
    return output_files


def _resource_y_scale(rows, column):
    """Return a y-axis scale that can represent all selected resource values."""
    return "linear" if any(float(row[column]) <= 0 for row in rows) else "log"


def _byte_plot_label(column, label, rows):
    """Return a human-readable byte label and its scale factor."""
    if column not in _BYTE_PLOT_COLUMNS:
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
    else:
        suffix = "/event"
    return f"{label} ({unit}{suffix})", scale


def _ratio_series(rows, column):
    """Return candidate-to-baseline ratios grouped by zenith and energy."""
    values = {}
    for row in rows:
        production = _comparison_role(row)
        if production not in {"baseline", "candidate"}:
            continue
        key = (
            production,
            float(row.get("zenith_angle_deg", 0.0)),
            row["energy_midpoint_gev"],
        )
        values.setdefault(key, []).append(float(row[column]))

    series = {}
    zenith_values = {key[1] for key in values}
    for zenith in zenith_values:
        baseline = _statistics_by_energy(values, "baseline", zenith)
        candidate = _statistics_by_energy(values, "candidate", zenith)
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
            series[zenith] = points
    return series


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
    changes = [
        (zenith, energy, ratio)
        for zenith, points in series.items()
        for energy, ratio, _ in points
        if np.isfinite(ratio)
        and ratio > 0
        and max(ratio, 1 / ratio) >= _RESOURCE_CHANGE_WARNING_FACTOR
    ]
    if not changes:
        return
    zenith, energy, ratio = max(changes, key=lambda item: max(item[2], 1 / item[2]))
    factor = max(ratio, 1 / ratio)
    severity = "Major" if factor >= _RESOURCE_CHANGE_MAJOR_FACTOR else "Large"
    direction = "increase" if ratio >= 1 else "decrease"
    _logger.warning(
        "%s resource change in %s for %s at za=%g deg and energy %.6g GeV: "
        "%s shows a %.3g-fold %s.",
        severity,
        plot_label,
        role,
        zenith,
        energy,
        ratio_label,
        factor,
        direction,
    )


def _statistics_by_energy(values, production, zenith):
    """Return means, RMS spreads, and sample counts keyed by energy."""
    grouped = {
        energy: samples
        for (label, sample_zenith, energy), samples in values.items()
        if label == production and sample_zenith == zenith
    }
    return {
        energy: (
            float(np.mean(samples)),
            float(np.sqrt(np.mean((np.asarray(samples) - np.mean(samples)) ** 2))),
            len(samples),
        )
        for energy, samples in grouped.items()
    }


def _plot_ratio(axis, series, role, zenith_colors):
    """Plot candidate-to-baseline ratios with propagated mean errors."""
    for series_index, (zenith, points) in enumerate(sorted(series.items())):
        energies, ratios, errors = zip(*points, strict=True)
        style = _ROLE_STYLE.get(role, {})
        axis.errorbar(
            energies,
            ratios,
            yerr=errors,
            fmt=style.get("marker", "o"),
            label=f"za={zenith:g} deg",
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
    bounds.append(1.0)
    lower = min(bounds)
    upper = max(bounds)
    margin = max((upper - lower) * 0.1, 0.1)
    axis.set_ylim(max(0.0, lower - margin), upper + margin)
    axis.set_yscale("linear")


def _plot_role(axis, rows, column, zenith_colors, value_scale=1.0):
    """Plot individual samples and averages for one process role."""
    labels = sorted(
        {
            (
                row.get("production_label", "production"),
                row.get("model_version", row.get("simtools_version", "unknown")),
                row.get("production_id", "unknown"),
                float(row.get("zenith_angle_deg", 0.0)),
            )
            for row in rows
        }
    )
    role = rows[0]["role"]
    show_production_labels = any(row.get("production_label") for row in rows)
    has_baseline = "baseline" in {_comparison_role(row) for row in rows}
    seen_series = set()
    for series_index, (production, version, production_id, zenith) in enumerate(labels):
        series_rows = [
            row
            for row in rows
            if (
                row.get("production_label", "production"),
                row.get("model_version", row.get("simtools_version", "unknown")),
                row.get("production_id", "unknown"),
                float(row.get("zenith_angle_deg", 0.0)),
            )
            == (production, version, production_id, zenith)
        ]
        color = zenith_colors[zenith]
        series_key = (production, zenith)
        comparison_role = _comparison_role(series_rows[0])
        if series_key in seen_series or (has_baseline and comparison_role != "baseline"):
            legend_label = "_nolegend_"
        elif has_baseline:
            legend_label = f"za={zenith:g} deg"
        elif show_production_labels:
            legend_label = f"{production}: za={zenith:g} deg"
        else:
            legend_label = f"za={zenith:g} deg"
        seen_series.add(series_key)
        _plot_averages(
            axis,
            series_rows,
            column,
            role,
            color,
            legend_label,
            series_index,
            value_scale,
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
        linestyle=("-", "--", ":", "-.")[series_index % 4],
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
