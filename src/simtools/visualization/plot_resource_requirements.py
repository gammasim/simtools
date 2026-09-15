"""Plot normalized CORSIKA and sim_telarray resource requirements."""

from pathlib import Path

import numpy as np
from matplotlib import colormaps

from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

_ROLE_STYLE = {
    "sim_telarray": {"marker": "s", "zorder": 3},
    "corsika": {"marker": "o", "zorder": 4},
}


def plot(rows, output_path, figure_format=None):
    """Write resource plots versus energy and return their base paths."""
    output_path = Path(output_path)
    plots = (
        ("wall_time_seconds_per_event", "Wall time (s/event)", "resource_wall_time"),
        ("cpu_time_seconds_per_event", "CPU time (s/event)", "resource_cpu_time"),
        ("peak_rss_bytes", "Peak RSS (bytes)", "resource_peak_rss"),
        (
            "sim_telarray_storage_bytes_per_event",
            "sim_telarray storage (bytes/event)",
            "resource_storage",
        ),
        (
            "corsika_output_bytes_per_event",
            "CORSIKA output (bytes/event)",
            "resource_corsika_output",
        ),
        (
            "sim_telarray_output_bytes_per_event",
            "sim_telarray output (bytes/event)",
            "resource_sim_telarray_output",
        ),
        (
            "reduced_event_data_bytes_per_event",
            "reduced event data (bytes/event)",
            "resource_reduced_event_data",
        ),
        (
            "sim_telarray_histogram_bytes_per_event",
            "sim_telarray histogram (bytes/event)",
            "resource_sim_telarray_histogram",
        ),
    )
    output_files = []
    for column, label, filename in plots:
        available = [
            row
            for row in rows
            if row.get(column) is not None
            and (not column.startswith("sim_telarray_storage") or row["role"] == "sim_telarray")
        ]
        if not available:
            continue
        zenith_values = sorted({float(row.get("zenith_angle_deg", 0.0)) for row in available})
        zenith_colors = _zenith_colors(zenith_values)
        for role in sorted({row["role"] for row in available}):
            role_rows = [row for row in available if row["role"] == role]
            fig, axis = plt.subplots(figsize=(8, 5))
            contexts = sorted(
                {
                    (
                        row.get("production_label", "production"),
                        row.get("simtools_version", "unknown"),
                    )
                    for row in role_rows
                }
            )
            _plot_role(
                axis,
                role_rows,
                column,
                zenith_colors,
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("Energy midpoint (GeV)")
            axis.set_ylabel(label)
            context_title = "; ".join(
                f"{production}: {version}" for production, version in contexts
            )
            axis.set_title(f"{label}: {role} ({context_title})")
            axis.grid(alpha=0.25)
            axis.legend()
            output_file = output_path / f"{filename}_{role}"
            save_figure(fig, output_file, figure_format=figure_format, dpi=300, close=True)
            output_files.append(output_file)
    return output_files


def _plot_role(axis, rows, column, zenith_colors):
    """Plot individual samples and averages for one process role."""
    labels = sorted(
        {
            (
                row.get("production_label", "production"),
                row.get("simtools_version", "unknown"),
                float(row.get("zenith_angle_deg", 0.0)),
            )
            for row in rows
        }
    )
    role = rows[0]["role"]
    seen_zenith = set()
    for production, version, zenith in labels:
        series_rows = [
            row
            for row in rows
            if (
                row.get("production_label", "production"),
                row.get("simtools_version", "unknown"),
                float(row.get("zenith_angle_deg", 0.0)),
            )
            == (production, version, zenith)
        ]
        color = zenith_colors[zenith]
        legend_label = f"za={zenith:g} deg" if zenith not in seen_zenith else "_nolegend_"
        seen_zenith.add(zenith)
        _plot_averages(axis, series_rows, column, role, color, legend_label)


def _plot_averages(axis, rows, column, role, color, legend_label):
    """Overlay one mean and RMS error bar for each energy in a series."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["energy_midpoint_gev"], []).append(row[column])
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
