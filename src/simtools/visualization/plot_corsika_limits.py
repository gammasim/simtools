"""Plotting utilities for CORSIKA limits tables."""

import logging
from pathlib import Path

import numpy as np

from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

lines = lazy_module("matplotlib.lines")

_logger = logging.getLogger(__name__)

ZENITH_LABEL = "Zenith [deg]"
BROAD_RANGE_COLUMN_ALIASES = {
    "lower_energy_limit": ["br_energy_min", "br_lower_energy_limit"],
    "upper_radius_limit": ["br_core_scatter_max", "br_upper_radius_limit"],
    "viewcone_radius": ["br_viewcone_max", "br_viewcone_radius"],
}
_LIMIT_COLUMNS = ("lower_energy_limit", "upper_radius_limit", "viewcone_radius")
_PANELS = (
    ("Lower Energy Limit vs Zenith", "Lower Energy Limit [TeV]"),
    ("Upper Radius Limit vs Zenith", "Upper Radius Limit [m]"),
    ("Viewcone Radius vs Zenith", "Viewcone Radius [deg]"),
)
_AZIMUTH_LINESTYLES = ("-", "--")


def _resolve_broad_range_columns(limits_table):
    """Return broad-range column names per limit, or None if any limit is missing."""
    resolved = {
        key: next((alias for alias in aliases if alias in limits_table.colnames), None)
        for key, aliases in BROAD_RANGE_COLUMN_ALIASES.items()
    }
    if any(column is None for column in resolved.values()):
        return None
    return resolved


def _mean_by_zenith(group, columns):
    """Average the given columns per zenith angle, sorted by zenith."""
    agg_data = group[columns].group_by("zenith").groups.aggregate(np.mean)
    return agg_data[np.argsort(agg_data["zenith"])]


def _plot_broad_range(axes, azimuth_group, zeniths, broad_range_columns):
    """Plot the broad-range limit series of one azimuth direction as dashed gray lines."""
    broad_columns = [broad_range_columns[column] for column in _LIMIT_COLUMNS]
    broad_data = _mean_by_zenith(azimuth_group, ["zenith", *broad_columns])
    for axis, column in zip(axes, broad_columns):
        axis.plot(zeniths, broad_data[column], linestyle="--", color="gray", linewidth=1.5)


def _plot_azimuth_group(axes, azimuth_group, color, azimuth_index, broad_range_columns):
    """Plot the limit series, and optionally the broad-range series, of one azimuth direction."""
    linestyle = _AZIMUTH_LINESTYLES[azimuth_index % len(_AZIMUTH_LINESTYLES)]
    agg_data = _mean_by_zenith(azimuth_group, ["zenith", *_LIMIT_COLUMNS])
    zeniths = agg_data["zenith"].value
    for axis, column in zip(axes, _LIMIT_COLUMNS):
        axis.plot(
            zeniths,
            agg_data[column],
            "o",
            color=color,
            linestyle=linestyle,
            markerfacecolor=color if azimuth_index == 0 else "none",
        )
    if broad_range_columns:
        _plot_broad_range(axes, azimuth_group, zeniths, broad_range_columns)


def _plot_limit_group(axes, group, broad_range_columns):
    """Plot all NSB and azimuth limit series for one layout and particle."""
    legend_handles, legend_labels = [], []
    palette = plt.get_cmap("Set1").colors  # don't expect more than 9 NSB levels
    azimuth_values = [
        getattr(g["azimuth"][0], "value", g["azimuth"][0]) for g in group.group_by("azimuth").groups
    ]

    for index, nsb_group in enumerate(group.group_by("nsb_level").groups):
        color = palette[index]
        legend_handles.append(lines.Line2D([0], [0], color=color))
        legend_labels.append(f"NSB={nsb_group['nsb_level'][0]} GHz")

        for azimuth_group in nsb_group.group_by("azimuth").groups:
            azimuth = getattr(azimuth_group["azimuth"][0], "value", azimuth_group["azimuth"][0])
            azimuth_index = azimuth_values.index(azimuth)
            _plot_azimuth_group(axes, azimuth_group, color, azimuth_index, broad_range_columns)

    return legend_handles, legend_labels, azimuth_values


def plot_limits(limits_table, output_dir):
    """
    Create plots of derived CORSIKA limits for each array and primary particle.

    NSB levels are distinguished by color. Azimuth directions share a plot and
    use filled and open markers in ascending azimuth order.

    Parameters
    ----------
    limits_table (Table)
        An astropy Table containing the CORSIKA limits data.
    output_dir (str or Path)
        Directory where the generated plots will be saved.

    Returns
    -------
    list of Path
        List of file paths to the generated plots.
    """
    _logger.info("Generating limit plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []

    has_particle = "primary_particle" in limits_table.colnames
    group_columns = ["array_name", "primary_particle"] if has_particle else ["array_name"]
    broad_range_columns = _resolve_broad_range_columns(limits_table)

    for group in limits_table.group_by(group_columns).groups:
        array_name = group["array_name"][0]
        particles = np.unique(np.array(group["primary_particle"], dtype=str))
        primary_particle = (
            "unknown" if "primary_particle" not in group.colnames else "/".join(particles)
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        legend_handles, legend_labels, azimuth_values = _plot_limit_group(
            axes, group, broad_range_columns
        )

        for axis, (title, ylabel) in zip(axes, _PANELS):
            axis.relim()
            axis.autoscale_view()
            axis.set_title(title)
            axis.set_xlabel(ZENITH_LABEL)
            axis.set_ylabel(ylabel)
            axis.grid(True)

        if broad_range_columns:
            legend_handles.append(lines.Line2D([0], [0], linestyle="--", color="gray"))
            legend_labels.append("broad-range limits")
        legend_handles += [
            lines.Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle=_AZIMUTH_LINESTYLES[index % len(_AZIMUTH_LINESTYLES)],
                markerfacecolor="black" if index == 0 else "none",
            )
            for index in range(len(azimuth_values))
        ]
        legend_labels += [f"Az={azimuth} deg" for azimuth in azimuth_values]
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels))
        plt.suptitle(f"CORSIKA Limits: {array_name}, {primary_particle}")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        output_file = output_dir / f"limits_{array_name}_{primary_particle}.png"
        save_figure(fig, output_file.with_suffix(""), figure_format=["png"], close=True)
        output_files.append(output_file)

    return output_files
