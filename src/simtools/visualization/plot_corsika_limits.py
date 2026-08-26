"""Plotting utilities for CORSIKA limits tables."""

import logging
from itertools import product
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from simtools.visualization.matplotlib_backend import pyplot as plt

_logger = logging.getLogger(__name__)

ZENITH_LABEL = "Zenith [deg]"
BROAD_RANGE_COLUMN_ALIASES = {
    "lower_energy_limit": ["br_energy_min", "br_lower_energy_limit"],
    "upper_radius_limit": ["br_core_scatter_max", "br_upper_radius_limit"],
    "viewcone_radius": ["br_viewcone_max", "br_viewcone_radius"],
}
_LIMIT_COLUMNS = (
    "lower_energy_limit",
    "upper_radius_limit",
    "viewcone_radius",
)


def _get_primary_particle_label(table):
    """Return a primary particle label derived from table content."""
    if "primary_particle" not in table.colnames:
        return "unknown"

    unique_particles = np.unique(np.array(table["primary_particle"], dtype=str))
    if len(unique_particles) == 1:
        return unique_particles[0]

    return "/".join(unique_particles)


def _resolve_broad_range_columns(limits_table):
    """Resolve broad-range column names from supported aliases."""
    resolved_columns = {}
    for column_key, aliases in BROAD_RANGE_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in limits_table.colnames:
                resolved_columns[column_key] = alias
                break

    if len(resolved_columns) != len(BROAD_RANGE_COLUMN_ALIASES):
        return None

    return resolved_columns


def _plot_single_grid_coverage(
    ax, zeniths, azimuths, nsb, array_name, found_combinations_str, primary_particle
):
    """Plot grid coverage for a single NSB and array name."""
    z_grid = np.zeros((len(zeniths), len(azimuths)))
    for i, zenith in enumerate(zeniths):
        for j, azimuth in enumerate(azimuths):
            point_str = (str(zenith), str(azimuth), str(nsb), str(array_name))
            if point_str in found_combinations_str:
                z_grid[i, j] = 1

    az_vals = azimuths.value if hasattr(azimuths, "value") else azimuths
    zen_vals = zeniths.value if hasattr(zeniths, "value") else zeniths
    extent = [
        min(az_vals) - 0.5,
        max(az_vals) + 0.5,
        max(zen_vals) + 0.5,
        min(zen_vals) - 0.5,
    ]

    im = ax.imshow(z_grid, cmap=ListedColormap(["red", "green"]), vmin=0, vmax=1, extent=extent)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], label="Coverage", shrink=0.25, pad=0.02)
    cbar.set_ticklabels(["Missing", "Present"])

    ax.set_title(
        f"Grid Coverage: NSB={nsb}, Array Name={array_name}, Primary Particle={primary_particle}"
    )
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel(ZENITH_LABEL)
    ax.set_xticks(az_vals)
    ax.set_yticks(zen_vals)
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="black", alpha=0.3)


def plot_grid_coverage(limits_table, grid_definition, output_dir):
    """
    Generate grid coverage plots for each NSB level and array name combination.

    Parameters
    ----------
    limits_table : Table
        An astropy Table containing the CORSIKA limits data.
    grid_definition : dict or None
        A dictionary defining the expected grid points for zenith,
        azimuth, NSB level, and array name.
    output_dir : str or Path
        Directory where the generated grid coverage plots will be saved.

    Returns
    -------
    list of Path
        List of file paths to the generated grid coverage plots.
    """
    if not grid_definition:
        _logger.info("No grid definition provided, skipping grid coverage plots.")
        return []

    _logger.info("Generating grid coverage plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    primary_particle = _get_primary_particle_label(limits_table)

    found_combinations_str = set(
        zip(
            np.array(limits_table["zenith"].value, dtype=str),
            np.array(limits_table["azimuth"].value, dtype=str),
            np.array(limits_table["nsb_level"], dtype=str),
            np.array(limits_table["array_name"], dtype=str),
        )
    )

    unique_values = {
        "zeniths": np.array(grid_definition.get("zenith", [])),
        "azimuths": np.array(grid_definition.get("azimuth", [])),
        "nsb_levels": np.array(grid_definition.get("nsb_level", [])),
        "array_names": np.array(grid_definition.get("array_name", [])),
    }

    for nsb, array_name in product(unique_values["nsb_levels"], unique_values["array_names"]):
        _, ax = plt.subplots(figsize=(10, 8))
        _plot_single_grid_coverage(
            ax,
            unique_values["zeniths"],
            unique_values["azimuths"],
            nsb,
            array_name,
            found_combinations_str,
            primary_particle,
        )
        output_file = output_dir / f"grid_coverage_{nsb}_{array_name}.png"
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        output_files.append(output_file)

    return output_files


def _plot_limit_series(axes, zeniths, values, color, filled_marker):
    """Plot the three derived limit series with a common azimuth marker style."""
    for axis, value in zip(axes, values, strict=True):
        axis.plot(
            zeniths,
            value,
            "o-",
            color=color,
            markerfacecolor=color if filled_marker else "none",
        )


def _plot_broad_range_series(axes, zeniths, broad_data, broad_range_columns):
    """Plot broad-range reference limits on the corresponding derived-limit axes."""
    for axis, column in zip(axes, _LIMIT_COLUMNS, strict=True):
        axis.plot(
            zeniths,
            broad_data[broad_range_columns[column]],
            linestyle="--",
            color="gray",
            linewidth=1.5,
        )


def _value_in_degrees(value):
    """Return a plain azimuth value for labels and filenames."""
    return value.value if hasattr(value, "value") else value


def _plot_limit_group(axes, group, broad_range_columns):
    """Plot all NSB and azimuth limit series for one layout and particle."""
    legend_handles, legend_labels = [], []
    grouped_by_nsb = group.group_by("nsb_level")
    colors = plt.get_cmap("Set1").colors  # don't expect more than 9 NSB levels
    azimuth_values = [
        _value_in_degrees(azimuth_group["azimuth"][0])
        for azimuth_group in group.group_by("azimuth").groups
    ]

    for index, nsb_group in enumerate(grouped_by_nsb.groups):
        nsb_level = nsb_group["nsb_level"][0]
        color = colors[index]
        legend_handles.append(Line2D([0], [0], color=color))
        legend_labels.append(f"NSB={nsb_level} GHz")

        for azimuth_group in nsb_group.group_by("azimuth").groups:
            plot_columns = ["zenith", *_LIMIT_COLUMNS]
            agg_data = azimuth_group[plot_columns].group_by("zenith").groups.aggregate(np.mean)
            agg_data.sort("zenith")
            zeniths = agg_data["zenith"].value
            azimuth_value = _value_in_degrees(azimuth_group["azimuth"][0])
            _plot_limit_series(
                axes,
                zeniths,
                [agg_data[column] for column in _LIMIT_COLUMNS],
                color,
                filled_marker=azimuth_values.index(azimuth_value) == 0,
            )

            if broad_range_columns:
                broad_columns = [
                    "zenith",
                    *(broad_range_columns[column] for column in _LIMIT_COLUMNS),
                ]
                broad_data = (
                    azimuth_group[broad_columns].group_by("zenith").groups.aggregate(np.mean)
                )
                broad_data.sort("zenith")
                _plot_broad_range_series(axes, zeniths, broad_data, broad_range_columns)

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

    group_columns = ["array_name"]
    if "primary_particle" in limits_table.colnames:
        group_columns.append("primary_particle")
    grouped_by_layout = limits_table.group_by(group_columns)
    broad_range_columns = _resolve_broad_range_columns(limits_table)

    for group in grouped_by_layout.groups:
        array_name = group["array_name"][0]
        primary_particle = _get_primary_particle_label(group)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        legend_handles, legend_labels, azimuth_values = _plot_limit_group(
            axes, group, broad_range_columns
        )

        for axis in axes:
            axis.relim()
            axis.autoscale_view()

        axes[0].set_title("Lower Energy Limit vs Zenith")
        axes[0].set_xlabel(ZENITH_LABEL)
        axes[0].set_ylabel("Lower Energy Limit [TeV]")
        axes[0].grid(True)
        axes[1].set_title("Upper Radius Limit vs Zenith")
        axes[1].set_xlabel(ZENITH_LABEL)
        axes[1].set_ylabel("Upper Radius Limit [m]")
        axes[1].grid(True)
        axes[2].set_title("Viewcone Radius vs Zenith")
        axes[2].set_xlabel(ZENITH_LABEL)
        axes[2].set_ylabel("Viewcone Radius [deg]")
        axes[2].grid(True)

        if broad_range_columns:
            legend_handles += [
                Line2D([0], [0], linestyle="--", color="gray"),
            ]
            legend_labels += ["broad-range limits"]
        legend_handles += [
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                linestyle="none",
                markerfacecolor="black" if index == 0 else "none",
            )
            for index, _ in enumerate(azimuth_values)
        ]
        legend_labels += [f"Az={azimuth} deg" for azimuth in azimuth_values]
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels))
        plt.suptitle(f"CORSIKA Limits: {array_name}, {primary_particle}")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        output_file = output_dir / f"limits_{array_name}_{primary_particle}.png"
        plt.savefig(output_file)
        plt.close(fig)
        output_files.append(output_file)

    return output_files
