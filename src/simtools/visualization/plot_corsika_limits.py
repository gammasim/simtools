"""Plotting utilities for CORSIKA limits tables."""

import logging
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

_logger = logging.getLogger(__name__)

ZENITH_LABEL = "Zenith [deg]"
BROAD_RANGE_COLUMN_ALIASES = {
    "lower_energy_limit": ["br_energy_min", "br_lower_energy_limit"],
    "upper_radius_limit": ["br_core_scatter_max", "br_upper_radius_limit"],
    "viewcone_radius": ["br_viewcone_max", "br_viewcone_radius"],
}


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


def _plot_single_grid_coverage(ax, zeniths, azimuths, nsb, array_name, found_combinations_str):
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

    ax.set_title(f"Grid Coverage: NSB={nsb}, Array Name={array_name}")
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

    """
    if not grid_definition:
        _logger.info("No grid definition provided, skipping grid coverage plots.")
        return []

    _logger.info("Generating grid coverage plots")
    output_dir = Path(output_dir)
    output_files = []

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
        )
        output_file = output_dir / f"grid_coverage_{nsb}_{array_name}.png"
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        output_files.append(output_file)

    return output_files


def plot_limits(limits_table, output_dir):
    """
    Create plots of derived CORSIKA limits for each array name and azimuth.

    Parameters
    ----------
    limits_table (Table)
        An astropy Table containing the CORSIKA limits data.
    output_dir (str or Path)
        Directory where the generated plots will be saved.
    """
    _logger.info("Generating limit plots")
    output_dir = Path(output_dir)
    output_files = []

    grouped_by_layout_az = limits_table.group_by(["array_name", "azimuth"])
    broad_range_columns = _resolve_broad_range_columns(limits_table)

    for group in grouped_by_layout_az.groups:
        array_name = group["array_name"][0]
        azimuth = group["azimuth"][0]
        azimuth_value = azimuth.value if hasattr(azimuth, "value") else azimuth

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        legend_handles, legend_labels = [], []

        grouped_by_nsb = group.group_by("nsb_level")
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(grouped_by_nsb.groups)))

        for i, nsb_group in enumerate(grouped_by_nsb.groups):
            nsb_level = nsb_group["nsb_level"][0]
            plot_columns = [
                "zenith",
                "lower_energy_limit",
                "upper_radius_limit",
                "viewcone_radius",
            ]
            agg_data = nsb_group[plot_columns].group_by("zenith").groups.aggregate(np.mean)
            agg_data.sort("zenith")
            zeniths = agg_data["zenith"].value

            (line,) = axes[0].plot(zeniths, agg_data["lower_energy_limit"], "o-", color=colors[i])
            axes[1].plot(zeniths, agg_data["upper_radius_limit"], "o-", color=colors[i])
            axes[2].plot(zeniths, agg_data["viewcone_radius"], "o-", color=colors[i])
            legend_handles.append(line)
            legend_labels.append(f"NSB={nsb_level}")

            if broad_range_columns:
                broad_columns = [
                    "zenith",
                    broad_range_columns["lower_energy_limit"],
                    broad_range_columns["upper_radius_limit"],
                    broad_range_columns["viewcone_radius"],
                ]
                broad_data = nsb_group[broad_columns].group_by("zenith").groups.aggregate(np.mean)
                broad_data.sort("zenith")

                axes[0].plot(
                    zeniths,
                    broad_data[broad_range_columns["lower_energy_limit"]],
                    linestyle="--",
                    color="gray",
                    linewidth=1.5,
                )
                axes[1].plot(
                    zeniths,
                    broad_data[broad_range_columns["upper_radius_limit"]],
                    linestyle="--",
                    color="gray",
                    linewidth=1.5,
                )
                axes[2].plot(
                    zeniths,
                    broad_data[broad_range_columns["viewcone_radius"]],
                    linestyle="--",
                    color="gray",
                    linewidth=1.5,
                )

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

        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels))
        plt.suptitle(f"CORSIKA Limits: Array Name={array_name}, Azimuth={azimuth_value} deg")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        output_file = output_dir / f"limits_{array_name}_azimuth{azimuth_value}.png"
        plt.savefig(output_file)
        plt.close(fig)
        output_files.append(output_file)

    return output_files
