#!/usr/bin/python3

r"""
Merge CORSIKA limit tables from multiple grid points and check grid completeness.

This tool merges multiple CORSIKA limit tables produced by the
simtools-production-derive-corsika-limits application for different grid points into
a single table. It also checks if the grid is complete by verifying that all expected
grid points (combinations of zenith, azimuth, NSB level, etc.) are covered in the
merged table.

The tool can optionally create plots showing the grid coverage and/or visualization
of the merged limits.

Command line arguments
----------------------
input_files (str, required)
    Directory containing corsika_simulation_limits_lookup*.ecsv files or path to a specific file.
grid_definition (str, required)
    Path to a YAML file defining the expected grid points.
output_file (str, optional)
    Name of the output file for the merged limits table. Default is "merged_corsika_limits.ecsv".
plot_grid_coverage (bool, optional)
    Flag to generate plots showing grid coverage.
plot_limits (bool, optional)
    Flag to generate plots showing the derived limits.

Example
-------

Merge CORSIKA limit tables from a directory and check grid completeness:

.. code-block:: console

    simtools-production-merge-corsika-limits \\
        --input_files "simtools-output/corsika_limits/" \\
        --grid_definition grid_definition.yaml \\
        --output_file merged_corsika_limits.ecsv \\
        --plot_grid_coverage \\
        --plot_limits

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import data_reader
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler

_logger = logging.getLogger(__name__)


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        description="Merge CORSIKA limit tables and check grid completeness."
    )
    config.parser.add_argument(
        "--input_files",
        type=str,
        required=True,
        help="Directory containing corsika_simulation_limits_lookup*.ecsv files or path.",
    )
    config.parser.add_argument(
        "--grid_definition",
        type=str,
        required=True,
        help="Path to YAML file defining the expected grid points.",
    )
    config.parser.add_argument(
        "--plot_grid_coverage",
        help="Generate plots showing grid coverage.",
        action="store_true",
        default=False,
    )
    config.parser.add_argument(
        "--plot_limits",
        help="Generate plots showing the derived limits.",
        action="store_true",
        default=False,
    )
    return config.initialize(output=True)


def merge_corsika_limit_tables(input_files):
    """Merge multiple CORSIKA limit tables into a single table."""
    _logger.info(f"Merging {len(input_files)} CORSIKA limit tables")

    tables = []
    metadata = {}

    for file_path in input_files:
        table = data_reader.read_table_from_file(file_path)
        tables.append(table)

        if not metadata:
            metadata = table.meta

    merged_table = vstack(tables, metadata_conflicts="silent")
    merged_table.meta.update(metadata)

    return merged_table


def check_grid_completeness(merged_table, grid_definition):
    """Check if the grid is complete by verifying all expected combinations exist."""
    layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
    nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

    expected_combinations = []
    for zenith in grid_definition.get("zenith", []):
        for azimuth in grid_definition.get("azimuth", []):
            for nsb_level in grid_definition.get("nsb_level", []):
                for layout in grid_definition.get("layouts", []):
                    expected_combinations.append((zenith, azimuth, nsb_level, layout))

    _logger.info(f"Expected {len(expected_combinations)} grid point combinations")

    missing_combinations = []
    for zenith, azimuth, nsb_level, layout in expected_combinations:
        mask = (
            (merged_table["zenith"] == zenith)
            & (merged_table["azimuth"] == azimuth)
            & (merged_table[nsb_column] == nsb_level)
            & (merged_table[layout_column] == layout)
        )
        if not any(mask):
            missing_combinations.append((zenith, azimuth, nsb_level, layout))

    is_complete = len(missing_combinations) == 0

    return is_complete, {
        "expected": len(expected_combinations),
        "found": len(expected_combinations) - len(missing_combinations),
        "missing": missing_combinations,
    }


def plot_grid_coverage(merged_table, output_dir):
    """Generate plots showing grid coverage."""
    _logger.info("Generating grid coverage plots")

    layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
    nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

    zeniths = np.unique(merged_table["zenith"])
    azimuths = np.unique(merged_table["azimuth"])
    nsb_levels = np.unique(merged_table[nsb_column])
    layouts = np.unique(merged_table[layout_column])

    for nsb in nsb_levels:
        for layout in layouts:
            _, ax = plt.subplots(figsize=(10, 8))

            z_grid = np.zeros((len(zeniths), len(azimuths)))
            for i, zenith in enumerate(zeniths):
                for j, azimuth in enumerate(azimuths):
                    mask = (
                        (merged_table["zenith"] == zenith)
                        & (merged_table["azimuth"] == azimuth)
                        & (merged_table[nsb_column] == nsb)
                        & (merged_table[layout_column] == layout)
                    )
                    if any(mask):
                        z_grid[i, j] = 1

            azimuths_values = azimuths.value if hasattr(azimuths, "value") else azimuths
            zeniths_values = zeniths.value if hasattr(zeniths, "value") else zeniths

            extent = [
                min(azimuths_values) - 0.5,
                max(azimuths_values) + 0.5,
                max(zeniths_values) + 0.5,
                min(zeniths_values) - 0.5,
            ]
            im = ax.imshow(z_grid, cmap="RdYlGn", vmin=0, vmax=1, extent=extent)

            plt.colorbar(im, ax=ax, label="Coverage (1=Present, 0=Missing)")
            ax.set_title(f"Grid Coverage: NSB={nsb}, Layout={layout}")
            ax.set_xlabel("Azimuth [deg]")
            ax.set_ylabel("Zenith [deg]")
            ax.set_xticks(azimuths_values)
            ax.set_yticks(zeniths_values)
            ax.grid(which="major", linestyle="-", linewidth="0.5", color="black", alpha=0.3)

            output_file = output_dir / f"grid_coverage_{nsb}_{layout}.png"
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()


def plot_limits(merged_table, output_dir):
    """Generate plots showing the derived limits.

    Creates overview plots organized by layout, combining different NSB levels in the same plot
    with different colors. Each azimuth angle gets its own set of plots.
    """
    _logger.info("Generating limit plots")

    layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
    nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

    layouts = np.unique(merged_table[layout_column])
    azimuths = np.unique(merged_table["azimuth"])
    nsb_levels = np.unique(merged_table[nsb_column])

    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nsb_levels)))

    for layout in layouts:
        for azimuth in azimuths:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            layout_az_mask = (merged_table[layout_column] == layout) & (
                merged_table["azimuth"] == azimuth
            )
            layout_az_data = merged_table[layout_az_mask]

            if len(layout_az_data) == 0:
                plt.close(fig)
                continue

            legend_handles = []
            legend_labels = []

            for i, nsb in enumerate(nsb_levels):
                nsb_mask = layout_az_data[nsb_column] == nsb
                filtered_data = layout_az_data[nsb_mask]

                if len(filtered_data) == 0:
                    continue

                zeniths = np.unique(filtered_data["zenith"])
                zeniths_values = zeniths.value if hasattr(zeniths, "value") else zeniths

                sort_idx = np.argsort(zeniths_values)
                zeniths_values = zeniths_values[sort_idx]
                zeniths = zeniths[sort_idx]

                energy_limits = []
                for zenith in zeniths:
                    zenith_mask = filtered_data["zenith"] == zenith
                    energy_values = filtered_data[zenith_mask]["lower_energy_limit"]
                    if hasattr(energy_values, "value"):
                        energy_limits.append(np.mean(energy_values.value))
                    else:
                        energy_limits.append(np.mean(energy_values))

                (line1,) = axes[0].plot(
                    zeniths_values, energy_limits, "o-", color=colors[i], label=f"NSB={nsb}"
                )

                radius_limits = []
                for zenith in zeniths:
                    zenith_mask = filtered_data["zenith"] == zenith
                    radius_values = filtered_data[zenith_mask]["upper_radius_limit"]
                    if hasattr(radius_values, "value"):
                        radius_limits.append(np.mean(radius_values.value))
                    else:
                        radius_limits.append(np.mean(radius_values))

                axes[1].plot(zeniths_values, radius_limits, "o-", color=colors[i])

                viewcone_limits = []
                for zenith in zeniths:
                    zenith_mask = filtered_data["zenith"] == zenith
                    viewcone_values = filtered_data[zenith_mask]["viewcone_radius"]
                    if hasattr(viewcone_values, "value"):
                        viewcone_limits.append(np.mean(viewcone_values.value))
                    else:
                        viewcone_limits.append(np.mean(viewcone_values))

                axes[2].plot(zeniths_values, viewcone_limits, "o-", color=colors[i])

                legend_handles.append(line1)
                legend_labels.append(f"NSB={nsb}")

            axes[0].set_title("Lower Energy Limit vs Zenith")
            axes[0].set_xlabel("Zenith [deg]")
            axes[0].set_ylabel("Lower Energy Limit [TeV]")
            axes[0].grid(True)

            axes[1].set_title("Upper Radius Limit vs Zenith")
            axes[1].set_xlabel("Zenith [deg]")
            axes[1].set_ylabel("Upper Radius Limit [m]")
            axes[1].grid(True)

            axes[2].set_title("Viewcone Radius vs Zenith")
            axes[2].set_xlabel("Zenith [deg]")
            axes[2].set_ylabel("Viewcone Radius [deg]")
            axes[2].grid(True)

            fig.legend(legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels))

            azimuth_value = azimuth.value if hasattr(azimuth, "value") else azimuth
            plt.suptitle(f"CORSIKA Limits: Layout={layout}, Azimuth={azimuth_value}°")
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            output_file = output_dir / f"limits_{layout}_azimuth{azimuth_value}.png"
            plt.savefig(output_file)
            plt.close(fig)


def main():
    """Merge CORSIKA limit tables and check grid completeness."""
    args_dict, _ = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict.get("log_level", "info")))

    input_dir = Path(args_dict["input_files"]).expanduser()

    if input_dir.is_dir():
        input_files = list(input_dir.glob("corsika_simulation_limits_lookup*.ecsv"))
    else:
        input_files = [input_dir]

    output_dir = io_handler.IOHandler().get_output_directory()
    merged_table = merge_corsika_limit_tables(input_files)

    grid_definition = gen.collect_data_from_file(args_dict["grid_definition"])

    is_complete, grid_completeness = check_grid_completeness(merged_table, grid_definition)

    if args_dict.get("plot_grid_coverage"):
        plot_grid_coverage(merged_table, output_dir)

    if args_dict.get("plot_limits"):
        plot_limits(merged_table, output_dir)

    output_file_name = args_dict.get("output_file", "merged_corsika_limits.ecsv")
    output_file = output_dir / output_file_name

    if "description" not in merged_table.meta:
        merged_table.meta["description"] = (
            "Lookup table for CORSIKA limits computed from simulations."
        )
    if "loss_fraction" not in merged_table.meta:
        merged_table.meta["loss_fraction"] = 1.0e-6

    merged_table.write(output_file, format="ascii.ecsv", overwrite=True)

    metadata = {
        "input_files": [str(f) for f in input_files],
        "grid_completeness": is_complete,
        "missing_points": len(grid_completeness.get("missing", [])),
        "total_expected_points": grid_completeness.get("expected", 0),
        "found_points": grid_completeness.get("found", 0),
    }
    MetadataCollector.dump(metadata, output_file)


if __name__ == "__main__":
    main()
