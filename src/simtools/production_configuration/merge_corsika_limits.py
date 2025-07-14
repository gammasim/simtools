"""Class for merging CORSIKA limit tables and checking grid completeness."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack

import simtools.utils.general as gen
from simtools.data_model import data_reader
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler

_logger = logging.getLogger(__name__)


class CorsikaMergeLimits:
    """Class for merging CORSIKA limit tables and checking grid completeness."""

    def __init__(self):
        """Initialize CorsikaMergeLimits."""
        self.output_dir = io_handler.IOHandler().get_output_directory()

    def merge_tables(self, input_files):
        """Merge multiple CORSIKA limit tables into a single table.

        This method reads all tables from the input files, checks for duplicate grid points,
        merges the tables, and sorts the result by layout and other relevant columns.

        Parameters
        ----------
        input_files : list
            List of paths to CORSIKA limit tables files

        Returns
        -------
        astropy.table.Table
            Merged and sorted table
        """
        _logger.info(f"Merging {len(input_files)} CORSIKA limit tables")

        tables = []
        metadata = {}
        loss_fractions = set()
        grid_points = set()
        duplicate_points = []

        for file_path in input_files:
            table = data_reader.read_table_from_file(file_path)

            layout_column = "layout" if "layout" in table.colnames else "array_name"
            nsb_column = "nsb_level" if "nsb_level" in table.colnames else "nsb"

            for row in table:
                grid_point = (row["zenith"], row["azimuth"], row[nsb_column], row[layout_column])

                if grid_point in grid_points:
                    duplicate_points.append(grid_point)
                else:
                    grid_points.add(grid_point)

            if "loss_fraction" in table.meta:
                loss_fractions.add(table.meta["loss_fraction"])

            if not metadata:
                metadata = table.meta

            tables.append(table)

        # Report duplicates
        if duplicate_points:
            _logger.warning(f"Found {len(duplicate_points)} duplicate grid points across tables")
            _logger.warning(f"First few duplicates: {duplicate_points[:5]}")
            _logger.warning("When duplicates exist, only the last occurrence will be kept")

        # Report different loss_fraction values
        if len(loss_fractions) > 1:
            _logger.warning(f"Found different loss_fraction values across tables: {loss_fractions}")
            _logger.warning(
                f"Using loss_fraction from the first table: {metadata.get('loss_fraction')}"
            )

        # Merge
        merged_table = vstack(tables, metadata_conflicts="silent")
        merged_table.meta.update(metadata)

        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

        # Sort by layout, zenith, azimuth, and nsb_level
        merged_table.sort([layout_column, "zenith", "azimuth", nsb_column])

        _logger.info(
            f"Merged table has {len(merged_table)} rows with {len(grid_points)} unique grid points"
        )

        return merged_table

    def check_grid_completeness(self, merged_table, grid_definition=None):
        """Check if the grid is complete by verifying all expected combinations exist.

        Parameters
        ----------
        merged_table : astropy.table.Table
            Merged table with all grid points
        grid_definition : dict, optional
            Grid definition with expected combinations. If None, will be extracted from the table

        Returns
        -------
        tuple
            (is_complete, grid_completeness_details)
        """
        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

        if grid_definition is None:
            grid_definition = {
                "zenith": np.unique(merged_table["zenith"]),
                "azimuth": np.unique(merged_table["azimuth"]),
                "nsb_level": np.unique(merged_table[nsb_column]),
                "layouts": np.unique(merged_table[layout_column]),
            }

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

    def plot_grid_coverage(self, merged_table):
        """Generate plots showing grid coverage.

        Parameters
        ----------
        merged_table : astropy.table.Table
            Merged table with all grid points

        Returns
        -------
        list
            List of paths to generated plot files
        """
        _logger.info("Generating grid coverage plots")
        output_files = []

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

                output_file = self.output_dir / f"grid_coverage_{nsb}_{layout}.png"
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()
                output_files.append(output_file)

        return output_files

    def plot_limits(self, merged_table):
        """Generate plots showing the derived limits.

        Creates overview plots organized by layout, combining different NSB levels in the same plot
        with different colors. Each azimuth angle gets its own set of plots.

        Parameters
        ----------
        merged_table : astropy.table.Table
            Merged table with all grid points

        Returns
        -------
        list
            List of paths to generated plot files
        """
        _logger.info("Generating limit plots")
        output_files = []

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

                fig.legend(
                    legend_handles, legend_labels, loc="lower center", ncol=len(legend_labels)
                )

                azimuth_value = azimuth.value if hasattr(azimuth, "value") else azimuth
                plt.suptitle(f"CORSIKA Limits: Layout={layout}, Azimuth={azimuth_value} deg")
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)

                output_file = self.output_dir / f"limits_{layout}_azimuth{azimuth_value}.png"
                plt.savefig(output_file)
                plt.close(fig)
                output_files.append(output_file)

        return output_files

    def write_merged_table(self, merged_table, output_file, input_files, grid_completeness):
        """Write the merged table to file and save metadata.

        Parameters
        ----------
        merged_table : astropy.table.Table
            Merged table with all grid points
        output_file : Path
            Path to write the merged table
        input_files : list
            List of input files used to create the merged table
        grid_completeness : dict
            Results of the grid completeness check

        Returns
        -------
        Path
            Path to the written file
        """
        if "description" not in merged_table.meta:
            merged_table.meta["description"] = (
                "Lookup table for CORSIKA limits computed from simulations."
            )

        if "loss_fraction" not in merged_table.meta:
            _logger.warning("No loss_fraction found in any of the input tables")

        merged_table.meta["created_by"] = "simtools-production-merge-corsika-limits"
        merged_table.meta["creation_date"] = gen.now_date_time_in_isoformat()
        merged_table.meta["input_files_count"] = len(input_files)

        merged_table.write(output_file, format="ascii.ecsv", overwrite=True)
        _logger.info(f"Merged table written to {output_file}")

        metadata = {
            "input_files": [str(f) for f in input_files],
            "grid_completeness": grid_completeness.get("is_complete", False),
            "missing_points": len(grid_completeness.get("missing", [])),
            "total_expected_points": grid_completeness.get("expected", 0),
            "found_points": grid_completeness.get("found", 0),
            "row_count": len(merged_table),
        }
        MetadataCollector.dump(metadata, output_file)

        return output_file
