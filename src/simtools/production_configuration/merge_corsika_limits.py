"""Class for merging CORSIKA limit tables and checking grid completeness."""

import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack

import simtools.utils.general as gen
from simtools.data_model import data_reader
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler

_logger = logging.getLogger(__name__)

ZENITH_LABEL = "Zenith [deg]"


class CorsikaMergeLimits:
    """Class for merging CORSIKA limit tables and checking grid completeness."""

    def __init__(self, output_dir=None):
        """Initialize CorsikaMergeLimits.

        Parameters
        ----------
        output_dir : Path or str, optional
            Output directory path. If None, will use the default from IOHandler.
        """
        self.output_dir = (
            io_handler.IOHandler().get_output_directory() if output_dir is None else output_dir
        )

    def _read_and_collect_tables(self, input_files):
        """Read tables from files and collect metadata."""
        tables = []
        metadata = {}
        loss_fractions = set()
        grid_points = set()
        duplicate_points = []

        for file_path in input_files:
            table = data_reader.read_table_from_file(file_path)
            tables.append(table)

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

        return tables, metadata, loss_fractions, grid_points, duplicate_points

    def _report_and_merge(self, tables, metadata, loss_fractions, duplicate_points):
        """Report issues and merge tables."""
        if duplicate_points:
            _logger.warning(f"Found {len(duplicate_points)} duplicate grid points across tables")
            _logger.warning(f"First few duplicates: {duplicate_points[:5]}")
            _logger.warning("When duplicates exist, only the last occurrence will be kept")

        if len(loss_fractions) > 1:
            _logger.warning(f"Found different loss_fraction values across tables: {loss_fractions}")
            _logger.warning(
                f"Using loss_fraction from the first table: {metadata.get('loss_fraction')}"
            )

        merged_table = vstack(tables, metadata_conflicts="silent")
        merged_table.meta.update(metadata)
        return merged_table

    def _remove_duplicates(self, merged_table):
        """Remove duplicate grid points from the merged table."""
        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

        grid_keys = [
            (row["zenith"], row["azimuth"], row[nsb_column], row[layout_column])
            for row in merged_table
        ]

        indices_to_keep = []
        seen_keys = set()
        for i in range(len(merged_table) - 1, -1, -1):
            key = grid_keys[i]
            if key not in seen_keys:
                indices_to_keep.append(i)
                seen_keys.add(key)

        indices_to_keep.sort()
        return merged_table[indices_to_keep]

    def merge_tables(self, input_files):
        """Merge multiple CORSIKA limit tables into a single table."""
        _logger.info(f"Merging {len(input_files)} CORSIKA limit tables")

        tables, metadata, loss_fractions, grid_points, duplicate_points = (
            self._read_and_collect_tables(input_files)
        )
        merged_table = self._report_and_merge(tables, metadata, loss_fractions, duplicate_points)

        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"
        merged_table.sort([layout_column, "zenith", "azimuth", nsb_column])

        if duplicate_points:
            original_count = len(merged_table)
            merged_table = self._remove_duplicates(merged_table)
            _logger.info(f"Removed {original_count - len(merged_table)} duplicate grid points")

        _logger.info(
            f"Merged table has {len(merged_table)} rows with {len(grid_points)} unique grid points"
        )
        return merged_table

    def check_grid_completeness(self, merged_table, grid_definition=None):
        """Check if the grid is complete by verifying all expected combinations exist."""
        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

        if grid_definition is None:
            grid_definition = {
                "zenith": np.unique(merged_table["zenith"]),
                "azimuth": np.unique(merged_table["azimuth"]),
                "nsb_level": np.unique(merged_table[nsb_column]),
                "layouts": np.unique(merged_table[layout_column]),
            }

        expected_combinations = list(
            product(
                grid_definition.get("zenith", []),
                grid_definition.get("azimuth", []),
                grid_definition.get("nsb_level", []),
                grid_definition.get("layouts", []),
            )
        )
        _logger.info(f"Expected {len(expected_combinations)} grid point combinations")

        found_combinations = set(
            zip(
                merged_table["zenith"],
                merged_table["azimuth"],
                merged_table[nsb_column],
                merged_table[layout_column],
            )
        )
        missing_combinations = [
            combo for combo in expected_combinations if combo not in found_combinations
        ]

        is_complete = not missing_combinations
        return is_complete, {
            "expected": len(expected_combinations),
            "found": len(found_combinations),
            "missing": missing_combinations,
        }

    def _plot_single_grid_coverage(self, ax, data, zeniths, azimuths, nsb, layout):
        """Plot grid coverage for a single NSB and layout."""
        z_grid = np.zeros((len(zeniths), len(azimuths)))
        for i, zenith in enumerate(zeniths):
            for j, azimuth in enumerate(azimuths):
                mask = (data["zenith"] == zenith) & (data["azimuth"] == azimuth)
                if np.any(mask):
                    z_grid[i, j] = 1

        az_vals = azimuths.value if hasattr(azimuths, "value") else azimuths
        zen_vals = zeniths.value if hasattr(zeniths, "value") else zeniths
        extent = [
            min(az_vals) - 0.5,
            max(az_vals) + 0.5,
            max(zen_vals) + 0.5,
            min(zen_vals) - 0.5,
        ]
        im = ax.imshow(z_grid, cmap="RdYlGn", vmin=0, vmax=1, extent=extent)

        plt.colorbar(im, ax=ax, label="Coverage (1=Present, 0=Missing)")
        ax.set_title(f"Grid Coverage: NSB={nsb}, Layout={layout}")
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel(ZENITH_LABEL)
        ax.set_xticks(az_vals)
        ax.set_yticks(zen_vals)
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="black", alpha=0.3)

    def plot_grid_coverage(self, merged_table):
        """Generate plots showing grid coverage."""
        _logger.info("Generating grid coverage plots")
        output_files = []
        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

        unique_values = {
            "zeniths": np.unique(merged_table["zenith"]),
            "azimuths": np.unique(merged_table["azimuth"]),
            "nsb_levels": np.unique(merged_table[nsb_column]),
            "layouts": np.unique(merged_table[layout_column]),
        }

        for nsb, layout in product(unique_values["nsb_levels"], unique_values["layouts"]):
            _, ax = plt.subplots(figsize=(10, 8))
            mask = (merged_table[nsb_column] == nsb) & (merged_table[layout_column] == layout)
            self._plot_single_grid_coverage(
                ax,
                merged_table[mask],
                unique_values["zeniths"],
                unique_values["azimuths"],
                nsb,
                layout,
            )
            output_file = self.output_dir / f"grid_coverage_{nsb}_{layout}.png"
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            output_files.append(output_file)
        return output_files

    def _get_mean_values(self, data, zeniths, column_name):
        """Calculate mean values for a given column, grouped by zenith."""
        mean_values = []
        for z in zeniths:
            mask = data["zenith"] == z
            values = data[mask][column_name]
            if hasattr(values, "value"):
                mean_values.append(np.mean(values.value))
            else:
                mean_values.append(np.mean(values))
        return mean_values

    def _plot_limits_for_azimuth(self, axes, data, nsb_levels, colors):
        """Plot limits for a specific azimuth."""
        legend_handles, legend_labels = [], []
        nsb_column = "nsb_level" if "nsb_level" in data.colnames else "nsb"

        for i, nsb in enumerate(nsb_levels):
            nsb_mask = data[nsb_column] == nsb
            if not np.any(nsb_mask):
                continue

            filtered_data = data[nsb_mask]
            zeniths = np.unique(filtered_data["zenith"])
            zeniths_values = zeniths.value if hasattr(zeniths, "value") else zeniths
            sort_idx = np.argsort(zeniths_values)
            zeniths_values, zeniths = zeniths_values[sort_idx], zeniths[sort_idx]

            (line,) = axes[0].plot(
                zeniths_values,
                self._get_mean_values(filtered_data, zeniths, "lower_energy_limit"),
                "o-",
                color=colors[i],
                label=f"NSB={nsb}",
            )
            axes[1].plot(
                zeniths_values,
                self._get_mean_values(filtered_data, zeniths, "upper_radius_limit"),
                "o-",
                color=colors[i],
            )
            axes[2].plot(
                zeniths_values,
                self._get_mean_values(filtered_data, zeniths, "viewcone_radius"),
                "o-",
                color=colors[i],
            )
            legend_handles.append(line)
            legend_labels.append(f"NSB={nsb}")
        return legend_handles, legend_labels

    def plot_limits(self, merged_table):
        """Generate plots showing the derived limits."""
        _logger.info("Generating limit plots")
        output_files = []
        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        azimuths = np.unique(merged_table["azimuth"])
        nsb_levels = np.unique(
            merged_table["nsb_level" if "nsb_level" in merged_table.colnames else "nsb"]
        )
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nsb_levels)))

        for layout, azimuth in product(np.unique(merged_table[layout_column]), azimuths):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            mask = (merged_table[layout_column] == layout) & (merged_table["azimuth"] == azimuth)
            if not np.any(mask):
                plt.close(fig)
                continue

            legend_handles, legend_labels = self._plot_limits_for_azimuth(
                axes, merged_table[mask], nsb_levels, colors
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
        """Write the merged table to file and save metadata."""
        if "description" not in merged_table.meta:
            merged_table.meta["description"] = (
                "Lookup table for CORSIKA limits computed from simulations."
            )
        if "loss_fraction" not in merged_table.meta:
            _logger.warning("No loss_fraction found in any of the input tables")

        merged_table.meta.update(
            {
                "created_by": "simtools-production-merge-corsika-limits",
                "creation_date": gen.now_date_time_in_isoformat(),
                "input_files_count": len(input_files),
            }
        )
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
