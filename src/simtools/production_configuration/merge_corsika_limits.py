"""Class for merging CORSIKA limit tables and checking grid completeness."""

import logging
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import unique, vstack

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
        """Remove duplicate grid points from the merged table, keeping the last occurrence."""
        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"
        keys = [layout_column, "zenith", "azimuth", nsb_column]

        # Reverse table to keep the last occurrence with unique
        reversed_table = merged_table[::-1]
        unique_table = unique(reversed_table, keys=keys, keep="first")

        # Restore original order
        return unique_table[::-1]

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

    def check_grid_completeness(self, merged_table, grid_definition):
        """Check if the grid is complete by verifying all expected combinations exist."""
        if not grid_definition:
            _logger.info("No grid definition provided, skipping completeness check.")
            return True, {}

        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

        expected_combinations = list(
            product(
                grid_definition.get("zenith", []),
                grid_definition.get("azimuth", []),
                grid_definition.get("nsb_level", []),
                grid_definition.get("layouts", []),
            )
        )
        _logger.info(f"Expected {len(expected_combinations)} grid point combinations")

        found_combinations_set = set(
            zip(
                np.array(merged_table["zenith"].value, dtype=str),
                np.array(merged_table["azimuth"].value, dtype=str),
                np.array(merged_table[nsb_column], dtype=str),
                np.array(merged_table[layout_column], dtype=str),
            )
        )
        _logger.info(f"Found {len(found_combinations_set)} unique grid points in merged table")

        expected_combinations_str = {tuple(map(str, combo)) for combo in expected_combinations}

        missing_combinations_str = expected_combinations_str - found_combinations_set

        # Find the original missing combinations (with original types)
        missing_combinations = [
            combo
            for combo in expected_combinations
            if tuple(map(str, combo)) in missing_combinations_str
        ]

        is_complete = not missing_combinations
        return is_complete, {
            "expected": len(expected_combinations),
            "found": len(found_combinations_set),
            "missing": missing_combinations,
            "found_str": found_combinations_set,
            "expected_str": expected_combinations_str,
        }

    def _plot_single_grid_coverage(
        self, ax, zeniths, azimuths, nsb, layout, found_combinations_str
    ):
        """Plot grid coverage for a single NSB and layout."""
        z_grid = np.zeros((len(zeniths), len(azimuths)))
        for i, zenith in enumerate(zeniths):
            for j, azimuth in enumerate(azimuths):
                point_str = (str(zenith), str(azimuth), str(nsb), str(layout))
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
        im = ax.imshow(z_grid, cmap="RdYlGn", vmin=0, vmax=1, extent=extent)

        plt.colorbar(im, ax=ax, label="Coverage (1=Present, 0=Missing)")
        ax.set_title(f"Grid Coverage: NSB={nsb}, Layout={layout}")
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel(ZENITH_LABEL)
        ax.set_xticks(az_vals)
        ax.set_yticks(zen_vals)
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="black", alpha=0.3)

    def plot_grid_coverage(self, merged_table, grid_definition):
        """Generate plots showing grid coverage."""
        if not grid_definition:
            _logger.info("No grid definition provided, skipping grid coverage plots.")
            return []

        _logger.info("Generating grid coverage plots")
        output_files = []

        _, completeness_info = self.check_grid_completeness(merged_table, grid_definition)
        found_combinations_str = completeness_info.get("found_str", set())

        unique_values = {
            "zeniths": np.array(grid_definition.get("zenith", [])),
            "azimuths": np.array(grid_definition.get("azimuth", [])),
            "nsb_levels": np.array(grid_definition.get("nsb_level", [])),
            "layouts": np.array(grid_definition.get("layouts", [])),
        }

        for nsb, layout in product(unique_values["nsb_levels"], unique_values["layouts"]):
            _, ax = plt.subplots(figsize=(10, 8))
            self._plot_single_grid_coverage(
                ax,
                unique_values["zeniths"],
                unique_values["azimuths"],
                nsb,
                layout,
                found_combinations_str,
            )
            output_file = self.output_dir / f"grid_coverage_{nsb}_{layout}.png"
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            output_files.append(output_file)
        return output_files

    def plot_limits(self, merged_table):
        """Generate plots showing the derived limits."""
        _logger.info("Generating limit plots")
        output_files = []
        layout_column = "layout" if "layout" in merged_table.colnames else "array_name"
        nsb_column = "nsb_level" if "nsb_level" in merged_table.colnames else "nsb"

        # Group data by layout and azimuth for plotting
        grouped_by_layout_az = merged_table.group_by([layout_column, "azimuth"])

        for group in grouped_by_layout_az.groups:
            layout = group[layout_column][0]
            azimuth = group["azimuth"][0]
            azimuth_value = azimuth.value if hasattr(azimuth, "value") else azimuth

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            legend_handles, legend_labels = [], []

            # Further group by NSB level to plot lines
            grouped_by_nsb = group.group_by(nsb_column)
            colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(grouped_by_nsb.groups)))

            for i, nsb_group in enumerate(grouped_by_nsb.groups):
                nsb_level = nsb_group[nsb_column][0]
                # Aggregate data by zenith angle to get mean values for plotting
                plot_columns = [
                    "zenith",
                    "lower_energy_limit",
                    "upper_radius_limit",
                    "viewcone_radius",
                ]
                agg_data = nsb_group[plot_columns].group_by("zenith").groups.aggregate(np.mean)
                agg_data.sort("zenith")
                zeniths = agg_data["zenith"].value

                (line,) = axes[0].plot(
                    zeniths, agg_data["lower_energy_limit"], "o-", color=colors[i]
                )
                axes[1].plot(zeniths, agg_data["upper_radius_limit"], "o-", color=colors[i])
                axes[2].plot(zeniths, agg_data["viewcone_radius"], "o-", color=colors[i])
                legend_handles.append(line)
                legend_labels.append(f"NSB={nsb_level}")

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
