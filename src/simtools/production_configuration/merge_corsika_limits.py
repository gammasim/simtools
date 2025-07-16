"""Class for merging CORSIKA limit tables and checking grid completeness."""

import logging
from itertools import product
from pathlib import Path

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

    def read_file_list(self, file_list_path):
        """Read a list of input files from a text file.

        The text file should contain one file path per line.
        Lines starting with '#' are treated as comments and ignored.
        Empty lines are also ignored.

        Parameters
        ----------
        file_list_path : Path or str
            Path to the text file containing the list of input files.

        Returns
        -------
        list
            List of Path objects for the input files.

        Raises
        ------
        FileNotFoundError
            If the file list does not exist.
        """
        file_list_path = Path(file_list_path).expanduser()
        _logger.info(f"Reading input files from list file: {file_list_path}")

        if not file_list_path.exists():
            raise FileNotFoundError(f"Input files list not found: {file_list_path}")

        files = []
        with open(file_list_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    file_path = Path(line).expanduser()
                    files.append(file_path)

        _logger.info(f"Found {len(files)} files in list file {file_list_path}")
        return files

    def _read_and_collect_tables(self, input_files):
        """Read tables from files and collect metadata. Move loss_fraction from meta to column."""
        tables = []
        metadata = {}
        grid_points = set()
        duplicate_points = []

        for file_path in input_files:
            table = data_reader.read_table_from_file(file_path)
            # Move loss_fraction from meta to column
            lf = table.meta.pop("loss_fraction", None)
            if lf is not None:
                table["loss_fraction"] = lf
            tables.append(table)

            for row in table:
                grid_point = (row["zenith"], row["azimuth"], row["nsb_level"], row["array_name"])
                if grid_point in grid_points:
                    duplicate_points.append(grid_point)
                else:
                    grid_points.add(grid_point)

            if not metadata:
                metadata = table.meta

        return tables, metadata, grid_points, duplicate_points

    def _report_and_merge(self, tables, metadata, duplicate_points):
        """Report issues and merge tables."""
        if duplicate_points:
            _logger.warning(f"Found {len(duplicate_points)} duplicate grid points across tables")
            _logger.warning(f"First few duplicates: {duplicate_points[:5]}")
            _logger.warning("When duplicates exist, only the last occurrence will be kept")

        merged_table = vstack(tables, metadata_conflicts="silent")
        merged_table.meta.update(metadata)
        return merged_table

    def _remove_duplicates(self, merged_table):
        """Remove duplicate grid points from the merged table, keeping the last occurrence."""
        keys = ["array_name", "zenith", "azimuth", "nsb_level"]

        reversed_table = merged_table[::-1]
        unique_table = unique(reversed_table, keys=keys, keep="first")

        return unique_table[::-1]

    def merge_tables(self, input_files):
        """Merge multiple CORSIKA limit tables into a single table.

        This function reads and merges CORSIKA limit tables from multiple files,
        handling duplicate grid points by keeping only the last occurrence.
        It also converts the loss_fraction value from metadata to a table column
        and logs a message if multiple loss_fraction values are found.

        Parameters
        ----------
        input_files : list of Path or str
            List of paths to CORSIKA limit table files to merge.

        Returns
        -------
        astropy.table.Table
            The merged table with duplicates removed, containing all rows from input files.
            The table will be sorted by array_name, zenith, azimuth, and nsb_level.
        """
        _logger.info(f"Merging {len(input_files)} CORSIKA limit tables")

        tables, metadata, grid_points, duplicate_points = self._read_and_collect_tables(input_files)
        merged_table = self._report_and_merge(tables, metadata, duplicate_points)

        if "loss_fraction" in merged_table.colnames:
            unique_loss_fractions = np.unique(merged_table["loss_fraction"])
            if len(unique_loss_fractions) > 1:
                _logger.info(
                    f"Found multiple loss_fraction values in merged table: {unique_loss_fractions}."
                    " Make sure this is intended."
                )

        merged_table.sort(["array_name", "zenith", "azimuth", "nsb_level"])

        if duplicate_points:
            original_count = len(merged_table)
            merged_table = self._remove_duplicates(merged_table)
            _logger.info(f"Removed {original_count - len(merged_table)} duplicate grid points")

        _logger.info(
            f"Merged table has {len(merged_table)} rows with {len(grid_points)} unique grid points"
        )
        return merged_table

    def check_grid_completeness(self, merged_table, grid_definition):
        """Check if the grid is complete by verifying all expected combinations exist.

        This function checks whether all combinations of zenith, azimuth, nsb_level, and array_name
        specified in the grid_definition are present in the merged_table.

        Parameters
        ----------
        merged_table : astropy.table.Table
            The merged table containing CORSIKA limit data.
        grid_definition : dict
            Dictionary defining the grid dimensions with keys:
            'zenith': list of zenith angles,
            'azimuth': list of azimuth angles,
            'nsb_level': list of NSB levels,
            'array_names': list of array names

        Returns
        -------
        tuple
            A tuple containing: is_complete (bool) that is True if all expected combinations
            are found in the table, and info_dict (dict) with detailed information about the
            completeness check including expected points, found points, and missing combinations.
        """
        if not grid_definition:
            _logger.info("No grid definition provided, skipping completeness check.")
            return True, {}

        expected_combinations = list(
            product(
                grid_definition.get("zenith", []),
                grid_definition.get("azimuth", []),
                grid_definition.get("nsb_level", []),
                grid_definition.get("array_names", []),
            )
        )
        _logger.info(f"Expected {len(expected_combinations)} grid point combinations")

        found_combinations_set = set(
            zip(
                np.array(merged_table["zenith"].value, dtype=str),
                np.array(merged_table["azimuth"].value, dtype=str),
                np.array(merged_table["nsb_level"], dtype=str),
                np.array(merged_table["array_name"], dtype=str),
            )
        )
        _logger.info(f"Found {len(found_combinations_set)} unique grid points in merged table")

        expected_combinations_str = {tuple(map(str, combo)) for combo in expected_combinations}

        missing_combinations_str = expected_combinations_str - found_combinations_set

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
        self, ax, zeniths, azimuths, nsb, array_name, found_combinations_str
    ):
        """Plot grid coverage for a single NSB and array_name."""
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
        colors = ["red", "green"]
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        im = ax.imshow(z_grid, cmap=cmap, vmin=0, vmax=1, extent=extent)

        cbar = plt.colorbar(
            im,
            ax=ax,
            ticks=[0, 1],
            label="Coverage",
            shrink=0.25,
            pad=0.02,
        )
        cbar.set_ticklabels(["Missing", "Present"])
        ax.set_title(f"Grid Coverage: NSB={nsb}, Array Name={array_name}")
        ax.set_xlabel("Azimuth [deg]")
        ax.set_ylabel(ZENITH_LABEL)
        ax.set_xticks(az_vals)
        ax.set_yticks(zen_vals)
        ax.grid(which="major", linestyle="-", linewidth="0.5", color="black", alpha=0.3)

    def plot_grid_coverage(self, merged_table, grid_definition):
        """Generate plots showing grid coverage for each combination of NSB level and array name.

        Creates a series of heatmap plots showing which grid points (combinations of zenith and
        azimuth angles) are present or missing in the merged table, for each combination of
        NSB level and array name.

        Parameters
        ----------
        merged_table : astropy.table.Table
            The merged table containing CORSIKA limit data.
        grid_definition : dict
            Dictionary defining the grid dimensions with keys:
            'zenith': list of zenith angles,
            'azimuth': list of azimuth angles,
            'nsb_level': list of NSB levels,
            'array_names': list of array names

        Returns
        -------
        list
            List of Path objects pointing to the saved plot files.
        """
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
            "array_names": np.array(grid_definition.get("array_name", [])),
        }

        for nsb, array_name in product(unique_values["nsb_levels"], unique_values["array_names"]):
            _, ax = plt.subplots(figsize=(10, 8))
            self._plot_single_grid_coverage(
                ax,
                unique_values["zeniths"],
                unique_values["azimuths"],
                nsb,
                array_name,
                found_combinations_str,
            )
            output_file = self.output_dir / f"grid_coverage_{nsb}_{array_name}.png"
            plt.tight_layout()
            plt.savefig(output_file, bbox_inches="tight")
            plt.close()
            output_files.append(output_file)
        return output_files

    def plot_limits(self, merged_table):
        """Create plots showing the derived limits for each combination of array_name and azimuth.

        Creates plots showing the lower energy limit, upper radius limit, and viewcone radius
        versus zenith angle for each combination of array_name and azimuth angle. Each plot has
        lines for different NSB levels.

        Parameters
        ----------
        merged_table : astropy.table.Table
            The merged table containing CORSIKA limit data.

        Returns
        -------
        list
            List of Path objects pointing to the saved plot files.
        """
        _logger.info("Generating limit plots")
        output_files = []

        grouped_by_layout_az = merged_table.group_by(["array_name", "azimuth"])

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
            plt.suptitle(f"CORSIKA Limits: Array Name={array_name}, Azimuth={azimuth_value} deg")
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

            output_file = self.output_dir / f"limits_{array_name}_azimuth{azimuth_value}.png"
            plt.savefig(output_file)
            plt.close(fig)
            output_files.append(output_file)
        return output_files

    def write_merged_table(self, merged_table, output_file, input_files, grid_completeness):
        """Write the merged table to file and save metadata.

        Writes the merged table to the specified output file in ECSV format and
        saves relevant metadata about the merge process, including input files,
        grid completeness statistics, and row count.

        Parameters
        ----------
        merged_table : astropy.table.Table
            The merged table to write to file.
        output_file : Path or str
            Path where the merged table will be written.
        input_files : list of Path or str
            List of input files used to create the merged table.
        grid_completeness : dict
            Dictionary with grid completeness information from check_grid_completeness.

        Returns
        -------
        Path or str
            The path to the written file (same as output_file).
        """
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
