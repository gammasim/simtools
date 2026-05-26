"""Class for merging CORSIKA limit tables and checking grid completeness."""

import logging
from itertools import product
from pathlib import Path

import numpy as np
from astropy.table import unique, vstack

import simtools.utils.general as gen
from simtools.data_model import data_reader
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler, io_handler

_logger = logging.getLogger(__name__)


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
        # Track grid points and their associated values to check for inconsistencies
        grid_point_values = {}
        duplicate_points = []
        inconsistent_points = []

        for file_path in input_files:
            table = data_reader.read_table_from_file(file_path)
            # Move loss_fraction from meta to column
            table["loss_fraction"] = table.meta.pop("loss_fraction")
            tables.append(table)

            for row in table:
                grid_point = (row["zenith"], row["azimuth"], row["nsb_level"], row["array_name"])

                if grid_point in grid_point_values:
                    duplicate_points.append(grid_point)

                    current_values = {
                        col: row[col]
                        for col in row.colnames
                        if col not in ["zenith", "azimuth", "nsb_level", "array_name"]
                    }
                    previous_values = grid_point_values[grid_point]

                    keys_to_compare = set(current_values.keys()) & set(previous_values.keys()) - {
                        "telescope_ids"
                    }
                    if any(
                        not np.array_equal(current_values[k], previous_values[k])
                        for k in keys_to_compare
                    ):
                        inconsistent_points.append(
                            {
                                "grid_point": grid_point,
                                "file": str(file_path),
                                "previous_file": grid_point_values[grid_point]["__file__"],
                            }
                        )

                    grid_point_values[grid_point] = current_values
                    grid_point_values[grid_point]["__file__"] = str(file_path)
                else:
                    values = {
                        col: row[col]
                        for col in row.colnames
                        if col not in ["zenith", "azimuth", "nsb_level", "array_name"]
                    }
                    values["__file__"] = str(file_path)
                    grid_point_values[grid_point] = values

            if not metadata:
                metadata = table.meta

        return (
            tables,
            metadata,
            set(grid_point_values.keys()),
            duplicate_points,
            inconsistent_points,
        )

    def _report_and_merge(self, tables, metadata, duplicate_points, inconsistent_points):
        """Report issues and merge tables.

        Parameters
        ----------
        tables : list
            List of tables to merge.
        metadata : dict
            Metadata to include in the merged table.
        duplicate_points : list
            List of grid points that occur in multiple tables.
        inconsistent_points : list
            List of grid points with inconsistent values across tables.

        Returns
        -------
        astropy.table.Table
            The merged table.

        Raises
        ------
        ValueError
            If inconsistent duplicate grid points are found.
        """
        if duplicate_points:
            _logger.warning(f"Found {len(duplicate_points)} duplicate grid points across tables")
            _logger.warning(f"First few duplicates: {duplicate_points[:5]}")

            if inconsistent_points:
                message = (
                    f"Found {len(inconsistent_points)} grid points with inconsistent values in "
                    "tables. This likely indicates an issue with the input data. "
                    f"First inconsistent point: {inconsistent_points[0]}"
                )
                _logger.error(message)
                raise ValueError(message)

            _logger.info("All duplicates have consistent values. Last occurrence will be kept.")

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
        handling duplicate grid points by checking for consistency and raising an
        error if inconsistent duplicates are found. It also converts the loss_fraction
        value from metadata to a table column and logs a message if multiple
        loss_fraction values are found.

        Parameters
        ----------
        input_files : list of Path or str
            List of paths to CORSIKA limit table files to merge.

        Returns
        -------
        astropy.table.Table
            The merged table with duplicates removed, containing all rows from input files.
            The table will be sorted by array_name, zenith, azimuth, and nsb_level.

        Raises
        ------
        ValueError
            If inconsistent duplicate grid points are found.
        """
        _logger.info(f"Merging {len(input_files)} CORSIKA limit tables")

        tables, metadata, grid_points, duplicate_points, inconsistent_points = (
            self._read_and_collect_tables(input_files)
        )
        merged_table = self._report_and_merge(
            tables, metadata, duplicate_points, inconsistent_points
        )

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
            'array_name': list of array name

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
                grid_definition.get("array_name", []),
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


def _read_grid_definition(grid_definition):
    """Read grid definition from file if provided."""
    return ascii_handler.collect_data_from_file(grid_definition) if grid_definition else None


def resolve_input_files_and_table(args_dict, merger):
    """Resolve input files and merged table from command line arguments."""
    if args_dict.get("merged_table"):
        merged_table_path = Path(args_dict["merged_table"]).expanduser()
        merged_table = data_reader.read_table_from_file(merged_table_path)
        return merged_table, [merged_table_path], True

    if not args_dict.get("input_files") and not args_dict.get("input_files_list"):
        raise ValueError(
            "Either --input_files, --input_files_list, or --merged_table must be provided."
        )

    input_files = []
    if args_dict.get("input_files"):
        raw_paths = args_dict.get("input_files")
        if len(raw_paths) == 1 and Path(raw_paths[0]).expanduser().is_dir():
            input_dir = Path(raw_paths[0]).expanduser()
            input_files.extend(input_dir.glob("*.ecsv"))
        else:
            input_files.extend(Path(file_name).expanduser() for file_name in raw_paths)

    if args_dict.get("input_files_list"):
        input_files.extend(merger.read_file_list(args_dict["input_files_list"]))

    if not input_files:
        raise FileNotFoundError(
            "No input files found. Check --input_files or --input_files_list arguments."
        )

    return merger.merge_tables(input_files), input_files, False


def merge_corsika_limits(args_dict, merger=None):
    """
    Run table merge, completeness checks, and optional write-out.

    Parameters
    ----------
    args_dict : dict
        Dictionary with command line arguments.
    merger : CorsikaMergeLimits, optional
        An instance of CorsikaMergeLimits to use for merging and plotting. If None, a
        new instance will be created.
    """
    merger = merger or CorsikaMergeLimits()
    grid_definition = _read_grid_definition(args_dict.get("grid_definition"))

    merged_table, input_files, from_merged_table = resolve_input_files_and_table(args_dict, merger)

    is_complete, grid_completeness = merger.check_grid_completeness(merged_table, grid_definition)

    if not from_merged_table:
        output_file = merger.output_dir / args_dict["output_file"]
        merger.write_merged_table(
            merged_table,
            output_file,
            input_files,
            {
                "is_complete": is_complete,
                "expected": grid_completeness.get("expected", 0),
                "found": grid_completeness.get("found", 0),
                "missing": grid_completeness.get("missing", []),
            },
        )
