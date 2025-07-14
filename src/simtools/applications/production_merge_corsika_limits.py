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

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.production_configuration.merge_corsika_limits import CorsikaMergeLimits

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

    merger = CorsikaMergeLimits()

    merged_table = merger.merge_tables(input_files)

    grid_definition = gen.collect_data_from_file(args_dict["grid_definition"])

    is_complete, grid_completeness = merger.check_grid_completeness(merged_table, grid_definition)

    if args_dict.get("plot_grid_coverage"):
        merger.plot_grid_coverage(merged_table)

    if args_dict.get("plot_limits"):
        merger.plot_limits(merged_table)

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


if __name__ == "__main__":
    main()
