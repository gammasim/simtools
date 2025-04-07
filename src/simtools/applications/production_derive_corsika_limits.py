#!/usr/bin/python3

r"""
Derives the limits for energy, radial distance, and viewcone to be used in CORSIKA simulations.

The limits are derived based on the event loss fraction specified by the user.

Command line arguments
----------------------
event_data_files (str, required)
    Path to a file containing event data file paths.
telescope_ids (str, required)
    Path to a file containing telescope configurations.
loss_fraction (float, required)
    Fraction of events to be lost.
plot_histograms (bool, optional)
    Plot histograms of the event data.
output_file (str, optional)
    Path to the output file for the derived limits.

Example
-------
Derive limits for a given file with a specified loss fraction.

.. code-block:: console

    simtools-production-derive-corsika-limits \\
        --event_data_files path/to/event_data_files.yaml \\
        --telescope_ids path/to/telescope_configs.yaml \\
        --loss_fraction 1e-6 \\
        --plot_histograms \\
        --output_file corsika_simulation_limits_lookup.ecsv
"""

import datetime
import logging
import re

from astropy.table import Table

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler
from simtools.production_configuration.derive_corsika_limits import LimitCalculator

_logger = logging.getLogger(__name__)


def _parse():
    """
    Parse command line configuration.

    Parameters
    ----------
    event_data_files: str
        Path to a file listing event data file paths. These files contain the
        simulation data used for deriving the limits.
    loss_fraction: float
        Fraction of events to be excluded during limit computation. Determines
        thresholds for energy, radial distance, and viewcone.
    telescope_ids: str
        Path to a file defining telescope configurations. Specifies telescope
        arrays or IDs used to filter events during processing.
    plot_histograms: bool
        If True, generates and saves histograms of the event data to visualize
        the computed limits and distributions.

    Returns
    -------
    CommandLineParser
        Command line parser object
    """
    config = configurator.Configurator(
        description="Derive limits for energy, radial distance, and viewcone."
    )
    config.parser.add_argument(
        "--event_data_files",
        type=str,
        required=True,
        help="Path to a file containing event data file paths.",
    )
    config.parser.add_argument(
        "--telescope_ids",
        type=str,
        required=True,
        help="Path to a file containing telescope configurations.",
    )
    config.parser.add_argument(
        "--loss_fraction", type=float, required=True, help="Fraction of events to be lost."
    )
    config.parser.add_argument(
        "--plot_histograms",
        help="Plot histograms of the event data.",
        action="store_true",
        default=False,
    )
    config.parser.add_argument(
        "--output_file",
        type=str,
        default="corsika_simulation_limits_lookup.ecsv",
        help="Output file for the derived limits (default: "
        "'corsika_simulation_limits_lookup.ecsv').",
    )
    return config.initialize(db_config=False)


def process_file(file_path, telescope_ids, loss_fraction, plot_histograms):
    """
    Process a single file and compute limits.

    Parameters
    ----------
    file_path : str
        Path to the event data file.
    telescope_ids : list[int]
        List of telescope IDs to filter the events.
    loss_fraction : float
        Fraction of events to be lost.
    plot_histograms : bool
        Whether to plot histograms.

    Returns
    -------
    dict
        Dictionary containing the computed limits and metadata.
    """
    match = re.search(r"za(\d+)deg.*azm(\d+)deg", file_path)
    if not match:
        raise ValueError(f"Could not extract zenith and azimuth from file path: {file_path}")
    zenith = int(match.group(1))
    azimuth = int(match.group(2))

    if "dark" in file_path:
        nsb = "dark"
    elif "moon" in file_path:
        nsb = "moon"
    else:
        _logger.warning(f"Could not determine NSB (dark or moon) from file path: {file_path}")
        nsb = "unknown"

    calculator = LimitCalculator(file_path, telescope_list=telescope_ids)

    lower_energy_limit = calculator.compute_lower_energy_limit(loss_fraction)
    upper_radial_distance = calculator.compute_upper_radial_distance(loss_fraction)
    viewcone = calculator.compute_viewcone(loss_fraction)

    if plot_histograms:
        _logger.info(
            f"Plotting histograms written to {io_handler.IOHandler().get_output_directory()}"
        )
        calculator.plot_data(
            lower_energy_limit,
            upper_radial_distance,
            viewcone,
            io_handler.IOHandler().get_output_directory(),
        )

    return {
        "telescope_ids": telescope_ids,
        "zenith": zenith,
        "azimuth": azimuth,
        "nsb": nsb,
        "lower_energy_threshold": lower_energy_limit,
        "upper_radius_threshold": upper_radial_distance,
        "viewcone_radius": viewcone,
    }


def create_results_table(results, loss_fraction):
    """
    Create an Astropy Table from the results.

    Parameters
    ----------
    results : list[dict]
        List of dictionaries containing the computed limits for each file
        and telescope configuration.
    loss_fraction : float
        Fraction of events to be lost, added as metadata to the table.

    Returns
    -------
    astropy.table.Table
        An Astropy Table containing the results with appropriate units and metadata.
    """
    table = Table(
        rows=[
            (
                res["file_path"],
                res["telescope_ids"],
                res["zenith"],
                res["azimuth"],
                res["nsb"],
                res["lower_energy_threshold"],
                res["upper_radius_threshold"],
                res["viewcone_radius"],
            )
            for res in results
        ],
        names=[
            "file_path",
            "telescope_ids",
            "zenith",
            "azimuth",
            "nsb",
            "lower_energy_threshold",
            "upper_radius_threshold",
            "viewcone_radius",
        ],
    )

    table["lower_energy_threshold"].unit = "TeV"
    table["upper_radius_threshold"].unit = "m"
    table["viewcone_radius"].unit = "deg"

    table.meta["created"] = datetime.datetime.now().isoformat()
    table.meta["description"] = (
        "Lookup table for CORSIKA limits computed from gamma-ray shower simulations "
        "using simtool production_derive_corsika_limits"
    )
    table.meta["loss_fraction"] = loss_fraction

    return table


def main():
    """Derive limits for energy, radial distance, and viewcone."""
    args_dict, _ = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict.get("log_level", "info")))

    event_data_files = gen.collect_data_from_file(args_dict["event_data_files"])["files"]
    telescope_configs = gen.collect_data_from_file(args_dict["telescope_ids"])["telescope_configs"]

    results = []
    for file_path in event_data_files:
        for array_name, telescope_ids in telescope_configs.items():
            _logger.info(f"Processing file: {file_path} with telescope config: {array_name}")
            result = process_file(
                file_path,
                telescope_ids,
                args_dict["loss_fraction"],
                args_dict["plot_histograms"],
            )
            result["layout"] = array_name
            results.append(result)

    table = create_results_table(results, args_dict["loss_fraction"])

    output_dir = io_handler.IOHandler().get_output_directory("corsika_limits")
    output_file = f"{output_dir}/{args_dict['output_file']}"

    table.write(output_file, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Results saved to {output_file}")

    metadata_file = f"{output_dir}/metadata.yml"
    MetadataCollector.dump(args_dict, metadata_file)
    _logger.info(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
