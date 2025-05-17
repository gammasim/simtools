#!/usr/bin/python3

r"""
Derive CORSIKA configuration limits for energy, core distance, and viewcone angle.

This tool determines configuration limits based on triggered events from broad-range
simulations. It supports setting:

- **ERANGE**: Derives the lower energy limit; upper limit is user-defined.
- **CSCAT**: Derives the upper core distance; lower limit is user-defined.
- **VIEWCONE**: Derives the viewcone radius; upper limit is user-defined.

Limits are computed based on a user-defined maximum event loss fraction.

- particle_type: Particle type (e.g., gamma, proton, electron).
- telescope_ids: List of telescope IDs used in the simulation.
- zenith: Zenith angle.
- azimuth: Azimuth angle.
- nsb: Night sky background level (e.g., 'dark', 'half moon', 'moon').
- layout: Layout of the telescope array used in the simulation.
- lower_energy_threshold: Derived lower energy limit.
- upper_radius_threshold: Derived upper radial distance limit.
- viewcone_radius: Derived viewcone radius limit.

The input event data files are generated using the application simtools-generate-simtel-event-data
and is required for each point in the lookup table.

Command line arguments
----------------------
event_data_files (str, required)
    Path to a file containing event data files derived with 'simtools-generate-simtel-event-data'.
telescope_ids (str, required)
    Path to a file containing telescope configurations.
loss_fraction (float, required)
    Maximum event-loss fraction for limit computation.
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

from astropy.table import Table

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler
from simtools.production_configuration.derive_corsika_limits import LimitCalculator

_logger = logging.getLogger(__name__)


def _parse():
    """Parse command line configuration."""
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
        "--loss_fraction",
        type=float,
        required=True,
        help="Maximum event-loss fraction for limit computation.",
    )
    config.parser.add_argument(
        "--plot_histograms",
        help="Plot histograms of the event data.",
        action="store_true",
        default=False,
    )
    # TODO
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
    Compute limits for a single file.

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
    calculator = LimitCalculator(file_path, telescope_list=telescope_ids)
    limits = calculator.compute_limits(loss_fraction)

    if plot_histograms:
        calculator.plot_data(io_handler.IOHandler().get_output_directory())

    return limits


def create_results_table(results, loss_fraction):
    """
    Convert list of simulation results to an Astropy Table with metadata.

    Parameters
    ----------
    results : list[dict]
        Computed limits per file and telescope configuration.
    loss_fraction : float
        Fraction of lost events (added to metadata).

    Returns
    -------
    astropy.table.Table
        Table of results with units and metadata.
    """
    cols = [
        "primary_particle",
        "telescope_ids",
        "zenith",
        "azimuth",
        "nsb_level",
        "lower_energy_threshold",
        "upper_radius_threshold",
        "viewcone_radius",
    ]
    table = Table(
        rows=[
            [res[k].value if hasattr(res[k], "value") else res[k] for k in cols] for res in results
        ],
        names=cols,
    )

    # TODO - is there a better way to set units? To know them from the metadata?
    # TODO - from simtel_config_io_write / reader?
    table["zenith"].unit = "deg"
    table["azimuth"].unit = "deg"
    table["lower_energy_threshold"].unit = "TeV"
    table["upper_radius_threshold"].unit = "m"
    table["viewcone_radius"].unit = "deg"

    table.meta.update(
        {
            "created": datetime.datetime.now().isoformat(),
            "description": "Lookup table for CORSIKA limits computed from gamma-ray simulations.",
            "loss_fraction": loss_fraction,
        }
    )

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

    # TODO add metadata to the table
    # TODO command line arguments should be in metadata?
    metadata_file = f"{output_dir}/metadata.yml"
    MetadataCollector.dump(args_dict, metadata_file)


if __name__ == "__main__":
    main()
