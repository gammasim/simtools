#!/usr/bin/python3

r"""
Derive CORSIKA configuration limits for energy, core distance, and viewcone radius.

This tool determines configuration limits based on triggered events from broad-range
simulations. It supports setting:

- **ERANGE**: Derives the lower energy limit; upper limit is user-defined.
- **CSCAT**: Derives the upper core distance; lower limit is user-defined.
- **VIEWCONE**: Derives the viewcone radius; lower limit is user-defined.

Limits are computed based on a configurable maximum event loss fraction.
Results are provided as a table with the following columns:

- particle_type: Particle type (e.g., gamma, proton, electron).
- telescope_ids: List of telescope IDs used in the simulation.
- zenith: Zenith angle.
- azimuth: Azimuth angle.
- nsb: Night sky background level
- layout: Layout of the telescope array used in the simulation.
- lower_energy_limit: Derived lower energy limit.
- upper_radius_limit: Derived upper radial distance limit.
- viewcone_radius: Derived upper viewcone radius limit.

The input event data files are generated using the application simtools-generate-simtel-event-data
and are required for each point in the lookup table.

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

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.production_configuration.derive_corsika_limits_grid import (
    generate_corsika_limits_grid,
)


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        description="Derive limits for energy, radial distance, and viewcone."
    )
    config.parser.add_argument(
        "--event_data_files",
        type=str,
        nargs="+",
        required=True,
        help="List of event data files or ascii file listing data files ",
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
    return config.initialize(db_config=False, output=True)


def main():
    """Derive limits for energy, radial distance, and viewcone."""
    args_dict, _ = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict.get("log_level", "info")))

    generate_corsika_limits_grid(args_dict)


if __name__ == "__main__":
    main()
