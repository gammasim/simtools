#!/usr/bin/python3

r"""
Derive CORSIKA configuration limits for energy, core distance, and viewcone radius.

This tool determines configuration limits based on triggered events from broad-range
simulations. It supports the derivation of the following CORSIKA configuration parameters:

- **ERANGE**: lower energy limit
- **CSCAT**: upper core distance
- **VIEWCONE**: viewcone radius

Broad-range simulations in this context are simulation sets generated with wide-ranging
definitions for above parameters.
Limits are computed based on a configurable maximum event loss fraction.
Results are provided as a table with the following columns:

+---------------------+-----------+--------+-----------------------------------------------+
| Field               | Data Type | Units  | Description                                   |
+=====================+===========+========+===============================================+
| primary_particle    | string    |        | Particle type (e.g., gamma, proton).          |
+---------------------+-----------+--------+-----------------------------------------------+
| array_name          | string    |        | Array name (custom or as defined in           |
|                     |           |        | 'array_layouts').                             |
+---------------------+-----------+--------+-----------------------------------------------+
| telescope_ids       | string    |        | Comma-separated list of telescope IDs         |
|                     |           |        | of this array.                                |
+---------------------+-----------+--------+-----------------------------------------------+
| zenith              | float64   | deg    | Direction of array pointing zenith.           |
+---------------------+-----------+--------+-----------------------------------------------+
| azimuth             | float64   | deg    | Direction of array pointing azimuth.          |
+---------------------+-----------+--------+-----------------------------------------------+
| nsb_level           | float64   |        | Night sky background level.                   |
+---------------------+-----------+--------+-----------------------------------------------+
| lower_energy_limit  | float64   | TeV    | Derived lower energy limit (**ERANGE**)       |
+---------------------+-----------+--------+-----------------------------------------------+
| upper_radius_limit  | float64   | m      | Derived upper core distance limit (**CSCAT**) |
+---------------------+-----------+--------+-----------------------------------------------+
| viewcone_radius     | float64   | deg    | Derived viewcone radius limit (**VIEWCONE**)  |
+---------------------+-----------+--------+-----------------------------------------------+

The input event data files are generated using the application simtools-generate-simtel-event-data
and are required for each point in the observational parameter space (e.g., array pointing
directions, level of night sky background, etc.).

Command line arguments
----------------------
event_data_file (str, required)
    Path to reduced event data file.
telescope_ids (str, optional)
    Custom array layout file containing telescope IDs.
loss_fraction (float, required)
    Maximum event-loss fraction for limit computation.
plot_histograms (bool, optional)
    Plot histograms of the event data.
output_file (str, optional)
    Path to the output file for the derived limits.

Example
-------

Derive limits for a list of array layouts (use 'all' to derive limits for all layouts):

.. code-block:: console

    simtools-production-derive-corsika-limits \\
        --event_data_file event_dat_file.hdf5 \\
        --array_layout_name alpha,beta \\
        --loss_fraction 1e-6 \\
        --plot_histograms \\
        --output_file corsika_simulation_limits_lookup.ecsv

Derive limits for a given file for custom defined array layouts:

.. code-block:: console

    simtools-production-derive-corsika-limits \\
        --event_data_file event_dat_file.hdf5 \\
        --telescope_ids path/to/telescope_configs.yaml \\
        --loss_fraction 1e-6 \\
        --plot_histograms \\
        --output_file corsika_simulation_limits_lookup.ecsv
"""

import logging

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.production_configuration.derive_corsika_limits import (
    generate_corsika_limits_grid,
)


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        description="Derive limits for energy, radial distance, and viewcone."
    )
    config.parser.add_argument(
        "--event_data_file",
        type=str,
        required=True,
        help="Event data file containing reduced event data.",
    )
    config.parser.add_argument(
        "--telescope_ids",
        type=str,
        required=False,
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
    return config.initialize(
        db_config=True,
        output=True,
        simulation_model=[
            "site",
            "model_version",
            "layout",
        ],
    )


def main():
    """Derive limits for energy, radial distance, and viewcone."""
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict.get("log_level", "info")))

    generate_corsika_limits_grid(args_dict, db_config)


if __name__ == "__main__":
    main()
