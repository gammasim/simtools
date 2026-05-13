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

Multi-production support
------------------------
To process multiple independent production datasets in parallel, provide multiple glob patterns
via ``--event_data_file``. Each pattern is processed as a separate production, with results
merged into a single output ECSV file. Plot files are organized into per-production subdirectories.

Command line arguments
----------------------
event_data_file (str or list of str, required)
    Path or glob pattern for reduced event data files. Can be a single pattern or
    multiple patterns (one per ``--event_data_file`` argument) to enable parallel
    multi-production processing.
telescope_ids (str, optional)
    Custom array layout file containing telescope IDs.
loss_fraction (float, required)
    Maximum event-loss fraction for limit computation.
plot_histograms (bool, optional)
    Plot histograms of the event data.
output_file (str, optional)
    Path to the output file for the derived limits.
n_workers (int, optional)
    Number of worker processes to use for execution. Default is 1.

Example
-------

Derive limits for a single production with a list of array layouts:

.. code-block:: console

    simtools-production-derive-corsika-limits \\
        --event_data_file event_dat_file.hdf5 \\
        --array_layout_name alpha,beta \\
        --loss_fraction 1e-6 \\
        --plot_histograms \\
        --output_file corsika_simulation_limits.ecsv

Derive limits for a single production with a given file for custom defined array layouts:

.. code-block:: console

    simtools-production-derive-corsika-limits \\
        --event_data_file event_dat_file.hdf5 \\
        --telescope_ids path/to/telescope_configs.yaml \\
        --loss_fraction 1e-6 \\
        --plot_histograms \\
        --output_file corsika_simulation_limits.ecsv

Derive limits for multiple independent productions in parallel:

.. code-block:: console

    simtools-production-derive-corsika-limits \\
        --event_data_file pattern_1_*.hdf5 \\
        --event_data_file pattern_2_*.hdf5 \\
        --array_layout_name alpha \\
        --loss_fraction 1e-6 \\
        --plot_histograms \\
        --n_workers 4 \\
        --output_file corsika_simulation_limits.ecsv

When multiple ``--event_data_file`` patterns are provided, results are merged into a single output
ECSV with a ``production_index`` column, and plot files are organized into per-production
subdirectories derived from the input pattern names (for example ``production_production_a/``).
"""

from simtools.application_control import build_application
from simtools.production_configuration.derive_corsika_limits import (
    generate_corsika_limits_grid,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.initialize_application_arguments(["telescope_ids"])
    # Override event_data_file to allow multiple patterns for multi-production support
    parser.add_argument(
        "--event_data_file",
        help=(
            "Event data file or glob pattern (one or more patterns for "
            "multi-production processing)."
        ),
        nargs="+",
        action="extend",
        required=True,
    )
    parser.add_argument(
        "--loss_fraction",
        type=float,
        required=True,
        help="Maximum event-loss fraction for limit computation.",
    )
    parser.add_argument(
        "--plot_histograms",
        help="Plot histograms of the event data.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_workers",
        help=(
            "Number of worker processes to use for execution "
            "(default: 1; set to 0 for auto-detection of available cores)."
        ),
        type=int,
        required=False,
        default=1,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "output": True,
            "simulation_model": [
                "site",
                "model_version",
                "layout",
            ],
        },
    )

    generate_corsika_limits_grid(app_context.args)


if __name__ == "__main__":
    main()
