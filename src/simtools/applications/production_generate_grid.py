#!/usr/bin/python3

r"""
Generate simulation job grid for production configurations.

This application expands simulation job rows and writes them to disk.
It supports both:

- explicit cartesian job-grid configuration (primary, zenith, energy range, etc.), and
- axes-based production-grid configuration with optional ``ra_dec`` coordinate handling
  and lookup-table interpolation.

Allow for flexible scaling of showers per run and total showers across the grid.
``showers_per_run_power_law`` scales the baseline showers per run with
``(E_mid / E_ref) ** power_index``, using the logarithmic midpoint energy of each bin.
``total_showers_scaling=zenith_scaled`` applies ``total_showers * exp(factor * (cos(ZD) - 1))``.

Command line arguments
----------------------
axis (repeatable)
    Compact axis definition in the form
    ``--axis <name> <min> <unit> <max> <unit> <binning> [scaling]``.
    Example: ``--axis azimuth 310 deg 20 deg 3 linear``.
    Options for scaling are: linear, log, 1/cos.
time_of_observation (str, optional)
    Time of the observation in UTC (format: 'YYYY-MM-DD HH:MM:SS').
    Used only if RA/Dec axes are provided (for coordinate transforms and sidereal-time
    sampling). Ignored otherwise.
corsika_limits (str, optional)
    Path to the lookup table for simulation limits. The table should contain
    varying azimuth and/or zenith angles for the selected array layout.
output_file (str, optional, default='job_grid.ecsv')
    Output file for the generated executable job grid.
showers_per_run_power_law (tuple, optional)
    Scale showers per run with energy as
    ``<power_index> <reference_energy_value> <reference_energy_unit>``
    (example: ``--showers_per_run_power_law -2.0 1 TeV``).


Example
-------
To generate a standard zenith/azimuth grid of simulation points, execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --axis azimuth 310 deg 20 deg 3 linear \
            --axis zenith 30 deg 40 deg 2 linear \
            --axis nsb 4 MHz 5 MHz 2 linear \
            --axis offset 0 deg 10 deg 2 linear \
            --corsika_limits tests/resources/corsika_simulation_limits/merged_corsika_limits.ecsv

To generate an all-sky RA/Dec direction grid and serialize output in RA/Dec,
execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --axis ra 0 deg 360 deg 36 linear \
            --axis dec -90 deg 90 deg 18 linear \
            --axis nsb 4 MHz 4 MHz 1 linear \
            --axis offset 0 deg 10 deg 2 linear \
            --time_of_observation "2017-09-16 00:00:00" \
            --corsika_limits tests/resources/corsika_simulation_limits/merged_corsika_limits.ecsv
"""

from simtools.application_control import build_application
from simtools.configuration import defaults
from simtools.production_configuration.job_grid_io import serialize_job_grid
from simtools.production_configuration.simulation_jobs import (
    build_job_grid_metadata,
    build_simulation_jobs,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--axis",
        action="append",
        nargs="+",
        required=False,
        help=(
            "Compact axis definition: --axis <name> <min> <unit> <max> <unit> <binning> "
            "[scaling]. May be repeated. "
            "Options for scaling are: linear, log, 1/cos"
        ),
    )
    parser.add_argument(
        "--time_of_observation",
        type=str,
        required=False,
        help=(
            "Observation time in UTC (format: 'YYYY-MM-DD HH:MM:SS'). Used only in 'ra_dec' mode."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="job_grid.ecsv",
        help="Output file for the generated executable job grid.",
    )
    parser.add_argument(
        "--corsika_limits",
        type=str,
        required=False,
        help="Path to the lookup table for simulation limits. "
        "Table required with varying azimuth and or zenith angle. ",
    )
    parser.add_argument(
        "--number_of_runs",
        help="Number of runs to be simulated.",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--total_showers",
        help="Total number of showers to simulate.",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--total_showers_scaling",
        help="Scaling mode for total showers.",
        type=str,
        choices=["fixed", "zenith_scaled"],
        required=False,
        default="fixed",
    )
    parser.add_argument(
        "--zenith_angle_scaling_factor",
        help=(
            "Scaling factor for zenith-dependent total_showers scaling. "
            "Used only when --total_showers_scaling is 'zenith_scaled'."
        ),
        type=float,
        required=False,
        default=defaults.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT,
    )
    parser.add_argument(
        "--showers_per_run_power_law",
        help=(
            "Scale showers_per_run by (E_mid / E_ref) ** power_index using the bin midpoint: "
            "<power_index> <reference_energy_value> <reference_energy_unit> "
            "(for example: --showers_per_run_power_law -2.0 1 TeV)."
        ),
        nargs=3,
        type=str,
        metavar=("POWER_INDEX", "REFERENCE_ENERGY_VALUE", "REFERENCE_ENERGY_UNIT"),
        required=False,
        default=None,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "preserve_by_version_keys": ["array_layout_name"],
            "simulation_model": ["site", "layout", "telescope", "model_version"],
            "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
        },
    )

    job_rows = build_simulation_jobs(app_context.args)
    serialize_job_grid(
        job_rows=job_rows,
        output_file=app_context.io_handler.get_output_file(app_context.args["output_file"]),
        metadata=build_job_grid_metadata(app_context.args),
    )


if __name__ == "__main__":
    main()
