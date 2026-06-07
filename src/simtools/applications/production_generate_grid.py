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
``showers_per_run_scaling=cosine_zenith`` scales showers per run with
``cos(zenith_angle)`` to control run duration at larger zenith angles.
``total_showers_scaling=zenith_scaled`` applies ``total_showers * exp(factor * (cos(ZD) - 1))``.

Command line arguments
----------------------
axis (repeatable)
    Compact axis definition in the form
    ``--axis <name> <min> <unit> <max> <unit> <binning> [scaling]``.
    Example: ``--axis azimuth 310 deg 20 deg 3 linear``.
    Options for scaling are: linear, log, 1/cos.
direction_grid_density (float or quantity, optional)
    Direction-grid density (typically in ``1/deg^2``).
    If provided, direction-axis binning (azimuth/zenith or ra/dec) is derived from
    range and density, while min/max values are kept from ``--axis`` definitions.
    In ``ra_dec`` mode, local-sky constraints can be defined with
    ``local_zenith_range`` and ``local_azimuth_range``.
local_zenith_range (quantity pair, optional)
    Zenith range (deg) used to filter generated RA/Dec density nodes in local sky,
    for example ``0 deg 70 deg``.
local_azimuth_range (quantity pair, optional)
    Directed azimuth range (deg) used to filter generated RA/Dec density nodes in
    local sky, for example ``300 deg 60 deg``.
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
    Use for fixed energy simulations only.
showers_per_run_scaling (str, optional)
    Zenith-angle showers-per-run scaling mode.
    ``fixed`` keeps showers per run unchanged.
    ``cosine_zenith`` applies ``showers_per_run * cos(zenith_angle)``
    (example: ``--showers_per_run_scaling cosine_zenith``).
energy_max_scaling_index (float, optional)
    Scale the configured max energy with zenith angle as
    ``energy_max_zenith * cos(zenith_angle) ** energy_max_scaling_index``,
    where ``energy_max_zenith`` is the configured max value of ``energy_range``.
    Disabled by default (``None``).
    Example: ``--energy_max_scaling_index -2.5``.


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

To generate an RA/Dec density grid constrained to local sky ranges (for example
full zenith coverage from 0 to 70 deg and a directed azimuth window), execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --axis ra 0 deg 360 deg 1 linear \
            --axis dec -40 deg 80 deg 1 linear \
            --axis nsb 4 MHz 4 MHz 1 linear \
            --axis offset 0 deg 10 deg 2 linear \
            --direction_grid_density 0.25 1/deg^2 \
            --local_zenith_range 0 deg 70 deg \
            --local_azimuth_range 300 deg 60 deg \
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
        "--direction_grid_density",
        nargs="+",
        required=False,
        default=None,
        help=(
            "Direction-grid density in 1/deg^2. If set, direction-axis binning is "
            "derived from axis ranges and this density. In ra_dec mode, use "
            "local_zenith_range/local_azimuth_range to filter generated points in local sky."
        ),
    )
    parser.add_argument(
        "--local_zenith_range",
        nargs="+",
        required=False,
        default=None,
        help=(
            "Local zenith range (quantity pair) used to filter RA/Dec density points, "
            "for example: --local_zenith_range 0 deg 70 deg"
        ),
    )
    parser.add_argument(
        "--local_azimuth_range",
        nargs="+",
        required=False,
        default=None,
        help=(
            "Local azimuth range (quantity pair) used to filter RA/Dec density points, "
            "for example: --local_azimuth_range 300 deg 60 deg"
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
    parser.add_argument(
        "--showers_per_run_scaling",
        help=(
            "Zenith-angle scaling mode for showers_per_run: "
            "'fixed' keeps the baseline value, "
            "'cosine_zenith' applies showers_per_run * cos(zenith_angle)."
        ),
        type=str,
        choices=["fixed", "cosine_zenith"],
        required=False,
        default="fixed",
    )
    parser.add_argument(
        "--energy_max_scaling_index",
        help=(
            "Scale max energy with zenith angle as "
            "energy_max_zenith * cos(zenith_angle) ** energy_max_scaling_index, "
            "using the configured energy_range max value as energy_max_zenith "
            "(for example: --energy_max_scaling_index -2.5)."
        ),
        type=float,
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
