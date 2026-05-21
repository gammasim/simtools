#!/usr/bin/python3

r"""
Generate simulation job grid for production configurations.

This application expands simulation job rows and writes them to disk as ECSV files.
It supports both:

- explicit cartesian job-grid configuration (primary, zenith, energy range, etc.), and
- axes-based production-grid configuration with optional ``ra_dec`` coordinate handling
  and lookup-table interpolation.

Command line arguments
----------------------
azimuth_range, zenith_range, ra_range, dec_range, nsb_range, offset_range (2 quantities)
    Axis ranges for grid generation, with explicit units provided on the command line.
    Unitless angular values are interpreted as deg and unitless nsb values as MHz.
azimuth_binning, zenith_binning, ra_binning, dec_binning, nsb_binning, offset_binning (int)
    Number of bins per axis.
azimuth_scaling, zenith_scaling, ra_scaling, dec_scaling, nsb_scaling, offset_scaling (str)
    Axis scaling mode (choices: ``linear``, ``log``, ``1/cos``).
time_of_observation (str, optional)
    Time of the observation in UTC (format: 'YYYY-MM-DD HH:MM:SS').
    Used only if RA/Dec axes are provided (for coordinate transforms and sidereal-time
    sampling). Ignored otherwise.
corsika_limits (str, optional)
    Path to the lookup table for simulation limits. The table should contain
    varying azimuth and/or zenith angles for the selected array layout.
output_file (str, optional, default='job_grid.ecsv')
    Output file for the generated executable job grid.


Example
-------
To generate a standard zenith/azimuth grid of simulation points, execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --azimuth_range 310 deg 20 deg --azimuth_binning 3 --azimuth_scaling linear \
            --zenith_range 30 deg 40 deg --zenith_binning 2 --zenith_scaling linear \
            --nsb_range 4 MHz 5 MHz --nsb_binning 2 --nsb_scaling linear \
            --offset_range 0 deg 10 deg --offset_binning 2 --offset_scaling linear \
            --corsika_limits tests/resources/corsika_simulation_limits/merged_corsika_limits.ecsv

To generate an all-sky RA/Dec direction grid and serialize output in RA/Dec,
execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --ra_range 0 deg 360 deg --ra_binning 36 --ra_scaling linear \
            --dec_range -90 deg 90 deg --dec_binning 18 --dec_scaling linear \
            --nsb_range 4 MHz 4 MHz --nsb_binning 1 --nsb_scaling linear \
            --offset_range 0 deg 10 deg --offset_binning 2 --offset_scaling linear \
            --time_of_observation "2017-09-16 00:00:00" \
            --corsika_limits tests/resources/corsika_simulation_limits/merged_corsika_limits.ecsv
"""

from simtools.application_control import build_application
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.production_configuration.job_grid_io import serialize_job_grid
from simtools.production_configuration.simulation_jobs import (
    build_job_grid_metadata,
    build_simulation_jobs,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    axis_defs = [
        ("azimuth", "deg", "Azimuth range (deg)"),
        ("zenith", "deg", "Zenith angle range (deg)"),
        ("ra", "deg", "Right ascension range (deg)"),
        ("dec", "deg", "Declination range (deg)"),
        ("nsb", "MHz", "NSB level range (MHz)"),
        ("offset", "deg", "Offset range (deg)"),
    ]
    scaling_choices = ["linear", "log", "1/cos"]
    for axis, unit, help_str in axis_defs:
        parser.add_argument(
            f"--{axis}_range",
            type=CommandLineParser.quantity(unit),
            nargs=2,
            help=help_str,
        )
        parser.add_argument(
            f"--{axis}_binning",
            type=int,
            required=False,
            help=f"Number of bins for {axis}",
        )
        parser.add_argument(
            f"--{axis}_scaling",
            type=str,
            default="linear",
            required=False,
            choices=scaling_choices,
            help=f"Scaling for {axis} (choices: {', '.join(scaling_choices)})",
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
        default=1,
    )
    parser.add_argument(
        "--nshow_power_index",
        help=(
            "Power-law index used to scale the baseline nshow with the geometric-mean energy "
            "of each energy_range entry."
        ),
        type=float,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--nshow_reference_energy",
        help=(
            "Reference energy for nshow power-law scaling (for example: '100 GeV'). "
            "Required together with --nshow_power_index."
        ),
        type=str,
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
