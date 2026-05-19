#!/usr/bin/python3

r"""
Generate executable simulation job grids.

This application expands executable simulation job rows and writes them to disk
as ECSV files. It supports both:

- explicit cartesian job-grid configuration (primary, zenith, energy range, etc.), and
- axes-based production-grid configuration with optional ``ra_dec`` coordinate handling
  and lookup-table interpolation.

Command line arguments
----------------------
axes (str, optional)
    Path to a YAML or JSON file defining the axes of the grid.
coordinate_system (str, optional, default='horizontal')
    The coordinate system for the grid generation ('horizontal' or 'ra_dec').
    In ``ra_dec`` mode, observing location/time are used to build sky directions and
    derive corresponding zenith/azimuth values for interpolation (ICRS/J2000 frame).
observing_time (str, optional)
    Time of the observation in UTC (format: 'YYYY-MM-DD HH:MM:SS').
    Used only in ``ra_dec`` mode (for coordinate transforms and sidereal-time
    sampling). Ignored in ``horizontal`` mode.
lookup_table (str, optional)
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
            --axes tests/resources/production_grid_generation_axes_definition.yml \
            --coordinate_system horizontal \
            --lookup_table tests/resources/corsika_simulation_limits/
                merged_corsika_limits_for_test.ecsv

To generate an all-sky RA/Dec direction grid and serialize output in RA/Dec,
execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --array_layout_name alpha \
            --axes tests/resources/production_grid_generation_axes_definition_ra_dec.yml \
            --coordinate_system ra_dec --observing_time "2017-09-16 00:00:00" \
            --lookup_table \
            tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv
"""

from simtools.application_control import build_application
from simtools.production_configuration.build_grid import (
    build_job_grid_metadata,
    build_simulation_jobs,
)
from simtools.production_configuration.job_grid_io import serialize_job_grid


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--axes",
        type=str,
        required=False,
        help="Path to a file defining the grid axes.",
    )
    parser.add_argument(
        "--coordinate_system",
        type=str,
        default="horizontal",
        help=(
            "Coordinate system ('horizontal' or 'ra_dec'). "
            "In 'ra_dec' mode, sky directions are generated using observing"
            " location/time and converted to zenith/azimuth for interpolation."
        ),
    )
    parser.add_argument(
        "--observing_time",
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
        "--lookup_table",
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
