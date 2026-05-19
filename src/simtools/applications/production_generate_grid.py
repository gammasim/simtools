#!/usr/bin/python3

r"""
Generate a grid of simulation points using flexible axes definitions.

This application generates a grid of simulation points based on the provided axes
definitions. The axes definitions (range, binning) are specified in a file.
The viewcone, radius and energy thresholds are provided as a lookup table and
are interpolated based on the generated grid points. The generated grid points are
filtered based on the specified telescope IDs and the limits from the lookup table.
The generated grid points are saved to a file.
It can also convert the generated points to RA/Dec coordinates if the selected
coordinate system is 'ra_dec'.

For ``coordinate_system='ra_dec'``, the underlying grid generation supports
declination-line sampling with hour-angle spacing and applies zenith-angle
filtering based on the configured zenith range in that mode.
When explicit ``ra`` / ``dec`` axes are provided, all YAML-defined grid points are
preserved in the serialized output.

Command line arguments
----------------------
axes (str, required)
    Path to a YAML or JSON file defining the axes of the grid.
coordinate_system (str, optional, default='zenith_azimuth')
    The coordinate system for the grid generation ('zenith_azimuth' or 'ra_dec').
    In ``ra_dec`` mode, observing location/time are used to build sky directions and
    derive corresponding zenith/azimuth values for interpolation (ICRS/J2000 frame).
observing_time (str, optional)
    Time of the observation in UTC (format: 'YYYY-MM-DD HH:MM:SS').
    Used only in ``ra_dec`` mode (for coordinate transforms and sidereal-time
    sampling). Ignored in ``zenith_azimuth`` mode.
lookup_table (str, required)
    Path to the lookup table for simulation limits. The table should contain
    varying azimuth and/or zenith angles.
telescope_ids (list of str, optional)
    List of telescope names used to filter the lookup table rows
    (e.g. ``MSTN-15``).
simtel_file (str, optional)
    Path to a sim_telarray file used only when lookup-table telescope selections
    are stored as numeric telescope IDs.
output_file (str, optional, default='grid_output.ecsv')
    Output file for the generated grid points (default: 'grid_output.ecsv').


Example
-------
To generate a standard zenith/azimuth grid of simulation points, execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --axes tests/resources/production_grid_generation_axes_definition.yml \
            --coordinate_system zenith_azimuth \
            --lookup_table tests/resources/corsika_simulation_limits/
                merged_corsika_limits_for_test.ecsv \
            --telescope_ids MSTN-15

To generate an all-sky RA/Dec direction grid and serialize output in RA/Dec,
execute:

.. code-block:: console

        simtools-production-generate-grid --site North --model_version 6.0.2 \
            --axes tests/resources/production_grid_generation_axes_definition_radec.yml \
            --coordinate_system ra_dec --observing_time "2017-09-16 00:00:00" \
            --lookup_table tests/resources/corsika_simulation_limits/
                merged_corsika_limits_for_test.ecsv \
            --telescope_ids MSTN-15
"""

from simtools.application_control import build_application
from simtools.production_configuration.build_grid import build_production_grid_engine


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--axes",
        type=str,
        required=True,
        help="Path to a file defining the grid axes.",
    )
    parser.add_argument(
        "--coordinate_system",
        type=str,
        default="zenith_azimuth",
        help=(
            "Coordinate system ('zenith_azimuth' or 'ra_dec'). "
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
        default="grid_output.ecsv",
        help="Output file for the generated grid points (default: 'grid_output.ecsv').",
    )
    parser.add_argument(
        "--telescope_ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "List of telescope names used to get specific limits from the lookup table "
            "(e.g. MSTN-15)."
        ),
    )
    parser.add_argument(
        "--lookup_table",
        type=str,
        required=True,
        help="Path to the lookup table for simulation limits. "
        "Table required with varying azimuth and or zenith angle. ",
    )
    parser.add_argument(
        "--simtel_file",
        type=str,
        required=False,
        help=(
            "Optional path to a sim_telarray file used to map sim_telarray telescope IDs "
            "to telescope names when lookup-table selections are numeric IDs."
        ),
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "simulation_model": ["version", "site", "model_version"],
        },
    )

    output_filepath = app_context.io_handler.get_output_file(app_context.args["output_file"])
    grid_gen = build_production_grid_engine(app_context.args)

    grid_points = grid_gen.generate_grid()
    grid_gen.serialize_grid_points(grid_points, output_file=output_filepath)


if __name__ == "__main__":
    main()
