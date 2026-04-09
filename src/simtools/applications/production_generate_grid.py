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

from pathlib import Path

from astropy.coordinates import EarthLocation
from astropy.time import Time

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.io.ascii_handler import collect_data_from_file
from simtools.model.site_model import SiteModel
from simtools.production_configuration.generate_production_grid import GridGeneration


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Generate a grid of simulation points using flexible axes definitions.",
    )

    config.parser.add_argument(
        "--axes",
        type=str,
        required=True,
        help="Path to a file defining the grid axes.",
    )
    config.parser.add_argument(
        "--coordinate_system",
        type=str,
        default="zenith_azimuth",
        help=(
            "Coordinate system ('zenith_azimuth' or 'ra_dec'). "
            "In 'ra_dec' mode, sky directions are generated using observing"
            " location/time and converted to zenith/azimuth for interpolation."
        ),
    )
    config.parser.add_argument(
        "--observing_time",
        type=str,
        required=False,
        help=(
            "Observation time in UTC (format: 'YYYY-MM-DD HH:MM:SS'). Used only in 'ra_dec' mode."
        ),
    )
    config.parser.add_argument(
        "--output_file",
        type=str,
        default="grid_output.ecsv",
        help="Output file for the generated grid points (default: 'grid_output.ecsv').",
    )
    config.parser.add_argument(
        "--telescope_ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "List of telescope names used to get specific limits from the lookup table "
            "(e.g. MSTN-15)."
        ),
    )
    config.parser.add_argument(
        "--lookup_table",
        type=str,
        required=True,
        help="Path to the lookup table for simulation limits. "
        "Table required with varying azimuth and or zenith angle. ",
    )
    config.parser.add_argument(
        "--simtel_file",
        type=str,
        required=False,
        help=(
            "Optional path to a sim_telarray file used to map sim_telarray telescope IDs "
            "to telescope names when lookup-table selections are numeric IDs."
        ),
    )

    return config.initialize(db_config=True, simulation_model=["version", "site", "model_version"])


def load_axes(file_path: str):
    """
    Load axes definitions from a YAML or JSON file.

    Parameters
    ----------
    file_path : str
        Path to the axes YAML or JSON file.

    Returns
    -------
    list[dict]
        List of axes definitions with Quantity values.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Axes file {file_path} not found.")

    return collect_data_from_file(file_path)


def main():
    """Run the Grid Generation application."""
    app_context = startup_application(_parse)

    output_filepath = app_context.io_handler.get_output_file(app_context.args["output_file"])

    axes = load_axes(app_context.args["axes"])
    site_model = SiteModel(
        model_version=app_context.args["model_version"],
        site=app_context.args["site"],
    )

    ref_lat = site_model.get_parameter_value_with_unit("reference_point_latitude")
    ref_long = site_model.get_parameter_value_with_unit("reference_point_longitude")
    altitude = site_model.get_parameter_value_with_unit("reference_point_altitude")

    observing_location = EarthLocation(lat=ref_lat, lon=ref_long, height=altitude)

    coordinate_system = app_context.args["coordinate_system"]
    observing_time = None
    if app_context.args.get("observing_time"):
        observing_time = Time(app_context.args["observing_time"], scale="utc")
    elif coordinate_system == "ra_dec":
        observing_time = Time.now()

    grid_gen = GridGeneration(
        axes=axes,
        coordinate_system=coordinate_system,
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=app_context.args["lookup_table"],
        telescope_ids=app_context.args["telescope_ids"],
        simtel_file=app_context.args.get("simtel_file"),
    )

    grid_points = grid_gen.generate_grid()

    if coordinate_system == "ra_dec":
        grid_points = grid_gen.convert_coordinates(grid_points)
    grid_gen.serialize_grid_points(grid_points, output_file=output_filepath)


if __name__ == "__main__":
    main()
