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

Command line arguments
----------------------
axes (str, required)
    Path to a YAML or JSON file defining the axes of the grid.
coordinate_system (str, optional, default='zenith_azimuth')
    The coordinate system for the grid generation ('zenith_azimuth' or 'ra_dec').
observing_time (str, optional, default=now)
    Time of the observation (format: 'YYYY-MM-DD HH:MM:SS').
lookup_table (str, required)
    Path to the lookup table for simulation limits. The table should contain
    varying azimuth and/or zenith angles.
telescope_ids (list of int, optional)
    List of telescope IDs as used in sim_telarray to filter the events.
output_file (str, optional, default='grid_output.json')
    Output file for the generated grid points (default: 'grid_output.json').


Example
-------
To generate a grid of simulation points, execute the script as follows:

.. code-block:: console

    simtools-production-generate-grid --site North --model_version 6.0.0 \
      --axes  tests/resources/production_grid_generation_axes_definition.yml \
      --coordinate_system ra_dec --observing_time "2017-09-16 00:00:00" \
      --lookup_table tests/resources/corsika_simulation_limits_lookup.ecsv \
        --telescope_ids 1
"""

import logging
from pathlib import Path

from astropy.coordinates import EarthLocation
from astropy.time import Time

from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.io_operations.ascii_handler import collect_data_from_file
from simtools.model.site_model import SiteModel
from simtools.production_configuration.generate_production_grid import GridGeneration


def _parse(label, description):
    """
    Parse command line configuration.

    Parameters
    ----------
    axes (str, required)
        Path to a YAML or JSON file defining the axes of the grid.
    coordinate_system (str, optional, default='zenith_azimuth')
        The coordinate system for the grid generation ('zenith_azimuth' or 'ra_dec').
    observing_time (str, optional, default=now)
        Time of the observation (format: 'YYYY-MM-DD HH:MM:SS').
    lookup_table (str, required)
        Path to the lookup table for simulation limits. The table should contain
        varying azimuth and/or zenith angles.
    telescope_ids (list of int, optional)
        List of telescope IDs as used in sim_telarray to filter the events.
    output_file (str, optional, default='grid_output.json')
        Output file for the generated grid points (default: 'grid_output.json').


    Returns
    -------
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description)

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
        help="Coordinate system ('zenith_azimuth' or 'ra_dec').",
    )
    config.parser.add_argument(
        "--observing_time",
        type=str,
        required=False,
        help="Time of the observation (format: 'YYYY-MM-DD HH:MM:SS').",
    )
    config.parser.add_argument(
        "--output_file",
        type=str,
        default="grid_output.json",
        help="Output file for the generated grid points (default: 'grid_output.json').",
    )
    config.parser.add_argument(
        "--telescope_ids",
        type=int,
        nargs="*",
        default=None,
        help="List of telescope IDs as used in sim_telarray to get the specific limits from the "
        "lookup table.",
    )
    config.parser.add_argument(
        "--lookup_table",
        type=str,
        required=True,
        help="Path to the lookup table for simulation limits. "
        "Table required with varying azimuth and or zenith angle. ",
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
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label,
        "Generate a grid of simulation points using flexible axes definitions.",
    )

    _logger = logging.getLogger()
    _logger.setLevel(logging.INFO)

    output_path = io_handler.IOHandler().get_output_directory(label)
    output_filepath = Path(output_path).joinpath(f"{args_dict['output_file']}")

    axes = load_axes(args_dict["axes"])
    site_model = SiteModel(
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        site=args_dict["site"],
    )

    ref_lat = site_model.get_parameter_value_with_unit("reference_point_latitude")
    ref_long = site_model.get_parameter_value_with_unit("reference_point_longitude")
    altitude = site_model.get_parameter_value_with_unit("reference_point_altitude")

    observing_location = EarthLocation(lat=ref_lat, lon=ref_long, height=altitude)

    observing_time = (
        Time(args_dict["observing_time"]) if args_dict.get("observing_time") else Time.now()
    )

    grid_gen = GridGeneration(
        axes=axes,
        coordinate_system=args_dict["coordinate_system"],
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=args_dict["lookup_table"],
        telescope_ids=args_dict["telescope_ids"],
    )

    grid_points = grid_gen.generate_grid()

    if args_dict["coordinate_system"] == "ra_dec":
        grid_points = grid_gen.convert_coordinates(grid_points)
    grid_gen.serialize_grid_points(grid_points, output_file=output_filepath)


if __name__ == "__main__":
    main()
