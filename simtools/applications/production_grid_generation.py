#!/usr/bin/python3

r"""
Generate a grid of simulation points using flexible axes definitions.

This application generates a grid of simulation points based on input parameters
such as energy, azimuth, zenith angle, night-sky background, and camera offset.
It can also convert the generated points to RA/Dec coordinates if the selected
coordinate system is 'ra_dec'.

Command line arguments
----------------------
axes (str, required)
    Path to a YAML or JSON file defining the axes of the grid.
data_level (str, required)
    The data level for the grid generation (e.g., 'A', 'B', 'C').
science_case (str, required)
    The science case for the grid generation (e.g., 'high_precision').
coordinate_system (str, optional, default='zenith_azimuth')
    The coordinate system for the grid generation ('zenith_azimuth' or 'ra_dec').
latitude (float, required)
    Latitude of the observing location in degrees.
longitude (float, required)
    Longitude of the observing location in degrees.
height (float, required)
    Height of the observing location in meters.
observing_time (str, optional, default=now)
    Time of the observation (format: 'YYYY-MM-DD HH:MM:SS').

Example
-------
To generate a grid of simulation points, execute the script as follows:

.. code-block:: console

    simtools-production-grid-generation --axes "path/to/axes.yaml" --data_level "B" \
      --science_case "high_precision" --latitude 28.7622 --longitude -17.8920 \
      --height 2000 --coordinate_system "ra_dec" --observing_time "2017-09-16 00:00:00"

The output will display the generated grid points and their RA/Dec coordinates
(if applicable).
"""

import argparse
import json
import logging
import os

import yaml
from astropy.coordinates import EarthLocation
from astropy.time import Time

from simtools.production_configuration.generate_production_grid import GridGeneration


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a grid of simulation points using flexible axes definitions."
    )
    parser.add_argument(
        "--axes",
        type=str,
        required=True,
        help="Path to a YAML or JSON file defining the grid axes.",
    )
    parser.add_argument(
        "--data_level", type=str, required=True, help="Data level (e.g., 'A', 'B', 'C')."
    )
    parser.add_argument(
        "--science_case", type=str, required=True, help="Science case for the grid generation."
    )
    parser.add_argument(
        "--coordinate_system",
        type=str,
        default="zenith_azimuth",
        help="Coordinate system ('zenith_azimuth' or 'ra_dec').",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        required=True,
        help="Latitude of the observing location (degrees).",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        required=True,
        help="Longitude of the observing location (degrees).",
    )
    parser.add_argument(
        "--height", type=float, required=True, help="Height of the observing location (meters)."
    )
    parser.add_argument(
        "--observing_time",
        type=str,
        default=None,
        help="Time of the observation (format: 'YYYY-MM-DD HH:MM:SS').",
    )

    return parser.parse_args()


def load_axes(file_path: str) -> list[dict]:
    """
    Load axes definitions from a YAML or JSON file.

    Parameters
    ----------
    file_path : str
        Path to the axes YAML or JSON file.

    Returns
    -------
    list[dict]
        List of axes definitions.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Axes file {file_path} not found.")

    with open(file_path, encoding="utf-8") as file:
        if file_path.endswith(".yaml") or file_path.endswith(".yml"):
            return yaml.safe_load(file)
        if file_path.endswith(".json"):
            return json.load(file)
        raise ValueError("Unsupported file format. Use a YAML or JSON file.")


def main():
    """Run the Grid Generation application."""
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO)

    axes = load_axes(args.axes)

    observing_location = EarthLocation(lat=args.latitude, lon=args.longitude, height=args.height)

    observing_time = Time(args.observing_time) if args.observing_time else Time.now()

    grid_gen = GridGeneration(
        axes=axes,
        data_level=args.data_level,
        science_case=args.science_case,
        coordinate_system=args.coordinate_system,
        observing_location=observing_location,
        observing_time=observing_time,
    )

    grid_points = grid_gen.generate_grid()

    if args.coordinate_system == "ra_dec":
        grid_points = grid_gen.convert_coordinates(grid_points)

    for point in grid_points:
        print(point)


if __name__ == "__main__":
    main()
