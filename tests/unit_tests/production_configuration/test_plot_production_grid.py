"""Tests for production grid plotting in production_configuration."""

import json
from pathlib import Path

import astropy.units as u
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from simtools.production_configuration.plot_production_grid import (
    DEFAULT_OUTPUT_FILE_STEM,
    ProductionGridPlotter,
)


def _write_grid_file(tmp_test_directory, file_name, grid_points):
    """Write grid points to a temporary JSON file."""
    file_path = Path(tmp_test_directory) / file_name
    file_path.write_text(json.dumps(grid_points), encoding="utf-8")
    return file_path


def test_normalize_altaz_point_creates_radec_coordinates(tmp_test_directory):
    """Test that native Alt/Az points are converted to RA/Dec for the equatorial panel."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz.json",
        [
            {
                "azimuth": {"value": 180.0, "unit": "deg"},
                "zenith_angle": {"value": 20.0, "unit": "deg"},
                "nsb": {"value": 0.0, "unit": "MHz"},
            }
        ],
    )

    plotter = ProductionGridPlotter(
        grid_points_file=grid_file,
        site_location_lat=28.76,
        site_location_lon=-17.89,
        site_location_height=2200.0,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "altaz"
    assert normalized_points[0]["visible_in_altaz"] is True
    assert normalized_points[0]["azimuth"] == pytest.approx(180.0)
    assert normalized_points[0]["zenith"] == pytest.approx(20.0)
    assert normalized_points[0]["ra"] is not None
    assert normalized_points[0]["dec"] is not None


def test_normalize_radec_point_projects_to_altaz(tmp_test_directory):
    """Test that native RA/Dec points are converted to Alt/Az for the local panel."""
    location = EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2200.0 * u.m)
    observation_time = Time("2025-01-01 00:00:00")
    source_altaz = SkyCoord(
        AltAz(
            alt=65.0 * u.deg,
            az=210.0 * u.deg,
            obstime=observation_time,
            location=location,
        )
    )
    source_radec = source_altaz.icrs

    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_radec.json",
        [
            {
                "ra": {"value": source_radec.ra.deg, "unit": "deg"},
                "dec": {"value": source_radec.dec.deg, "unit": "deg"},
            }
        ],
    )

    plotter = ProductionGridPlotter(
        grid_points_file=grid_file,
        site_location_lat=28.76,
        site_location_lon=-17.89,
        site_location_height=2200.0,
        observation_time=str(observation_time.value),
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "radec"
    assert normalized_points[0]["visible_in_altaz"] is True
    assert normalized_points[0]["azimuth"] == pytest.approx(210.0, abs=0.05)
    assert normalized_points[0]["zenith"] == pytest.approx(25.0, abs=0.05)
    assert normalized_points[0]["ra"] == pytest.approx(source_radec.ra.deg, abs=0.05)
    assert normalized_points[0]["dec"] == pytest.approx(source_radec.dec.deg, abs=0.05)


def test_infer_radec_grid_tracks_from_native_points(tmp_test_directory):
    """Infer RA and Dec grid tracks from a native RA/Dec mesh."""
    location = EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2200.0 * u.m)
    observation_time = Time("2025-01-01 00:00:00")
    lst = observation_time.sidereal_time("apparent", longitude=location.lon).deg
    ra_values = [(lst - 5.0) % 360.0, (lst + 5.0) % 360.0]
    dec_values = [20.0, 30.0]

    grid_points = []
    for dec_value in dec_values:
        for ra_value in ra_values:
            grid_points.append(
                {
                    "ra": {"value": ra_value, "unit": "deg"},
                    "dec": {"value": dec_value, "unit": "deg"},
                }
            )

    grid_file = _write_grid_file(tmp_test_directory, "grid_radec_mesh.json", grid_points)
    plotter = ProductionGridPlotter(
        grid_points_file=grid_file,
        site_location_lat=28.76,
        site_location_lon=-17.89,
        site_location_height=2200.0,
        observation_time=str(observation_time.value),
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()
    track_groups = plotter.infer_radec_grid_tracks(normalized_points)

    assert len(track_groups["declination_tracks"]) == 2
    assert len(track_groups["right_ascension_tracks"]) == 2
    assert all(len(group) == 2 for group in track_groups["declination_tracks"])
    assert all(len(group) == 2 for group in track_groups["right_ascension_tracks"])


def test_plot_sky_projection_creates_outputs(tmp_test_directory):
    """Test that the sky projection plot is written to disk."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_wrapped.json",
        [
            {
                "grid_point": {
                    "azimuth": {"value": 310.0, "unit": "deg"},
                    "zenith_angle": {"value": 30.0, "unit": "deg"},
                    "nsb": {"value": 4.0, "unit": "MHz"},
                    "offset": {"value": 0.0, "unit": "deg"},
                },
                "interpolated_production_statistics": 1234.0,
            }
        ],
    )
    output_path = Path(tmp_test_directory) / "output"

    plotter = ProductionGridPlotter(
        grid_points_file=grid_file,
        site_location_lat=28.76,
        site_location_lon=-17.89,
        site_location_height=2200.0,
        observation_time="2025-01-01 00:00:00",
        output_path=output_path,
    )

    plotter.plot_sky_projection(plot_ra_dec_tracks=True, dec_values=[20.0, 30.0])

    assert (output_path / f"{DEFAULT_OUTPUT_FILE_STEM}.png").exists()


def test_plot_sky_projection_infers_radec_grid_tracks(tmp_test_directory):
    """Plot inferred RA/Dec grid tracks."""
    location = EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2200.0 * u.m)
    observation_time = Time("2025-01-01 00:00:00")
    lst = observation_time.sidereal_time("apparent", longitude=location.lon).deg
    ra_values = [(lst - 5.0) % 360.0, (lst + 5.0) % 360.0]
    dec_values = [20.0, 30.0]

    grid_points = []
    for dec_value in dec_values:
        for ra_value in ra_values:
            grid_points.append(
                {
                    "ra": {"value": ra_value, "unit": "deg"},
                    "dec": {"value": dec_value, "unit": "deg"},
                }
            )

    grid_file = _write_grid_file(tmp_test_directory, "grid_radec_tracks.json", grid_points)
    output_path = Path(tmp_test_directory) / "output"
    plotter = ProductionGridPlotter(
        grid_points_file=grid_file,
        site_location_lat=28.76,
        site_location_lon=-17.89,
        site_location_height=2200.0,
        observation_time=str(observation_time.value),
        output_path=output_path,
    )

    plotter.plot_sky_projection(plot_ra_dec_tracks=True)

    assert (output_path / f"{DEFAULT_OUTPUT_FILE_STEM}.png").exists()
