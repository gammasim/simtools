import logging
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units import Quantity

from simtools.production_configuration.generate_production_grid import GridGeneration


@pytest.fixture
def axes_definition():
    """Load the axes definition from the YAML file."""
    axes_file = Path("tests/resources/production_grid_generation_axes_definition.yml")
    with open(axes_file) as f:
        return yaml.safe_load(f)


@pytest.fixture
def lookup_table():
    """Load the lookup table from the resources directory."""
    return str(Path("tests/resources/corsika_simulation_limits_lookup.ecsv"))


@pytest.fixture
def observing_location():
    """Return a mock observing location."""
    latitude = 28.7622  # degrees
    longitude = -17.8920  # degrees
    return EarthLocation(lon=longitude * u.deg, lat=latitude * u.deg, height=2000 * u.m)


@pytest.fixture
def observing_time():
    """Return a mock observing time."""
    return Time("2017-09-16 00:00:00")


@pytest.fixture
def grid_gen(axes_definition, lookup_table, observing_location, observing_time):
    """Create a GridGeneration object with the provided fixtures."""
    return GridGeneration(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
        telescope_ids=[1],
    )


def test_generate_grid(grid_gen):
    grid_points = grid_gen.generate_grid()
    assert isinstance(grid_points, list)
    assert len(grid_points) > 0
    assert all(isinstance(point, dict) for point in grid_points)
    assert all("zenith_angle" in point for point in grid_points)
    assert all("azimuth" in point for point in grid_points)
    assert all("nsb" in point for point in grid_points)


def test_generate_grid_log_scaling(
    axes_definition, lookup_table, observing_location, observing_time
):
    """Test grid generation with logarithmic scaling for nsb axis."""
    axes_definition["axes"]["nsb"] = {
        "range": [2, 5],
        "binning": 4,
        "scaling": "log",
        "units": "MHz",
    }

    grid_gen = GridGeneration(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
        telescope_ids=[1],
    )

    grid_points = grid_gen.generate_grid()
    nsb_values = [point["nsb"].value for point in grid_points]
    unique_nsb_values = np.unique(nsb_values)

    expected_values = np.logspace(
        np.log10(axes_definition["axes"]["nsb"]["range"][0]),
        np.log10(axes_definition["axes"]["nsb"]["range"][1]),
        axes_definition["axes"]["nsb"]["binning"],
    )

    assert len(unique_nsb_values) == len(expected_values)
    assert np.allclose(unique_nsb_values, expected_values, rtol=1e-4)


def test_generate_grid_1_over_cos_scaling(
    axes_definition, lookup_table, observing_location, observing_time
):
    """Test grid generation with 1/cos scaling for zenith_angle axis."""
    axes_definition["axes"]["zenith_angle"] = {
        "range": [30, 60],
        "binning": 5,
        "scaling": "1/cos",
        "units": "deg",
    }

    grid_gen = GridGeneration(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
        telescope_ids=[1],
    )

    grid_points = grid_gen.generate_grid()
    zenith_values = [point["zenith_angle"].value for point in grid_points]
    unique_zenith_values = np.unique(zenith_values)

    cos_min = np.cos(np.radians(axes_definition["axes"]["zenith_angle"]["range"][0]))
    cos_max = np.cos(np.radians(axes_definition["axes"]["zenith_angle"]["range"][1]))
    cos_values = np.linspace(
        1 / cos_min, 1 / cos_max, axes_definition["axes"]["zenith_angle"]["binning"]
    )
    expected_values = np.degrees(np.arccos(1 / cos_values))

    assert len(unique_zenith_values) == len(expected_values)
    assert np.allclose(unique_zenith_values, expected_values, rtol=1e-4)


def test_interpolated_limits(grid_gen):
    grid_gen.interpolated_limits = {
        "energy": np.array(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]
        ),
        "radius": np.array(
            [[[100, 200], [300, 400], [500, 600]], [[700, 800], [900, 1000], [1100, 1200]]]
        ),
        "viewcone": np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
        "target_zeniths": np.array([30, 40]),
        "target_azimuths": np.array([310, 345, 20]),
        "target_nsb": np.array([4, 5]),
    }

    grid_points = grid_gen.generate_grid()

    # Check that interpolated values are correctly assigned
    for point in grid_points:
        assert "energy_threshold" in point
        assert "radius" in point
        assert "viewcone" in point
        assert isinstance(point["energy_threshold"]["lower"], Quantity)
        assert isinstance(point["radius"], Quantity)
        assert isinstance(point["viewcone"], Quantity)


def test_serialize_grid_points_with_output_file(grid_gen, tmp_test_directory, caplog):
    """Test serialize_grid_points when an output file is provided."""
    grid_points = [
        {
            "zenith_angle": 30 * u.deg,
            "azimuth": 310 * u.deg,
            "nsb": 4,
            "energy_threshold": {"lower": 0.1 * u.TeV},
            "radius": 100 * u.m,
            "viewcone": 1 * u.deg,
        },
        {
            "zenith_angle": 40 * u.deg,
            "azimuth": 345 * u.deg,
            "nsb": 5,
            "energy_threshold": {"lower": 0.2 * u.TeV},
            "radius": 200 * u.m,
            "viewcone": 2 * u.deg,
        },
    ]

    output_file = tmp_test_directory / "grid_output.json"
    with caplog.at_level(logging.INFO):
        grid_gen.serialize_grid_points(grid_points, output_file=output_file)
    assert output_file.exists()

    with open(output_file, encoding="utf-8") as f:
        file_content = f.read()
        assert '"zenith_angle"' in file_content
        assert '"energy_threshold"' in file_content

    assert f"Output saved to {output_file}" in caplog.text


def test_serialize_quantity(grid_gen, caplog):
    # Case 1: Value is a Quantity (single value)
    quantity = 5 * u.m
    serialized = grid_gen.serialize_quantity(quantity)
    assert serialized == {"value": 5, "unit": "m"}

    # Case 2: Value is not a Quantity (single value)
    value = 5

    with caplog.at_level(logging.WARNING):
        grid_gen.serialize_quantity(value)

    assert "Unsupported type" in caplog.text
    assert str(type(value)) in caplog.text


def test_convert_altaz_to_radec_and_coordinates(grid_gen):
    # Case 1: Valid AltAz to RA/Dec conversion
    alt, az = 45.0 * u.deg, 30.0 * u.deg
    radec = grid_gen.convert_altaz_to_radec(alt, az)
    assert isinstance(radec, SkyCoord)
    assert radec.ra.unit == u.deg
    assert radec.dec.unit == u.deg

    # Case 2: Valid coordinate conversion
    grid_gen.coordinate_system = "ra_dec"
    grid_points = [
        {"zenith_angle": 30 * u.deg, "azimuth": 45 * u.deg},
        {"zenith_angle": 20 * u.deg, "azimuth": 60 * u.deg},
    ]
    converted_points = grid_gen.convert_coordinates(grid_points)
    assert "ra" in converted_points[0]
    assert "dec" in converted_points[0]

    # Case 3: Missing zenith_angle or azimuth
    grid_points = [{"azimuth": 45 * u.deg}]
    converted_points = grid_gen.convert_coordinates(grid_points)
    assert "ra" not in converted_points[0]
    assert "dec" not in converted_points[0]


def test_create_circular_binning(grid_gen):
    # Case 1: No wraparound
    bins = grid_gen.create_circular_binning((0, 150), 6)
    assert len(bins) == 6
    assert bins[0] == 0
    assert bins[-1] == 150
    assert bins[1] == 30
    assert bins[2] == 60

    # Case 2: Wraparound
    bins = grid_gen.create_circular_binning((300, 20), 5)
    assert len(bins) == 5
    assert bins[0] == 300
    assert bins[-1] == 20
    assert bins[1] == 320

    # Case 3: Single bin
    bins = grid_gen.create_circular_binning((0, 360), 1)
    assert len(bins) == 1
    assert bins[0] == 0


def test_create_circular_binning_with_shortest_path(grid_gen):
    # Case 1: Clockwise path (shortest distance)
    bins = grid_gen.create_circular_binning((350, 10), 3)
    expected_bins = [350, 0, 10]
    assert np.allclose(bins, expected_bins)

    # Case 2: No wraparound
    bins = grid_gen.create_circular_binning((30, 150), 4)
    expected_bins = [30, 70, 110, 150]
    assert np.allclose(bins, expected_bins)

    # Case 3: Counterclockwise path (shortest distance)
    bins = grid_gen.create_circular_binning((10, 350), 3)
    expected_bins = [10, 0, 350]
    assert np.allclose(bins, expected_bins)


def test_apply_lookup_table_limits(grid_gen):
    grid_gen._apply_lookup_table_limits()
    assert "energy" in grid_gen.interpolated_limits
    assert "radius" in grid_gen.interpolated_limits
    assert "viewcone" in grid_gen.interpolated_limits
    assert np.isclose(grid_gen.interpolated_limits["energy"][0][0][0], 0.00459, rtol=1e-2)
    assert np.isclose(grid_gen.interpolated_limits["radius"][0][0][0], 2047.8, rtol=1e-2)
    assert np.isclose(grid_gen.interpolated_limits["viewcone"][0][0][0], 9.98, rtol=1e-2)

    assert np.shape(grid_gen.interpolated_limits["energy"]) == (2, 3, 2)
    assert np.shape(grid_gen.interpolated_limits["radius"]) == (2, 3, 2)
    assert np.shape(grid_gen.interpolated_limits["viewcone"]) == (2, 3, 2)


def test_no_matching_rows_in_lookup_table(axes_definition, observing_location, observing_time):
    """Test behavior when no matching rows are found in the lookup table."""
    with pytest.raises(
        ValueError, match=r"No matching rows in the lookup table for telescope_ids: \[999\]"
    ):
        GridGeneration(
            axes=axes_definition,
            coordinate_system="zenith_azimuth",
            observing_location=observing_location,
            observing_time=observing_time,
            lookup_table="tests/resources/corsika_simulation_limits_lookup.ecsv",
            telescope_ids=[999],
        )


def test_missing_observing_time(grid_gen):
    """Test behavior when observing_time is not set."""
    grid_gen.observing_time = None

    with pytest.raises(ValueError, match="Observing time is not set"):
        grid_gen.convert_altaz_to_radec(45 * u.deg, 30 * u.deg)
