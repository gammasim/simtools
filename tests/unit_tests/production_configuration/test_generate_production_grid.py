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


def test_interpolated_limits(grid_gen):
    # Mock interpolated limits
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


def test_clean_grid_output(grid_gen):
    # Case 1: Valid grid points
    grid_points = [
        {
            "zenith_angle": 30 * u.deg,
            "azimuth": 310 * u.deg,
            "nsb": 4 * u.MHz,
            "energy_threshold": {"lower": 0.1 * u.TeV},
            "radius": 100 * u.m,
            "viewcone": 1 * u.deg,
        },
        {
            "zenith_angle": 40 * u.deg,
            "azimuth": 345 * u.deg,
            "nsb": 5 * u.MHz,
            "energy_threshold": {"lower": 0.2 * u.TeV},
            "radius": 200 * u.m,
            "viewcone": 2 * u.deg,
        },
    ]
    cleaned_points = grid_gen.clean_grid_output(grid_points)
    assert isinstance(cleaned_points, str)  # JSON string
    assert '"zenith_angle"' in cleaned_points
    assert '"energy_threshold"' in cleaned_points


def test_serialize_quantity(grid_gen):
    # Case 1: Value is a numpy array
    np_array = np.array([1, 2, 3])
    serialized = grid_gen.serialize_quantity(np_array)
    assert serialized == [1, 2, 3]

    # Case 2: Value is a Quantity (single value)
    quantity = 5 * u.m
    serialized = grid_gen.serialize_quantity(quantity)
    assert serialized == {"value": 5, "unit": "m"}

    # Case 3: Value is a Quantity (array)
    quantity_array = np.array([1, 2, 3]) * u.s
    serialized = grid_gen.serialize_quantity(quantity_array)
    assert serialized == {
        "value": [1, 2, 3],
        "unit": "s",
    }

    # Case 4: Value is neither numpy array nor Quantity
    normal_value = "test_string"
    serialized = grid_gen.serialize_quantity(normal_value)
    assert serialized == "test_string"


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


def test_generate_power_law_values(grid_gen):
    # Case 1: Valid input
    values = grid_gen.generate_power_law_values((1e9, 1e12), 10, 3)
    assert len(values) == 10
    assert values[0] >= 1e9
    assert values[-1] <= 1e12

    # Case 3: Zero bins
    values = grid_gen.generate_power_law_values((1e9, 1e12), 0, 3)
    assert len(values) == 0


def test_apply_lookup_table_limits(grid_gen):
    # Ensure the lookup table is applied correctly
    grid_gen._apply_lookup_table_limits()
    assert "energy" in grid_gen.interpolated_limits
    assert "radius" in grid_gen.interpolated_limits
    assert "viewcone" in grid_gen.interpolated_limits
