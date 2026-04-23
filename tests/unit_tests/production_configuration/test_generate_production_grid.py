import logging
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from astropy.utils import iers
from astropy.utils.iers import IERSWarning
from scipy.spatial import QhullError

from simtools.production_configuration.generate_production_grid import GridGeneration


@pytest.fixture(autouse=True, scope="module")
def disable_iers_auto_download():
    """Disable IERS auto-download during tests to avoid network dependency."""
    previous_auto_download = iers.conf.auto_download
    iers.conf.auto_download = False
    try:
        yield
    finally:
        iers.conf.auto_download = previous_auto_download


def _create_grid_generation(
    axes,
    coordinate_system,
    observing_location,
    observing_time,
    lookup_table,
    simtel_file=None,
):
    """Create a GridGeneration instance with a standard telescope selection."""
    return GridGeneration(
        axes=axes,
        coordinate_system=coordinate_system,
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
        telescope_ids=["LSTN-01"],
        simtel_file=simtel_file,
    )


def _build_single_point_radec_axes_definition(source_radec):
    """Build one-bin RA/Dec axes around a source coordinate."""
    return {
        "axes": {
            "ra": {
                "range": [source_radec.ra.deg, source_radec.ra.deg],
                "binning": 1,
                "scaling": "linear",
                "units": "deg",
            },
            "dec": {
                "range": [source_radec.dec.deg, source_radec.dec.deg],
                "binning": 1,
                "scaling": "linear",
                "units": "deg",
            },
            "nsb_level": {
                "range": [4, 4],
                "binning": 1,
                "scaling": "linear",
                "units": "MHz",
            },
        }
    }


@pytest.fixture
def axes_definition():
    """Load the axes definition from the YAML file."""
    axes_file = Path("tests/resources/production_grid_generation_axes_definition.yml")
    with open(axes_file) as f:
        return yaml.safe_load(f)


@pytest.fixture
def lookup_table():
    """Load the lookup table from the resources directory."""
    return str(
        Path("tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv")
    )


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
    return _create_grid_generation(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
    )


def test_generate_grid(grid_gen):
    grid_points = grid_gen.generate_grid()
    assert isinstance(grid_points, list)
    assert len(grid_points) > 0
    assert all(isinstance(point, dict) for point in grid_points)
    assert all("zenith_angle" in point for point in grid_points)
    assert all("azimuth" in point for point in grid_points)
    assert all("nsb_level" in point for point in grid_points)


def test_generate_grid_log_scaling(
    axes_definition, lookup_table, observing_location, observing_time
):
    """Test grid generation with logarithmic scaling for nsb_level axis."""
    axes_definition["axes"]["nsb_level"] = {
        "range": [2, 5],
        "binning": 4,
        "scaling": "log",
        "units": "MHz",
    }

    grid_gen = _create_grid_generation(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
    )

    grid_points = grid_gen.generate_grid()
    nsb_values = [point["nsb_level"].value for point in grid_points]
    unique_nsb_values = np.unique(nsb_values)

    expected_values = np.logspace(
        np.log10(axes_definition["axes"]["nsb_level"]["range"][0]),
        np.log10(axes_definition["axes"]["nsb_level"]["range"][1]),
        axes_definition["axes"]["nsb_level"]["binning"],
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

    grid_gen = _create_grid_generation(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
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


def test_generate_grid_radec_mode_minimal(observing_location, observing_time):
    """Generate a minimal RA/Dec-native grid and apply zenith cut."""
    axes_definition = {
        "axes": {
            "zenith_angle": {
                "range": [0, 10],
                "binning": 2,
                "scaling": "linear",
                "units": "deg",
            },
            "azimuth": {
                "range": [0, 10],
                "binning": 2,
                "scaling": "linear",
                "units": "deg",
            },
            "nsb_level": {
                "range": [4, 4],
                "binning": 1,
                "scaling": "linear",
                "units": "MHz",
            },
        }
    }

    grid_gen = _create_grid_generation(
        axes=axes_definition,
        coordinate_system="ra_dec",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IERSWarning)
        grid_points = grid_gen.generate_grid()

    assert len(grid_points) > 0
    assert all("zenith_angle" in point and "azimuth" in point for point in grid_points)
    assert all(point["zenith_angle"].value <= 10 for point in grid_points)


def test_generate_grid_radec_axes_mode(observing_location, observing_time):
    """Generate a grid directly from explicit RA/Dec axes."""
    source_altaz = SkyCoord(
        alt=65.0 * u.deg,
        az=210.0 * u.deg,
        frame="altaz",
        obstime=observing_time,
        location=observing_location,
    )
    source_radec = source_altaz.icrs

    axes_definition = _build_single_point_radec_axes_definition(source_radec)

    grid_gen = _create_grid_generation(
        axes=axes_definition,
        coordinate_system="ra_dec",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IERSWarning)
        grid_points = grid_gen.generate_grid()

    assert len(grid_points) == 1
    assert "ra" in grid_points[0]
    assert "dec" in grid_points[0]
    assert "zenith_angle" not in grid_points[0]
    assert "azimuth" not in grid_points[0]
    assert grid_points[0]["ra"].value == pytest.approx(source_radec.ra.deg, abs=0.05)
    assert grid_points[0]["dec"].value == pytest.approx(source_radec.dec.deg, abs=0.05)


def test_generate_grid_radec_axes_mode_keeps_below_horizon_points(
    observing_location, observing_time
):
    """Keep explicit RA/Dec YAML points even when they are below the horizon."""
    source_altaz = SkyCoord(
        alt=-20.0 * u.deg,
        az=45.0 * u.deg,
        frame="altaz",
        obstime=observing_time,
        location=observing_location,
    )
    source_radec = source_altaz.icrs

    axes_definition = _build_single_point_radec_axes_definition(source_radec)

    grid_gen = _create_grid_generation(
        axes=axes_definition,
        coordinate_system="ra_dec",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IERSWarning)
        grid_points = grid_gen.generate_grid()

    assert len(grid_points) == 1
    assert grid_points[0]["ra"].value == pytest.approx(source_radec.ra.deg, abs=0.05)
    assert grid_points[0]["dec"].value == pytest.approx(source_radec.dec.deg, abs=0.05)


def test_generate_grid_radec_axes_mode_with_lookup(
    lookup_table, observing_location, observing_time
):
    """Interpolate production limits for explicit RA/Dec axes points."""
    source_altaz = SkyCoord(
        alt=50.0 * u.deg,
        az=0.0 * u.deg,
        frame="altaz",
        obstime=observing_time,
        location=observing_location,
    )
    source_radec = source_altaz.icrs
    axes_definition = _build_single_point_radec_axes_definition(source_radec)
    axes_definition["axes"]["nsb_level"]["range"] = [1, 1]

    grid_gen = _create_grid_generation(
        axes=axes_definition,
        coordinate_system="ra_dec",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IERSWarning)
        grid_points = grid_gen.generate_grid()

    assert len(grid_points) == 1
    assert "lower_energy_threshold" in grid_points[0]
    assert "scatter_radius" in grid_points[0]
    assert "viewcone_radius" in grid_points[0]
    assert grid_points[0]["lower_energy_threshold"].value == pytest.approx(0.007, rel=1e-2)
    assert grid_points[0]["scatter_radius"].value == pytest.approx(1100.0, rel=1e-2)
    assert grid_points[0]["viewcone_radius"].value == pytest.approx(7.0, rel=1e-2)


def test_generate_grid_radec_axes_mode_with_sparse_lookup_raises_clear_error(
    lookup_table, observing_location, observing_time, monkeypatch
):
    """Raise a user-facing error when lookup points are too sparse for 3D interpolation."""
    source_altaz = SkyCoord(
        alt=50.0 * u.deg,
        az=0.0 * u.deg,
        frame="altaz",
        obstime=observing_time,
        location=observing_location,
    )
    source_radec = source_altaz.icrs
    axes_definition = _build_single_point_radec_axes_definition(source_radec)

    def _raise_qhull(*args, **kwargs):
        raise QhullError("mocked sparse triangulation failure")

    monkeypatch.setattr(
        "simtools.production_configuration.generate_production_grid.LinearNDInterpolator",
        _raise_qhull,
    )

    with pytest.raises(ValueError, match="does not contain enough unique points"):
        _create_grid_generation(
            axes=axes_definition,
            coordinate_system="ra_dec",
            observing_location=observing_location,
            observing_time=observing_time,
            lookup_table=lookup_table,
        )


def test_interpolated_limits(grid_gen):
    grid_gen.interpolated_limits = {
        "lower_energy_threshold": np.array(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]]
        ),
        "upper_scatter_radius": np.array(
            [[[100, 200], [300, 400], [500, 600]], [[700, 800], [900, 1000], [1100, 1200]]]
        ),
        "viewcone_radius": np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
        "target_zeniths": np.array([30, 40]),
        "target_azimuths": np.array([310, 345, 20]),
        "target_nsb": np.array([4, 5]),
    }

    grid_points = grid_gen.generate_grid()

    # Check that interpolated values are correctly assigned
    for point in grid_points:
        assert "lower_energy_threshold" in point
        assert "scatter_radius" in point
        assert "viewcone_radius" in point
        assert isinstance(point["lower_energy_threshold"], Quantity)
        assert isinstance(point["scatter_radius"], Quantity)
        assert isinstance(point["viewcone_radius"], Quantity)


def test_load_matching_lookup_arrays_with_simtel_id_mapping(grid_gen, tmp_test_directory):
    """Match lookup rows when telescope IDs are stored as sim_telarray numeric IDs."""
    lookup_table_path = tmp_test_directory / "lookup_simtel_ids.ecsv"
    Table(
        {
            "telescope_ids": ["[1]"],
            "zenith": [20.0],
            "azimuth": [0.0],
            "nsb_level": [4.0],
            "lower_energy_limit": [0.01],
            "upper_radius_limit": [1200.0],
            "viewcone_radius": [5.0],
        }
    ).write(lookup_table_path, format="ascii.ecsv", overwrite=True)

    grid_gen.lookup_table = str(lookup_table_path)
    grid_gen.telescope_ids = ["LSTN-01"]
    grid_gen._simtel_id_to_name = {1: "LSTN-01"}

    lookup_arrays = grid_gen._load_matching_lookup_arrays()

    assert lookup_arrays["points"].shape[0] == 1
    assert lookup_arrays["lower_energy_threshold"][0] == pytest.approx(0.01)


def test_load_matching_lookup_arrays_numeric_ids_require_simtel_file(grid_gen, tmp_test_directory):
    """Raise a clear error when numeric lookup telescope IDs are present without mapping input."""
    lookup_table_path = tmp_test_directory / "lookup_simtel_ids_no_mapping.ecsv"
    Table(
        {
            "telescope_ids": ["[1]"],
            "zenith": [20.0],
            "azimuth": [0.0],
            "nsb_level": [4.0],
            "lower_energy_limit": [0.01],
            "upper_radius_limit": [1200.0],
            "viewcone_radius": [5.0],
        }
    ).write(lookup_table_path, format="ascii.ecsv", overwrite=True)

    grid_gen.lookup_table = str(lookup_table_path)
    grid_gen.telescope_ids = ["LSTN-01"]
    grid_gen.simtel_file = None
    grid_gen._simtel_id_to_name = {}

    with pytest.raises(ValueError, match="Provide --simtel_file"):
        grid_gen._load_matching_lookup_arrays()


def test_serialize_grid_points_with_output_file(grid_gen, tmp_test_directory, caplog):
    """Test serialize_grid_points when an output file is provided."""
    grid_points = [
        {
            "zenith_angle": 30 * u.deg,
            "azimuth": 310 * u.deg,
            "nsb_level": 4,
            "lower_energy_threshold": 0.1 * u.TeV,
            "scatter_radius": 100 * u.m,
            "viewcone_radius": 1 * u.deg,
        },
        {
            "zenith_angle": 40 * u.deg,
            "azimuth": 345 * u.deg,
            "nsb_level": 5,
            "lower_energy_threshold": 0.2 * u.TeV,
            "scatter_radius": 200 * u.m,
            "viewcone_radius": 2 * u.deg,
        },
    ]

    output_file = tmp_test_directory / "grid_output.ecsv"
    with caplog.at_level(logging.INFO):
        grid_gen.serialize_grid_points(grid_points, output_file=output_file)
    assert output_file.exists()

    output_data = Table.read(output_file, format="ascii.ecsv")
    assert "zenith_angle" in output_data.colnames
    assert "lower_energy_threshold" in output_data.colnames
    assert output_data.meta["coordinate_system"] == grid_gen.coordinate_system
    assert output_data.meta["reference_frame"] == "ICRS (J2000)"

    assert f"Output saved to {output_file}" in caplog.text


def test_serialize_quantity(grid_gen):
    # Case 1: Value is a Quantity (single value)
    quantity = 5 * u.m
    serialized = grid_gen.serialize_quantity(quantity)
    assert serialized == {"value": 5, "unit": "m"}

    # Case 2: Value is not a Quantity (single value)
    value = 5
    serialized_value = grid_gen.serialize_quantity(value)
    assert serialized_value == value


@pytest.mark.xfail(reason="May fail due to IERS data download timeout", strict=False)
def test_convert_altaz_to_radec_and_coordinates(grid_gen):
    warnings.simplefilter("error", IERSWarning)
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

    # Case 2: Counterclockwise path (shortest distance)
    bins = grid_gen.create_circular_binning((10, 350), 3)
    expected_bins = [10, 0, 350]
    assert np.allclose(bins, expected_bins)


def test_apply_lookup_table_limits(lookup_table, observing_location, observing_time):
    axes_definition = {
        "axes": {
            "azimuth": {
                "range": [0, 180],
                "binning": 2,
                "scaling": "linear",
                "units": "deg",
            },
            "zenith_angle": {
                "range": [20, 40],
                "binning": 2,
                "scaling": "linear",
                "units": "deg",
            },
            "nsb_level": {
                "range": [1, 1],
                "binning": 1,
                "scaling": "linear",
                "units": "MHz",
            },
        }
    }
    grid_gen = _create_grid_generation(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
    )

    assert "lower_energy_threshold" in grid_gen.interpolated_limits
    assert "upper_scatter_radius" in grid_gen.interpolated_limits
    assert "viewcone_radius" in grid_gen.interpolated_limits
    assert np.isclose(
        grid_gen.interpolated_limits["lower_energy_threshold"][0][0][0], 0.007, rtol=1e-2
    )
    assert np.isclose(
        grid_gen.interpolated_limits["upper_scatter_radius"][0][0][0], 925.0, rtol=1e-2
    )
    assert np.isclose(grid_gen.interpolated_limits["viewcone_radius"][0][0][0], 9.25, rtol=1e-2)

    assert np.shape(grid_gen.interpolated_limits["lower_energy_threshold"]) == (2, 2, 1)
    assert np.shape(grid_gen.interpolated_limits["upper_scatter_radius"]) == (2, 2, 1)
    assert np.shape(grid_gen.interpolated_limits["viewcone_radius"]) == (2, 2, 1)


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
            lookup_table="tests/resources/corsika_simulation_limits/merged_corsika_limits_for_test.ecsv",
            telescope_ids=[999],
        )


def test_matching_rows_with_string_telescope_id(
    axes_definition, lookup_table, observing_location, observing_time
):
    """Match legacy numeric lookup-table rows using string telescope IDs."""
    grid_gen = GridGeneration(
        axes=axes_definition,
        coordinate_system="zenith_azimuth",
        observing_location=observing_location,
        observing_time=observing_time,
        lookup_table=lookup_table,
        telescope_ids=["LSTN-01"],
    )

    lookup_arrays = grid_gen._load_matching_lookup_arrays()
    assert lookup_arrays["points"].shape[0] > 0


def test_missing_observing_time(grid_gen):
    """Test behavior when observing_time is not set."""
    grid_gen.observing_time = None

    with pytest.raises(ValueError, match="Observing time is not set"):
        grid_gen.convert_altaz_to_radec(45 * u.deg, 30 * u.deg)


def test_iers_not_modified_without_env(monkeypatch):
    from astropy.utils import iers

    iers.conf.auto_download = True
    iers.conf.auto_max_age = 30

    monkeypatch.delenv("SIMTOOLS_OFFLINE_IERS", raising=False)

    from simtools.production_configuration.generate_production_grid import GridGeneration

    GridGeneration(axes={"axes": {}})

    assert iers.conf.auto_download is True
    assert iers.conf.auto_max_age == 30


def test_iers_disabled_with_env(monkeypatch):
    from astropy.utils import iers

    iers.conf.auto_download = True
    iers.conf.auto_max_age = 30

    monkeypatch.setenv("SIMTOOLS_OFFLINE_IERS", "1")

    from simtools.production_configuration.generate_production_grid import GridGeneration

    GridGeneration(axes={"axes": {}})

    assert iers.conf.auto_download is False
    assert iers.conf.auto_max_age is None
