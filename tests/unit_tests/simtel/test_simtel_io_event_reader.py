import logging
from unittest.mock import patch

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.simtel.simtel_io_event_reader import (
    SimtelIOEventDataReader,
)


@pytest.fixture
def mock_tables():
    """Create mock tables with test data."""
    # Create SHOWERS table
    shower_table = Table()
    shower_table.meta["EXTNAME"] = "SHOWERS"
    shower_table["shower_id"] = [1, 2]
    shower_table["event_id"] = [1, 2]
    shower_table["file_id"] = [0, 0]
    shower_table["simulated_energy"] = [1.0, 2.0] * u.TeV
    shower_table["x_core"] = [100.0, 200.0]
    shower_table["y_core"] = [150.0, 250.0]
    shower_table["shower_azimuth"] = [0.1, 0.2]
    shower_table["shower_altitude"] = [1.0, 1.1]
    shower_table["area_weight"] = [1.0, 1.0]

    # Create TRIGGERS table
    trigger_table = Table()
    trigger_table.meta["EXTNAME"] = "TRIGGERS"
    trigger_table["shower_id"] = [1, 2]
    trigger_table["event_id"] = [1, 2]
    trigger_table["file_id"] = [0, 0]
    trigger_table["array_altitude"] = [1.1, 1.2] * u.rad
    trigger_table["array_azimuth"] = [0.2, 0.3] * u.rad
    trigger_table["telescope_list"] = ["LSTN-01,LSTN-02,LSTN-03", "LSTN-03, LSTN-04, MSTN-01"]

    # Create FILE_INFO table
    file_info_table = Table()
    file_info_table.meta["EXTNAME"] = "FILE_INFO"
    file_info_table["file_name"] = ["test.fits"]
    file_info_table["file_id"] = [0]
    file_info_table["particle_id"] = [1]
    file_info_table["zenith"] = [20.0] * u.deg
    file_info_table["azimuth"] = [0.0] * u.deg
    file_info_table["nsb_level"] = [1.0]
    file_info_table["energy_min"] = [0.001] * u.TeV
    file_info_table["energy_max"] = [100.0] * u.TeV
    file_info_table["viewcone_min"] = [0.0] * u.deg
    file_info_table["viewcone_max"] = [10.0] * u.deg
    file_info_table["core_scatter_min"] = [0.0] * u.m
    file_info_table["core_scatter_max"] = [1.0e3] * u.m

    return shower_table, trigger_table, file_info_table


@pytest.fixture
def mock_fits_file(mock_tables, tmp_path):
    """Create a mock FITS file with test data."""
    test_file = tmp_path / "test.fits"
    shower_table, trigger_table, file_info_table = mock_tables

    shower_table.write(test_file, format="fits", overwrite=True)
    trigger_table.write(test_file, format="fits", append=True)
    file_info_table.write(test_file, format="fits", append=True)

    return str(test_file)


def test_reader_initialization(mock_fits_file):
    """Test basic reader initialization."""
    reader = SimtelIOEventDataReader(mock_fits_file)
    data_sets = reader.data_sets

    assert len(data_sets) > 0
    assert all("SHOWERS" in ds and "TRIGGERS" in ds for ds in data_sets)


def test_telescope_filtering(mock_fits_file):
    """Test filtering by telescope list."""
    # Should only keep events with telescope 1
    reader = SimtelIOEventDataReader(mock_fits_file, telescope_list=["LSTN-01"])
    _, _, _, triggered_data = reader.read_event_data(mock_fits_file)

    assert len(triggered_data.telescope_list) == 1
    assert "LSTN-01" in triggered_data.telescope_list[0]

    # Should keep both events (all have telescope "LSTN-03")
    reader = SimtelIOEventDataReader(mock_fits_file, telescope_list=["LSTN-03"])
    _, _, _, triggered_data = reader.read_event_data(mock_fits_file)

    assert len(triggered_data.telescope_list) == 2
    for tel_list in triggered_data.telescope_list:
        assert "LSTN-03" in tel_list


def test_shower_coordinate_transformation(mock_fits_file):
    """Test transformation of core positions to shower coordinates."""
    reader = SimtelIOEventDataReader(mock_fits_file)
    _, _, triggered_shower, _ = reader.read_event_data(mock_fits_file)

    assert hasattr(triggered_shower, "x_core_shower")
    assert hasattr(triggered_shower, "y_core_shower")
    assert hasattr(triggered_shower, "core_distance_shower")


def test_angular_separation_calculation(mock_fits_file):
    """Test calculation of angular separation."""
    reader = SimtelIOEventDataReader(mock_fits_file)
    _, _, _, triggered_data = reader.read_event_data(mock_fits_file)

    assert hasattr(triggered_data, "angular_distance")
    assert len(triggered_data.angular_distance) == 2


def test_get_reduced_simulation_info(mock_fits_file):
    """Test getting reduced simulation information."""
    reader = SimtelIOEventDataReader(mock_fits_file)
    file_info, _, _, _ = reader.read_event_data(mock_fits_file)
    info = reader.get_reduced_simulation_file_info(file_info)

    assert info["primary_particle"] == "gamma"
    assert info["zenith"] == 20.0 * u.deg
    assert info["azimuth"] == 0.0 * u.deg
    assert info["nsb_level"] == 1.0


@patch("simtools.simtel.simtel_io_event_reader.PrimaryParticle")
def test_get_reduced_simulation_info_with_warning(mock_primary_particle, mock_fits_file, caplog):
    """Test get_reduced_simulation_info with multiple values that trigger warning."""

    reader = SimtelIOEventDataReader(mock_fits_file)

    new_file_info = Table()
    new_file_info.meta["EXTNAME"] = "FILE_INFO"
    new_file_info["file_name"] = ["test1.fits", "test2.fits"]
    new_file_info["file_id"] = [0, 1]
    new_file_info["particle_id"] = [1, 1]  # Same value, no warning
    new_file_info["zenith"] = [20.0, 30.0]  # Different value, warning
    new_file_info["azimuth"] = [0.0, 0.0]
    new_file_info["nsb_level"] = [1.0, 1.0]
    new_file_info["energy_min"] = [1.0, 1.0]
    new_file_info["energy_max"] = [1.0, 1.0]
    new_file_info["viewcone_min"] = [1.0, 1.0]
    new_file_info["viewcone_max"] = [1.0, 1.0]
    new_file_info["core_scatter_min"] = [1.0, 1.0]
    new_file_info["core_scatter_max"] = [1.0, 1.0]

    # Replace the existing table
    reader.simulation_file_info = new_file_info
    mock_primary_particle.return_value.name = "gamma"

    with caplog.at_level(logging.WARNING):
        info = reader.get_reduced_simulation_file_info(new_file_info)

    assert "Simulation file info has non-unique values" in caplog.text
    assert info["primary_particle"] == "gamma"
    assert info["zenith"] == 20.0  # Should use first value


def test_get_triggered_shower_data_single_match(mock_fits_file):
    """Test _get_triggered_shower_data with single matches."""
    reader = SimtelIOEventDataReader(mock_fits_file)
    _, shower_data, triggered_shower, _ = reader.read_event_data(mock_fits_file)

    # Get triggered shower data
    triggered_shower = reader._get_triggered_shower_data(
        shower_data,
        [0],  # file_id
        [1],  # event_id
        [1],  # shower_id
    )

    assert len(triggered_shower.shower_id) == 1
    assert triggered_shower.shower_id[0] == 1
    assert triggered_shower.simulated_energy[0] == 1.0


def test_get_triggered_shower_data_no_matches(mock_fits_file, caplog):
    """Test _get_triggered_shower_data when no matches are found."""
    reader = SimtelIOEventDataReader(mock_fits_file)
    _, shower_data, triggered_shower, _ = reader.read_event_data(mock_fits_file)

    with caplog.at_level(logging.WARNING):
        triggered_shower = reader._get_triggered_shower_data(
            shower_data,
            [999],  # file_id
            [999],  # event_id
            [999],  # shower_id
        )

        assert len(triggered_shower.shower_id) == 0
        assert len(triggered_shower.simulated_energy) == 0
        assert "Found 0 matches" in caplog.text
