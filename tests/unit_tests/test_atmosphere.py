import astropy.units as u
import numpy as np
import pytest

from simtools.atmosphere import AtmosphereProfile


def test_read_valid_file(tmp_path):
    """Test reading a valid atmosphere profile file."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n"
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
        "2.0 1.007 900.0 0.00008 275.0 79501.0 0.016\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    assert profile.data.shape == (3, 7)
    assert profile.columns == {
        "alt": 0,
        "rho": 1,
        "thick": 2,
        "n_minus_1": 3,
        "T": 4,
        "p": 5,
        "pw_over_p": 6,
    }
    assert profile.data[0, 0] == 0.0
    assert profile.data[1, 1] == 1.112
    assert profile.data[2, 2] == 900.0


def test_read_with_comments_and_empty_lines(tmp_path):
    """Test reading a file with comments and empty lines."""
    atmosphere_file = tmp_path / "atmosphere_comments.txt"
    atmosphere_file.write_text(
        "# This is a comment\n"
        "\n"
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n"
        "  # Indented comment\n"
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    assert profile.data.shape == (2, 7)
    assert profile.data[0, 0] == 0.0
    assert profile.data[1, 0] == 1.0


def test_read_converts_strings_to_floats(tmp_path):
    """Test that all data is converted to float values."""
    atmosphere_file = tmp_path / "atmosphere_float.txt"
    atmosphere_file.write_text("0 1 2 3 4 5 6\n")

    profile = AtmosphereProfile(str(atmosphere_file))

    assert profile.data.dtype == np.float64
    assert profile.data[0, 0] == 0.0


def test_interpolate_valid_altitude(tmp_path):
    """Test interpolating at a valid altitude."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n"
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
        "2.0 1.007 900.0 0.00008 275.0 79501.0 0.016\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    result = profile.interpolate(0.5 * u.km, column="thick")

    assert 950.0 < result < 1000.0


def test_interpolate_at_exact_altitude(tmp_path):
    """Test interpolating at an altitude that exists in the data."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n"
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
        "2.0 1.007 900.0 0.00008 275.0 79501.0 0.016\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    result = profile.interpolate(1.0 * u.km, column="thick")

    assert result == 950.0


def test_interpolate_different_columns(tmp_path):
    """Test interpolating different columns."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    assert profile.interpolate(0.5 * u.km, column="rho") == 1.1685
    assert profile.interpolate(0.5 * u.km, column="T") == 284.5


def test_interpolate_altitude_below_minimum(tmp_path):
    """Test that interpolation raises ValueError for altitude below minimum."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n2.0 1.007 900.0 0.00008 275.0 79501.0 0.016\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    with pytest.raises(ValueError, match="Altitude out of bounds"):
        profile.interpolate(0.5 * u.km, column="thick")


def test_interpolate_altitude_above_maximum(tmp_path):
    """Test that interpolation raises ValueError for altitude above maximum."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    with pytest.raises(ValueError, match="Altitude out of bounds"):
        profile.interpolate(5.0 * u.km, column="thick")


def test_interpolate_invalid_column(tmp_path):
    """Test that interpolation raises KeyError for unknown column."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    with pytest.raises(KeyError, match="Unknown column: unknown_col"):
        profile.interpolate(0.5 * u.km, column="unknown_col")


def test_interpolate_with_different_units(tmp_path):
    """Test interpolation with different altitude units."""
    atmosphere_file = tmp_path / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    result_km = profile.interpolate(500 * u.m, column="thick")
    result_m = profile.interpolate(0.5 * u.km, column="thick")

    assert result_km == result_m
