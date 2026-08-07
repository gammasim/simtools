from pathlib import Path

import astropy.units as u
import pytest

from simtools.atmosphere import AtmosphereProfile


def test_read_valid_file(tmp_test_directory):
    atmosphere_file = Path(str(tmp_test_directory)) / "atmosphere.txt"
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
    assert profile.data[0, 0] == pytest.approx(0.0)
    assert profile.data[1, 1] == pytest.approx(1.112)
    assert profile.data[2, 2] == pytest.approx(900.0)


def test_read_with_comments_and_empty_lines(tmp_test_directory):
    atmosphere_file = Path(str(tmp_test_directory)) / "atmosphere_comments.txt"
    atmosphere_file.write_text(
        "# This is a comment\n"
        "\n"
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n"
        "  # Indented comment\n"
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    assert profile.data.shape == (2, 7)
    assert profile.data[0, 0] == pytest.approx(0.0)
    assert profile.data[1, 0] == pytest.approx(1.0)


def test_interpolate_valid_altitude(tmp_test_directory):
    atmosphere_file = Path(str(tmp_test_directory)) / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n"
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
        "2.0 1.007 900.0 0.00008 275.0 79501.0 0.016\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    result = profile.interpolate(0.5 * u.km, column="thick")

    assert 950.0 < result < 1000.0


def test_interpolate_altitude_below_minimum(tmp_test_directory):
    atmosphere_file = Path(str(tmp_test_directory)) / "atmosphere.txt"
    atmosphere_file.write_text(
        "1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n2.0 1.007 900.0 0.00008 275.0 79501.0 0.016\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    with pytest.raises(ValueError, match="Altitude out of bounds"):
        profile.interpolate(0.5 * u.km, column="thick")


def test_interpolate_altitude_above_maximum(tmp_test_directory):
    atmosphere_file = Path(str(tmp_test_directory)) / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    with pytest.raises(ValueError, match="Altitude out of bounds"):
        profile.interpolate(5.0 * u.km, column="thick")


def test_interpolate_invalid_column(tmp_test_directory):
    atmosphere_file = Path(str(tmp_test_directory)) / "atmosphere.txt"
    atmosphere_file.write_text(
        "0.0 1.225 1000.0 0.0001 288.0 101325.0 0.02\n1.0 1.112 950.0 0.00009 281.0 89876.0 0.018\n"
    )

    profile = AtmosphereProfile(str(atmosphere_file))

    with pytest.raises(KeyError, match="Unknown column: unknown_col"):
        profile.interpolate(0.5 * u.km, column="unknown_col")
