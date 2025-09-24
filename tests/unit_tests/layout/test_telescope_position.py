#!/usr/bin/python3

import logging
import math

import astropy.units as u
import numpy as np
import pyproj
import pytest

from simtools.layout.telescope_position import (
    InvalidCoordSystemErrorError,
    TelescopePosition,
)

logger = logging.getLogger()


@pytest.fixture
def crs_wgs84():
    return pyproj.CRS("EPSG:4326")


@pytest.fixture
def crs_local():
    center_lon = -17.8920302
    center_lat = 28.7621661
    proj4_string = "+proj=tmerc +ellps=WGS84 +datum=WGS84"
    proj4_string += f" +lon_0={center_lon} +lat_0={center_lat}"
    proj4_string += " +axis=nwu +units=m +k_0=1.0"
    return pyproj.CRS.from_proj4(proj4_string)


@pytest.fixture
def crs_utm():
    return pyproj.CRS.from_user_input(32628)


def position_for_testing():
    return {
        "pos_x": 0.0 * u.m,
        "pos_y": 0.0 * u.m,
        "pos_z": 43.00 * u.m,
        "altitude": 2.177 * u.km,
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "utm_east": 217611.227 * u.m,
        "utm_north": 3185066.278 * u.m,
    }


def test_str(crs_wgs84, crs_local, crs_utm):
    tel = TelescopePosition(name="LSTN-01")

    _tcors = tel.__str__()
    assert _tcors == "LSTN-01"

    tel.set_coordinates("ground", 50, -25.0, 2158.0 * u.m)
    _tcors = tel.__str__()
    _test_string = "LSTN-01\t Ground x(->North): 50.00 y(->West): -25.00"
    altitude = "\t Alt: 2158.00"
    assert _tcors == (_test_string + altitude)
    tel.convert_all(crs_local=crs_local, crs_wgs84=crs_wgs84)
    _tcors = tel.__str__()
    _test_string += "\t Longitude: 28.76262 Latitude: -17.89177"
    assert _tcors == (_test_string + altitude)
    tel.convert_all(crs_local=crs_local, crs_wgs84=crs_wgs84, crs_utm=crs_utm)
    _tcors = tel.__str__()
    _test_string = "LSTN-01\t Ground x(->North): 50.00 y(->West): -25.00"
    _test_string += "\t UTM East: 217635.45 UTM North: 3185116.68"
    _test_string += "\t Longitude: 28.76262 Latitude: -17.89177"
    assert _tcors == (_test_string + altitude)


def test_get_coordinates(crs_wgs84, crs_local, crs_utm):
    tel = TelescopePosition(name="LSTN-01")

    with pytest.raises(InvalidCoordSystemErrorError):
        tel.get_coordinates("not_valid_crs")
    with pytest.raises(InvalidCoordSystemErrorError):
        tel.get_coordinates("not_valid_crs", coordinate_field="value")

    tel.set_coordinates("ground", 50, -25.0, 2158.0 * u.m)
    tel.convert_all(crs_wgs84=crs_wgs84, crs_local=crs_local, crs_utm=crs_utm)

    _x, _y, _z = tel.get_coordinates("ground")
    assert _x.unit == "m"
    assert _y.unit == "m"
    assert _z.unit == "m"
    assert _x.value == pytest.approx(50.0, 0.1)
    assert _y.value == pytest.approx(-25.0, 0.1)
    assert _z.value == pytest.approx(2178, 0.1)
    _lat, _lon, _z = tel.get_coordinates("mercator")
    assert _lat.unit == "deg"
    assert _lon.unit == "deg"
    assert _z.unit == "m"
    _x, _y, _z = tel.get_coordinates("utm")
    assert _x.unit == "m"
    assert _y.unit == "m"
    assert _z.unit == "m"

    _x, _y, _z = tel.get_coordinates("ground", coordinate_field="value")
    assert _x == pytest.approx(50.0, 0.1)
    assert _y == pytest.approx(-25.0, 0.1)
    assert _z == pytest.approx(2178.0, 0.1)


def test_get_coordinate_variable():
    tel = TelescopePosition(name="LSTN-01")

    # value should stay a value
    assert tel._get_coordinate_value(5.0, None) == pytest.approx(5.0, 1.0e-6)
    # quantity should become value
    assert tel._get_coordinate_value(5.0 * u.m, u.Unit("m")) == pytest.approx(5.0, 1.0e-6)
    # quantity should become value (plus unit conversion)
    assert tel._get_coordinate_value(5.0 * u.km, u.Unit("m")) == pytest.approx(5.0e3, 1.0e-6)
    # nan should be isnan
    assert np.isnan(tel._get_coordinate_value(np.nan * u.km, u.m))
    # some units can't be converted
    with pytest.raises(u.UnitsError):
        tel._get_coordinate_value(5.0 * u.deg, u.Unit("m"))


def test_set_coordinates():
    tel = TelescopePosition(name="LSTN-01")

    with pytest.raises(InvalidCoordSystemErrorError):
        tel.set_coordinates("not_valid_crs", 5.0, 2.0, 3.0)
    tel.set_coordinates("utm", 217611 * u.m, 3185066 * u.m)
    assert tel.crs["utm"]["xx"]["value"] == pytest.approx(217611, 0.1)
    assert tel.crs["utm"]["yy"]["value"] == pytest.approx(3185066, 0.1)
    assert np.isnan(tel.crs["utm"]["zz"]["value"])
    tel.set_coordinates("utm", 217611 * u.m, 3185066 * u.m, 22.0 * u.km)
    assert tel.crs["utm"]["zz"]["value"] == pytest.approx(22.0e3, 0.1)


def test_set_altitude():
    tel = TelescopePosition(name="LSTN-01")

    tel.set_altitude(5.0)
    for key, _crs in tel.crs.items():
        if tel.is_coordinate_system(key):
            assert _crs["zz"]["value"] == pytest.approx(5.0, 1.0e-6)
    tel.set_altitude(5.0 * u.cm)
    for key, _crs in tel.crs.items():
        if tel.is_coordinate_system(key):
            assert _crs["zz"]["value"] == pytest.approx(0.05, 1.0e-6)


def test_convert(crs_wgs84, crs_local, crs_utm):
    test_position = position_for_testing()

    tel = TelescopePosition(name="LSTN-01")

    # local to mercator
    _lat, _lon = tel._convert(crs_local, crs_wgs84, test_position["pos_x"], test_position["pos_y"])
    assert math.isclose(_lat, test_position["center_lat"].value, abs_tol=0.000001)
    assert math.isclose(_lon, test_position["center_lon"].value, abs_tol=0.000001)
    # mercator to local
    _x, _y = tel._convert(crs_wgs84, crs_local, _lat, _lon)
    assert math.isclose(_x, test_position["pos_x"].value, abs_tol=0.000001)
    assert math.isclose(_y, test_position["pos_y"].value, abs_tol=0.000001)
    # local to UTM
    _utm_e, _utm_n = tel._convert(crs_local, crs_utm, 0.0, 0.0)
    assert math.isclose(_utm_e, test_position["utm_east"].value, abs_tol=3)
    assert math.isclose(_utm_n, test_position["utm_north"].value, abs_tol=5)
    # UTM to mercator
    _lat, _lon = tel._convert(crs_utm, crs_wgs84, _utm_e, _utm_n)
    assert math.isclose(_lat, test_position["center_lat"].value, abs_tol=0.000001)
    assert math.isclose(_lon, test_position["center_lon"].value, abs_tol=0.000001)

    # errors
    with pytest.raises(pyproj.exceptions.CRSError):
        _lat, _lon = tel._convert("crs_local", crs_wgs84, 0.0, 0.0)
    with pytest.raises(pyproj.exceptions.CRSError):
        _lat, _lon = tel._convert(None, None, 0.0, 0.0)

    _lat, _lon = tel._convert(crs_local, crs_wgs84, test_position["pos_x"], None)
    assert np.isnan(_lat)
    assert np.isnan(_lon)
    _lat, _lon = tel._convert(crs_local, crs_wgs84, None, test_position["pos_y"])
    assert np.isnan(_lat)
    assert np.isnan(_lon)

    # (invalid) mercator to local
    _x, _y = tel._convert(crs_wgs84, crs_local, +95.0, _lon)
    assert np.isnan(_x)
    assert np.isnan(_y)


def test_get_reference_system_from(crs_utm):
    tel = TelescopePosition(name="LSTS-01")

    assert tel._get_reference_system_from() == (None, None)

    tel.set_coordinates("utm", 217611 * u.m, 3185066 * u.m)
    assert tel._get_reference_system_from() == (None, None)

    tel.crs["utm"]["crs"] = crs_utm

    _crs_name, _crs = tel._get_reference_system_from()

    assert _crs_name == "utm"
    assert _crs is not None


def test_has_coordinates(crs_wgs84, crs_local, crs_utm):
    tel = TelescopePosition(name="LSTS-01")

    with pytest.raises(InvalidCoordSystemErrorError):
        tel.has_coordinates("not_a_system")

    assert not tel.has_coordinates("ground")
    assert not tel.has_coordinates("utm")
    assert not tel.has_coordinates("mercator")

    tel.set_coordinates("ground", 0.0, 0.0, 2158.0 * u.m)
    assert tel.has_coordinates("ground")
    assert not tel.has_coordinates("ground", True)
    tel.convert_all(crs_wgs84=crs_wgs84, crs_local=crs_local, crs_utm=crs_utm)
    assert tel.has_coordinates("ground", True)
    assert tel.has_coordinates("utm", True)
    assert tel.has_coordinates("mercator", True)
    assert tel.has_coordinates("ground", False)
    assert tel.has_coordinates("utm", False)
    assert tel.has_coordinates("mercator", False)


def test_has_altitude():
    tel = TelescopePosition(name="LSTS-01")

    with pytest.raises(InvalidCoordSystemErrorError):
        tel.has_altitude("not_a_system")

    assert not tel.has_altitude("utm")
    assert not tel.has_altitude()

    tel.set_coordinates("utm", 217611 * u.m, 3185066 * u.m, 1.0 * u.km)
    assert tel.has_altitude("utm")
    assert not tel.has_altitude("ground")
    assert not tel.has_altitude("mercator")
    tel.set_coordinates("utm", 217611 * u.m, 3185066 * u.m, np.nan)
    assert not tel.has_altitude("utm")
    tel.set_altitude(1 * u.km)
    assert tel.has_altitude("utm")
    assert tel.has_altitude("ground")
    assert tel.has_altitude("mercator")
    assert tel.has_altitude()


def test_set_coordinate_system(crs_wgs84):
    tel = TelescopePosition(name="LSTN-01")

    with pytest.raises(InvalidCoordSystemErrorError):
        tel._set_coordinate_system("not_a_system", crs_wgs84)

    tel._set_coordinate_system("mercator", crs_wgs84)

    assert tel.crs["mercator"]["crs"] == crs_wgs84


def test_altitude_transformations():
    tel = TelescopePosition(name="LSTN-01")

    _z = tel.convert_telescope_altitude_to_corsika_system(
        tel_altitude=2.177 * u.km,
        corsika_observation_level=2158.0 * u.m,
        telescope_axis_height=16.0 * u.m,
    )
    assert _z.value == pytest.approx(35.0, 0.1)

    with pytest.raises(TypeError):
        tel.convert_telescope_altitude_to_corsika_system(
            tel_altitude=2177,
            corsika_observation_level=2158.0 * u.m,
            telescope_axis_height=16.0 * u.m,
        )

    _alt = tel.convert_telescope_altitude_from_corsika_system(
        tel_corsika_z=35.0 * u.m,
        corsika_observation_level=2.158 * u.km,
        telescope_axis_height=16.0 * u.m,
    )
    assert _alt.value == pytest.approx(2177.0, 0.1)
    with pytest.raises(TypeError):
        tel.convert_telescope_altitude_from_corsika_system(
            tel_corsika_z=35.0 * u.m,
            corsika_observation_level=2.158,
            telescope_axis_height=16.0 * u.m,
        )


def test_convert_all(crs_wgs84, crs_local, crs_utm):
    tel = TelescopePosition(name="LSTN-01")

    tel.set_coordinates("ground", 0.0, 0.0, 2158.0 * u.m)
    tel.convert_all(crs_wgs84=crs_wgs84, crs_local=crs_local, crs_utm=crs_utm)

    assert tel.crs["mercator"]["xx"]["value"] == pytest.approx(28.7621, 1.0e-4)
    assert tel.crs["mercator"]["yy"]["value"] == pytest.approx(-17.8920302, 1.0e-7)
    assert tel.crs["utm"]["yy"]["value"] == pytest.approx(3185067.2783240844, 1.0e-9)
    assert tel.crs["utm"]["xx"]["value"] == pytest.approx(217609.2270142641, 1.0e-9)
    assert tel.crs["utm"]["yy"]["value"] == pytest.approx(3185067.2783240844, 1.0e-9)
    assert tel.crs["utm"]["zz"]["value"] == pytest.approx(2158.0, 1.0e-9)

    tel_nan = TelescopePosition(name="LSTN-02")
    tel_nan.set_coordinates("ground", np.nan, np.nan, 2158.0 * u.m)
    tel_nan.convert_all(crs_wgs84=crs_wgs84, crs_local=crs_local, crs_utm=crs_utm)
    assert np.isnan(tel_nan.crs["mercator"]["xx"]["value"])
    assert np.isnan(tel_nan.crs["mercator"]["yy"]["value"])
    assert np.isnan(tel_nan.crs["utm"]["xx"]["value"])
    assert np.isnan(tel_nan.crs["utm"]["yy"]["value"])


def test_get_altitude():
    telescope = TelescopePosition(name="LSTS-01")
    assert np.isnan(telescope.get_altitude())

    telescope.set_coordinates("ground", xx=100.0, yy=200.0, zz=2100.0)
    assert telescope.get_altitude().value == pytest.approx(2100.0, 0.1)


def test_print_compact_format(capsys):
    telescope = TelescopePosition(name="LSTS-01")
    telescope.set_auxiliary_parameter("telescope_axis_height", 16.0 * u.m)
    telescope.set_auxiliary_parameter("telescope_sphere_radius", 12.5 * u.m)
    telescope.set_coordinates("ground", xx=100.0, yy=200.0, zz=2100.0)
    expected_output = "LSTS-01 100.00 200.00 2100.00"
    telescope.print_compact_format(
        crs_name="ground",
        corsika_observation_level=None,
    )
    _output = capsys.readouterr().out
    # ignore differences in spaces
    assert "".join(expected_output.split()) == "".join(_output.split())

    expected_output = "LSTS-01 100.00 200.00 116.00"
    telescope.print_compact_format(
        crs_name="ground",
        corsika_observation_level=2000.0 * u.m,
    )
    _output = capsys.readouterr().out
    assert "".join(expected_output.split()) == "".join(_output.split())

    expected_output = (
        "telescope_name position_x position_y position_z\nLSTS-01 100.00 200.00 116.00"
    )
    telescope.print_compact_format(
        crs_name="ground",
        corsika_observation_level=2000.0 * u.m,
        print_header=True,
    )
    _output = capsys.readouterr().out
    assert "".join(expected_output.split()) == "".join(_output.split())

    telescope.set_coordinates("mercator", xx=28.7621661, yy=-17.8920302, zz=2100.0)
    expected_output = "LSTS-01 28.76216610 -17.89203020    2100.00"
    telescope.print_compact_format("mercator")
    _output = capsys.readouterr().out
    assert "".join(expected_output.split()) == "".join(_output.split())
    # corsika_sphere should have no impact on output
    telescope.print_compact_format("mercator")
    _output = capsys.readouterr().out
    assert "".join(expected_output.split()) == "".join(_output.split())

    telescope.set_coordinates("utm", xx=217611.227, yy=3185066.278, zz=2100.0)
    expected_output = "LSTS-01 217611.23 3185066.28    2100.00"
    telescope.print_compact_format("utm")
    _output = capsys.readouterr().out
    assert "".join(expected_output.split()) == "".join(_output.split())

    telescope.set_coordinates("utm", xx=217611.227, yy=3185066.278, zz=2100.0)
    telescope.geo_code = "ABC"
    expected_output = "LSTS-01 217611.23 3185066.28    2100.00  ABC"
    telescope.print_compact_format("utm")
    _output = capsys.readouterr().out
    assert "".join(expected_output.split()) == "".join(_output.split())

    with pytest.raises(InvalidCoordSystemErrorError):
        telescope.print_compact_format("not_a_crs")
