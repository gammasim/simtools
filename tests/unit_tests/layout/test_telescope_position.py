#!/usr/bin/python3

import logging
import math

import astropy.units as u
import numpy as np
import pyproj
import pytest

from simtools.layout.telescope_position import (
    InvalidCoordSystem,
    MissingInputForConvertion,
    TelescopePosition,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def crs_wgs84():
    return pyproj.CRS("EPSG:4326")


@pytest.fixture
def crs_local():
    center_lon = -17.8920302
    center_lat = 28.7621661
    proj4_string = "+proj=tmerc +ellps=WGS84 +datum=WGS84"
    proj4_string += " +lon_0={} +lat_0={}".format(center_lon, center_lat)
    proj4_string += " +axis=nwu +units=m +k_0=1.0"
    return pyproj.CRS.from_proj4(proj4_string)


@pytest.fixture
def crs_utm():
    return pyproj.CRS.from_user_input(32628)


def position_for_testing():

    return {
        "posX": 0.0 * u.m,
        "posY": 0.0 * u.m,
        "posZ": 43.00 * u.m,
        "altitude": 2.177 * u.km,
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "utm_east": 217611 * u.m,
        "utm_north": 3185066 * u.m,
    }


def test_repr(crs_wgs84, crs_local, crs_utm):

    tel = TelescopePosition(name="L-01")

    _tcors = tel.__repr__()
    assert _tcors == "L-01"

    tel.setCoordinates("corsika", 50, -25.0, 2158.0 * u.m)
    _tcors = tel.__repr__()
    _test_string = "L-01\t CORSIKA x(->North): 50.00 y(->West): -25.00"
    assert _tcors == (_test_string + "\t Alt: 2158.00")
    tel.convertAll(crsLocal=crs_local, crsWgs84=crs_wgs84)
    _tcors = tel.__repr__()
    _test_string += "\t Longitude: 28.76262 Latitude: -17.89177"
    assert _tcors == (_test_string + "\t Alt: 2158.00")
    tel.convertAll(crsLocal=crs_local, crsWgs84=crs_wgs84, crsUtm=crs_utm)
    _tcors = tel.__repr__()
    _test_string = "L-01\t CORSIKA x(->North): 50.00 y(->West): -25.00"
    _test_string += "\t UTM East: 217635.45 UTM North: 3185116.68"
    _test_string += "\t Longitude: 28.76262 Latitude: -17.89177"
    assert _tcors == (_test_string + "\t Alt: 2158.00")


def test_getCoordinates(crs_wgs84, crs_local, crs_utm):

    tel = TelescopePosition(name="L-01")

    with pytest.raises(InvalidCoordSystem):
        tel.getCoordinates("not_valid_crs")

    tel.setCoordinates("corsika", 50, -25.0, 2158.0 * u.m)
    tel.convertAll(crsWgs84=crs_wgs84, crsLocal=crs_local, crsUtm=crs_utm)

    _x, _y, _z = tel.getCoordinates("corsika")
    assert _x.unit == "m"
    assert _y.unit == "m"
    assert _z.unit == "m"
    assert _x.value == pytest.approx(50.0, 0.1)
    assert _y.value == pytest.approx(-25.0, 0.1)
    assert _z.value == pytest.approx(2178, 0.1)
    _lat, _lon, _z = tel.getCoordinates("mercator")
    assert _lat.unit == "deg"
    assert _lon.unit == "deg"
    assert _z.unit == "m"
    _x, _y, _z = tel.getCoordinates("utm")
    assert _x.unit == "m"
    assert _y.unit == "m"
    assert _z.unit == "m"


def test_setCoordinateVariable():

    tel = TelescopePosition(name="L-01")

    # value should stay a value
    assert tel._getCoordinateValue(5.0, None) == pytest.approx(5.0, 1.0e-6)
    # quantity should become value
    assert tel._getCoordinateValue(5.0 * u.m, u.Unit("m")) == pytest.approx(5.0, 1.0e-6)
    # quantity should become value (plus unit conversion)
    assert tel._getCoordinateValue(5.0 * u.km, u.Unit("m")) == pytest.approx(5.0e3, 1.0e-6)
    # some units can't be converted
    with pytest.raises(u.UnitsError):
        tel._getCoordinateValue(5.0 * u.deg, u.Unit("m"))


def test_setCoordinates():

    tel = TelescopePosition(name="L-01")

    with pytest.raises(InvalidCoordSystem):
        tel.setCoordinates("not_valid_crs", 5.0, 2.0, 3.0)
    tel.setCoordinates("utm", 217611 * u.m, 3185066 * u.m)
    assert tel.crs["utm"]["xx"]["value"] == pytest.approx(217611, 0.1)
    assert tel.crs["utm"]["yy"]["value"] == pytest.approx(3185066, 0.1)
    assert np.isnan(tel.crs["utm"]["zz"]["value"])
    tel.setCoordinates("utm", 217611 * u.m, 3185066 * u.m, 22.0 * u.km)
    assert tel.crs["utm"]["zz"]["value"] == pytest.approx(22.0e3, 0.1)


def test_setAltitude():

    tel = TelescopePosition(name="L-01")

    tel.setAltitude(5.0)
    for _crs in tel.crs.values():
        assert _crs["zz"]["value"] == pytest.approx(5.0, 1.0e-6)
    tel.setAltitude(5.0 * u.cm)
    for _crs in tel.crs.values():
        assert _crs["zz"]["value"] == pytest.approx(0.05, 1.0e-6)


def test_convert(crs_wgs84, crs_local, crs_utm):

    test_position = position_for_testing()

    tel = TelescopePosition(name="L-01")

    # local to mercator
    _lat, _lon = tel._convert(crs_local, crs_wgs84, test_position["posX"], test_position["posY"])
    assert math.isclose(_lat, test_position["center_lat"].value, abs_tol=0.000001)
    assert math.isclose(_lon, test_position["center_lon"].value, abs_tol=0.000001)
    # mercator to local
    _x, _y = tel._convert(crs_wgs84, crs_local, _lat, _lon)
    assert math.isclose(_x, test_position["posX"].value, abs_tol=0.000001)
    assert math.isclose(_y, test_position["posY"].value, abs_tol=0.000001)
    # local to UTM
    _utmE, _utmN = tel._convert(crs_local, crs_utm, 0.0, 0.0)
    assert math.isclose(_utmE, test_position["utm_east"].value, abs_tol=3)
    assert math.isclose(_utmN, test_position["utm_north"].value, abs_tol=5)
    # UTM to mercator
    _lat, _lon = tel._convert(crs_utm, crs_wgs84, _utmE, _utmN)
    assert math.isclose(_lat, test_position["center_lat"].value, abs_tol=0.000001)
    assert math.isclose(_lon, test_position["center_lon"].value, abs_tol=0.000001)

    # errors
    with pytest.raises(pyproj.exceptions.CRSError):
        _lat, _lon = tel._convert("crs_local", crs_wgs84, 0.0, 0.0)
    with pytest.raises(pyproj.exceptions.CRSError):
        _lat, _lon = tel._convert(None, None, 0.0, 0.0)

    _lat, _lon = tel._convert(crs_local, crs_wgs84, test_position["posX"], None)
    assert np.isnan(_lat)
    assert np.isnan(_lon)


def test_get_reference_system_from(crs_utm):

    tel = TelescopePosition(name="L-01")

    assert tel._get_reference_system_from() == (None, None)

    tel.setCoordinates("utm", 217611 * u.m, 3185066 * u.m)
    assert tel._get_reference_system_from() == (None, None)

    tel.crs["utm"]["crs"] = crs_utm

    _crs_name, _crs = tel._get_reference_system_from()

    assert _crs_name == "utm"
    # TODO: somehow fail to check the type of _crs
    assert _crs is not None


def test_hasCoordinates(crs_wgs84, crs_local, crs_utm):

    tel = TelescopePosition(name="L-01")

    with pytest.raises(InvalidCoordSystem):
        tel.hasCoordinates("not_a_system")

    assert not tel.hasCoordinates("corsika")
    assert not tel.hasCoordinates("utm")
    assert not tel.hasCoordinates("mercator")

    tel.setCoordinates("corsika", 0.0, 0.0, 2158.0 * u.m)
    assert tel.hasCoordinates("corsika")
    assert not tel.hasCoordinates("corsika", True)
    tel.convertAll(crsWgs84=crs_wgs84, crsLocal=crs_local, crsUtm=crs_utm)
    assert tel.hasCoordinates("corsika", True)
    assert tel.hasCoordinates("utm", True)
    assert tel.hasCoordinates("mercator", True)
    assert tel.hasCoordinates("corsika", False)
    assert tel.hasCoordinates("utm", False)
    assert tel.hasCoordinates("mercator", False)


def test_hasAltitude():

    tel = TelescopePosition(name="L-01")

    with pytest.raises(InvalidCoordSystem):
        tel.hasAltitude("not_a_system")

    assert not tel.hasAltitude("utm")

    tel.setCoordinates("utm", 217611 * u.m, 3185066 * u.m, 1.0 * u.km)
    assert tel.hasAltitude("utm")
    assert not tel.hasAltitude("corsika")
    assert not tel.hasAltitude("mercator")
    tel.setCoordinates("utm", 217611 * u.m, 3185066 * u.m, np.nan)
    assert not tel.hasAltitude("utm")
    tel.setAltitude(1 * u.km)
    assert tel.hasAltitude("utm")
    assert tel.hasAltitude("corsika")
    assert tel.hasAltitude("mercator")


def test_setCoordinateSystem(crs_wgs84):

    tel = TelescopePosition(name="L-01")

    with pytest.raises(InvalidCoordSystem):
        tel._setCoordinateSystem("not_a_system", crs_wgs84)

    tel._setCoordinateSystem("mercator", crs_wgs84)

    assert tel.crs["mercator"]["crs"] == crs_wgs84


def test_altitude_transformations():

    tel = TelescopePosition(name="L-01")

    _z = tel.convertTelescopeAltitudeToCorsikaSystem(
        telAltitude=2.177 * u.km,
        corsikaObsLevel=2158.0 * u.m,
        corsikaSphereCenter=16.0 * u.m,
    )
    assert _z.value == pytest.approx(35.0, 0.1)

    with pytest.raises(TypeError):
        tel.convertTelescopeAltitudeToCorsikaSystem(
            telAltitude=2177,
            corsikaObsLevel=2158.0 * u.m,
            corsikaSphereCenter=16.0 * u.m,
        )

    _alt = tel.convertTelescopeAltitudeFromCorsikaSystem(
        telCorsikaZ=35.0 * u.m,
        corsikaObsLevel=2.158 * u.km,
        corsikaSphereCenter=16.0 * u.m,
    )
    assert _alt.value == pytest.approx(2177.0, 0.1)
    with pytest.raises(TypeError):
        tel.convertTelescopeAltitudeFromCorsikaSystem(
            telCorsikaZ=35.0 * u.m,
            corsikaObsLevel=2.158,
            corsikaSphereCenter=16.0 * u.m,
        )


def test_convert_all(cfg_setup, crs_wgs84, crs_local, crs_utm):

    tel = TelescopePosition(name="L-01")

    with pytest.raises(MissingInputForConvertion):
        tel.convertAll(crsWgs84=crs_wgs84, crsLocal=crs_local, crsUtm=crs_utm)

    tel.setCoordinates("corsika", 0.0, 0.0, 2158.0 * u.m)
    tel.convertAll(crsWgs84=crs_wgs84, crsLocal=crs_local, crsUtm=crs_utm)

    assert 28.7621 == pytest.approx(tel.crs["mercator"]["xx"]["value"], 1.0e-4)
    assert -17.8920302 == pytest.approx(tel.crs["mercator"]["yy"]["value"], 1.0e-7)
    assert 3185067.2783240844 == pytest.approx(tel.crs["utm"]["yy"]["value"], 1.0e-9)
    assert 217609.2270142641 == pytest.approx(tel.crs["utm"]["xx"]["value"], 1.0e-9)
    assert 3185067.2783240844 == pytest.approx(tel.crs["utm"]["yy"]["value"], 1.0e-9)
    assert 2158.0 == pytest.approx(tel.crs["utm"]["zz"]["value"], 1.0e-9)
