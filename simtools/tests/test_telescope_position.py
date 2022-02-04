#!/usr/bin/python3

import logging
import math
import astropy.units as u

import pyproj

from simtools.layout.telescope_position import TelescopePosition

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input():
    configData = {
        "posX": -70.93 * u.m,
        "posY": -52.07 * u.m,
        "posZ": 43.00 * u.m,
        "altitude": 2.177 * u.km,
    }
    tel = TelescopePosition(name="L-01", configData=configData)

    assert tel._posX == -70.93
    # Testing default unit convertion
    # altitude should be converted to m
    assert tel._altitude == 2177


def test_coordinate_transformations():
    configData = {
        "posX": 0 * u.m,
        "posY": 0 * u.m,
        "posZ": 43.00 * u.m,
        "altitude": 2.177 * u.km,
    }
    tel = TelescopePosition(name="L-01", configData=configData)

    wgs84 = pyproj.CRS("EPSG:4326")

    center_lon = -17.8920302
    center_lat = 28.7621661
    proj4_string = "+proj=tmerc +ellps=WGS84 +datum=WGS84"
    proj4_string += " +lon_0={} +lat_0={}".format(center_lon, center_lat)
    proj4_string += " +axis=nwu +units=m +k_0=1.0"
    crsLocal = pyproj.CRS.from_proj4(proj4_string)

    tel.convertLocalToMercator(wgs84=wgs84, crsLocal=crsLocal)

    assert math.isclose(tel._latitude, center_lat, abs_tol=0.000001)
    assert math.isclose(tel._longitude, center_lon, abs_tol=0.000001)

    crsUtm = pyproj.CRS.from_user_input(32628)

    tel.convertLocalToUtm(crsUtm=crsUtm, crsLocal=crsLocal)

    assert math.isclose(tel._utmEast, 217611, abs_tol=3)
    assert math.isclose(tel._utmNorth, 3185066, abs_tol=5)

    tel._latitude = None
    tel._longitude = None

    tel.convertUtmToMercator(crsUtm=crsUtm, wgs84=wgs84)

    assert math.isclose(tel._latitude, center_lat, abs_tol=0.000001)
    assert math.isclose(tel._longitude, center_lon, abs_tol=0.000001)

    print(tel._latitude, tel._longitude)


def test_corsika_transformations():
    configData = {
        "posX": 0 * u.m,
        "posY": 0 * u.m,
        "altitude": 2.177 * u.km,
    }
    tel = TelescopePosition(name="L-01", configData=configData)

    # ASL -> CORSIKA
    tel.convertAslToCorsika(corsikaObsLevel=2158 * u.m, corsikaSphereCenter=16 * u.m)
    assert math.isclose(tel._posZ, 35.0, abs_tol=0.1)

    # CORSIKA -> ASL
    tel._posZ = 35.0
    tel._altitude = None

    tel.convertCorsikaToAsl(corsikaObsLevel=2158 * u.m, corsikaSphereCenter=16 * u.m)
    assert math.isclose(tel._altitude, 2177, abs_tol=0.1)


def test_convert_all():
    configData = {
        "posX": 0 * u.m,
        "posY": 0 * u.m,
        "posZ": 43.00 * u.m,
    }
    tel = TelescopePosition(name="L-01", configData=configData)

    wgs84 = pyproj.CRS("EPSG:4326")

    center_lon = -17.8920302
    center_lat = 28.7621661
    proj4_string = "+proj=tmerc +ellps=WGS84 +datum=WGS84"
    proj4_string += " +lon_0={} +lat_0={}".format(center_lon, center_lat)
    proj4_string += " +axis=nwu +units=m +k_0=1.0"
    crsLocal = pyproj.CRS.from_proj4(proj4_string)

    crsUtm = pyproj.CRS.from_user_input(32628)

    tel.convertAll(
        wgs84=wgs84,
        crsLocal=crsLocal,
        crsUtm=crsUtm,
        corsikaObsLevel=2158 * u.m,
        corsikaSphereCenter=16 * u.m,
    )


if __name__ == "__main__":

    test_input()
    # test_coordinate_transformations()
    # test_corsika_transformations()
    # test_convert_all()
    pass
