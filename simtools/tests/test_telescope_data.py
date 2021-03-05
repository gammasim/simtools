#!/usr/bin/python3

import logging
import math
import astropy.units as u

import pyproj

from simtools.layout.telescope_data import TelescopeData

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input():
    inp = {
        'name': 'L-01',
        'posX': -70.93 * u.m,
        'posY': -52.07 * u.m,
        'posZ': 43.00 * u.m,
        'altitude': 2.177 * u.km
    }
    tel = TelescopeData(**inp)

    assert tel._posX == -70.93
    # Testing default unit convertion
    # altitude should be converted to m
    assert tel._altitude == 2177


def test_convertion():
    inp = {
        'name': 'L-01',
        'posX': 0 * u.m,
        'posY': 0 * u.m,
        'posZ': 43.00 * u.m,
        'altitude': 2.177 * u.km,
    }
    tel = TelescopeData(**inp)

    wgs84 = pyproj.CRS('EPSG:4326')

    center_lon = -17.8920302
    center_lat = 28.7621661
    proj4_string = '+proj=tmerc +ellps=WGS84 +datum=WGS84'
    proj4_string += ' +lon_0={} +lat_0={}'.format(center_lon, center_lat)
    proj4_string += ' +axis=nwu +units=m +k_0=1.0'
    crs_local = pyproj.CRS.from_proj4(proj4_string)

    tel.convertLocalToMercator(wgs84=wgs84, crs_local=crs_local)

    assert math.isclose(tel._latitude, center_lat, abs_tol=0.000001)
    assert math.isclose(tel._longitude, center_lon, abs_tol=0.000001)

    crs_utm = pyproj.CRS.from_user_input(32628)

    tel.convertLocalToUtm(crs_utm=crs_utm, crs_local=crs_local)

    assert math.isclose(tel._utmEast, 217611, abs_tol=3)
    assert math.isclose(tel._utmNorth, 3185066, abs_tol=5)

    tel._latitude = None
    tel._longitude = None

    tel.convertUtmToMercator(crs_utm=crs_utm, wgs84=wgs84)

    assert math.isclose(tel._latitude, center_lat, abs_tol=0.000001)
    assert math.isclose(tel._longitude, center_lon, abs_tol=0.000001)

    print(tel._latitude, tel._longitude)


if __name__ == '__main__':

    # test_input()
    test_convertion()
    pass
