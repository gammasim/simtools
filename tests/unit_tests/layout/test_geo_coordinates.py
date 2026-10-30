#!/usr/bin/python3

import warnings

import numpy as np
import pyproj
import pytest

from simtools.layout.geo_coordinates import GeoCoordinates
from simtools.layout.telescope_position import TelescopePosition


def test_crs_utm():
    geo = GeoCoordinates()
    utm_crs = geo.crs_utm(32719)
    assert isinstance(utm_crs, pyproj.CRS)
    assert utm_crs.to_epsg() == 32719

    utm_none = geo.crs_utm(None)
    assert utm_none is None


def test_crs_local():
    geo = GeoCoordinates()

    lapalma = TelescopePosition(name="LaPalma")
    lapalma.set_coordinates("mercator", 28.7621661, -17.8920302, 2177.0)
    lapalma_crs = geo.crs_local(lapalma)
    assert isinstance(lapalma_crs, pyproj.CRS)

    _crs_test_dict = {
        "proj": "tmerc",
        "lat_0": 28.7621661,
        "lon_0": -17.8920302,
        "k": 1.00034158569596,
        "x_0": 0,
        "y_0": 0,
        "datum": "WGS84",
        "units": "m",
        "no_defs": None,
        "type": "crs",
    }
    # ignore warnings from pyproj that the crs.to_dict()
    # method is not sufficient to fully describe a
    # coordinate system - this is not relevant for the
    # simple comparison of the crs parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lapalma_dict = lapalma_crs.to_dict()
    assert lapalma_dict["proj"] == _crs_test_dict["proj"]
    assert lapalma_dict["k"] == pytest.approx(_crs_test_dict["k"], rel=1.0e-6)

    with pytest.raises(AttributeError):
        geo.crs_local(None)

    nan_reference = TelescopePosition(name="NaN")
    nan_reference.set_coordinates("mercator", np.nan, -70.316345, 2147.0)
    assert geo.crs_local(nan_reference) is None


def test_coordinate_scale_factor():
    geo = GeoCoordinates()
    reference = TelescopePosition(name="LaPalma")
    reference.set_coordinates("mercator", 28.7621661, -17.8920302, 2177.0)
    assert geo._coordinate_scale_factor(reference) == pytest.approx(1.0003415856959632)


def test_valid_reference_point():
    geo = GeoCoordinates()
    reference = TelescopePosition(name="LaPalma")
    reference.set_coordinates("mercator", 28.7621661, -17.8920302, 2177.0)
    assert geo._valid_reference_point(reference)
    reference.set_coordinates("mercator", np.nan, -17.8920302, 2177.0)
    assert not geo._valid_reference_point(reference)


def test_coordinate_parameter_names():
    assert "epsg_code" in GeoCoordinates.coordinate_parameter_names()
