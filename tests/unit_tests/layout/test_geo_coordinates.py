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
    _coord = GeoCoordinates()

    ref_equator = TelescopePosition(name="Equator")
    ref_equator.set_coordinates("mercator", 0.0, 0.0, 0.0)
    scale_factor_equator = _coord._coordinate_scale_factor(ref_equator)
    assert scale_factor_equator == pytest.approx(1.0, rel=1.0e-4)

    ref_lapalma = TelescopePosition(name="LaPalma")
    ref_lapalma.set_coordinates("mercator", 28.7621661, -17.8920302, 2177.0)
    scale_factor_lapalma = _coord._coordinate_scale_factor(ref_lapalma)
    assert scale_factor_lapalma == pytest.approx(1.0003415856959632, rel=1.0e-6)

    ref_paranal = TelescopePosition(name="Paranal")
    ref_paranal.set_coordinates("mercator", -24.68342915, -70.316345, 2147.0)
    scale_factor_paranal = _coord._coordinate_scale_factor(ref_paranal)
    assert scale_factor_paranal == pytest.approx(1.0003368142476146, rel=1.0e-6)

    with pytest.raises(AttributeError):
        _coord._coordinate_scale_factor(None)


def test_valid_reference_point():
    geo_coords = GeoCoordinates()
    reference_point = TelescopePosition(name="LaPalma")
    reference_point.set_coordinates("mercator", 28.7621661, -17.8920302, 2177.0)
    assert geo_coords._valid_reference_point(reference_point)

    reference_point.set_coordinates("mercator", 28.7621661, -17.8920302, np.nan)
    assert not geo_coords._valid_reference_point(reference_point)

    reference_point.set_coordinates("mercator", np.nan, np.nan, 2177.0)
    assert not geo_coords._valid_reference_point(reference_point)

    reference_point.set_coordinates("mercator", np.nan, -17.8920302, 2177.0)
    assert not geo_coords._valid_reference_point(reference_point)

    reference_point.set_coordinates("mercator", 28.7621661, np.nan, 2177.0)
    assert not geo_coords._valid_reference_point(reference_point)
