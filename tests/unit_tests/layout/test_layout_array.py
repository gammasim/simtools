#!/usr/bin/python3

import logging

import astropy.units as u
import numpy as np
import pytest

import simtools.util.general as gen
from simtools.layout.layout_array import LayoutArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def layoutCenterDataDict():
    return {
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "center_easting": None,
        "center_northing": None,
        "EPSG": 32628,
        "center_alt": 2177 * u.m,
    }


@pytest.fixture
def corsikaTelescopeDataDict():
    return {
        "corsika_sphere_radius": {"LST": 12.5 * u.m, "MST": 9.15 * u.m, "SST": 3.0 * u.m},
        "corsika_sphere_center": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        "corsika_obs_level": 2158 * u.m,
    }


@pytest.fixture
def telescopeTestFile(db, args_dict, io_handler):
    testFileName = "telescope_positions-North-TestLayout.ecsv"
    db.export_file_db(
        dbName="test-data",
        dest=io_handler.get_output_directory(dirType="model", test=True),
        fileName=testFileName,
    )

    cfgFile = gen.find_file(
        testFileName,
        io_handler.get_output_directory(dirType="model", test=True),
    )
    return cfgFile


def test_from_layout_array_name(io_handler):

    layout = LayoutArray.from_layout_array_name("south-TestLayout")

    assert 99 == layout.get_number_of_telescopes()


def test_initialize_coordinate_systems(layoutCenterDataDict):

    layout = LayoutArray(name="testLayout")
    layout._initialize_coordinate_systems()
    _x, _y, _z = layout._arrayCenter.get_coordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == 0.0 * u.m
    _lat, _lon, _z = layout._arrayCenter.get_coordinates("mercator")
    assert np.isnan(_lat) and np.isnan(_lon)

    layout._initialize_coordinate_systems(layoutCenterDataDict, False)
    _x, _y, _z = layout._arrayCenter.get_coordinates("corsika")
    assert _x == 0.0 * u.m and _y == 0.0 * u.m and _z == layoutCenterDataDict["center_alt"]
    _lat, _lon, _z = layout._arrayCenter.get_coordinates("mercator")
    assert _lat.value == pytest.approx(layoutCenterDataDict["center_lat"].value, 1.0e-2)
    assert _lon.value == pytest.approx(layoutCenterDataDict["center_lon"].value, 1.0e-2)
    _E, _N, _z = layout._arrayCenter.get_coordinates("utm")
    assert _E.value == pytest.approx(217609.0, 1.0)
    assert _N.value == pytest.approx(3185067.0, 1.0)


def test_initialize_corsika_telescopeFromFile(corsikaTelescopeDataDict, args_dict, io_handler):

    layout = LayoutArray(name="testLayout")
    layout._initialize_corsika_telescope()

    for key, value in corsikaTelescopeDataDict["corsika_sphere_radius"].items():
        assert value == layout._corsikaTelescope["corsika_sphere_radius"][key]
    for key, value in corsikaTelescopeDataDict["corsika_sphere_center"].items():
        assert value == layout._corsikaTelescope["corsika_sphere_center"][key]


def test_read_tel_list(telescopeTestFile):

    layout = LayoutArray(name="testLayout")
    layout.read_telescope_list_file(telescopeTestFile)
    layout.convert_coordinates()
    assert 19 == layout.get_number_of_telescopes()

    layout_2 = LayoutArray(name="testLayout", telescopeListFile=telescopeTestFile)
    layout_2.convert_coordinates()
    assert 19 == layout_2.get_number_of_telescopes()


def test_add_tel(telescopeTestFile):

    layout = LayoutArray(name="testLayout")
    layout.read_telescope_list_file(telescopeTestFile)
    ntel_before = layout.get_number_of_telescopes()
    layout.add_telescope("LST-05", "corsika", 100.0 * u.m, 50.0 * u.m, 2177.0 * u.m)
    ntel_after = layout.get_number_of_telescopes()
    assert ntel_before + 1 == ntel_after

    layout.add_telescope("LST-05", "corsika", 100.0 * u.m, 50.0 * u.m, None, 50.0 * u.m)
    assert layout._telescopeList[-1].get_altitude().value == pytest.approx(2192.0)


def test_build_layout(
    layoutCenterDataDict, corsikaTelescopeDataDict, tmp_test_directory, io_handler
):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )

    layout.add_telescope(
        telescopeName="LST-01", crsName="corsika", xx=57.5 * u.m, yy=57.5 * u.m, telCorsikaZ=0 * u.m
    )
    layout.add_telescope(
        telescopeName="LST-02",
        crsName="corsika",
        xx=-57.5 * u.m,
        yy=57.5 * u.m,
        telCorsikaZ=0 * u.m,
    )
    layout.add_telescope(
        telescopeName="LST-02",
        crsName="corsika",
        xx=57.5 * u.m,
        yy=-57.5 * u.m,
        telCorsikaZ=0 * u.m,
    )
    layout.add_telescope(
        telescopeName="LST-04",
        crsName="corsika",
        xx=-57.5 * u.m,
        yy=-57.5 * u.m,
        telCorsikaZ=0 * u.m,
    )

    layout.convert_coordinates()
    layout.print_telescope_list()
    layout.export_telescope_list(crsName="corsika")

    # Building a second layout from the file exported by the first one
    layout_2 = LayoutArray("test_layout_2")
    layout_2.read_telescope_list_file(layout.telescopeListFile)

    assert 4 == layout_2.get_number_of_telescopes()
    assert layout_2._arrayCenter.get_altitude().value == pytest.approx(
        layout._arrayCenter.get_altitude().value, 1.0e-2
    )


def test_converting_center_coordinates(layoutCenterDataDict, corsikaTelescopeDataDict):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )

    _lat, _lon, _ = layout._arrayCenter.get_coordinates("mercator")
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _east, _north, _ = layout._arrayCenter.get_coordinates("utm")
    assert _north.value == pytest.approx(3185067.28)
    assert _east.value == pytest.approx(217609.23)

    assert layout._arrayCenter.get_altitude().value == pytest.approx(2177.0)


def test_get_corsika_input_list(layoutCenterDataDict, corsikaTelescopeDataDict, telescopeTestFile):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )
    layout.add_telescope(
        telescopeName="LST-01", crsName="corsika", xx=57.5 * u.m, yy=57.5 * u.m, telCorsikaZ=0 * u.m
    )
    corsikaInputList = layout.get_corsika_input_list()

    assert corsikaInputList == "TELESCOPE\t 57.500E2\t 57.500E2\t 0.000E2\t 12.500E2\t # LST-01\n"


def test_altitude_from_corsika_z(layoutCenterDataDict, corsikaTelescopeDataDict):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )

    layout.add_telescope(
        telescopeName="LST-01", crsName="corsika", xx=57.5 * u.m, yy=57.5 * u.m, telCorsikaZ=0 * u.m
    )

    assert layout._altitude_from_corsika_z(5.0 * u.m, None, "LST-01").value == pytest.approx(2147.0)
    assert layout._altitude_from_corsika_z(None, 8848.0 * u.m, "LST-01").value == pytest.approx(
        6706.0
    )
    with pytest.raises(TypeError):
        layout._altitude_from_corsika_z(5.0, None, "LST-01")
