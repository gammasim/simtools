#!/usr/bin/python3

import logging

import astropy.units as u
import pytest

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
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
        "corsika_sphere_radius": {"LST": 12.5 * u.m, "MST": 9.6 * u.m, "SST": 3 * u.m},
        "corsika_sphere_center": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        "corsika_obs_level": 2158 * u.m,
    }


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


@pytest.fixture
def telescopeTestFile(db):
    testFileName = "telescope_positions_prod5_north.ecsv"
    db.exportFileDB(dbName="test-data", dest=io.getTestModelDirectory(), fileName=testFileName)

    cfgFile = cfg.findFile(testFileName, io.getTestModelDirectory())
    return cfgFile


def test_read_tel_list(cfg_setup, telescopeTestFile):

    layout = LayoutArray(name="testLayout")
    layout.readTelescopeListFile(telescopeTestFile)
    layout.convertCoordinates()

    assert 19 == layout.getNumberOfTelescopes()


def test_add_tel(cfg_setup, telescopeTestFile):

    layout = LayoutArray(name="testLayout")
    layout.readTelescopeListFile(telescopeTestFile)
    ntel_before = layout.getNumberOfTelescopes()
    layout.addTelescope(telescopeName="L-05", posX=100 * u.m, posY=100 * u.m, posZ=100 * u.m)
    ntel_after = layout.getNumberOfTelescopes()

    layout.printTelescopeList()

    assert ntel_before + 1 == ntel_after


def test_build_layout(cfg_setup):

    layoutCenterData = {
        "center_lon": -17.8920302 * u.deg,
        "center_lat": 28.7621661 * u.deg,
        "center_alt": 2177 * u.m,
        "center_northing": 3185066.0 * u.m,
        "center_easting": 217611.0 * u.m,
        "EPSG": 32628,
    }
    corsikaTelescopeData = {
        "corsika_sphere_radius": {"LST": 12.5 * u.m, "MST": 9.6 * u.m, "SST": 3 * u.m},
        "corsika_sphere_center": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        "corsika_obs_level": 2158 * u.m,
    }

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterData,
        corsikaTelescopeData=corsikaTelescopeData,
    )

    # Adding 4 LST on a regular grid
    layout.addTelescope(telescopeName="L-01", posX=57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName="L-02", posX=-57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName="L-03", posX=57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m)
    layout.addTelescope(telescopeName="L-04", posX=-57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m)

    layout.convertCoordinates()
    layout.printTelescopeList()
    layout.exportTelescopeList()

    # Building a second layout from the file exported by the first one
    # TODO


def test_classmethod(cfg_setup):

    layout = LayoutArray.fromLayoutArrayName("south-Prod5")

    # assume that south-prod5 is the only array with 99 telescopes
    assert 99 == layout.getNumberOfTelescopes()


def test_converting_center_coordinates(cfg_setup, layoutCenterDataDict, corsikaTelescopeDataDict):

    layout = LayoutArray(
        label="test_layout",
        name="LST4",
        layoutCenterData=layoutCenterDataDict,
        corsikaTelescopeData=corsikaTelescopeDataDict,
    )
    layout.printTelescopeList()

    _lat, _lon = layout._arrayCenter.getMercatorCoordinates()
    assert _lat.value == pytest.approx(28.7621661)
    assert _lon.value == pytest.approx(-17.8920302)

    _north, _east = layout._arrayCenter.getUtmCoordinates()
    assert _north.value == pytest.approx(3185067.28)
    assert _east.value == pytest.approx(217609.23)

    assert layout._arrayCenter.getAltitude().value == pytest.approx(2177.0)


#    assert layout._arrayCenter["._corsikaObsLevel == pytest.approx(2158.0)
