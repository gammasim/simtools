#!/usr/bin/python3

import logging
import pytest

import astropy.units as u

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.layout.layout_array import LayoutArray

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def configData():
    return {
        "centerLongitude": -17.8920302 * u.deg,
        "centerLatitude": 28.7621661 * u.deg,
        "epsg": 32628,
        "centerAltitude": 2177 * u.m,
        "corsikaSphereRadius": {"LST": 12.5 * u.m, "MST": 9.6 * u.m, "SST": 3 * u.m},
        "corsikaSphereCenter": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        "corsikaObsLevel": 2158 * u.m,
    }


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


@pytest.fixture
def testTelescopeFile(db):
    testFileName = "telescope_positions_prod5_north.ecsv"
    db.exportFileDB(
        dbName="test-data",
        dest=io.getTestModelDirectory(),
        fileName=testFileName
    )

    cfgFile = cfg.findFile(
        testFileName,
        io.getTestModelDirectory()
    )
    return cfgFile


def test_read_tel_list(cfg_setup, configData, testTelescopeFile):

    layout = LayoutArray(name="testLayout",
                         configData=configData)
    layout.readTelescopeListFile(testTelescopeFile)
    layout.convertCoordinates()

    assert 19 == layout.getNumberOfTelescopes()


def test_dict_input(cfg_setup):

    configData = {
        "corsikaSphereRadius": {"LST": 1 * u.m, "MST": 1 * u.m, "SST": 1 * u.m}
    }
    layout = LayoutArray(name="testLayout", configData=configData)
    layout.printTelescopeList()

    for tel, sphere in configData['corsikaSphereRadius'].items():
        assert sphere.value == layout._corsikaSphereRadius[tel]


def test_add_tel(cfg_setup, testTelescopeFile):

    layout = LayoutArray(name="testLayout")
    layout.readTelescopeListFile(testTelescopeFile)
    ntel_before = layout.getNumberOfTelescopes()
    layout.addTelescope(
        telescopeName="L-05", posX=100 * u.m, posY=100 * u.m, posZ=100 * u.m
    )
    ntel_after = layout.getNumberOfTelescopes()

    layout.printTelescopeList()

    assert ntel_before + 1 == ntel_after


def test_build_layout(cfg_setup):

    configData = {
        "centerLongitude": -17.8920302 * u.deg,
        "centerLatitude": 28.7621661 * u.deg,
        "epsg": 32628,
        "centerAltitude": 2177 * u.m,
        "centerNorthing": 3185066.0 * u.m,
        "centerEasting": 217611.0 * u.m,
        "corsikaSphereRadius": {"LST": 12.5 * u.m, "MST": 9.6 * u.m, "SST": 3 * u.m},
        "corsikaSphereCenter": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        "corsikaObsLevel": 2158 * u.m,
    }

    layout = LayoutArray(label="test_layout", name="LST4", configData=configData)

    # Adding 4 LST on a regular grid
    layout.addTelescope(
        telescopeName="L-01", posX=57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m
    )
    layout.addTelescope(
        telescopeName="L-02", posX=-57.5 * u.m, posY=57.5 * u.m, posZ=0 * u.m
    )
    layout.addTelescope(
        telescopeName="L-03", posX=57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m
    )
    layout.addTelescope(
        telescopeName="L-04", posX=-57.5 * u.m, posY=-57.5 * u.m, posZ=0 * u.m
    )

    layout.convertCoordinates()
    layout.printTelescopeList()
    layout.exportTelescopeList()

    # Building a second layout from the file exported by the first one
    layout_copy = LayoutArray(label="test_layout", name="LST4-copy")
    layout_copy.readTelescopeListFile(layout.telescopeListFile)
    layout_copy.printTelescopeList()
    layout_copy.exportTelescopeList()

    # Comparing both layouts
    for par in ["_centerEasting", "_centerLatitude", "_epsg", "_corsikaObsLevel"]:
        assert layout.__dict__[par] == layout_copy.__dict__[par]

    assert layout.getNumberOfTelescopes() == layout_copy.getNumberOfTelescopes()


def test_classmethod(cfg_setup):

    layout = LayoutArray.fromLayoutArrayName("south-Prod5")

    # assume that south-prod5 is the only array with 99 telescopes
    assert 99 == layout.getNumberOfTelescopes()


def test_converting_center_coordinates(cfg_setup, configData):

    layout = LayoutArray(label="test_layout", name="LST4", configData=configData)
    layout.printTelescopeList()

    assert layout._centerNorthing == pytest.approx(3185067.28)
    assert layout._centerEasting == pytest.approx(217609.23)
    assert layout._centerLongitude == pytest.approx(-17.8920302)
    assert layout._centerLatitude == pytest.approx(28.7621661)
    assert layout._corsikaObsLevel == pytest.approx(2158.)
    assert layout._centerAltitude == pytest.approx(2177.)
