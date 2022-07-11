#!/usr/bin/python3

import pytest
import logging
# import subprocess
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def db(set_db):
    db = db_handler.DatabaseHandler()
    return db


@pytest.fixture
def DB_CTA_SIMULATION_MODEL():
    return "CTA-Simulation-Model"


def test_reading_db_lst(db):

    logger.info("----Testing reading LST-----")
    pars = db.getModelParameters("north", "lst-1", "Current")
    if cfg.get("useMongoDB"):
        assert pars["parabolic_dish"]["Value"] == 1
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["parabolic_dish"] == 1
        assert pars["camera_pixels"] == 1855

    # TODO - is this part of testing the reading from the DB?
    db.exportModelFiles(pars, io.getTestOutputDirectory())
    logger.info("Listing files written in {}".format(io.getTestOutputDirectory()))
    # subprocess.call(["ls -lh {}".format(io.getTestOutputDirectory())], shell=True)


def test_reading_db_mst_nc(db):

    logger.info("----Testing reading MST-NectarCam-----")
    pars = db.getModelParameters("north", "mst-NectarCam-D", "Current")
    if cfg.get("useMongoDB"):
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_fc(db):

    logger.info("----Testing reading MST-FlashCam-----")
    pars = db.getModelParameters("north", "mst-FlashCam-D", "Current")
    if cfg.get("useMongoDB"):
        assert pars["camera_pixels"]["Value"] == 1764
    else:
        assert pars["camera_pixels"] == 1764


def test_reading_db_sst(db):

    logger.info("----Testing reading SST-----")
    pars = db.getModelParameters("south", "sst-D", "Current")
    if cfg.get("useMongoDB"):
        assert pars["camera_pixels"]["Value"] == 2048
    else:
        assert pars["camera_pixels"] == 2048


def test_copy_telescope_db(db, DB_CTA_SIMULATION_MODEL):

    logger.info("----Testing copying a whole telescope-----")
    db.copyTelescope(
        DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.copyDocuments(
        DB_CTA_SIMULATION_MODEL,
        "metadata",
        {"Entry": "Simulation-Model-Tags"},
        "sandbox",
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "Current", io.getTestOutputDirectory(), False
    )
    assert pars["camera_pixels"]["Value"] == 1855

    # TODO - no sure what is tested below
    logger.info(
        "Testing deleting a query (a whole telescope in this case and metadata)"
    )
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)
    query = {"Entry": "Simulation-Model-Tags"}
    db.deleteQuery("sandbox", "metadata", query)


def test_adding_parameter_version_db(db, DB_CTA_SIMULATION_MODEL):

    logger.info("----Testing adding a new version of a parameter-----")
    db.copyTelescope(
        DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.addParameter(
        "sandbox", "North-LST-Test", "camera_config_version", "test", 42
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "test", io.getTestOutputDirectory(), False
    )
    assert pars["camera_config_version"]["Value"] == 42

    # TODO - what is tested here? or is this cleanup?
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)


def test_update_parameter_db(db, DB_CTA_SIMULATION_MODEL):

    logger.info("----Testing updating a parameter-----")
    db.copyTelescope(
        DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.addParameter(
        "sandbox", "North-LST-Test", "camera_config_version", "test", 42
    )
    db.updateParameter(
        "sandbox", "North-LST-Test", "test", "camera_config_version", 999
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "test", io.getTestOutputDirectory(), False
    )
    assert pars["camera_config_version"]["Value"] == 999

    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)


def test_adding_new_parameter_db(db, DB_CTA_SIMULATION_MODEL):

    logger.info("----Testing adding a new parameter-----")
    db.copyTelescope(
        DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.addNewParameter(
        "sandbox", "North-LST-Test", "test", "camera_config_version_test", 999
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "test", io.getTestOutputDirectory(), False
    )
    assert pars["camera_config_version_test"]["Value"] == 999

    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)


def test_update_parameter_field_db(db, DB_CTA_SIMULATION_MODEL):

    logger.info("----Testing modifying a field of a parameter-----")
    db.copyTelescope(
        DB_CTA_SIMULATION_MODEL,
        "North-LST-1",
        "Current",
        "North-LST-Test",
        "sandbox",
    )
    db.copyDocuments(
        DB_CTA_SIMULATION_MODEL,
        "metadata",
        {"Entry": "Simulation-Model-Tags"},
        "sandbox",
    )

    db.updateParameterField(
        dbName="sandbox",
        telescope="North-LST-Test",
        version="Current",
        parameter="camera_pixels",
        field="Applicable",
        newValue=False
    )
    pars = db.readMongoDB(
        "sandbox", "North-LST-Test", "Current", io.getTestOutputDirectory(), False
    )
    assert pars["camera_pixels"]["Applicable"] is False

    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes", query)
    query = {"Entry": "Simulation-Model-Tags"}
    db.deleteQuery("sandbox", "metadata", query)


def test_reading_db_sites(db):

    logger.info("----Testing reading La Palma parameters-----")
    pars = db.getSiteParameters("North", "Current")
    if cfg.get("useMongoDB"):
        assert pars["altitude"]["Value"] == 2158
    else:
        assert pars["altitude"] == 2158

    logger.info("Listing files written in {}".format(io.getTestOutputDirectory()))
    # subprocess.call(["ls -lh {}".format(io.getTestOutputDirectory())], shell=True)

    logger.info(
        "Removing the files written in {}".format(io.getTestOutputDirectory())
    )
    # subprocess.call(["rm -f {}/*".format(io.getTestOutputDirectory())], shell=True)

    logger.info("----Testing reading Paranal parameters-----")
    pars = db.getSiteParameters("South", "Current")
    if cfg.get("useMongoDB"):
        assert pars["altitude"]["Value"] == 2147
    else:
        assert pars["altitude"] == 2147

    logger.info("Listing files written in {}".format(io.getTestOutputDirectory()))
    # subprocess.call(["ls -lh {}".format(io.getTestOutputDirectory())], shell=True)

    logger.info(
        "Removing the files written in {}".format(io.getTestOutputDirectory())
    )
    # subprocess.call(["rm -f {}/*".format(io.getTestOutputDirectory())], shell=True)

    return


def test_separating_get_and_write(db):
    pars = db.getModelParameters("north", "lst-1", "prod4")

    logger.info("Listing files written in {}".format(io.getTestOutputDirectory()))
    # subprocess.call(["ls -lh {}".format(io.getTestOutputDirectory())], shell=True)

    db.exportModelFiles(pars, io.getTestOutputDirectory())

    logger.info("Listing files written in {}".format(io.getTestOutputDirectory()))
    # subprocess.call(["ls -lh {}".format(io.getTestOutputDirectory())], shell=True)


def test_insert_files_db(db):

    logger.info("----Testing inserting files to the DB-----")
    logger.info(
        "Creating a temporary file in {}".format(io.getTestOutputDirectory())
    )
    fileName = Path(io.getTestOutputDirectory()).joinpath("test_file.dat")
    with open(fileName, "w") as f:
        f.write("# This is a test file")

    file_id = db.insertFileToDB(fileName, "sandbox")
    assert file_id == db._getFileMongoDB("sandbox", "test_file.dat")._id

    # subprocess.call(["rm -f {}".format(fileName)], shell=True)

    logger.info("Dropping the temporary files in the sandbox")
    db.dbClient["sandbox"]["fs.chunks"].drop()
    db.dbClient["sandbox"]["fs.files"].drop()
