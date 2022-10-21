#!/usr/bin/python3

import logging
import uuid

import pytest

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def random_id():
    random_id = uuid.uuid4().hex
    return random_id


@pytest.fixture()
def db_cleanup(db, random_id):
    yield
    # Cleanup
    logger.info(f"dropping the telescopes_{random_id} and metadata_{random_id} collections")
    db.dbClient["sandbox"]["telescopes_" + random_id].drop()
    db.dbClient["sandbox"]["metadata_" + random_id].drop()


def test_reading_db_lst(db):

    logger.info("----Testing reading LST-----")
    assert 1 == 1
    pars = db.getModelParameters("north", "lst-1", "Current")
    if db.useMongoDB:
        assert pars["parabolic_dish"]["Value"] == 1
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["parabolic_dish"] == 1
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_nc(db):

    logger.info("----Testing reading MST-NectarCam-----")
    pars = db.getModelParameters("north", "mst-NectarCam-D", "Current")
    if db.useMongoDB:
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_fc(db):

    logger.info("----Testing reading MST-FlashCam-----")
    pars = db.getModelParameters("north", "mst-FlashCam-D", "Current")
    if db.useMongoDB:
        assert pars["camera_pixels"]["Value"] == 1764
    else:
        assert pars["camera_pixels"] == 1764


def test_reading_db_sst(db):

    logger.info("----Testing reading SST-----")
    pars = db.getModelParameters("south", "sst-D", "Current")
    if db.useMongoDB:
        assert pars["camera_pixels"]["Value"] == 2048
    else:
        assert pars["camera_pixels"] == 2048


def test_get_reference_data(db):

    logger.info("----Testing reading reference data-----")
    pars = db.getReferenceData("south", "Prod5")
    assert pars["nsb_reference_value"]["Value"] == pytest.approx(0.24)


def test_get_derived_values(db):

    logger.info("----Testing reading derived values-----")
    pars = db.getDerivedValues("north", "lst-1", "Prod5")
    assert (
        pars["ray_tracing"]["Value"] == "ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv"
    )


def test_copy_telescope_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing copying a whole telescope-----")
    db.copyTelescope(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        telToCopy="North-LST-1",
        versionToCopy="Current",
        newTelName="North-LST-Test",
        collectionName="telescopes",
        dbToCopyTo="sandbox",
        collectionToCopyTo="telescopes_" + random_id,
    )
    db.copyDocuments(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        collection="metadata",
        query={"Entry": "Simulation-Model-Tags"},
        dbToCopyTo="sandbox",
        collectionToCopyTo="metadata_" + random_id,
    )
    pars = db.readMongoDB(
        dbName="sandbox",
        telescopeModelNameDB="North-LST-Test",
        modelVersion="Current",
        runLocation=io_handler.getOutputDirectory(dirType="model", test=True),
        collectionName="telescopes_" + random_id,
        writeFiles=False,
    )
    assert pars["camera_pixels"]["Value"] == 1855

    logger.info("Testing deleting a query (a whole telescope in this case and metadata)")
    query = {"Telescope": "North-LST-Test"}
    db.deleteQuery("sandbox", "telescopes_" + random_id, query)
    query = {"Entry": "Simulation-Model-Tags"}
    db.deleteQuery("sandbox", "metadata_" + random_id, query)

    # After deleting the copied telescope
    # we always expect to get a ValueError (query returning zero results)
    with pytest.raises(ValueError):
        db.readMongoDB(
            dbName="sandbox",
            telescopeModelNameDB="North-LST-Test",
            modelVersion="Current",
            runLocation=io_handler.getOutputDirectory(dirType="model", test=True),
            collectionName="telescopes_" + random_id,
            writeFiles=False,
        )


def test_adding_parameter_version_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing adding a new version of a parameter-----")
    db.copyTelescope(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        telToCopy="North-LST-1",
        versionToCopy="Current",
        newTelName="North-LST-Test",
        collectionName="telescopes",
        dbToCopyTo="sandbox",
        collectionToCopyTo="telescopes_" + random_id,
    )
    db.addParameter(
        dbName="sandbox",
        telescope="North-LST-Test",
        parameter="camera_config_version",
        newVersion="test",
        newValue=42,
        collectionName="telescopes_" + random_id,
    )
    pars = db.readMongoDB(
        dbName="sandbox",
        telescopeModelNameDB="North-LST-Test",
        modelVersion="test",
        runLocation=io_handler.getOutputDirectory(dirType="model", test=True),
        collectionName="telescopes_" + random_id,
        writeFiles=False,
    )
    assert pars["camera_config_version"]["Value"] == 42


def test_update_parameter_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing updating a parameter-----")
    db.copyTelescope(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        telToCopy="North-LST-1",
        versionToCopy="Current",
        newTelName="North-LST-Test",
        collectionName="telescopes",
        dbToCopyTo="sandbox",
        collectionToCopyTo="telescopes_" + random_id,
    )
    db.addParameter(
        dbName="sandbox",
        telescope="North-LST-Test",
        parameter="camera_config_version",
        newVersion="test",
        newValue=42,
        collectionName="telescopes_" + random_id,
    )
    db.updateParameter(
        dbName="sandbox",
        telescope="North-LST-Test",
        version="test",
        parameter="camera_config_version",
        newValue=999,
        collectionName="telescopes_" + random_id,
    )
    pars = db.readMongoDB(
        dbName="sandbox",
        telescopeModelNameDB="North-LST-Test",
        modelVersion="test",
        runLocation=io_handler.getOutputDirectory(dirType="model", test=True),
        collectionName="telescopes_" + random_id,
        writeFiles=False,
    )
    assert pars["camera_config_version"]["Value"] == 999


def test_adding_new_parameter_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing adding a new parameter-----")
    db.copyTelescope(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        telToCopy="North-LST-1",
        versionToCopy="Current",
        newTelName="North-LST-Test",
        collectionName="telescopes",
        dbToCopyTo="sandbox",
        collectionToCopyTo="telescopes_" + random_id,
    )
    db.addNewParameter(
        dbName="sandbox",
        telescope="North-LST-Test",
        version="test",
        parameter="camera_config_version_test",
        value=999,
        collectionName="telescopes_" + random_id,
    )
    pars = db.readMongoDB(
        dbName="sandbox",
        telescopeModelNameDB="North-LST-Test",
        modelVersion="test",
        runLocation=io_handler.getOutputDirectory(dirType="model", test=True),
        collectionName="telescopes_" + random_id,
        writeFiles=False,
    )
    assert pars["camera_config_version_test"]["Value"] == 999


def test_update_parameter_field_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing modifying a field of a parameter-----")
    db.copyTelescope(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        telToCopy="North-LST-1",
        versionToCopy="Current",
        newTelName="North-LST-Test",
        collectionName="telescopes",
        dbToCopyTo="sandbox",
        collectionToCopyTo="telescopes_" + random_id,
    )
    db.copyDocuments(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        collection="metadata",
        query={"Entry": "Simulation-Model-Tags"},
        dbToCopyTo="sandbox",
        collectionToCopyTo="metadata_" + random_id,
    )
    db.updateParameterField(
        dbName="sandbox",
        telescope="North-LST-Test",
        version="Current",
        parameter="camera_pixels",
        field="Applicable",
        newValue=False,
        collectionName="telescopes_" + random_id,
    )
    pars = db.readMongoDB(
        dbName="sandbox",
        telescopeModelNameDB="North-LST-Test",
        modelVersion="Current",
        runLocation=io_handler.getOutputDirectory(dirType="model", test=True),
        collectionName="telescopes_" + random_id,
        writeFiles=False,
    )
    assert pars["camera_pixels"]["Applicable"] is False


def test_reading_db_sites(db):

    logger.info("----Testing reading La Palma parameters-----")
    pars = db.getSiteParameters("North", "Current")
    if db.useMongoDB:
        assert pars["altitude"]["Value"] == 2158
    else:
        assert pars["altitude"] == 2158

    logger.info("----Testing reading Paranal parameters-----")
    pars = db.getSiteParameters("South", "Current")
    if db.useMongoDB:
        assert pars["altitude"]["Value"] == 2147
    else:
        assert pars["altitude"] == 2147


def test_separating_get_and_write(db, io_handler):

    logger.info("----Testing getting parameters and exporting model files-----")
    pars = db.getModelParameters("north", "lst-1", "Current")

    fileList = list()
    for parNow in pars.values():
        if parNow["File"]:
            fileList.append(parNow["Value"])
    db.exportModelFiles(
        pars,
        io_handler.getOutputDirectory(dirType="model", test=True),
    )
    logger.debug(
        "Checking files were written to {}".format(
            io_handler.getOutputDirectory(dirType="model", test=True)
        )
    )
    for fileNow in fileList:
        assert io_handler.getOutputFile(fileNow, dirType="model", test=True).exists()


def test_insert_files_db(db, io_handler):

    logger.info("----Testing inserting files to the DB-----")
    logger.info(
        "Creating a temporary file in {}".format(
            io_handler.getOutputDirectory(dirType="model", test=True)
        )
    )
    fileName = io_handler.getOutputDirectory(dirType="model", test=True) / "test_file.dat"
    with open(fileName, "w") as f:
        f.write("# This is a test file")

    file_id = db.insertFileToDB(fileName, "sandbox")
    assert file_id == db._getFileMongoDB("sandbox", "test_file.dat")._id

    logger.info("Dropping the temporary files in the sandbox")
    db.dbClient["sandbox"]["fs.chunks"].drop()
    db.dbClient["sandbox"]["fs.files"].drop()


def test_get_all_versions(db):

    allVersions = db.getAllVersions(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        telescopeModelName="LST-1",
        site="North",
        parameter="camera_config_file",
        collectionName="telescopes",
    )

    # Check only a subset of the versions so that this test doesn't fail when we add more versions.
    assert all(
        _v in allVersions for _v in ["2018-11-07", "prod3_compatible", "prod4", "2020-06-28"]
    )

    allVersions = db.getAllVersions(
        dbName=db.DB_CTA_SIMULATION_MODEL,
        site="North",
        parameter="altitude",
        collectionName="sites",
    )

    # Check only a subset of the versions so that this test doesn't fail when we add more versions.
    assert all(
        _v in allVersions for _v in ["2015-07-21", "prod3_compatible", "prod4", "2020-06-28"]
    )


def test_get_descriptions(db):

    descriptions = db.getDescriptions()

    assert (
        descriptions["quantum_efficiency"]["description"]
        == "File name for the quantum efficiency curve."
    )
    assert descriptions["camera_pixels"]["description"] == "Number of pixels per camera."
