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
    db.db_client["sandbox"]["telescopes_" + random_id].drop()
    db.db_client["sandbox"]["metadata_" + random_id].drop()


def test_reading_db_lst(db):

    logger.info("----Testing reading LST-----")
    assert 1 == 1
    pars = db.get_model_parameters("north", "lst-1", "Current")
    if db.mongo_db_config:
        assert pars["parabolic_dish"]["Value"] == 1
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["parabolic_dish"] == 1
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_nc(db):

    logger.info("----Testing reading MST-NectarCam-----")
    pars = db.get_model_parameters("north", "mst-NectarCam-D", "Current")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_fc(db):

    logger.info("----Testing reading MST-FlashCam-----")
    pars = db.get_model_parameters("north", "mst-FlashCam-D", "Current")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["Value"] == 1764
    else:
        assert pars["camera_pixels"] == 1764


def test_reading_db_sst(db):

    logger.info("----Testing reading SST-----")
    pars = db.get_model_parameters("south", "sst-D", "Current")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["Value"] == 2048
    else:
        assert pars["camera_pixels"] == 2048


def test_get_reference_data(db):

    logger.info("----Testing reading reference data-----")
    pars = db.get_reference_data("south", "Prod5")
    assert pars["nsb_reference_value"]["Value"] == pytest.approx(0.24)


def test_get_derived_values(db):

    logger.info("----Testing reading derived values-----")
    pars = db.get_derived_values("north", "lst-1", "Prod5")
    assert (
        pars["ray_tracing"]["Value"] == "ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv"
    )


def test_copy_telescope_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing copying a whole telescope-----")
    db.copy_telescope(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Current",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to="sandbox",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db.copy_documents(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        collection="metadata",
        query={"Entry": "Simulation-Model-Tags"},
        db_to_copy_to="sandbox",
        collection_to_copy_to="metadata_" + random_id,
    )
    pars = db.read_mongo_db(
        db_name="sandbox",
        telescope_model_name_db="North-LST-Test",
        model_version="Current",
        run_location=io_handler.get_output_directory(dir_type="model", test=True),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_pixels"]["Value"] == 1855

    logger.info("Testing deleting a query (a whole telescope in this case and metadata)")
    query = {"Telescope": "North-LST-Test"}
    db.delete_query("sandbox", "telescopes_" + random_id, query)
    query = {"Entry": "Simulation-Model-Tags"}
    db.delete_query("sandbox", "metadata_" + random_id, query)

    # After deleting the copied telescope
    # we always expect to get a ValueError (query returning zero results)
    with pytest.raises(ValueError):
        db.read_mongo_db(
            db_name="sandbox",
            telescope_model_name_db="North-LST-Test",
            model_version="Current",
            run_location=io_handler.get_output_directory(dir_type="model", test=True),
            collection_name="telescopes_" + random_id,
            write_files=False,
        )


def test_adding_parameter_version_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing adding a new version of a parameter-----")
    db.copy_telescope(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Current",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to="sandbox",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db.add_parameter(
        db_name="sandbox",
        telescope="North-LST-Test",
        parameter="camera_config_version",
        new_version="test",
        new_value=42,
        collection_name="telescopes_" + random_id,
    )
    pars = db.read_mongo_db(
        db_name="sandbox",
        telescope_model_name_db="North-LST-Test",
        model_version="test",
        run_location=io_handler.get_output_directory(dir_type="model", test=True),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_config_version"]["Value"] == 42


def test_update_parameter_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing updating a parameter-----")
    db.copy_telescope(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Current",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to="sandbox",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db.add_parameter(
        db_name="sandbox",
        telescope="North-LST-Test",
        parameter="camera_config_version",
        new_version="test",
        new_value=42,
        collection_name="telescopes_" + random_id,
    )
    db.update_parameter(
        db_name="sandbox",
        telescope="North-LST-Test",
        version="test",
        parameter="camera_config_version",
        new_value=999,
        collection_name="telescopes_" + random_id,
    )
    pars = db.read_mongo_db(
        db_name="sandbox",
        telescope_model_name_db="North-LST-Test",
        model_version="test",
        run_location=io_handler.get_output_directory(dir_type="model", test=True),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_config_version"]["Value"] == 999


def test_adding_new_parameter_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing adding a new parameter-----")
    db.copy_telescope(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Current",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to="sandbox",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db.add_new_parameter(
        db_name="sandbox",
        telescope="North-LST-Test",
        version="test",
        parameter="camera_config_version_test",
        value=999,
        collection_name="telescopes_" + random_id,
    )
    pars = db.read_mongo_db(
        db_name="sandbox",
        telescope_model_name_db="North-LST-Test",
        model_version="test",
        run_location=io_handler.get_output_directory(dir_type="model", test=True),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_config_version_test"]["Value"] == 999


def test_update_parameter_field_db(db, random_id, db_cleanup, io_handler):

    logger.info("----Testing modifying a field of a parameter-----")
    db.copy_telescope(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Current",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to="sandbox",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db.copy_documents(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        collection="metadata",
        query={"Entry": "Simulation-Model-Tags"},
        db_to_copy_to="sandbox",
        collection_to_copy_to="metadata_" + random_id,
    )
    db.update_parameter_field(
        db_name="sandbox",
        telescope="North-LST-Test",
        version="Current",
        parameter="camera_pixels",
        field="Applicable",
        new_value=False,
        collection_name="telescopes_" + random_id,
    )
    pars = db.read_mongo_db(
        db_name="sandbox",
        telescope_model_name_db="North-LST-Test",
        model_version="Current",
        run_location=io_handler.get_output_directory(dir_type="model", test=True),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_pixels"]["Applicable"] is False


def test_reading_db_sites(db):

    logger.info("----Testing reading La Palma parameters-----")
    pars = db.get_site_parameters("North", "Current")
    if db.mongo_db_config:
        assert pars["altitude"]["Value"] == 2158
    else:
        assert pars["altitude"] == 2158

    logger.info("----Testing reading Paranal parameters-----")
    pars = db.get_site_parameters("South", "Current")
    if db.mongo_db_config:
        assert pars["altitude"]["Value"] == 2147
    else:
        assert pars["altitude"] == 2147


def test_separating_get_and_write(db, io_handler):

    logger.info("----Testing getting parameters and exporting model files-----")
    pars = db.get_model_parameters("north", "lst-1", "Current")

    file_list = list()
    for par_now in pars.values():
        if par_now["File"]:
            file_list.append(par_now["Value"])
    db.export_model_files(
        pars,
        io_handler.get_output_directory(dir_type="model", test=True),
    )
    logger.debug(
        "Checking files were written to "
        f"{io_handler.get_output_directory(dir_type='model', test=True)}"
    )
    for file_now in file_list:
        assert io_handler.get_output_file(file_now, dir_type="model", test=True).exists()


def test_export_file_db(db, io_handler):
    logger.info("----Testing exporting files from the DB-----")
    output_dir = io_handler.get_output_directory(dir_type="model", test=True)
    file_name = "mirror_CTA-S-LST_v2020-04-07.dat"
    file_to_export = output_dir / file_name
    db.export_file_db(db.DB_CTA_SIMULATION_MODEL, output_dir, file_name)
    assert file_to_export.exists()


def test_insert_files_db(db, io_handler, db_cleanup_file_sandbox, random_id, caplog):

    logger.info("----Testing inserting files to the DB-----")
    logger.info(
        "Creating a temporary file in "
        f"{io_handler.get_output_directory(dir_type='model', test=True)}"
    )
    file_name = (
        io_handler.get_output_directory(dir_type="model", test=True) / f"test_file_{random_id}.dat"
    )
    with open(file_name, "w") as f:
        f.write("# This is a test file")

    file_id = db.insert_file_to_db(file_name, "sandbox")
    assert file_id == db._get_file_mongo_db("sandbox", f"test_file_{random_id}.dat")._id
    logger.info("Now test inserting the same file again, this time expect a warning")
    with caplog.at_level(logging.WARNING):
        file_id = db.insert_file_to_db(file_name, "sandbox")
    assert "exists in the DB. Returning its ID" in caplog.text
    assert file_id == db._get_file_mongo_db("sandbox", f"test_file_{random_id}.dat")._id


def test_get_all_versions(db):

    all_versions = db.get_all_versions(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        telescope_model_name="LST-1",
        site="North",
        parameter="camera_config_file",
        collection_name="telescopes",
    )

    # Check only a subset of the versions so that this test doesn't fail when we add more versions.
    assert all(
        _v in all_versions for _v in ["2018-11-07", "prod3_compatible", "prod4", "2020-06-28"]
    )

    all_versions = db.get_all_versions(
        db_name=db.DB_CTA_SIMULATION_MODEL,
        site="North",
        parameter="altitude",
        collection_name="sites",
    )

    # Check only a subset of the versions so that this test doesn't fail when we add more versions.
    assert all(
        _v in all_versions for _v in ["2015-07-21", "prod3_compatible", "prod4", "2020-06-28"]
    )


def test_get_descriptions(db):

    descriptions = db.get_descriptions()

    assert (
        descriptions["quantum_efficiency"]["description"]
        == "File name for the quantum efficiency curve."
    )
    assert descriptions["camera_pixels"]["description"] == "Number of pixels per camera."
