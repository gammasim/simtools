#!/usr/bin/python3

import logging
import uuid

import pytest
from astropy import units as u

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def random_id():
    random_id = uuid.uuid4().hex
    return random_id


@pytest.fixture()
@pytest.mark.parametrize("db_config", [True], indirect=True)
def db_cleanup(db_write, random_id):
    yield
    # Cleanup
    logger.info(f"dropping the telescopes_{random_id} and metadata_{random_id} collections")
    db_write.db_client[f"sandbox_{random_id}"]["telescopes_" + random_id].drop()
    db_write.db_client[f"sandbox_{random_id}"]["metadata_" + random_id].drop()


@pytest.fixture()
def db_cleanup_file_sandbox(db_no_config_file, random_id):
    yield
    # Cleanup
    logger.info("Dropping the temporary files in the sandbox")
    db_no_config_file.db_client[f"sandbox_{random_id}"]["fs.chunks"].drop()
    db_no_config_file.db_client[f"sandbox_{random_id}"]["fs.files"].drop()


def test_reading_db_lst(db):
    logger.info("----Testing reading LST-----")
    assert 1 == 1
    pars = db.get_model_parameters("north", "lst-1", "Released")
    if db.mongo_db_config:
        assert pars["parabolic_dish"]["Value"] == 1
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["parabolic_dish"] == 1
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_nc(db):
    logger.info("----Testing reading MST-NectarCam-----")
    pars = db.get_model_parameters("north", "mst-NectarCam-D", "Released")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["Value"] == 1855
    else:
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_fc(db):
    logger.info("----Testing reading MST-FlashCam-----")
    pars = db.get_model_parameters("north", "mst-FlashCam-D", "Released")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["Value"] == 1764
    else:
        assert pars["camera_pixels"] == 1764


def test_reading_db_sst(db):
    logger.info("----Testing reading SST-----")
    pars = db.get_model_parameters("south", "sst-D", "Released")
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


@pytest.mark.parametrize("db_config", [True], indirect=True)
def test_copy_telescope_db(db_write, random_id, db_cleanup, io_handler):
    logger.info("----Testing copying a whole telescope-----")
    db_write.copy_telescope(
        db_name=db_write.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Released",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db_write.copy_documents(
        db_name=db_write.DB_CTA_SIMULATION_MODEL,
        collection="metadata",
        query={"Entry": "Simulation-Model-Tags"},
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="metadata_" + random_id,
    )
    pars = db_write.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        telescope_model_name_db="North-LST-Test",
        model_version="Released",
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_pixels"]["Value"] == 1855

    logger.info("Testing deleting a query (a whole telescope in this case and metadata)")
    query = {"Telescope": "North-LST-Test"}
    db_write.delete_query(f"sandbox_{random_id}", "telescopes_" + random_id, query)
    query = {"Entry": "Simulation-Model-Tags"}
    db_write.delete_query(f"sandbox_{random_id}", "metadata_" + random_id, query)

    # After deleting the copied telescope
    # we always expect to get a ValueError (query returning zero results)
    with pytest.raises(ValueError):
        db_write.read_mongo_db(
            db_name=f"sandbox_{random_id}",
            telescope_model_name_db="North-LST-Test",
            model_version="Released",
            run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
            collection_name="telescopes_" + random_id,
            write_files=False,
        )


@pytest.mark.parametrize("db_config", [True], indirect=True)
def test_adding_parameter_version_db(db_write, random_id, db_cleanup, io_handler):
    logger.info("----Testing adding a new version of a parameter-----")
    db_write.copy_telescope(
        db_name=db_write.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Released",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db_write.add_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        parameter="camera_config_version",
        new_version="test",
        new_value=42,
        collection_name="telescopes_" + random_id,
    )
    pars = db_write.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        telescope_model_name_db="North-LST-Test",
        model_version="test",
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_config_version"]["Value"] == 42


@pytest.mark.parametrize("db_config", [True], indirect=True)
def test_update_parameter_db(db_write, random_id, db_cleanup, io_handler):
    logger.info("----Testing updating a parameter-----")
    db_write.copy_telescope(
        db_name=db_write.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Released",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db_write.add_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        parameter="camera_config_version",
        new_version="test",
        new_value=42,
        collection_name="telescopes_" + random_id,
    )
    db_write.update_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        version="test",
        parameter="camera_config_version",
        new_value=999,
        collection_name="telescopes_" + random_id,
    )
    pars = db_write.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        telescope_model_name_db="North-LST-Test",
        model_version="test",
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_config_version"]["Value"] == 999


@pytest.mark.parametrize("db_config", [True], indirect=True)
def test_adding_new_parameter_db(db_write, random_id, db_cleanup, io_handler):
    logger.info("----Testing adding a new parameter-----")
    db_write.copy_telescope(
        db_name=db_write.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Released",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db_write.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        version="test",
        parameter="new_test_parameter_str",
        value="hello",
        collection_name="telescopes_" + random_id,
    )
    db_write.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        version="test",
        parameter="new_test_parameter_int",
        value=999,
        collection_name="telescopes_" + random_id,
    )
    db_write.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        version="test",
        parameter="new_test_parameter_float",
        value=999.9,
        collection_name="telescopes_" + random_id,
    )
    db_write.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        version="test",
        parameter="new_test_parameter_quantity",
        value=999.9 * u.m,
        collection_name="telescopes_" + random_id,
    )
    db_write.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        version="test",
        parameter="new_test_parameter_quantity_str",
        value="999.9 cm",
        collection_name="telescopes_" + random_id,
    )
    pars = db_write.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        telescope_model_name_db="North-LST-Test",
        model_version="test",
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["new_test_parameter_str"]["Value"] == "hello"
    assert pars["new_test_parameter_str"]["Type"] == "str"
    assert pars["new_test_parameter_int"]["Value"] == 999
    assert pars["new_test_parameter_int"]["Type"] == "int"
    assert pars["new_test_parameter_float"]["Value"] == pytest.approx(999.9)
    assert pars["new_test_parameter_float"]["Type"] == "float"
    assert pars["new_test_parameter_quantity"]["Value"] == pytest.approx(999.9)
    assert pars["new_test_parameter_quantity"]["Type"] == "float"
    assert pars["new_test_parameter_quantity"]["units"] == "m"
    assert pars["new_test_parameter_quantity_str"]["Value"] == pytest.approx(999.9)
    assert pars["new_test_parameter_quantity_str"]["Type"] == "float"
    assert pars["new_test_parameter_quantity_str"]["units"] == "cm"


@pytest.mark.parametrize("db_config", [True], indirect=True)
def test_update_parameter_field_db(db_write, random_id, db_cleanup, io_handler):
    logger.info("----Testing modifying a field of a parameter-----")
    db_write.copy_telescope(
        db_name=db_write.DB_CTA_SIMULATION_MODEL,
        tel_to_copy="North-LST-1",
        version_to_copy="Released",
        new_tel_name="North-LST-Test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes_" + random_id,
    )
    db_write.copy_documents(
        db_name=db_write.DB_CTA_SIMULATION_MODEL,
        collection="metadata",
        query={"Entry": "Simulation-Model-Tags"},
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="metadata_" + random_id,
    )
    db_write.update_parameter_field(
        db_name=f"sandbox_{random_id}",
        telescope="North-LST-Test",
        version="Released",
        parameter="camera_pixels",
        field="Applicable",
        new_value=False,
        collection_name="telescopes_" + random_id,
    )
    pars = db_write.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        telescope_model_name_db="North-LST-Test",
        model_version="Released",
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes_" + random_id,
        write_files=False,
    )
    assert pars["camera_pixels"]["Applicable"] is False


def test_reading_db_sites(db):
    logger.info("----Testing reading La Palma parameters-----")
    pars = db.get_site_parameters("North", "Released")
    if db.mongo_db_config:
        assert pars["altitude"]["Value"] == 2158
    else:
        assert pars["altitude"] == 2158

    logger.info("----Testing reading Paranal parameters-----")
    pars = db.get_site_parameters("South", "Released")
    if db.mongo_db_config:
        assert pars["altitude"]["Value"] == 2147
    else:
        assert pars["altitude"] == 2147


def test_separating_get_and_write(db, io_handler):
    logger.info("----Testing getting parameters and exporting model files-----")
    pars = db.get_model_parameters("north", "lst-1", "Released")

    file_list = list()
    for par_now in pars.values():
        if par_now["File"]:
            file_list.append(par_now["Value"])
    db.export_model_files(
        pars,
        io_handler.get_output_directory(sub_dir="model", dir_type="test"),
    )
    logger.debug(
        "Checking files were written to "
        f"{io_handler.get_output_directory(sub_dir='model', dir_type='test')}"
    )
    for file_now in file_list:
        assert io_handler.get_output_file(file_now, sub_dir="model", dir_type="test").exists()


def test_export_file_db(db, io_handler):
    logger.info("----Testing exporting files from the DB-----")
    output_dir = io_handler.get_output_directory(sub_dir="model", dir_type="test")
    file_name = "mirror_CTA-S-LST_v2020-04-07.dat"
    file_to_export = output_dir / file_name
    db.export_file_db(db.DB_CTA_SIMULATION_MODEL, output_dir, file_name)
    assert file_to_export.exists()


@pytest.mark.parametrize("db_config", [True], indirect=True)
def test_insert_files_db(db_write, io_handler, db_cleanup_file_sandbox, random_id, caplog):
    logger.info("----Testing inserting files to the DB-----")
    logger.info(
        "Creating a temporary file in "
        f"{io_handler.get_output_directory(sub_dir='model', dir_type='test')}"
    )
    file_name = (
        io_handler.get_output_directory(sub_dir="model", dir_type="test")
        / f"test_file_{random_id}.dat"
    )
    with open(file_name, "w") as f:
        f.write("# This is a test file")

    file_id = db_write.insert_file_to_db(file_name, f"sandbox_{random_id}")
    assert (
        file_id
        == db_write._get_file_mongo_db(f"sandbox_{random_id}", f"test_file_{random_id}.dat")._id
    )
    logger.info("Now test inserting the same file again, this time expect a warning")
    with caplog.at_level(logging.WARNING):
        file_id = db_write.insert_file_to_db(file_name, f"sandbox_{random_id}")
    assert "exists in the db_write. Returning its ID" in caplog.text
    assert (
        file_id
        == db_write._get_file_mongo_db(f"sandbox_{random_id}", f"test_file_{random_id}.dat")._id
    )


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
