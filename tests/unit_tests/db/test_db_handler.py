#!/usr/bin/python3

import copy
import logging
import uuid

import pytest
from astropy import units as u

from simtools.db import db_handler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def random_id():
    return uuid.uuid4().hex


@pytest.fixture
def db_no_config_file():
    """Database object (without configuration)."""
    return db_handler.DatabaseHandler(mongo_db_config=None)


@pytest.fixture
def _db_cleanup(db, random_id):
    yield
    # Cleanup
    logger.info(f"dropping sandbox_{random_id} collections")
    db.db_client[f"sandbox_{random_id}"]["telescopes"].drop()
    db.db_client[f"sandbox_{random_id}"]["calibration_devices"].drop()
    db.db_client[f"sandbox_{random_id}"]["sites"].drop()


@pytest.fixture
def _db_cleanup_file_sandbox(db_no_config_file, random_id):
    yield
    # Cleanup
    logger.info("Dropping the temporary files in the sandbox")
    db_no_config_file.db_client[f"sandbox_{random_id}"]["fs.chunks"].drop()
    db_no_config_file.db_client[f"sandbox_{random_id}"]["fs.files"].drop()


def test_find_latest_simulation_model_db(db, db_no_config_file, mocker):

    db_no_config_file._find_latest_simulation_model_db()
    assert db_no_config_file.mongo_db_config is None

    db_name = db.mongo_db_config["db_simulation_model"]
    db._find_latest_simulation_model_db()
    assert db_name == db.mongo_db_config["db_simulation_model"]

    db_copy = copy.deepcopy(db)
    db_copy.mongo_db_config["db_simulation_model"] = "DB_NAME-LATEST"
    with pytest.raises(
        ValueError, match=r"Found LATEST in the DB name but no matching versions found in DB."
    ):
        db_copy._find_latest_simulation_model_db()

    db_names = [
        "CTAO-Simulation-Model-v0-3-0",
        "CTAO-Simulation-Model-v0-2-0",
        "CTAO-Simulation-Model-v0-1-19",
        "CTAO-Simulation-Model-v0-3-9",
        "CTAO-Simulation-Model-v0-3-19",
        "CTAO-Simulation-Model-v0-3-0",
        "CTAO-Simulation-Model-v0-3-0-alpha-2",
        "CTAO-Simulation-Model-v0-4-19-alpha-1",
        "CTAO-Simulation-Model-v0-4-19-dev1",
    ]
    mocker.patch.object(db_copy.db_client, "list_database_names", return_value=db_names)
    db_copy.mongo_db_config["db_simulation_model"] = "CTAO-Simulation-Model-LATEST"
    db_copy._find_latest_simulation_model_db()
    assert db_copy.mongo_db_config["db_simulation_model"] == "CTAO-Simulation-Model-v0-3-19"


def test_reading_db_lst_without_simulation_repo(db, model_version):

    db_copy = copy.deepcopy(db)
    db_copy.mongo_db_config["db_simulation_model_url"] = None
    pars = db.get_model_parameters("North", "LSTN-01", model_version, collection="telescopes")
    assert pars["parabolic_dish"]["value"] == 1
    assert pars["camera_pixels"]["value"] == 1855


def test_reading_db_lst(db, model_version):
    logger.info("----Testing reading LST-North-----")
    pars = db.get_model_parameters("North", "LSTN-01", model_version, collection="telescopes")
    if db.mongo_db_config:
        assert pars["parabolic_dish"]["value"] == 1
        assert pars["camera_pixels"]["value"] == 1855
    else:
        assert pars["parabolic_dish"] == 1
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_nc(db, model_version):
    logger.info("----Testing reading MST-North-----")
    pars = db.get_model_parameters("North", "MSTN-design", model_version, collection="telescopes")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["value"] == 1855
    else:
        assert pars["camera_pixels"] == 1855


def test_reading_db_mst_fc(db, model_version):
    logger.info("----Testing reading MST-South-----")
    pars = db.get_model_parameters("South", "MSTS-design", model_version, collection="telescopes")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["value"] == 1764
    else:
        assert pars["camera_pixels"] == 1764


def test_reading_db_sst(db, model_version):
    logger.info("----Testing reading SST-----")
    pars = db.get_model_parameters("South", "SSTS-design", model_version, collection="telescopes")
    if db.mongo_db_config:
        assert pars["camera_pixels"]["value"] == 2048
    else:
        assert pars["camera_pixels"] == 2048


@pytest.mark.xfail(reason="Test requires Derived-Values Database")
def test_get_derived_values(db, model_version_prod5):
    logger.info("----Testing reading derived values-----")
    try:
        pars = db.get_derived_values("North", "LSTN-01", model_version_prod5)
        assert (
            pars["ray_tracing"]["value"]
            == "ray-tracing-North-LST-1-d10.0-za20.0_validate_optics.ecsv"
        )
    except ValueError:
        logger.error("Derived DB not updated for new telescope names. Expect failure")
        raise AssertionError

    with pytest.raises(ValueError, match=r"^abc"):
        pars = db.get_derived_values("North", None, model_version_prod5)


def test_get_sim_telarray_configuration_parameters(db, model_version):

    _pars = db.get_sim_telarray_configuration_parameters("North", "LSTN-01", model_version)
    assert "min_photoelectrons" in _pars

    _pars = db.get_sim_telarray_configuration_parameters("North", "LSTN-design", model_version)
    assert "min_photoelectrons" in _pars


@pytest.mark.usefixtures("_db_cleanup")
def test_copy_array_element_db(db, random_id, io_handler, model_version):
    logger.info("----Testing copying a whole telescope-----")
    db.copy_array_element(
        db_name=None,
        element_to_copy="LSTN-01",
        version_to_copy=model_version,
        new_array_element_name="LSTN-test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes",
    )
    pars = db.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        model_version=model_version,
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes",
        write_files=False,
    )
    assert pars["camera_pixels"]["value"] == 1855

    logger.info("Testing deleting a query (a telescope)")
    query = {"instrument": "LSTN-test"}
    db.delete_query(f"sandbox_{random_id}", "telescopes", query)

    # After deleting the copied telescope
    # we always expect to get a ValueError (query returning zero results)
    with pytest.raises(ValueError, match=r"The following query returned zero results"):
        db.read_mongo_db(
            db_name=f"sandbox_{random_id}",
            array_element_name="LSTN-test",
            model_version=model_version,
            run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
            collection_name="telescopes",
            write_files=False,
        )


@pytest.mark.usefixtures("_db_cleanup")
def test_adding_new_parameter_db(db, random_id, io_handler, model_version):
    logger.info("----Testing adding a new parameter-----")
    test_model_version = "0.0.9876"
    db.copy_array_element(
        db_name=None,
        element_to_copy="LSTN-01",
        version_to_copy=model_version,
        new_array_element_name="LSTN-test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes",
    )
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        version=test_model_version,
        parameter="new_test_parameter_str",
        value="hello",
        collection_name="telescopes",
    )
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        version=test_model_version,
        parameter="new_test_parameter_int",
        value=999,
        collection_name="telescopes",
    )
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        version=test_model_version,
        parameter="new_test_parameter_float",
        value=999.9,
        collection_name="telescopes",
    )
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        version=test_model_version,
        parameter="new_test_parameter_quantity",
        value=999.9 * u.m,
        collection_name="telescopes",
    )
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        version=test_model_version,
        parameter="new_test_parameter_quantity_str",
        value="999.9 cm",
        collection_name="telescopes",
    )
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        version=test_model_version,
        parameter="new_test_parameter_simtel_list",
        value="0.969 0.0 0.0 0.0 0.0 0.0",
        collection_name="telescopes",
        unit=None,
    )
    pars = db.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        model_version=test_model_version,
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes",
        write_files=False,
    )
    assert pars["new_test_parameter_str"]["value"] == "hello"
    assert pars["new_test_parameter_str"]["type"] == "str"
    assert pars["new_test_parameter_int"]["value"] == 999
    assert pars["new_test_parameter_int"]["type"] == "int"
    assert pars["new_test_parameter_float"]["value"] == pytest.approx(999.9)
    assert pars["new_test_parameter_float"]["type"] == "float"
    assert pars["new_test_parameter_quantity"]["value"] == pytest.approx(999.9)
    assert pars["new_test_parameter_quantity"]["type"] == "float"
    assert pars["new_test_parameter_quantity"]["unit"] == "m"
    assert pars["new_test_parameter_quantity_str"]["value"] == pytest.approx(999.9)
    assert pars["new_test_parameter_quantity_str"]["type"] == "float"
    assert pars["new_test_parameter_quantity_str"]["unit"] == "cm"

    # make sure that cache has been emptied after updating
    assert (
        db._parameter_cache_key("North", "LSTN-test", test_model_version)
        not in db.model_parameters_cached
    )

    # site parameters
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        site="North",
        version=test_model_version,
        parameter="corsika_observation_level",
        value="1800. m",
        collection_name="sites",
    )
    pars = db.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        array_element_name="North",
        model_version=test_model_version,
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="sites",
        write_files=False,
    )
    assert pars["corsika_observation_level"]["value"] == pytest.approx(1800.0)
    assert pars["corsika_observation_level"]["unit"] == "m"

    # calibration_devices parameters
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        array_element_name="ILLN-test",
        version=test_model_version,
        parameter="led_pulse_offset",
        value="0 ns",
        collection_name="calibration_devices",
    )
    pars = db.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        array_element_name="ILLN-test",
        model_version=test_model_version,
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="calibration_devices",
        write_files=False,
    )
    assert pars["led_pulse_offset"]["value"] == 0
    assert pars["led_pulse_offset"]["unit"] == "ns"

    # wrong collection
    with pytest.raises(ValueError, match=r"^Cannot add parameter to collection "):
        db.add_new_parameter(
            db_name=f"sandbox_{random_id}",
            site="North",
            version=test_model_version,
            parameter="corsika_observation_level",
            value="1800. m",
            collection_name="wrong_collection",
        )


@pytest.mark.usefixtures("_db_cleanup")
def test_update_parameter_field_db(db, random_id, io_handler, model_version):
    logger.info("----Testing modifying a field of a parameter-----")
    db.copy_array_element(
        db_name=None,
        element_to_copy="LSTN-01",
        version_to_copy=model_version,
        new_array_element_name="LSTN-test",
        collection_name="telescopes",
        db_to_copy_to=f"sandbox_{random_id}",
        collection_to_copy_to="telescopes",
    )
    db.update_parameter_field(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        model_version=model_version,
        parameter="camera_pixels",
        field="applicable",
        new_value=False,
        collection_name="telescopes",
    )
    pars = db.read_mongo_db(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        model_version=model_version,
        run_location=io_handler.get_output_directory(sub_dir="model", dir_type="test"),
        collection_name="telescopes",
        write_files=False,
    )
    assert pars["camera_pixels"]["applicable"] is False

    with pytest.raises(ValueError, match=r"You need to specify an array element or a site."):
        db.update_parameter_field(
            db_name=f"sandbox_{random_id}",
            array_element_name=None,
            site=None,
            model_version=model_version,
            parameter="not_important",
            field="applicable",
            new_value=False,
            collection_name="site",
        )

    # make sure that cache has been emptied after updating
    assert (
        db._parameter_cache_key("North", "LSTN-test", model_version)
        not in db.model_parameters_cached
    )


def test_reading_db_sites(db, db_config, simulation_model_url, model_version):
    logger.info("----Testing reading La Palma parameters-----")
    db.mongo_db_config["db_simulation_model_url"] = None
    pars = db.get_site_parameters("North", model_version)
    if db.mongo_db_config:
        _obs_level = pars["corsika_observation_level"].get("value")
        assert _obs_level == pytest.approx(2156.0)
    else:
        assert pars["altitude"] == 2156

    logger.info("----Testing reading Paranal parameters-----")
    pars = db.get_site_parameters("South", model_version)
    if db.mongo_db_config:
        _obs_level = pars["corsika_observation_level"].get("value")
        assert _obs_level == pytest.approx(2147.0)
    else:
        assert pars["altitude"] == 2147

    db._reset_parameter_cache("South", None, model_version)
    if db.mongo_db_config.get("db_simulation_model_url", None) is None:
        db.mongo_db_config["db_simulation_model_url"] = simulation_model_url
    pars = db.get_site_parameters("South", model_version)
    assert pars["corsika_observation_level"]["value"] == 2147.0
    db.mongo_db_config["db_simulation_model_url"] = None  # make sure that this is reset


def test_separating_get_and_write(db, io_handler, model_version):
    logger.info("----Testing getting parameters and exporting model files-----")
    pars = db.get_model_parameters("North", "LSTN-01", model_version, collection="telescopes")

    file_list = []
    for par_now in pars.values():
        if par_now["file"] and par_now["value"] is not None:
            file_list.append(par_now["value"])
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
    db.export_file_db(None, output_dir, file_name)
    assert file_to_export.exists()


@pytest.mark.usefixtures("_db_cleanup_file_sandbox")
def test_insert_files_db(db, io_handler, random_id, caplog):
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

    file_id = db.insert_file_to_db(file_name, f"sandbox_{random_id}")
    assert (
        file_id == db._get_file_mongo_db(f"sandbox_{random_id}", f"test_file_{random_id}.dat")._id
    )
    logger.info("Now test inserting the same file again, this time expect a warning")
    with caplog.at_level(logging.WARNING):
        file_id = db.insert_file_to_db(file_name, f"sandbox_{random_id}")
    assert "exists in the DB. Returning its ID" in caplog.text
    assert (
        file_id == db._get_file_mongo_db(f"sandbox_{random_id}", f"test_file_{random_id}.dat")._id
    )


def test_get_all_versions(db, mocker, caplog):

    # not specifying any database names, collections, or parameters
    all_versions = db.get_all_versions()
    assert all(_v in all_versions for _v in ["5.0.0", "6.0.0"])
    assert any(key.endswith("None") for key in db.model_versions_cached)

    # not specifying a telescope model name and parameter
    all_versions = db.get_all_versions(
        array_element_name=None,
        site="North",
        parameter=None,
        collection="telescopes",
    )
    assert all(_v in all_versions for _v in ["5.0.0", "6.0.0"])
    assert any("telescopes" in key for key in db.model_versions_cached)

    # using a specific parameter
    all_versions = db.get_all_versions(
        array_element_name="LSTN-01",
        site="North",
        parameter="camera_config_file",
        collection="telescopes",
    )
    assert all(_v in all_versions for _v in ["5.0.0", "6.0.0"])
    assert any(
        key.endswith("telescopes-camera_config_file-LSTN-01") for key in db.model_versions_cached
    )

    all_versions = db.get_all_versions(
        site="North",
        parameter="corsika_observation_level",
        collection="sites",
    )
    assert all(_v in all_versions for _v in ["5.0.0", "6.0.0"])
    assert any(
        key.endswith("sites-corsika_observation_level-North") for key in db.model_versions_cached
    )

    # no db_name defined
    mocker.patch.object(db, "_get_db_name", return_value=None)
    with caplog.at_level(logging.WARNING):
        assert db.get_all_versions() == []
    assert "No database name defined to determine" in caplog.text


def test_get_all_available_array_elements(db, model_version):
    available_telescopes = db.get_all_available_array_elements(
        model_version=model_version, collection="telescopes"
    )

    expected_telescope_names = [
        "LSTN-01",
        "LSTN-02",
        "LSTN-03",
        "LSTN-04",
        "LSTS-design",
        "MSTN-design",
        "MSTS-design",
        "SSTS-design",
    ]
    assert all(_t in available_telescopes for _t in expected_telescope_names)

    with pytest.raises(
        ValueError, match=r"^Query for collection name wrong_collection not implemented."
    ):
        db.get_all_available_array_elements(
            model_version=model_version, collection="wrong_collection"
        )


def test_get_available_array_elements_of_type(db, model_version):

    available_types = db.get_available_array_elements_of_type(
        array_element_type="LSTN", model_version=model_version, collection="telescopes"
    )
    assert "LSTN-04" in available_types
    assert len(available_types) == 4
    assert all("design" not in tel_type for tel_type in available_types)


def test_get_array_element_db_name(db, model_version_prod5):
    assert db.get_array_element_db_name("LSTN-01", model_version=model_version_prod5) == "LSTN-01"
    assert (
        db.get_array_element_db_name("LSTS-20", model_version=model_version_prod5) == "LSTS-design"
    )
    assert (
        db.get_array_element_db_name("LSTN-design", model_version=model_version_prod5)
        == "LSTN-design"
    )
    assert (
        db.get_array_element_db_name("LSTS-design", model_version=model_version_prod5)
        == "LSTS-design"
    )
    assert (
        db.get_array_element_db_name("SSTS-91", model_version=model_version_prod5) == "SSTS-design"
    )
    assert (
        db.get_array_element_db_name("SSTS-design", model_version=model_version_prod5)
        == "SSTS-design"
    )
    with pytest.raises(ValueError, match=r"Invalid name SSTN"):
        db.get_array_element_db_name(
            "SSTN-05", model_version=model_version_prod5, collection="telescopes"
        )

    with pytest.raises(ValueError, match=r"Invalid database name."):
        db.get_array_element_db_name(
            "ILLN-01", model_version=model_version_prod5, collection="telescopes"
        )


def test_parameter_cache_key(db, model_version_prod5):

    assert db._parameter_cache_key("North", "LSTN-01", model_version_prod5) == "North-LSTN-01-5.0.0"
    assert db._parameter_cache_key("North", None, model_version_prod5) == "North-5.0.0"
    assert db._parameter_cache_key(None, None, model_version_prod5) == "5.0.0"


def test_model_version(db):

    assert db.model_version(version="6.0.0") == "6.0.0"

    with pytest.raises(ValueError, match=r"Invalid model version test"):
        db.model_version(version="test")
    with pytest.raises(ValueError, match=r"Invalid model version 0.0.9876"):
        db.model_version(version="0.0.9876")


def test_get_collections(db, db_config):

    collections = db.get_collections()
    assert isinstance(collections, list)
    assert "telescopes" in collections

    collections_from_name = db.get_collections(db_config["db_simulation_model"])
    assert isinstance(collections_from_name, list)
    assert "telescopes" in collections_from_name
    assert "fs.files" in collections_from_name

    collections_no_model = db.get_collections(db_config["db_simulation_model"], True)
    assert isinstance(collections_no_model, list)
    assert "telescopes" in collections_no_model
    assert "fs.files" not in collections_no_model
    assert "metadata" not in collections_no_model


def test_model_version_empty(db, mocker):

    mocker.patch.object(db, "get_all_versions", return_value=[])
    assert db.model_version("6.0.0") is None
