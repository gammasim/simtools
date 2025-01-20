#!/usr/bin/python3

import copy
import logging
import re
import uuid
from unittest.mock import call

import pytest
from bson.objectid import ObjectId

from simtools.db import db_handler
from simtools.utils import names

logger = logging.getLogger()


@pytest.fixture(autouse=True)
def reset_db_client():
    """Reset db_client before each test."""
    # If using the class-level db_client:
    db_handler.DatabaseHandler.db_client = None
    db_handler.production_table_cached = {}
    db_handler.model_parameters_cached = {}
    yield  # allows the test to run
    # After the test, reset any side-effects (if necessary):
    db_handler.DatabaseHandler.db_client = None
    db_handler.production_table_cached.clear()
    db_handler.model_parameters_cached.clear()


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
def fs_files():
    return "fs.files"


# TODO remove?
@pytest.fixture
def _db_cleanup_file_sandbox(db_no_config_file, random_id, fs_files):
    yield
    # Cleanup
    logger.info("Dropping the temporary files in the sandbox")
    db_no_config_file.db_client[f"sandbox_{random_id}"]["fs.chunks"].drop()
    db_no_config_file.db_client[f"sandbox_{random_id}"][fs_files].drop()


def test_set_up_connection_no_config():
    """Test _set_up_connection with no configuration."""
    db = db_handler.DatabaseHandler(mongo_db_config=None)
    db._set_up_connection()
    assert db_handler.DatabaseHandler.db_client is None


def test_set_up_connection_with_config(db):
    """Test _set_up_connection with valid configuration."""
    db._set_up_connection()
    assert isinstance(db_handler.DatabaseHandler.db_client, db_handler.MongoClient)


def test_valid_db_config(db, db_config):
    assert db.mongo_db_config == db._validate_mongo_db_config(db_config)
    assert db._validate_mongo_db_config(None) is None
    none_db_dict = copy.deepcopy(db_config)
    for key in none_db_dict.keys():
        none_db_dict[key] = None
    assert db._validate_mongo_db_config(none_db_dict) is None
    assert db._validate_mongo_db_config({}) is None
    with pytest.raises(ValueError, match=r"Invalid MongoDB configuration"):
        db._validate_mongo_db_config({"wrong_config": "wrong"})


def test_open_mongo_db_direct_connection(mocker, db, db_config):
    """Test _open_mongo_db with direct connection configuration."""
    db_config["db_server"] = "localhost"
    mock_mongo_client = mocker.patch(
        "simtools.db.db_handler.MongoClient", return_value="mock_client"
    )
    db.mongo_db_config = db_config
    client = db._open_mongo_db()
    assert client == "mock_client"
    mock_mongo_client.assert_called_once_with(
        db_config["db_server"],
        port=db_config["db_api_port"],
        username=db_config["db_api_user"],
        password=db_config["db_api_pw"],
        authSource=db_config.get("db_api_authentication_database", "admin"),
        directConnection=True,
        ssl=False,
        tlsallowinvalidhostnames=True,
        tlsallowinvalidcertificates=True,
    )


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


def test_get_model_parameter(db, mocker):
    """Test get_model_parameter method."""
    mock_read_mongo_db = mocker.patch.object(
        db, "_read_mongo_db", return_value={"parameter": "value"}
    )

    parameter = "camera_pixels"
    parameter_version = "0.0.1"
    site = "North"
    array_element_name = "LSTN-01"
    collection = "telescopes"

    result = db.get_model_parameter(
        parameter, parameter_version, site, array_element_name, collection
    )

    query = {
        "parameter_version": parameter_version,
        "parameter": parameter,
        "instrument": array_element_name,
        "site": site,
    }

    mock_read_mongo_db.assert_called_once_with(query=query, collection_name=collection)
    assert result == {"parameter": "value"}


def test_get_model_parameter_no_site(db, mocker):
    """Test get_model_parameter method without site."""
    mock_read_mongo_db = mocker.patch.object(
        db, "_read_mongo_db", return_value={"parameter": "value"}
    )

    parameter = "mirror_list"
    parameter_version = "1.0.0"
    site = None
    array_element_name = "LSTS-01"
    collection = "telescopes"

    result = db.get_model_parameter(
        parameter, parameter_version, site, array_element_name, collection
    )

    query = {
        "parameter_version": parameter_version,
        "parameter": parameter,
        "instrument": array_element_name,
    }

    mock_read_mongo_db.assert_called_once_with(query=query, collection_name=collection)
    assert result == {"parameter": "value"}


def test_get_model_parameter_no_array_element_name(db, mocker):
    """Test get_model_parameter method without array element name."""
    mock_read_mongo_db = mocker.patch.object(
        db, "_read_mongo_db", return_value={"parameter": "value"}
    )

    parameter = "corsika_iact_io_buffer"
    parameter_version = "10.11.12"
    site = "South"
    array_element_name = None
    collection = "configuration_corsika"

    result = db.get_model_parameter(
        parameter, parameter_version, site, array_element_name, collection
    )

    query = {
        "parameter_version": parameter_version,
        "parameter": parameter,
        "site": site,
    }

    mock_read_mongo_db.assert_called_once_with(query=query, collection_name=collection)
    assert result == {"parameter": "value"}


def test_get_model_parameters(db, mocker):
    """Test get_model_parameters method."""
    mock_get_production_table = mocker.patch.object(
        db,
        "get_production_table_from_mongo_db",
        return_value={"parameters": {"LSTN-01": {"param1": "v1"}}},
    )
    mock_get_array_element_list = mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-design", "LSTN-01"]
    )
    mock_read_cache = mocker.patch.object(db, "_read_cache", return_value=("cache_key", None))
    mock_read_mongo_db = mocker.patch.object(
        db, "_read_mongo_db", return_value={"param1": {"value": "value1"}}
    )

    site = "North"
    array_element_name = "LSTN-01"
    model_version = "1.0.0"
    collection = "telescopes"

    result = db.get_model_parameters(site, array_element_name, model_version, collection)

    mock_get_production_table.assert_called_once_with(collection, model_version)
    mock_get_array_element_list.assert_called_once_with(
        array_element_name, site, {"parameters": {"LSTN-01": {"param1": "v1"}}}, collection
    )
    mock_read_cache.assert_has_calls(
        [
            call(
                db_handler.DatabaseHandler.model_parameters_cached,
                names.validate_site_name(site),
                "LSTN-design",
                model_version,
                collection,
            ),
            call(
                db_handler.DatabaseHandler.model_parameters_cached,
                names.validate_site_name(site),
                "LSTN-01",
                model_version,
                collection,
            ),
        ]
    )
    mock_read_mongo_db.assert_called_once_with(
        query={
            "$or": [{"parameter": "param1", "parameter_version": "v1"}],
            "instrument": "LSTN-01",
            "site": site,
        },
        collection_name=collection,
    )
    assert result == {"param1": {"value": "value1"}}


def test_get_model_parameters_with_cache(db, mocker):
    """Test get_model_parameters method with cache."""
    mock_get_production_table = mocker.patch.object(
        db,
        "get_production_table_from_mongo_db",
        return_value={"parameters": {"LSTN-01": {"param1": "v1"}}},
    )
    mock_get_array_element_list = mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-01"]
    )
    mock_read_cache = mocker.patch.object(
        db, "_read_cache", return_value=("cache_key", {"param1": {"value": "cached_value"}})
    )

    site = "North"
    array_element_name = "LSTN-01"
    model_version = "1.0.0"
    collection = "telescopes"

    result = db.get_model_parameters(site, array_element_name, model_version, collection)

    mock_get_production_table.assert_called_once_with(collection, model_version)
    mock_get_array_element_list.assert_called_once_with(
        array_element_name, site, {"parameters": {"LSTN-01": {"param1": "v1"}}}, collection
    )
    mock_read_cache.assert_called_once_with(
        db_handler.DatabaseHandler.model_parameters_cached,
        names.validate_site_name(site),
        "LSTN-01",
        model_version,
        collection,
    )
    assert result == {"param1": {"value": "cached_value"}}


def test_get_model_parameters_no_parameters(db, mocker):
    """Test get_model_parameters method with no parameters."""
    mock_get_production_table = mocker.patch.object(
        db, "get_production_table_from_mongo_db", return_value={"parameters": {}}
    )
    mock_get_array_element_list = mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-01"]
    )
    mock_read_cache = mocker.patch.object(db, "_read_cache", return_value=("cache_key", None))

    site = "North"
    array_element_name = "LSTN-01"
    model_version = "1.0.0"
    collection = "telescopes"

    result = db.get_model_parameters(site, array_element_name, model_version, collection)

    mock_get_production_table.assert_called_once_with(collection, model_version)
    mock_get_array_element_list.assert_called_once_with(
        array_element_name, site, {"parameters": {}}, collection
    )
    mock_read_cache.assert_called_once_with(
        db_handler.DatabaseHandler.model_parameters_cached,
        names.validate_site_name(site),
        "LSTN-01",
        model_version,
        collection,
    )
    assert result == {}


def test_get_collection(db, mocker):
    """Test get_collection method."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": {"test_collection": "mock_collection"}}
    )

    db_name = "test_db"
    collection_name = "test_collection"

    result = db.get_collection(db_name, collection_name)

    mock_get_db_name.assert_called_once_with(db_name)
    assert result == "mock_collection"


def test_get_collections(db, db_config, fs_files):
    collections = db.get_collections()
    assert isinstance(collections, list)
    assert "telescopes" in collections

    collections_from_name = db.get_collections(db_config["db_simulation_model"])
    assert isinstance(collections_from_name, list)
    assert "telescopes" in collections_from_name
    assert fs_files in collections_from_name

    collections_no_model = db.get_collections(db_config["db_simulation_model"], True)
    assert isinstance(collections_no_model, list)
    assert "telescopes" in collections_no_model
    assert fs_files not in collections_no_model
    assert "metadata" not in collections_no_model


def test_export_model_files_with_file_names(db, mocker):
    """Test export_model_files method with file names."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", return_value=mocker.Mock(_id="file_id")
    )
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")

    file_names = ["file1.dat", "file2.dat"]
    dest = "/tmp"

    result = db.export_model_files(file_names=file_names, dest=dest)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_has_calls(
        [call("test_db", "file1.dat"), call("test_db", "file2.dat")]
    )
    mock_write_file_from_mongo_to_disk.assert_has_calls(
        [
            call("test_db", dest, mock_get_file_mongo_db.return_value),
            call("test_db", dest, mock_get_file_mongo_db.return_value),
        ]
    )
    assert result == {"file1.dat": "file_id", "file2.dat": "file_id"}


def test_export_model_files_with_parameters(db, mocker):
    """Test export_model_files method with parameters."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", return_value=mocker.Mock(_id="file_id")
    )
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")

    parameters = {
        "param1": {"file": True, "value": "file1.dat"},
        "param2": {"file": True, "value": "file2.dat"},
    }
    dest = "/tmp"

    result = db.export_model_files(parameters=parameters, dest=dest)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_has_calls(
        [call("test_db", "file1.dat"), call("test_db", "file2.dat")]
    )
    mock_write_file_from_mongo_to_disk.assert_has_calls(
        [
            call("test_db", dest, mock_get_file_mongo_db.return_value),
            call("test_db", dest, mock_get_file_mongo_db.return_value),
        ]
    )
    assert result == {"file1.dat": "file_id", "file2.dat": "file_id"}


def test_export_model_files_file_exists(db, mocker):
    """Test export_model_files method when file already exists."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_file_mongo_db = mocker.patch.object(db, "_get_file_mongo_db")
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")
    mock_path_exists = mocker.patch("pathlib.Path.exists", return_value=True)

    file_names = ["file1.dat"]
    dest = "/tmp"

    result = db.export_model_files(file_names=file_names, dest=dest)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_not_called()
    mock_write_file_from_mongo_to_disk.assert_not_called()
    mock_path_exists.assert_called_once()
    assert result == {"file1.dat": "file exits"}


def test_export_model_files_file_not_found(db, mocker):
    """Test export_model_files method when file is not found in parameters."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", side_effect=FileNotFoundError
    )
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")

    parameters = {
        "param1": {"file": True, "value": "file1.dat"},
    }
    dest = "/tmp"

    with pytest.raises(FileNotFoundError):
        db.export_model_files(parameters=parameters, dest=dest)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_called_once_with("test_db", "file1.dat")
    mock_write_file_from_mongo_to_disk.assert_not_called()


def test_get_query_from_parameter_version_table(db):
    """Test _get_query_from_parameter_version_table method."""
    or_list = [
        {"parameter": "param1", "parameter_version": "v1"},
        {"parameter": "param2", "parameter_version": "v2"},
    ]
    parameter_version_table = {
        "param1": "v1",
        "param2": "v2",
    }
    array_element_name = "LSTN-01"
    site = "North"

    result = db._get_query_from_parameter_version_table(
        parameter_version_table, array_element_name, site
    )

    expected_query = {
        "$or": or_list,
        "instrument": array_element_name,
        "site": site,
    }

    assert result == expected_query

    # _get_query_from_parameter_version_table method without site.
    array_element_name = "LSTN-01"
    site = None

    result = db._get_query_from_parameter_version_table(
        parameter_version_table, array_element_name, site
    )

    expected_query = {
        "$or": or_list,
        "instrument": array_element_name,
    }

    assert result == expected_query

    #  _get_query_from_parameter_version_table method without array element name.
    array_element_name = None
    site = "North"

    result = db._get_query_from_parameter_version_table(
        parameter_version_table, array_element_name, site
    )

    expected_query = {
        "$or": or_list,
        "site": site,
    }

    assert result == expected_query

    # _get_query_from_parameter_version_table method without site and array element name.
    array_element_name = None
    site = None

    result = db._get_query_from_parameter_version_table(
        parameter_version_table, array_element_name, site
    )

    expected_query = {
        "$or": or_list,
    }

    assert result == expected_query

    # _get_query_from_parameter_version_table method with array element name 'xSTx-design'.
    array_element_name = "xSTx-design"
    site = "North"

    result = db._get_query_from_parameter_version_table(
        parameter_version_table, array_element_name, site
    )

    expected_query = {
        "$or": or_list,
        "site": site,
    }

    assert result == expected_query


def test_read_mongo_db(db, mocker):
    """Test read_mongo_db method."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mock_find = mocker.patch.object(
        db.get_collection.return_value,
        "find",
        return_value=[
            {"_id": ObjectId(), "parameter": "param1", "value": "value1"},
            {"_id": ObjectId(), "parameter": "param2", "value": "value2"},
        ],
    )

    query = {"parameter_version": "1.0.0"}
    collection_name = "test_collection"

    result = db._read_mongo_db(query, collection_name)

    mock_get_db_name.assert_called_once_with()
    mock_get_collection.assert_called_once_with("test_db", collection_name)
    mock_find.assert_called_once_with(query)
    assert result == {
        "param1": {
            "_id": mock_find.return_value[0]["_id"],
            "value": "value1",
            "entry_date": mock_find.return_value[0]["_id"].generation_time,
        },
        "param2": {
            "_id": mock_find.return_value[1]["_id"],
            "value": "value2",
            "entry_date": mock_find.return_value[1]["_id"].generation_time,
        },
    }

    mock_find = mocker.patch.object(db.get_collection.return_value, "find", return_value=[])
    with pytest.raises(
        ValueError,
        match=r"The following query for test_collection returned zero results: {'parameter_version': '1.0.0'}",
    ):
        db._read_mongo_db(query, collection_name)


def test_get_production_table_from_mongo_db_with_cache(db, mocker):
    """Test get_production_table_from_mongo_db method with cache."""
    collection_name = "telescopes"
    model_version = "1.0.0"
    param = {"param1": "value1"}
    mock_cache_key = mocker.patch.object(db, "_cache_key", return_value="cache_key")
    db_handler.DatabaseHandler.production_table_cached["cache_key"] = {
        "collection": model_version,
        "model_version": model_version,
        "parameters": param,
        "design_model": {},
        "entry_date": ObjectId().generation_time,
    }

    result = db.get_production_table_from_mongo_db(collection_name, model_version)

    mock_cache_key.assert_called_once_with(None, None, model_version, collection_name)
    assert result == db_handler.DatabaseHandler.production_table_cached["cache_key"]

    mock_cache_key = mocker.patch.object(db, "_cache_key", return_value="no_cache_key")
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_find_one = mocker.patch.object(
        db.get_collection.return_value,
        "find_one",
        return_value={
            "_id": ObjectId(),
            "collection": collection_name,
            "model_version": model_version,
            "parameters": param,
            "design_model": {},
        },
    )

    result = db.get_production_table_from_mongo_db(collection_name, model_version)

    mock_cache_key.assert_called_once_with(None, None, model_version, collection_name)
    mock_get_collection.assert_called_once_with("test_db", "production_tables")
    mock_find_one.assert_called_once_with(
        {"model_version": model_version, "collection": collection_name}
    )
    assert result == {
        "collection": collection_name,
        "model_version": model_version,
        "parameters": param,
        "design_model": {},
        "entry_date": mock_find_one.return_value["_id"].generation_time,
    }

    # test with no results
    mock_find_one = mocker.patch.object(
        db.get_collection.return_value, "find_one", return_value=None
    )
    with pytest.raises(
        ValueError,
        match=r"The following query returned zero results: {'model_version': '1.0.0', 'collection': 'telescopes'}",
    ):
        db.get_production_table_from_mongo_db(collection_name, model_version)


@pytest.mark.usefixtures("_db_cleanup")
def ff_test_adding_new_parameter_db(db, random_id, io_handler, model_version):
    logger.info("----Testing adding a new parameter-----")
    test_model_version = "0.0.9876"
    tmp_par_dict = {
        "parameter": None,
        "instrument": "LSTN-test",
        "site": "North",
        "version": test_model_version,
        "value": None,
        "unit": None,
        "type": None,
        "applicable": True,
        "file": False,
    }

    par_dict_int = copy.deepcopy(tmp_par_dict)
    par_dict_int["parameter"] = "num_gains"
    par_dict_int["value"] = 3
    par_dict_int["type"] = "int64"
    with pytest.raises(
        ValueError,
        match=re.escape("Value for column '0' out of range. ([3, 3], allowed_range: [1, 2])"),
    ):
        db.add_new_parameter(
            db_name=f"sandbox_{random_id}",
            par_dict=par_dict_int,
            collection_name="telescopes",
        )
    par_dict_int["value"] = 2
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        par_dict=par_dict_int,
        collection_name="telescopes",
    )

    par_dict_list = copy.deepcopy(tmp_par_dict)
    par_dict_list["parameter"] = "telescope_transmission"
    par_dict_list["value"] = [0.969, 0.01, 0.0, 0.0, 0.0, 0.0]
    par_dict_list["type"] = "float64"
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        par_dict=par_dict_list,
        collection_name="telescopes",
    )
    par_dict_quantity = copy.deepcopy(tmp_par_dict)
    par_dict_quantity["parameter"] = "focal_length"
    par_dict_quantity["value"] = 12.5  # test that value is converted to cm
    par_dict_quantity["type"] = "float64"
    par_dict_quantity["unit"] = "m"
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        par_dict=par_dict_quantity,
        collection_name="telescopes",
    )

    par_dict_file = copy.deepcopy(tmp_par_dict)
    par_dict_file["parameter"] = "mirror_list"
    par_dict_file["value"] = "mirror_list_CTA-N-LST1_v2019-03-31_rotated_simtel.dat"
    par_dict_file["type"] = "file"
    par_dict_file["file"] = True
    with pytest.raises(FileNotFoundError, match=r"^The location of the file to upload"):
        db.add_new_parameter(
            db_name=f"sandbox_{random_id}",
            par_dict=par_dict_file,
            collection_name="telescopes",
        )
    db.add_new_parameter(
        db_name=f"sandbox_{random_id}",
        par_dict=par_dict_file,
        collection_name="telescopes",
        file_prefix="tests/resources",
    )

    pars = db._read_mongo_db(
        db_name=f"sandbox_{random_id}",
        array_element_name="LSTN-test",
        model_version=test_model_version,
        run_location=io_handler.get_output_directory(sub_dir="model"),
        collection_name="telescopes",
        write_files=False,
    )
    assert pars["num_gains"]["value"] == 2
    assert pars["num_gains"]["type"] == "int64"
    assert isinstance(pars["telescope_transmission"]["value"], list)
    assert pars["focal_length"]["value"] == pytest.approx(1250.0)
    assert pars["focal_length"]["unit"] == "cm"
    assert pars["mirror_list"]["file"] is True

    # make sure that cache has been emptied after updating
    assert db._cache_key("North", "LSTN-test", test_model_version) not in db.model_parameters_cached


@pytest.mark.usefixtures("_db_cleanup_file_sandbox")
def ff_test_insert_files_db(db, io_handler, random_id, caplog):
    logger.info("----Testing inserting files to the DB-----")
    logger.info(
        "Creating a temporary file in " f"{io_handler.get_output_directory(sub_dir='model')}"
    )
    file_name = io_handler.get_output_directory(sub_dir="model") / f"test_file_{random_id}.dat"
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


def test_cache_key(db):
    """Test _cache_key method."""
    # Test with all parameters
    result = db._cache_key(
        site="North", array_element_name="LSTN-01", model_version="1.0.0", collection="telescopes"
    )
    assert result == "1.0.0-telescopes-North-LSTN-01"

    # Test with missing site
    result = db._cache_key(
        site=None, array_element_name="LSTN-01", model_version="1.0.0", collection="telescopes"
    )
    assert result == "1.0.0-telescopes-LSTN-01"

    # Test with missing array_element_name
    result = db._cache_key(
        site="North", array_element_name=None, model_version="1.0.0", collection="telescopes"
    )
    assert result == "1.0.0-telescopes-North"

    # Test with missing model_version
    result = db._cache_key(
        site="North", array_element_name="LSTN-01", model_version=None, collection="telescopes"
    )
    assert result == "telescopes-North-LSTN-01"

    # Test with missing collection
    result = db._cache_key(
        site="North", array_element_name="LSTN-01", model_version="1.0.0", collection=None
    )
    assert result == "1.0.0-North-LSTN-01"

    # Test with only model_version
    result = db._cache_key(
        site=None, array_element_name=None, model_version="1.0.0", collection=None
    )
    assert result == "1.0.0"

    # Test with only collection
    result = db._cache_key(
        site=None, array_element_name=None, model_version=None, collection="telescopes"
    )
    assert result == "telescopes"

    # Test with only site
    result = db._cache_key(
        site="North", array_element_name=None, model_version=None, collection=None
    )
    assert result == "North"

    # Test with only array_element_name
    result = db._cache_key(
        site=None, array_element_name="LSTN-01", model_version=None, collection=None
    )
    assert result == "LSTN-01"

    # Test with no parameters
    result = db._cache_key(site=None, array_element_name=None, model_version=None, collection=None)
    assert result == ""
