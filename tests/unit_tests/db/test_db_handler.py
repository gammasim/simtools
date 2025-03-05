#!/usr/bin/python3

import copy
import logging
from pathlib import Path
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
    yield  # allows the test to run
    # After the test, reset any side-effects (if necessary):
    db_handler.DatabaseHandler.db_client = None


@pytest.fixture
def db_no_config_file():
    """Database object (without configuration)."""
    return db_handler.DatabaseHandler(mongo_db_config=None)


@pytest.fixture
def test_db():
    return "test_db"


@pytest.fixture
def test_file():
    return "test_file.dat"


@pytest.fixture
def test_file_2():
    return "test_file_2.dat"


@pytest.fixture
def fs_files():
    return "fs.files"


@pytest.fixture
def value_unit_type():
    return "simtools.db.db_handler.value_conversion.get_value_unit_type"


@pytest.fixture
def mock_open(mocker):
    return mocker.patch("builtins.open", mocker.mock_open(read_data=b"file_content"))


@pytest.fixture
def validate_model_parameter():
    return "simtools.db.db_handler.validate_data.DataValidator.validate_model_parameter"


@pytest.fixture
def mock_gridfs(mocker):
    return mocker.patch("simtools.db.db_handler.gridfs.GridFS")


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
        "CTAO-Simulation-ModelParameters-v0-3-0",
        "CTAO-Simulation-ModelParameters-v0-2-0",
        "CTAO-Simulation-ModelParameters-v0-1-19",
        "CTAO-Simulation-ModelParameters-v0-3-9",
        "CTAO-Simulation-ModelParameters-v0-3-19",
        "CTAO-Simulation-ModelParameters-v0-3-0",
        "CTAO-Simulation-ModelParameters-v0-3-0-alpha-2",
        "CTAO-Simulation-ModelParameters-v0-4-19-alpha-1",
        "CTAO-Simulation-ModelParameters-v0-4-19-dev1",
    ]
    mocker.patch.object(db_copy.db_client, "list_database_names", return_value=db_names)
    db_copy.mongo_db_config["db_simulation_model"] = "CTAO-Simulation-ModelParameters-LATEST"
    db_copy._find_latest_simulation_model_db()
    assert (
        db_copy.mongo_db_config["db_simulation_model"] == "CTAO-Simulation-ModelParameters-v0-3-19"
    )


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

    result = db.get_model_parameter(parameter, parameter_version, site, array_element_name)

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

    result = db.get_model_parameter(parameter, parameter_version, site, array_element_name)

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

    result = db.get_model_parameter(parameter, parameter_version, site, array_element_name)

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
        "_read_production_table_from_mongo_db",
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

    result = db.get_model_parameters(site, array_element_name, collection, model_version)

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
        "_read_production_table_from_mongo_db",
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

    result = db.get_model_parameters(site, array_element_name, collection, model_version)

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
        db, "_read_production_table_from_mongo_db", return_value={"parameters": {}}
    )
    mock_get_array_element_list = mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-01"]
    )
    mock_read_cache = mocker.patch.object(db, "_read_cache", return_value=("cache_key", None))

    site = "North"
    array_element_name = "LSTN-01"
    model_version = "1.0.0"
    collection = "telescopes"

    result = db.get_model_parameters(site, array_element_name, collection, model_version)

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


def test_get_collection(db, mocker, test_db):
    """Test get_collection method."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value=test_db)
    mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {test_db: {"test_collection": "mock_collection"}}
    )
    collection_name = "test_collection"

    result = db.get_collection(test_db, collection_name)

    mock_get_db_name.assert_called_once_with(test_db)
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


def test_export_model_files_with_file_names(
    db, mocker, tmp_test_directory, test_db, test_file, test_file_2
):
    """Test export_model_files method with file names."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value=test_db)
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", return_value=mocker.Mock(_id="file_id")
    )
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")

    file_names = [test_file, test_file_2]

    result = db.export_model_files(file_names=file_names, dest=tmp_test_directory)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_has_calls([call(test_db, test_file), call(test_db, test_file_2)])
    mock_write_file_from_mongo_to_disk.assert_has_calls(
        [
            call(test_db, tmp_test_directory, mock_get_file_mongo_db.return_value),
            call(test_db, tmp_test_directory, mock_get_file_mongo_db.return_value),
        ]
    )
    assert result == {test_file: "file_id", test_file_2: "file_id"}


def test_export_model_files_with_parameters(
    db, mocker, tmp_test_directory, test_db, test_file, test_file_2
):
    """Test export_model_files method with parameters."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value=test_db)
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", return_value=mocker.Mock(_id="file_id")
    )
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")

    parameters = {
        "param1": {"file": True, "value": test_file},
        "param2": {"file": True, "value": test_file_2},
    }

    result = db.export_model_files(parameters=parameters, dest=tmp_test_directory)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_has_calls([call(test_db, test_file), call(test_db, test_file_2)])
    mock_write_file_from_mongo_to_disk.assert_has_calls(
        [
            call(test_db, tmp_test_directory, mock_get_file_mongo_db.return_value),
            call(test_db, tmp_test_directory, mock_get_file_mongo_db.return_value),
        ]
    )
    assert result == {test_file: "file_id", test_file_2: "file_id"}


def test_export_model_files_file_exists(db, mocker, tmp_test_directory, test_db, test_file):
    """Test export_model_files method when file already exists."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value=test_db)
    mock_get_file_mongo_db = mocker.patch.object(db, "_get_file_mongo_db")
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")
    mock_path_exists = mocker.patch("pathlib.Path.exists", return_value=True)

    file_names = [test_file]

    result = db.export_model_files(file_names=file_names, dest=tmp_test_directory)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_not_called()
    mock_write_file_from_mongo_to_disk.assert_not_called()
    mock_path_exists.assert_called_once()
    assert result == {test_file: "file exists"}


def test_export_model_files_file_not_found(db, mocker, tmp_test_directory, test_db, test_file):
    """Test export_model_files method when file is not found in parameters."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value=test_db)
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", side_effect=FileNotFoundError
    )
    mock_write_file_from_mongo_to_disk = mocker.patch.object(db, "_write_file_from_mongo_to_disk")

    parameters = {
        "param1": {"file": True, "value": test_file},
    }

    with pytest.raises(FileNotFoundError):
        db.export_model_files(parameters=parameters, dest=tmp_test_directory)

    mock_get_db_name.assert_called()
    mock_get_file_mongo_db.assert_called_once_with(test_db, test_file)
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


def test_read_mongo_db(db, mocker, test_db):
    """Test read_mongo_db method."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value=test_db)
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
    mock_get_collection.assert_called_once_with(test_db, collection_name)
    mock_find.assert_called_once_with(query)
    assert result == {
        "param1": {
            "_id": mock_find.return_value[0]["_id"],
            "parameter": "param1",
            "value": "value1",
            "entry_date": mock_find.return_value[0]["_id"].generation_time,
        },
        "param2": {
            "_id": mock_find.return_value[1]["_id"],
            "parameter": "param2",
            "value": "value2",
            "entry_date": mock_find.return_value[1]["_id"].generation_time,
        },
    }

    mocker.patch.object(db.get_collection.return_value, "find", return_value=[])
    with pytest.raises(
        ValueError,
        match=r"The following query for test_collection returned zero results: "
        r"{'parameter_version': '1.0.0'}",
    ):
        db._read_mongo_db(query, collection_name)


def test__read_production_table_from_mongo_db_with_cache(db, mocker, test_db):
    """Test _read_production_table_from_mongo_db method with cache."""
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

    result = db._read_production_table_from_mongo_db(collection_name, model_version)

    mock_cache_key.assert_called_once_with(None, None, model_version, collection_name)
    assert result == db_handler.DatabaseHandler.production_table_cached["cache_key"]

    mock_cache_key = mocker.patch.object(db, "_cache_key", return_value="no_cache_key")
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mocker.patch.object(db, "_get_db_name", return_value=test_db)
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

    result = db._read_production_table_from_mongo_db(collection_name, model_version)

    mock_cache_key.assert_called_once_with(None, None, model_version, collection_name)
    mock_get_collection.assert_called_once_with(test_db, "production_tables")
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
    mocker.patch.object(db.get_collection.return_value, "find_one", return_value=None)
    with pytest.raises(
        ValueError,
        match=r"The following query returned zero results: "
        r"{'model_version': '1.0.0', 'collection': 'telescopes'}",
    ):
        db._read_production_table_from_mongo_db(collection_name, model_version)


def test_get_array_elements_of_type(db, mocker):
    """Test get_array_elements_of_type method."""
    mock_get_production_table = mocker.patch.object(
        db,
        "_read_production_table_from_mongo_db",
        return_value={
            "parameters": {"LSTN-01": "value1", "LSTN-02": "value2", "MSTS-01": "value3"}
        },
    )

    array_element_type = "LSTN"
    model_version = "1.0.0"
    collection = "telescopes"

    result = db.get_array_elements_of_type(array_element_type, model_version, collection)

    mock_get_production_table.assert_called_once_with(collection, model_version)
    assert result == ["LSTN-01", "LSTN-02"]

    # Test with no matching array elements
    mock_get_production_table.return_value = {"parameters": {"MSTS-01": "value3"}}
    result = db.get_array_elements_of_type(array_element_type, model_version, collection)
    assert result == []

    # Test with array elements containing '-design'
    mock_get_production_table.return_value = {
        "parameters": {"LSTN-01": "value1", "LSTN-design": "value2"}
    }
    result = db.get_array_elements_of_type(array_element_type, model_version, collection)
    assert result == ["LSTN-01"]

    # Test with different array element type
    array_element_type = "MSTS"
    mock_get_production_table.return_value = {
        "parameters": {"LSTN-01": "value1", "MSTS-01": "value3", "MSTS-02": "value4"}
    }
    result = db.get_array_elements_of_type(array_element_type, model_version, collection)
    assert result == ["MSTS-01", "MSTS-02"]


def test_get_simulation_configuration_parameters(db, mocker):
    return_value = {"parameter": "value"}
    mock_get_model_parameters = mocker.patch.object(
        db, "get_model_parameters", return_value=return_value
    )

    assert (
        db.get_simulation_configuration_parameters("corsika", "North", "LSTN-design", "6.0.0")
        == return_value
    )
    assert mock_get_model_parameters.call_count == 1

    assert (
        db.get_simulation_configuration_parameters("simtel", "North", "LSTN-design", "6.0.0")
        == return_value
    )
    assert mock_get_model_parameters.call_count == 2
    assert db.get_simulation_configuration_parameters("simtel", "North", None, "6.0.0") == {}
    assert mock_get_model_parameters.call_count == 2
    assert db.get_simulation_configuration_parameters("simtel", None, "LSTN-design", "6.0.0") == {}
    assert mock_get_model_parameters.call_count == 2
    assert db.get_simulation_configuration_parameters("simtel", None, None, "6.0.0") == {}
    assert mock_get_model_parameters.call_count == 2

    with pytest.raises(ValueError, match=r"Unknown simulation software: wrong"):
        db.get_simulation_configuration_parameters("wrong", "North", "LSTN-design", "6.0.0")


def test_get_file_mongo_db_file(db, mocker, test_db, test_file, mock_gridfs):
    """Test _get_file_mongo_db method when file exists."""
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {test_db: mocker.Mock()}
    )
    mock_file_system = mock_gridfs.return_value
    mock_file_system.exists.return_value = True
    mock_file_instance = mocker.Mock()
    mock_file_system.find_one.return_value = mock_file_instance

    result = db._get_file_mongo_db(test_db, test_file)

    mock_gridfs.assert_called_once_with(mock_db_client[test_db])
    mock_file_system.exists.assert_called_once_with({"filename": test_file})
    mock_file_system.find_one.assert_called_once_with({"filename": test_file})
    assert result == mock_file_instance

    mock_file_system.exists.return_value = False
    with pytest.raises(
        FileNotFoundError, match=f"The file {test_file} does not exist in the database {test_db}"
    ):
        db._get_file_mongo_db(test_db, test_file)


def test_write_file_from_mongo_to_disk(
    db, mocker, tmp_test_directory, mock_open, test_db, test_file
):
    """Test _write_file_from_mongo_to_disk method."""
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": mocker.Mock()}
    )
    mock_gridfs_bucket = mocker.patch("simtools.db.db_handler.gridfs.GridFSBucket")
    mock_fs_output = mock_gridfs_bucket.return_value

    file = mocker.Mock()
    file.filename = test_file

    db._write_file_from_mongo_to_disk(test_db, tmp_test_directory, file)

    mock_gridfs_bucket.assert_called_once_with(mock_db_client[test_db])
    mock_open.assert_called_once_with(Path(tmp_test_directory).joinpath(file.filename), "wb")
    mock_fs_output.download_to_stream_by_name.assert_called_once_with(file.filename, mock_open())


def test_add_production_table(db, mocker, test_db):
    """Test add_production_table method."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mock_insert_one = mocker.patch.object(db.get_collection.return_value, "insert_one")

    production_table = {
        "collection": "telescopes",
        "model_version": "1.0.0",
        "parameters": {"param1": "value1"},
    }

    db.add_production_table(test_db, production_table)

    mock_get_db_name.assert_called_once_with(test_db)
    mock_get_collection.assert_called_once_with("test_db", "production_tables")
    mock_insert_one.assert_called_once_with(production_table)


def test_add_new_parameter(db, mocker, value_unit_type, validate_model_parameter, test_db):
    """Test add_new_parameter method."""
    mock_validate_model_parameter = mocker.patch(
        validate_model_parameter,
        return_value={"parameter": "param1", "value": "value1", "file": False},
    )
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mock_insert_one = mocker.patch.object(db.get_collection.return_value, "insert_one")
    mock_get_value_unit_type = mocker.patch(
        value_unit_type,
        return_value=("value1", "unit1", None),
    )
    mock_reset_parameter_cache = mocker.patch.object(db, "_reset_parameter_cache")

    par_dict = {"parameter": "param1", "value": "value1", "file": False}
    collection_name = "telescopes"
    file_prefix = None

    db.add_new_parameter(test_db, par_dict, collection_name, file_prefix)

    mock_validate_model_parameter.assert_called_once_with(par_dict)
    mock_get_db_name.assert_called_once_with(test_db)
    mock_get_collection.assert_called_once_with("test_db", collection_name)
    mock_get_value_unit_type.assert_called_once_with(value="value1", unit_str=None)
    mock_insert_one.assert_called_once_with(
        {"parameter": "param1", "value": "value1", "file": False, "unit": "unit1"}
    )
    mock_reset_parameter_cache.assert_called_once()


def test_add_new_parameter_with_file(
    db, mocker, tmp_test_directory, value_unit_type, validate_model_parameter, test_db
):
    """Test add_new_parameter method with file."""
    mock_validate_model_parameter = mocker.patch(
        validate_model_parameter,
        return_value={"parameter": "param1", "value": "value1", "file": True},
    )
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mock_insert_one = mocker.patch.object(db.get_collection.return_value, "insert_one")
    mock_get_value_unit_type = mocker.patch(
        value_unit_type,
        return_value=("value1", "unit1", None),
    )
    mock_insert_file_to_db = mocker.patch.object(db, "insert_file_to_db")
    mock_reset_parameter_cache = mocker.patch.object(db, "_reset_parameter_cache")

    par_dict = {"parameter": "param1", "value": "value1", "file": True}
    collection_name = "telescopes"

    db.add_new_parameter(test_db, par_dict, collection_name, tmp_test_directory)

    mock_validate_model_parameter.assert_called_once_with(par_dict)
    mock_get_db_name.assert_called_once_with(test_db)
    mock_get_collection.assert_called_once_with("test_db", collection_name)
    mock_get_value_unit_type.assert_called_once_with(value="value1", unit_str=None)
    mock_insert_one.assert_called_once_with(
        {"parameter": "param1", "value": "value1", "file": True, "unit": "unit1"}
    )
    mock_insert_file_to_db.assert_called_once_with(f"{tmp_test_directory!s}/value1", "test_db")
    mock_reset_parameter_cache.assert_called_once()


def test_add_new_parameter_with_file_no_prefix(
    db, mocker, value_unit_type, validate_model_parameter, test_db
):
    """Test add_new_parameter method with file but no file_prefix."""
    mock_validate_model_parameter = mocker.patch(
        validate_model_parameter,
        return_value={"parameter": "param1", "value": "value1", "file": True},
    )
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mock_get_value_unit_type = mocker.patch(
        value_unit_type,
        return_value=("value1", "unit1", None),
    )
    mock_reset_parameter_cache = mocker.patch.object(db, "_reset_parameter_cache")

    par_dict = {"parameter": "param1", "value": "value1", "file": True}
    collection_name = "telescopes"
    file_prefix = None

    with pytest.raises(
        FileNotFoundError,
        match=r"The location of the file to upload, corresponding to the param1 parameter, "
        r"must be provided.",
    ):
        db.add_new_parameter(test_db, par_dict, collection_name, file_prefix)

    mock_validate_model_parameter.assert_called_once_with(par_dict)
    mock_get_db_name.assert_called_once_with(test_db)
    mock_get_collection.assert_called_once_with("test_db", collection_name)
    mock_get_value_unit_type.assert_called_once_with(value="value1", unit_str=None)
    mock_reset_parameter_cache.assert_not_called()


def test_insert_file_to_db_file_exists(db, mocker, test_db, test_file, mock_gridfs):
    """Test insert_file_to_db method when file already exists in the DB."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": mocker.Mock()}
    )
    mock_file_system = mock_gridfs.return_value
    mock_file_system.exists.return_value = True
    mock_file_instance = mocker.Mock()
    mock_file_system.find_one.return_value = mock_file_instance

    result = db.insert_file_to_db(test_file, test_db)

    mock_get_db_name.assert_called_once_with(test_db)
    mock_gridfs.assert_called_once_with(mock_db_client[test_db])
    mock_file_system.exists.assert_called_once_with({"filename": test_file})
    mock_file_system.find_one.assert_called_once_with({"filename": test_file})
    assert result == mock_file_instance._id


def test_insert_file_to_db_new_file(db, mocker, mock_open, test_db, test_file, mock_gridfs):
    """Test insert_file_to_db method when file does not exist in the DB."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": mocker.Mock()}
    )
    mock_file_system = mock_gridfs.return_value
    mock_file_system.exists.return_value = False
    mock_file_system.put.return_value = "new_file_id"

    result = db.insert_file_to_db(test_file, test_db)

    mock_get_db_name.assert_called_once_with(test_db)
    mock_gridfs.assert_called_once_with(mock_db_client[test_db])
    mock_file_system.exists.assert_called_once_with({"filename": test_file})
    mock_open.assert_called_once_with(test_file, "rb")
    mock_file_system.put.assert_called_once_with(
        mock_open(), content_type="ascii/dat", filename=test_file
    )
    assert result == "new_file_id"


def test_insert_file_to_db_with_kwargs(db, mocker, mock_open, test_db, test_file, mock_gridfs):
    """Test insert_file_to_db method with additional kwargs."""
    mock_get_db_name = mocker.patch.object(db, "_get_db_name", return_value="test_db")
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": mocker.Mock()}
    )
    mock_file_system = mock_gridfs.return_value
    mock_file_system.exists.return_value = False
    mock_file_system.put.return_value = "new_file_id"

    kwargs = {"content_type": "application/octet-stream", "metadata": {"key": "value"}}

    result = db.insert_file_to_db(test_file, test_db, **kwargs)

    mock_get_db_name.assert_called_once_with(test_db)
    mock_gridfs.assert_called_once_with(mock_db_client[test_db])
    mock_file_system.exists.assert_called_once_with({"filename": test_file})
    mock_open.assert_called_once_with(test_file, "rb")
    mock_file_system.put.assert_called_once_with(
        mock_open(),
        content_type="application/octet-stream",
        filename=test_file,
        metadata={"key": "value"},
    )
    assert result == "new_file_id"


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


def test_read_cache(db):
    test_key = "1.0.0-telescopes-North-LSTN-01"
    test_param1 = {"param1": "value1"}
    cache_dict = {test_key: test_param1}

    # Test _read_cache method when cache hit occurs.
    site = "North"
    array_element_name = "LSTN-01"
    model_version = "1.0.0"
    collection = "telescopes"

    cache_key, result = db._read_cache(
        cache_dict, site, array_element_name, model_version, collection
    )
    assert cache_key == test_key
    assert result == test_param1

    # Test _read_cache method when cache miss occurs.
    cache_key, result = db._read_cache(cache_dict, site, "LSTN-02", model_version, collection)
    assert cache_key == "1.0.0-telescopes-North-LSTN-02"
    assert result is None

    # Test _read_cache method with empty cache.
    cache_key, result = db._read_cache({}, site, array_element_name, model_version, collection)
    assert cache_key == test_key
    assert result is None

    # Test _read_cache method with partial parameters.
    test_key = "1.0.0-telescopes-North"
    cache_dict = {test_key: test_param1}
    cache_key, result = db._read_cache(cache_dict, site, None, model_version, collection)
    assert cache_key == test_key
    assert result == test_param1

    # Test _read_cache method with no parameters.
    cache_dict = {"": test_param1}
    cache_key, result = db._read_cache(cache_dict, None, None, None, None)
    assert cache_key == ""
    assert result == test_param1


def test_reset_parameter_cache(db):
    """Test _reset_parameter_cache method."""
    # Populate the cache dictionaries
    db_handler.DatabaseHandler.model_parameters_cached = {"key2": "value2"}

    # Ensure the caches are populated
    assert db_handler.DatabaseHandler.model_parameters_cached

    # Call the method to reset the caches
    db._reset_parameter_cache()

    # Check that the caches are cleared
    assert not db_handler.DatabaseHandler.model_parameters_cached


def test_get_array_element_list_configuration_corsika(db):
    """Test _get_array_element_list method for configuration_corsika collection."""
    array_element_name = "LSTN-01"
    site = "North"
    production_table = {}
    collection = "configuration_corsika"

    result = db._get_array_element_list(array_element_name, site, production_table, collection)

    assert result == ["xSTx-design"]


def test_get_array_element_list_sites(db):
    """Test _get_array_element_list method for sites collection."""
    array_element_name = "LSTN-01"
    site = "North"
    production_table = {}
    collection = "sites"

    result = db._get_array_element_list(array_element_name, site, production_table, collection)

    assert result == ["OBS-North"]


def test_get_array_element_list_design_model(db):
    """Test _get_array_element_list method when array element name contains '-design'."""
    array_element_name = "LSTN-design"
    site = "North"
    production_table = {}
    collection = "telescopes"

    result = db._get_array_element_list(array_element_name, site, production_table, collection)

    assert result == ["LSTN-design"]


def test_get_array_element_list_with_design_model_in_production_table(db, mocker):
    """Test _get_array_element_list method with design model in production table."""
    array_element_name = "LSTN-01"
    site = "North"
    production_table = {"design_model": {"LSTN-01": "LSTN-design"}}
    collection = "telescopes"

    result = db._get_array_element_list(array_element_name, site, production_table, collection)

    assert result == ["LSTN-design", "LSTN-01"]


def test_get_model_versions(db):
    model_versions = db.get_model_versions()
    assert len(model_versions) > 0
    assert "5.0.0" in model_versions
    assert "6.0.0" in model_versions


def test_get_array_elements(db):
    prod5_elements = db.get_array_elements("5.0.0", "telescopes")
    assert len(prod5_elements) > 0
    assert "LSTN-01" in prod5_elements
    assert "MSTN-101" not in prod5_elements
    prod6_elements = db.get_array_elements("6.0.0", "telescopes")
    assert "MSTN-101" in prod6_elements
    prod6_calibration_devices = db.get_array_elements("6.0.0", "calibration_devices")
    assert "ILLN-02" in prod6_calibration_devices


def test_get_design_model(db):
    assert db.get_design_model("5.0.0", "LSTN-01") == "LSTN-design"
    assert db.get_design_model("6.0.0", "LSTN-01") == "LSTN-design"
    assert db.get_design_model("5.0.0", "SSTS-03") == "SSTS-design"
    assert db.get_design_model("6.0.0", "LSTN-design") == "LSTN-design"
