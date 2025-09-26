#!/usr/bin/python3

import copy
import logging
from pathlib import Path
from unittest.mock import call

import pytest
from bson.objectid import ObjectId

from simtools.db import db_handler
from simtools.utils import names

# Suppress warnings of type
# 'pytest.PytestUnraisableExceptionWarning: Exception ignored in: <function MongoClient'
pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")

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
def mock_get_collection_name(mocker):
    return mocker.patch(
        "simtools.db.db_handler.names.get_collection_name_from_parameter_name",
        return_value="telescopes",
    )


@pytest.fixture
def mock_read_simtel_table(mocker):
    return mocker.patch(
        "simtools.db.db_handler.simtel_table_reader.read_simtel_table",
        return_value="test_table",
    )


@pytest.fixture
def validate_model_parameter():
    return "simtools.db.db_handler.validate_data.DataValidator.validate_model_parameter"


@pytest.fixture
def mock_gridfs(mocker):
    return mocker.patch("simtools.db.db_handler.gridfs.GridFS")


@pytest.fixture
def standard_test_params():
    """Common test parameters used across multiple tests."""
    return {
        "site": "North",
        "array_element_name": "LSTN-01",
        "model_version": "1.0.0",
        "parameter_version": "1.0.0",
        "collection": "telescopes",
        "parameter": "test_param",
    }


@pytest.fixture
def mock_collection_setup(mocker, db):
    """Common fixture for mocking collection operations."""
    mock_collection = mocker.Mock()
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mock_collection)
    return {"collection": mock_collection, "get_collection": mock_get_collection}


@pytest.fixture
def mock_db_client(mocker, db, test_db):
    """Common fixture for mocking db_client."""
    mock_client = {test_db: mocker.Mock()}
    return mocker.patch.object(db_handler.DatabaseHandler, "db_client", mock_client)


@pytest.fixture
def mock_file_system(mocker, mock_gridfs):
    """Setup mock file system objects."""
    mock_fs = mock_gridfs.return_value
    mock_file_instance = mocker.Mock(_id="file_id")
    mock_fs.find_one.return_value = mock_file_instance
    return {"fs": mock_fs, "instance": mock_file_instance}


@pytest.fixture
def export_files_setup(db, mocker):
    """Common setup for export_model_files tests."""
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", return_value=mocker.Mock(_id="file_id")
    )
    mock_write_file = mocker.patch.object(db, "_write_file_from_mongo_to_disk")
    return {"get_file_mongo_db": mock_get_file_mongo_db, "write_file": mock_write_file}


@pytest.fixture
def add_parameter_mocks(db, mocker, test_db, value_unit_type, validate_model_parameter):
    """Common setup for add_new_parameter tests."""
    mock_validate = mocker.patch(
        validate_model_parameter,
        return_value={"parameter": "param1", "value": "value1", "file": False},
    )
    # db_handler._get_db_name was removed; tests should use the db.db_name attribute
    db.db_name = test_db
    mock_coll = mocker.Mock()
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mock_coll)
    mock_insert_one = mocker.patch.object(mock_coll, "insert_one")
    mock_value_unit = mocker.patch(
        value_unit_type,
        return_value=("value1", "unit1", None),
    )
    mock_reset_cache = mocker.patch.object(db, "_reset_parameter_cache")

    return {
        "validate": mock_validate,
        "db_name": test_db,
        "collection": mock_coll,
        "get_collection": mock_get_collection,
        "insert_one": mock_insert_one,
        "value_unit": mock_value_unit,
        "reset_cache": mock_reset_cache,
    }


@pytest.fixture
def common_mock_read_production_table(mocker, db):
    """Common fixture for mocking read_production_table_from_mongo_db."""
    return mocker.patch.object(
        db,
        "read_production_table_from_mongo_db",
        return_value={"parameters": {"LSTN-01": {"param1": "v1"}}},
    )


@pytest.fixture
def common_mock_get_array_element_list(mocker, db):
    """Common fixture for mocking _get_array_element_list."""
    return mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-design", "LSTN-01"]
    )


@pytest.fixture
def common_mock_read_cache(mocker, db):
    """Common fixture for mocking _read_cache."""
    return mocker.patch.object(db, "_read_cache", return_value=("cache_key", None))


@pytest.fixture
def common_mock_read_mongo_db(mocker, db):
    """Common fixture for mocking _read_mongo_db."""
    return mocker.patch.object(db, "_read_mongo_db", return_value={"param1": {"value": "value1"}})


def assert_model_parameter_calls(
    mock_get_collection_name,
    mock_read_mongo_db,
    test_param,
    site,
    array_element_name,
    parameter_version,
    collection="telescopes",
):
    """Helper function to verify common model parameter assertions."""
    mock_get_collection_name.assert_called_once_with(test_param)
    mock_read_mongo_db.assert_called_once_with(
        query={
            "parameter_version": parameter_version,
            "parameter": test_param,
            **({"instrument": array_element_name} if array_element_name else {}),
            **({"site": site} if site else {}),
        },
        collection_name=collection,
    )


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
        authSource=(
            db_config.get("db_api_authentication_database")
            if db_config.get("db_api_authentication_database")
            else "admin"
        ),
        directConnection=True,
        ssl=False,
        tlsallowinvalidhostnames=True,
        tlsallowinvalidcertificates=True,
    )


def test_get_model_parameters(
    db,
    common_mock_read_production_table,
    common_mock_get_array_element_list,
    common_mock_read_cache,
    common_mock_read_mongo_db,
    standard_test_params,
):
    """Test get_model_parameters method."""
    site = standard_test_params["site"]
    array_element_name = standard_test_params["array_element_name"]
    model_version = standard_test_params["model_version"]
    collection = standard_test_params["collection"]

    result = db.get_model_parameters(site, array_element_name, collection, model_version)

    common_mock_read_production_table.assert_called_once_with(collection, model_version)
    common_mock_get_array_element_list.assert_called_once_with(
        array_element_name, site, {"parameters": {"LSTN-01": {"param1": "v1"}}}, collection
    )
    common_mock_read_cache.assert_has_calls(
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
    common_mock_read_mongo_db.assert_called_once_with(
        query={
            "$or": [{"parameter": "param1", "parameter_version": "v1"}],
            "instrument": "LSTN-01",
            "site": site,
        },
        collection_name=collection,
    )
    assert result == {"param1": {"value": "value1"}}


def test_get_model_parameters_for_all_model_versions(
    db,
    mocker,
    standard_test_params,
):
    """Test get_model_parameters_for_all_model_versions method."""
    site = standard_test_params["site"]
    array_element_name = standard_test_params["array_element_name"]
    collection = standard_test_params["collection"]

    mock_get_model_versions = mocker.patch.object(
        db, "get_model_versions", return_value=["5.0.0", "6.0.0"]
    )

    mock_get_parameter = mocker.patch.object(
        db,
        "get_model_parameters",
        side_effect=lambda site, array_element_name, collection, version: {
            "param1": "val1" if version == "5.0.0" else "val5",
            "param2": "val4" if version == "5.0.0" else "val2",
            "param3": "val3" if version == "5.0.0" else "val6",
        },
    )

    result = db.get_model_parameters_for_all_model_versions(site, array_element_name, collection)

    # Verify that the mocks were called correctly
    mock_get_model_versions.assert_called_once_with(collection)
    assert mock_get_parameter.call_count == 2  # Called once for each version

    expected = {
        "5.0.0": {
            "param1": "val1",
            "param2": "val4",
            "param3": "val3",
        },
        "6.0.0": {
            "param1": "val5",
            "param2": "val2",
            "param3": "val6",
        },
    }

    assert result == expected


def test_get_model_parameters_for_all_model_versions_mst(
    db,
    mocker,
    standard_test_params,
):
    """Test get_model_parameters_for_all_model_versions method."""
    site = standard_test_params["site"]
    array_element_name = "MSTN-101"  # Using telescope that only exists in prod6
    collection = standard_test_params["collection"]

    mock_get_model_versions = mocker.patch.object(
        db, "get_model_versions", return_value=["5.0.0", "6.0.0"]
    )

    # Mock get_model_parameters to raise KeyError for version 5.0.0
    # but return valid data for version 6.0.0
    def mock_get_parameters(site, array_element_name, collection, version):
        if version == "5.0.0":
            raise KeyError(f"{array_element_name} not found")
        return {
            "param1": "val5",
            "param2": "val2",
            "param3": "val6",
        }

    mock_get_parameter = mocker.patch.object(
        db, "get_model_parameters", side_effect=mock_get_parameters
    )

    # Mock logger to verify debug message
    mock_logger = mocker.patch.object(db._logger, "debug")

    result = db.get_model_parameters_for_all_model_versions(site, array_element_name, collection)

    # Verify that the mocks were called correctly
    mock_get_model_versions.assert_called_once_with(collection)
    assert mock_get_parameter.call_count == 2  # Should still try both versions

    # Verify debug message for skipped version
    mock_logger.assert_called_once_with(
        f"Skipping model version 5.0.0 - {array_element_name} not found"
    )

    # Verify result only contains data for version 6.0.0
    expected = {
        "6.0.0": {
            "param1": "val5",
            "param2": "val2",
            "param3": "val6",
        }
    }

    assert result == expected


def test_get_model_parameters_with_cache(db, mocker, standard_test_params):
    """Test get_model_parameters method with cache."""
    site = standard_test_params["site"]
    array_element_name = standard_test_params["array_element_name"]
    model_version = standard_test_params["model_version"]
    collection = standard_test_params["collection"]

    mock_get_production_table = mocker.patch.object(
        db,
        "read_production_table_from_mongo_db",
        return_value={"parameters": {"LSTN-01": {"param1": "v1"}}},
    )
    mock_get_array_element_list = mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-01"]
    )
    mock_read_cache = mocker.patch.object(
        db, "_read_cache", return_value=("cache_key", {"param1": {"value": "cached_value"}})
    )

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


def test_get_model_parameters_no_parameters(db, mocker, standard_test_params):
    """Test get_model_parameters method with no parameters."""
    site = standard_test_params["site"]
    array_element_name = standard_test_params["array_element_name"]
    model_version = standard_test_params["model_version"]
    collection = standard_test_params["collection"]

    mock_get_production_table = mocker.patch.object(
        db, "read_production_table_from_mongo_db", return_value={"parameters": {}}
    )
    mock_get_array_element_list = mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-01"]
    )
    mock_read_cache = mocker.patch.object(db, "_read_cache", return_value=("cache_key", None))

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


def test_get_model_parameter_with_model_version_list(
    db, mock_get_collection_name, common_mock_read_mongo_db, mocker, standard_test_params
):
    """Test get_model_parameter method with model_version as list."""
    site = standard_test_params["site"]
    array_element_name = standard_test_params["array_element_name"]
    collection = standard_test_params["collection"]

    # Mock the production table reading
    mock_read_production_table = mocker.patch.object(
        db,
        "read_production_table_from_mongo_db",
        return_value={
            "parameters": {
                "LSTN-design": {"test_param": "2.0.0"},
                "LSTN-01": {"test_param": "1.0.0"},
            }
        },
    )

    # Mock array element list
    mock_get_array_element_list = mocker.patch.object(
        db, "_get_array_element_list", return_value=["LSTN-design", "LSTN-01"]
    )

    # Update common_mock_read_mongo_db to return the expected format
    common_mock_read_mongo_db.return_value = {"test_param": {"value": "test_value"}}

    # Test with single version as string - should work
    db.get_model_parameter(
        parameter="test_param",
        site=site,
        array_element_name=array_element_name,
        model_version="5.0.0",
    )
    mock_get_collection_name.assert_called_once_with("test_param")
    mock_read_production_table.assert_called_once_with(collection, "5.0.0")
    mock_get_array_element_list.assert_called_once_with(
        array_element_name,
        site,
        {
            "parameters": {
                "LSTN-design": {"test_param": "2.0.0"},
                "LSTN-01": {"test_param": "1.0.0"},
            },
        },
        collection,
    )
    common_mock_read_mongo_db.assert_called_once_with(
        query={
            "parameter_version": "1.0.0",
            "parameter": "test_param",
            "instrument": "LSTN-01",
            "site": site,
        },
        collection_name=collection,
    )

    error_message = "Only one model version can be passed to get_model_parameter, not a list."
    # Test with multiple versions - should raise ValueError
    with pytest.raises(ValueError, match=error_message):
        db.get_model_parameter(
            parameter="test_param",
            site=site,
            array_element_name=array_element_name,
            model_version=["5.0.0", "6.0.0"],
        )

    # Test with single version in list - should raise ValueError
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        db.get_model_parameter(
            parameter="test_param",
            site=site,
            array_element_name=array_element_name,
            model_version=["5.0.0"],
        )


def test_get_collection(db, test_db, mocker):
    """Test get_collection method."""
    mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {test_db: {"test_collection": "mock_collection"}}
    )
    collection_name = "test_collection"

    result = db.get_collection(collection_name, db_name=test_db)

    assert result == "mock_collection"


def test_get_collections(db, db_config, fs_files):
    collections = db.get_collections()
    assert isinstance(collections, list)
    assert "telescopes" in collections

    db_name = db.get_db_name(
        db_simulation_model_version=db_config["db_simulation_model_version"],
        model_name=db_config["db_simulation_model"],
    )
    collections_from_name = db.get_collections(db_name)
    assert isinstance(collections_from_name, list)
    assert "telescopes" in collections_from_name
    assert fs_files in collections_from_name

    collections_no_model = db.get_collections(db_name, True)
    assert isinstance(collections_no_model, list)
    assert "telescopes" in collections_no_model
    assert fs_files not in collections_no_model
    assert "metadata" not in collections_no_model


def test_export_model_files_with_file_names(
    db, export_files_setup, tmp_test_directory, test_db, test_file, test_file_2
):
    """Test export_model_files method with file names."""
    mocks = export_files_setup
    file_names = [test_file, test_file_2]

    result = db.export_model_files(file_names=file_names, dest=tmp_test_directory, db_name=test_db)

    mocks["get_file_mongo_db"].assert_has_calls(
        [call(test_db, test_file), call(test_db, test_file_2)]
    )
    mocks["write_file"].assert_has_calls(
        [
            call(test_db, tmp_test_directory, mocks["get_file_mongo_db"].return_value),
            call(test_db, tmp_test_directory, mocks["get_file_mongo_db"].return_value),
        ]
    )
    assert result == {test_file: "file_id", test_file_2: "file_id"}


def test_export_model_files_with_parameters(
    db, export_files_setup, tmp_test_directory, test_db, test_file, test_file_2
):
    """Test export_model_files method with parameters."""
    mocks = export_files_setup
    parameters = {
        "param1": {"file": True, "value": test_file},
        "param2": {"file": True, "value": test_file_2},
    }

    result = db.export_model_files(parameters=parameters, dest=tmp_test_directory, db_name=test_db)

    mocks["get_file_mongo_db"].assert_has_calls(
        [call(test_db, test_file), call(test_db, test_file_2)]
    )
    mocks["write_file"].assert_has_calls(
        [
            call(test_db, tmp_test_directory, mocks["get_file_mongo_db"].return_value),
            call(test_db, tmp_test_directory, mocks["get_file_mongo_db"].return_value),
        ]
    )
    assert result == {test_file: "file_id", test_file_2: "file_id"}


def test_export_model_files_file_exists(db, mocker, tmp_test_directory, test_db, test_file):
    """Test export_model_files method when file already exists."""
    mock_get_file_mongo_db = mocker.patch.object(db, "_get_file_mongo_db")
    mock_write_file = mocker.patch.object(db, "_write_file_from_mongo_to_disk")
    mock_path_exists = mocker.patch("pathlib.Path.exists", return_value=True)

    file_names = [test_file]
    result = db.export_model_files(file_names=file_names, dest=tmp_test_directory)

    mock_get_file_mongo_db.assert_not_called()
    mock_write_file.assert_not_called()
    mock_path_exists.assert_called_once()
    assert result == {test_file: "file exists"}


def test_export_model_files_file_not_found(db, mocker, tmp_test_directory, test_db, test_file):
    """Test export_model_files method when file is not found in parameters."""
    mock_get_file_mongo_db = mocker.patch.object(
        db, "_get_file_mongo_db", side_effect=FileNotFoundError
    )
    mock_write_file = mocker.patch.object(db, "_write_file_from_mongo_to_disk")

    parameters = {"param1": {"file": True, "value": test_file}}

    with pytest.raises(FileNotFoundError):
        db.export_model_files(parameters=parameters, dest=tmp_test_directory, db_name=test_db)

    mock_get_file_mongo_db.assert_called_once_with(test_db, test_file)
    mock_write_file.assert_not_called()


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

    test_cases = [
        # (array_element_name, site, expected_result)
        ("LSTN-01", "North", {"$or": or_list, "instrument": "LSTN-01", "site": "North"}),
        ("LSTN-01", None, {"$or": or_list, "instrument": "LSTN-01"}),
        (None, "North", {"$or": or_list, "site": "North"}),
        (None, None, {"$or": or_list}),
        ("xSTx-design", "North", {"$or": or_list, "site": "North"}),
    ]

    for array_element_name, site, expected in test_cases:
        result = db._get_query_from_parameter_version_table(
            parameter_version_table, array_element_name, site
        )
        assert result == expected


def test_read_mongo_db(db, mock_collection_setup, mocker, test_db):
    """Test read_mongo_db method."""
    mock_find = mocker.patch.object(
        mock_collection_setup["collection"],
        "find",
        return_value=[
            {"_id": ObjectId(), "parameter": "param1", "value": "value1"},
            {"_id": ObjectId(), "parameter": "param2", "value": "value2"},
        ],
    )

    query = {"parameter_version": "1.0.0"}
    collection_name = "test_collection"

    result = db._read_mongo_db(query, collection_name)

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

    # Test with no results
    mocker.patch.object(mock_collection_setup["collection"], "find", return_value=[])
    with pytest.raises(
        ValueError,
        match=r"The following query for test_collection returned zero results: "
        r"{'parameter_version': '1.0.0'}",
    ):
        db._read_mongo_db(query, collection_name)


def setup_production_table_cached(cache_key, model_version, param):
    """Helper to set up production table cache."""
    db_handler.DatabaseHandler.production_table_cached[cache_key] = {
        "collection": model_version,
        "model_version": model_version,
        "parameters": param,
        "design_model": {},
        "entry_date": ObjectId().generation_time,
    }
    return db_handler.DatabaseHandler.production_table_cached[cache_key]


def test_read_production_table_from_mongo_db_with_cache(db, mocker, test_db, mock_collection_setup):
    """Test read_production_table_from_mongo_db method with cache."""
    collection_name = "telescopes"
    model_version = "1.0.0"
    param = {"param1": "value1"}

    # Mock get_model_versions to return the expected model version
    mocker.patch.object(db, "get_model_versions", return_value=[model_version])

    # Test with cache hit
    mock_cache_key = mocker.patch.object(db, "_cache_key", return_value="cache_key")
    cached_result = setup_production_table_cached("cache_key", model_version, param)

    result = db.read_production_table_from_mongo_db(collection_name, model_version)

    mock_cache_key.assert_called_once_with(None, None, model_version, collection_name)
    assert result == cached_result

    # Test with cache miss
    mock_cache_key = mocker.patch.object(db, "_cache_key", return_value="no_cache_key")
    # _get_db_name removed; set db.db_name for tests that expect it
    db.db_name = test_db
    mock_find_one = mocker.patch.object(
        mock_collection_setup["collection"],
        "find_one",
        return_value={
            "_id": ObjectId(),
            "collection": collection_name,
            "model_version": model_version,
            "parameters": param,
            "design_model": {},
        },
    )

    result = db.read_production_table_from_mongo_db(collection_name, model_version)

    mock_cache_key.assert_called_once_with(None, None, model_version, collection_name)
    mock_collection_setup["get_collection"].assert_called_once_with(
        "production_tables", db_name=test_db
    )
    mock_find_one.assert_called_once_with(
        {"model_version": model_version, "collection": collection_name}
    )
    assert result["collection"] == collection_name
    assert result["model_version"] == model_version
    assert result["parameters"] == param
    assert result["design_model"] == {}
    assert "entry_date" in result

    # Test with no results
    mocker.patch.object(mock_collection_setup["collection"], "find_one", return_value=None)
    with pytest.raises(
        ValueError,
        match=r"The following query returned zero results: "
        r"{'model_version': '1.0.0', 'collection': 'telescopes'}",
    ):
        db.read_production_table_from_mongo_db(collection_name, model_version)


def test_get_array_elements_of_type(db, mocker):
    """Test get_array_elements_of_type method."""
    array_element_type = "LSTN"
    model_version = "1.0.0"
    collection = "telescopes"

    test_cases = [
        # Production table, expected result
        (
            {"parameters": {"LSTN-01": "value1", "LSTN-02": "value2", "MSTS-01": "value3"}},
            ["LSTN-01", "LSTN-02"],
        ),
        ({"parameters": {"MSTS-01": "value3"}}, []),
        ({"parameters": {"LSTN-01": "value1", "LSTN-design": "value2"}}, ["LSTN-01"]),
    ]

    for prod_table, expected in test_cases:
        mock_get_production_table = mocker.patch.object(
            db, "read_production_table_from_mongo_db", return_value=prod_table
        )
        result = db.get_array_elements_of_type(array_element_type, model_version, collection)
        mock_get_production_table.assert_called_once_with(collection, model_version)
        assert result == expected

    # Test with different array element type
    array_element_type = "MSTS"
    mocker.patch.object(
        db,
        "read_production_table_from_mongo_db",
        return_value={
            "parameters": {"LSTN-01": "value1", "MSTS-01": "value3", "MSTS-02": "value4"}
        },
    )
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

    software = "sim_telarray"
    assert (
        db.get_simulation_configuration_parameters(software, "North", "LSTN-design", "6.0.0")
        == return_value
    )
    assert mock_get_model_parameters.call_count == 2
    assert db.get_simulation_configuration_parameters(software, "North", None, "6.0.0") == {}
    assert mock_get_model_parameters.call_count == 2
    assert db.get_simulation_configuration_parameters(software, None, "LSTN-design", "6.0.0") == {}
    assert mock_get_model_parameters.call_count == 2
    assert db.get_simulation_configuration_parameters(software, None, None, "6.0.0") == {}
    assert mock_get_model_parameters.call_count == 2

    with pytest.raises(ValueError, match=r"Unknown simulation software: wrong"):
        db.get_simulation_configuration_parameters("wrong", "North", "LSTN-design", "6.0.0")


def test_get_file_mongo_db_file(db, test_db, test_file, mock_file_system, mock_db_client):
    """Test _get_file_mongo_db method when file exists."""
    mock_file_system["fs"].exists.return_value = True

    result = db._get_file_mongo_db(test_db, test_file)

    mock_file_system["fs"].exists.assert_called_once_with({"filename": test_file})
    mock_file_system["fs"].find_one.assert_called_once_with({"filename": test_file})
    assert result == mock_file_system["instance"]

    # Test when file does not exist
    mock_file_system["fs"].exists.return_value = False
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

    mock_file = mocker.Mock()
    mock_file.filename = test_file

    db._write_file_from_mongo_to_disk(test_db, tmp_test_directory, mock_file)

    mock_gridfs_bucket.assert_called_once_with(mock_db_client[test_db])
    mock_open.assert_called_once_with(Path(tmp_test_directory).joinpath(mock_file.filename), "wb")
    filename = mock_file.filename
    mock_fs_output.download_to_stream_by_name.assert_called_once_with(filename, mock_open())


def test_add_production_table(db, mocker, test_db):
    """Test add_production_table method."""
    # _get_db_name removed; set db.db_name for tests that expect it
    db.db_name = "test_db"
    mock_get_collection = mocker.patch.object(db, "get_collection", return_value=mocker.Mock())
    mock_insert_one = mocker.patch.object(db.get_collection.return_value, "insert_one")

    production_table = {
        "collection": "telescopes",
        "model_version": "1.0.0",
        "parameters": {"param1": "value1"},
    }

    db.add_production_table(production_table, db_name=test_db)

    assert db.db_name == test_db
    mock_get_collection.assert_called_once_with("production_tables", db_name=test_db)
    mock_insert_one.assert_called_once_with(production_table)


def test_add_new_parameter(db, add_parameter_mocks, test_db):
    """Test add_new_parameter method."""
    mocks = add_parameter_mocks
    par_dict = {"parameter": "param1", "value": "value1", "file": False}
    collection_name = "telescopes"
    file_prefix = None
    db.add_new_parameter(par_dict, test_db, collection_name, file_prefix)
    mocks["validate"].assert_called_once_with(par_dict)
    assert mocks["db_name"] == test_db
    mocks["get_collection"].assert_called_once_with(collection_name, db_name=test_db)
    mocks["value_unit"].assert_called_once_with(value="value1", unit_str=None)
    mocks["insert_one"].assert_called_once_with(
        {"parameter": "param1", "value": "value1", "file": False, "unit": "unit1"}
    )
    mocks["reset_cache"].assert_called_once()


def test_add_new_parameter_with_file(db, add_parameter_mocks, tmp_test_directory, test_db, mocker):
    """Test add_new_parameter method with file."""
    mocks = add_parameter_mocks
    mocks["validate"].return_value = {"parameter": "param1", "value": "value1", "file": True}
    mock_insert_file_to_db = mocker.patch.object(db, "insert_file_to_db")
    mock_is_utf8 = mocker.patch("simtools.io.ascii_handler.is_utf8_file", return_value=True)

    par_dict = {"parameter": "param1", "value": "value1", "file": True}
    collection_name = "telescopes"
    db.add_new_parameter(par_dict, test_db, collection_name, tmp_test_directory)
    mock_is_utf8.assert_called_once_with(Path(f"{tmp_test_directory!s}/value1"))
    mocks["validate"].assert_called_once_with(par_dict)
    assert mocks["db_name"] == test_db
    mocks["get_collection"].assert_called_once_with(collection_name, db_name=test_db)
    mocks["value_unit"].assert_called_once_with(value="value1", unit_str=None)
    mocks["insert_one"].assert_called_once_with(
        {"parameter": "param1", "value": "value1", "file": True, "unit": "unit1"}
    )
    mock_insert_file_to_db.assert_called_once_with(f"{tmp_test_directory!s}/value1", test_db)
    mocks["reset_cache"].assert_called_once()

    # non-utf8 file
    mock_insert_file_to_db.reset_mock()
    mock_is_utf8.return_value = False
    with pytest.raises(ValueError, match=r"File is not UTF-8 encoded"):
        db.add_new_parameter(par_dict, test_db, collection_name, tmp_test_directory)
    mock_insert_file_to_db.assert_not_called()


def test_add_new_parameter_with_file_no_prefix(db, add_parameter_mocks, test_db):
    """Test add_new_parameter method with file but no file_prefix."""
    mocks = add_parameter_mocks
    mocks["validate"].return_value = {"parameter": "param1", "value": "value1", "file": True}

    par_dict = {"parameter": "param1", "value": "value1", "file": True}
    collection_name = "telescopes"
    file_prefix = None

    with pytest.raises(
        FileNotFoundError,
        match=r"The location of the file to upload, corresponding to the param1 parameter, "
        r"must be provided.",
    ):
        db.add_new_parameter(par_dict, test_db, collection_name, file_prefix)

    mocks["validate"].assert_called_once_with(par_dict)
    assert mocks["db_name"] == test_db
    mocks["get_collection"].assert_called_once_with(collection_name, db_name=test_db)
    mocks["value_unit"].assert_called_once_with(value="value1", unit_str=None)
    mocks["reset_cache"].assert_not_called()


def test_insert_file_to_db_file_exists(db, mocker, test_db, test_file, mock_gridfs):
    """Test insert_file_to_db method when file already exists in the DB."""
    # _get_db_name removed; set db.db_name for tests that expect it
    db.db_name = "test_db"
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": mocker.Mock()}
    )
    mock_file_system = mock_gridfs.return_value
    mock_file_system.exists.return_value = True
    mock_file_instance = mocker.Mock()
    mock_file_system.find_one.return_value = mock_file_instance

    result = db.insert_file_to_db(test_file, test_db)

    assert db.db_name == test_db
    mock_gridfs.assert_called_once_with(mock_db_client[test_db])
    mock_file_system.exists.assert_called_once_with({"filename": test_file})
    mock_file_system.find_one.assert_called_once_with({"filename": test_file})
    assert result == mock_file_instance._id


def test_insert_file_to_db_new_file(db, mocker, mock_open, test_db, test_file, mock_gridfs):
    """Test insert_file_to_db method when file does not exist in the DB."""
    # _get_db_name removed; set db.db_name for tests that expect it
    db.db_name = "test_db"
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": mocker.Mock()}
    )
    mock_file_system = mock_gridfs.return_value
    mock_file_system.exists.return_value = False
    mock_file_system.put.return_value = "new_file_id"

    result = db.insert_file_to_db(test_file, test_db)

    assert db.db_name == test_db
    mock_gridfs.assert_called_once_with(mock_db_client[test_db])
    mock_file_system.exists.assert_called_once_with({"filename": test_file})
    mock_open.assert_called_once_with(test_file, "rb")
    mock_file_system.put.assert_called_once_with(
        mock_open(), content_type="ascii/dat", filename=test_file
    )
    assert result == "new_file_id"


def test_insert_file_to_db_with_kwargs(db, mocker, mock_open, test_db, test_file, mock_gridfs):
    """Test insert_file_to_db method with additional kwargs."""
    # _get_db_name removed; set db.db_name for tests that expect it
    db.db_name = "test_db"
    mock_db_client = mocker.patch.object(
        db_handler.DatabaseHandler, "db_client", {"test_db": mocker.Mock()}
    )
    mock_file_system = mock_gridfs.return_value
    mock_file_system.exists.return_value = False
    mock_file_system.put.return_value = "new_file_id"

    kwargs = {"content_type": "application/octet-stream", "metadata": {"key": "value"}}

    result = db.insert_file_to_db(test_file, test_db, **kwargs)

    assert db.db_name == test_db
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


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "with_parameter_version",
            "params": {
                "site": "North",
                "array_element_name": "LSTN-01",
                "parameter_version": "1.0.0",
                "model_version": None,
            },
        },
        {
            "name": "with_model_version",
            "params": {
                "site": "North",
                "array_element_name": "LSTN-01",
                "parameter_version": None,
                "model_version": "1.0.0",
            },
            "setup": {
                "prod_table": {
                    "parameters": {
                        "LSTN-design": {"test_param": "2.0.0"},
                        "LSTN-01": {"test_param": "1.0.0"},
                    }
                }
            },
        },
        {
            "name": "with_no_site_no_instrument",
            "params": {
                "site": None,
                "array_element_name": None,
                "parameter_version": "1.0.0",
                "model_version": None,
            },
        },
    ],
)
def test_get_model_parameter_variants(db, mocker, mock_get_collection_name, test_case):
    """Test get_model_parameter variations."""
    params = test_case["params"]
    mock_read_mongo_db = mocker.patch.object(
        db,
        "_read_mongo_db",
        return_value={"test_param": {"value": "test_value"}},
    )

    if "setup" in test_case:
        mocker.patch.object(
            db,
            "read_production_table_from_mongo_db",
            return_value=test_case["setup"]["prod_table"],
        )
        mocker.patch.object(
            db,
            "_get_array_element_list",
            return_value=["LSTN-design", "LSTN-01"],
        )

    result = db.get_model_parameter(parameter="test_param", **params)

    assert_model_parameter_calls(
        mock_get_collection_name,
        mock_read_mongo_db,
        "test_param",
        params["site"],
        params["array_element_name"],
        params["parameter_version"] or "1.0.0",
    )
    assert result == {"test_param": {"value": "test_value"}}


@pytest.fixture
def export_model_file_mocks(db, mocker, tmp_test_directory, test_file):
    """Common setup for export_model_file tests."""
    test_param = "test_param"
    mock_parameters = {test_param: {"value": test_file}}

    return {
        "test_param": test_param,
        "test_file": test_file,
        "parameters": mock_parameters,
        "get_model_parameter": mocker.patch.object(
            db, "get_model_parameter", return_value=mock_parameters
        ),
        "export_model_files": mocker.patch.object(db, "export_model_files"),
        "get_output_directory": mocker.patch.object(
            db.io_handler, "get_output_directory", return_value=tmp_test_directory
        ),
    }


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "without_table",
            "params": {
                "export_file_as_table": False,
                "parameter_version": None,
                "model_version": "1.0.0",
            },
            "expected_result": None,
        },
        {
            "name": "with_table",
            "params": {
                "export_file_as_table": True,
                "parameter_version": None,
                "model_version": "1.0.0",
            },
            "expected_result": "test_table",
        },
        {
            "name": "with_parameter_version",
            "params": {
                "export_file_as_table": False,
                "parameter_version": "1.0.0",
                "model_version": None,
            },
            "expected_result": None,
        },
    ],
)
def test_export_model_file_variants(
    db, export_model_file_mocks, mock_read_simtel_table, mocker, tmp_test_directory, test_case
):
    """Test export_model_file variations."""
    mocks = export_model_file_mocks
    params = test_case["params"]

    if params["export_file_as_table"]:
        path_obj = mocker.Mock()
        path_obj.joinpath.return_value = f"{tmp_test_directory}/{mocks['test_file']}"
        mocks["get_output_directory"].return_value = path_obj

    result = db.export_model_file(
        parameter=mocks["test_param"], site="North", array_element_name="LSTN-01", **params
    )

    mocks["get_model_parameter"].assert_called_once_with(
        mocks["test_param"],
        "North",
        "LSTN-01",
        parameter_version=params["parameter_version"],
        model_version=params["model_version"],
    )

    assert result == test_case["expected_result"]


def test_get_array_element_list_configuration_sim_telarray(db, mocker):
    """Test _get_array_element_list method for configuration_sim_telarray collection."""
    array_element_name = "LSTN-01"
    site = "North"
    model_version = "1.0.0"
    production_table = {"model_version": model_version}
    collection = "configuration_sim_telarray"

    mock_read_production_table = mocker.patch.object(
        db,
        "read_production_table_from_mongo_db",
        return_value={"design_model": {"LSTN-01": "LSTN-design"}},
    )

    result = db._get_array_element_list(array_element_name, site, production_table, collection)
    mock_read_production_table.assert_called_once_with("telescopes", model_version)
    assert result == ["LSTN-design", "LSTN-01"]

    mock_read_production_table.return_value = {"design_model": {}}  # No design model for LSTN-01
    with pytest.raises(
        KeyError, match=r"Failed generated array element list for db query for LSTN-01"
    ):
        db._get_array_element_list(array_element_name, site, production_table, collection)


def test_generate_compound_indexes(mocker, db):
    """Test generate_compound_indexes method."""
    mock_get_collection = mocker.patch.object(db, "get_collection")
    mock_create_index = mocker.Mock()
    mock_get_collection.return_value.create_index = mock_create_index

    db.generate_compound_indexes()

    expected_calls = [
        call([("instrument", 1), ("site", 1), ("parameter", 1), ("parameter_version", 1)]),
        call([("instrument", 1), ("site", 1), ("parameter", 1), ("parameter_version", 1)]),
        call([("instrument", 1), ("site", 1), ("parameter", 1), ("parameter_version", 1)]),
        call([("instrument", 1), ("site", 1), ("parameter", 1), ("parameter_version", 1)]),
        call([("instrument", 1), ("site", 1), ("parameter", 1), ("parameter_version", 1)]),
        call([("collection", 1), ("model_version", 1)]),
    ]

    assert mock_create_index.call_count == len(expected_calls)
    mock_create_index.assert_has_calls(expected_calls, any_order=True)


def test_get_ecsv_file_as_astropy_table(mocker, db):
    # Prevent any real DB calls by providing a fake db_client and mocking GridFSBucket
    mocker.patch.object(db_handler.DatabaseHandler, "db_client", {db.db_name: mocker.Mock()})

    mock_gridfs_bucket = mocker.patch("simtools.db.db_handler.gridfs.GridFSBucket")
    fs_instance = mock_gridfs_bucket.return_value

    # Minimal valid ECSV content so astropy.Table.read can parse it
    ecsv_content = (
        b"# %ECSV 0.9\n# ---\n# datatype:\n# - {name: x, datatype: float64}\n# schema: {}\nx\n1.0\n"
    )

    def download_side_effect(filename, buf):
        if filename == "nsb_spectrum_halfmoon.ecsv":
            buf.write(ecsv_content)
        else:
            # Raise the same error the real GridFS would raise
            raise db_handler.gridfs.errors.NoFile("no such file")

    fs_instance.download_to_stream_by_name.side_effect = download_side_effect

    # Positive case: file exists (simulated)
    assert db.get_ecsv_file_as_astropy_table("nsb_spectrum_halfmoon.ecsv") is not None

    # Negative case: simulate missing file
    with pytest.raises(FileNotFoundError, match=r"ECSV file 'test_file.ecsv' not found in DB."):
        db.get_ecsv_file_as_astropy_table("test_file.ecsv")


def test_get_db_name(db):
    """Test _get_db_name with valid configuration."""
    assert (
        db.get_db_name(db_simulation_model_version="v1.0.0", model_name="SimulationModel")
        == "SimulationModel-v1-0-0"
    )
    assert db.get_db_name(db_simulation_model_version="v1.0.0") is None
    assert db.get_db_name(model_name="SimulationModel") is None
    assert db.get_db_name() is not None
    assert db.get_db_name(db_name="test_db") == "test_db"
    assert (
        db.get_db_name(
            db_name="test_db", db_simulation_model_version="v1.0.0", model_name="SimulationModel"
        )
        == "test_db"
    )


def test_is_remote_database():
    test_db = db_handler.DatabaseHandler(None)

    assert test_db.is_remote_database() is False

    test_db.mongo_db_config = {"db_server": "remote.server.com"}
    assert test_db.is_remote_database() is True

    test_db.mongo_db_config = {"db_server": "localhost"}
    assert test_db.is_remote_database() is False

    test_db.mongo_db_config = {"db_server": ""}
    assert test_db.is_remote_database() is False
