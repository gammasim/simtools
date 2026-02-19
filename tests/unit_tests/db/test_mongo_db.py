#!/usr/bin/python3

import logging
from pathlib import Path

import pytest

from simtools.db import mongo_db

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")


@pytest.fixture(autouse=True)
def reset_db_client():
    """Reset db_client before each test."""
    mongo_db.MongoDBHandler.db_client = None
    yield
    mongo_db.MongoDBHandler.db_client = None


@pytest.fixture
def valid_db_config():
    """Valid MongoDB configuration for testing."""
    return {
        "db_server": "localhost",
        "db_api_port": 27017,
        "db_api_user": "test_user",
        "db_api_pw": "test_password",
        "db_api_authentication_database": "admin",
        "db_simulation_model": "test_model",
        "db_simulation_model_version": "1.0.0",
    }


@pytest.fixture
def mongo_handler(valid_db_config):
    """Create a MongoDBHandler instance."""
    return mongo_db.MongoDBHandler(valid_db_config)


def test_validate_db_config_valid(valid_db_config):
    """Test validation of valid MongoDB configuration."""
    result = mongo_db.MongoDBHandler.validate_db_config(valid_db_config)
    assert result == valid_db_config


def test_validate_db_config_none():
    """Test validation with None configuration."""
    result = mongo_db.MongoDBHandler.validate_db_config(None)
    assert result is None


def test_validate_db_config_empty():
    """Test validation with empty configuration."""
    result = mongo_db.MongoDBHandler.validate_db_config({})
    assert result is None


def test_validate_db_config_all_none_values(valid_db_config):
    """Test validation with all None values."""
    none_config = dict.fromkeys(valid_db_config.keys())
    result = mongo_db.MongoDBHandler.validate_db_config(none_config)
    assert result is None


def test_validate_db_config_invalid():
    """Test validation with invalid configuration."""
    invalid_config = {"wrong_key": "wrong_value"}
    with pytest.raises(ValueError, match="Invalid MongoDB configuration"):
        mongo_db.MongoDBHandler.validate_db_config(invalid_config)


def test_validate_db_config_missing_required():
    """Test validation with missing required fields."""
    incomplete_config = {
        "db_server": "localhost",
        "db_api_port": 27017,
    }
    with pytest.raises(ValueError, match="Invalid MongoDB configuration"):
        mongo_db.MongoDBHandler.validate_db_config(incomplete_config)


def test_get_db_name_with_version_and_model():
    """Test get_db_name with version and model."""
    result = mongo_db.MongoDBHandler.get_db_name(
        db_simulation_model_version="1.0.0", model_name="test_model"
    )
    assert result == "test_model-1-0-0"


def test_get_db_name_with_direct_name():
    """Test get_db_name with direct database name."""
    result = mongo_db.MongoDBHandler.get_db_name(db_name="direct_db")
    assert result == "direct_db"


def test_get_db_name_priority():
    """Test that direct db_name takes priority."""
    result = mongo_db.MongoDBHandler.get_db_name(
        db_name="direct_db", db_simulation_model_version="1.0.0", model_name="test_model"
    )
    assert result == "direct_db"


def test_get_db_name_incomplete():
    """Test get_db_name with incomplete parameters."""
    result = mongo_db.MongoDBHandler.get_db_name(db_simulation_model_version="1.0.0")
    assert result is None
    result = mongo_db.MongoDBHandler.get_db_name(model_name="test_model")
    assert result is None


def test_init_with_valid_config(valid_db_config):
    """Test initialization with valid configuration."""
    handler = mongo_db.MongoDBHandler(valid_db_config)
    assert handler.db_config == valid_db_config
    assert handler.list_of_collections == {}


def test_init_with_none_config():
    """Test initialization with None configuration."""
    handler = mongo_db.MongoDBHandler(None)
    assert handler.db_config is None


def test_build_uri_direct_connection(valid_db_config):
    """Test _build_uri with direct connection."""
    valid_db_config["db_server"] = "localhost"

    uri = mongo_db.MongoDBHandler._build_uri(valid_db_config)

    assert "mongodb://test_user:test_password@localhost:27017/" in uri
    assert "authSource=admin" in uri
    assert "directConnection=true" in uri
    assert "ssl=" not in uri


def test_build_uri_remote_connection(valid_db_config):
    """Test _build_uri with remote connection."""
    valid_db_config["db_server"] = "remote.server.com"

    uri = mongo_db.MongoDBHandler._build_uri(valid_db_config)

    assert "mongodb://test_user:test_password@remote.server.com:27017/" in uri
    assert "authSource=admin" in uri
    assert "ssl=true" in uri
    assert "tlsAllowInvalidHostnames=true" in uri
    assert "tlsAllowInvalidCertificates=true" in uri
    assert "directConnection=true" not in uri


@pytest.mark.parametrize("server_name", ["simtools-MongoDB", "local-simtools-mongodb", "127.0.0.1"])
def test_build_uri_direct_connection_local_aliases(valid_db_config, server_name):
    """Test _build_uri with local aliases requiring direct connection."""
    valid_db_config["db_server"] = server_name

    uri = mongo_db.MongoDBHandler._build_uri(valid_db_config)

    assert f"mongodb://test_user:test_password@{server_name}:27017/" in uri
    assert "authSource=admin" in uri
    assert "directConnection=true" in uri
    assert "ssl=true" not in uri


def test_initialize_client(mocker, valid_db_config):
    """Test _initialize_client method."""
    mock_mongo_client = mocker.patch("simtools.db.mongo_db.MongoClient")
    mock_client_instance = mocker.MagicMock()
    mock_mongo_client.return_value = mock_client_instance

    mongo_db.MongoDBHandler._initialize_client(valid_db_config)

    assert mongo_db.MongoDBHandler.db_client == mock_client_instance
    mock_mongo_client.assert_called_once()
    call_args = mock_mongo_client.call_args
    assert "maxIdleTimeMS" in call_args.kwargs
    assert call_args.kwargs["maxIdleTimeMS"] == 10000


def test_initialize_client_thread_safety(mocker, valid_db_config):
    """Test that _initialize_client is thread-safe and only creates one client."""
    mock_mongo_client = mocker.patch("simtools.db.mongo_db.MongoClient")
    mock_client_instance = mocker.MagicMock()
    mock_mongo_client.return_value = mock_client_instance

    mongo_db.MongoDBHandler._initialize_client(valid_db_config)
    first_client = mongo_db.MongoDBHandler.db_client

    mongo_db.MongoDBHandler._initialize_client(valid_db_config)
    second_client = mongo_db.MongoDBHandler.db_client

    assert first_client is second_client
    mock_mongo_client.assert_called_once()


def test_initialize_client_early_return(mocker, valid_db_config):
    """Test that _initialize_client returns early if client already exists."""
    mock_mongo_client = mocker.patch("simtools.db.mongo_db.MongoClient")
    mock_client_instance = mocker.MagicMock()
    mock_mongo_client.return_value = mock_client_instance

    mongo_db.MongoDBHandler.db_client = mock_client_instance

    mongo_db.MongoDBHandler._initialize_client(valid_db_config)

    mock_mongo_client.assert_not_called()


def test_initialize_client_failure(mocker, valid_db_config, caplog):
    """Test _initialize_client handles MongoClient initialization failure."""
    mock_mongo_client = mocker.patch("simtools.db.mongo_db.MongoClient")
    mock_mongo_client.side_effect = Exception("Connection failed")

    with pytest.raises(Exception, match="Connection failed"):
        mongo_db.MongoDBHandler._initialize_client(valid_db_config)

    assert "Failed to initialize MongoDB client" in caplog.text


def test_initialize_client_with_debug_logging(mocker, valid_db_config, caplog):
    """Test _initialize_client with DEBUG logging enables IdleConnectionMonitor."""
    mock_mongo_client = mocker.patch("simtools.db.mongo_db.MongoClient")
    mock_client_instance = mocker.MagicMock()
    mock_mongo_client.return_value = mock_client_instance

    with caplog.at_level(logging.DEBUG):
        mongo_db.MongoDBHandler._initialize_client(valid_db_config)

    call_args = mock_mongo_client.call_args
    assert "event_listeners" in call_args.kwargs
    assert len(call_args.kwargs["event_listeners"]) == 1
    assert isinstance(call_args.kwargs["event_listeners"][0], mongo_db.IdleConnectionMonitor)


def test_idle_connection_monitor(mocker):
    """Test IdleConnectionMonitor connection_created and connection_closed methods."""
    monitor = mongo_db.IdleConnectionMonitor()

    mock_event_created = mocker.MagicMock()
    mock_event_created.address = ("localhost", 27017)

    mock_event_closed = mocker.MagicMock()
    mock_event_closed.address = ("localhost", 27017)
    mock_event_closed.reason = "idle"

    monitor.connection_created(mock_event_created)
    assert monitor.open_connections == 1

    monitor.connection_closed(mock_event_closed)
    assert monitor.open_connections == 0


def test_is_remote_database_true(valid_db_config):
    """Test is_remote_database with remote server."""
    valid_db_config["db_server"] = "cta-simpipe-protodb.zeuthen.desy.de"
    handler = mongo_db.MongoDBHandler(valid_db_config)
    assert handler.is_remote_database() is True


def test_is_remote_database_false_localhost(valid_db_config):
    """Test is_remote_database with localhost."""
    valid_db_config["db_server"] = "localhost"
    handler = mongo_db.MongoDBHandler(valid_db_config)
    assert handler.is_remote_database() is False


def test_is_remote_database_false_no_config():
    """Test is_remote_database with no configuration."""
    handler = mongo_db.MongoDBHandler(None)
    assert handler.is_remote_database() is False


def test_print_connection_info(valid_db_config, caplog):
    """Test print_connection_info."""
    handler = mongo_db.MongoDBHandler(valid_db_config)
    with caplog.at_level(logging.INFO):
        handler.print_connection_info("test_db")

    assert "Connected to MongoDB at localhost:27017" in caplog.text
    assert "using database: test_db" in caplog.text


def test_print_connection_info_no_config(caplog):
    """Test print_connection_info with no configuration."""
    handler = mongo_db.MongoDBHandler(None)
    with caplog.at_level(logging.INFO):
        handler.print_connection_info("test_db")

    assert "No MongoDB configuration provided." in caplog.text


def test_get_entry_date_from_document():
    """Test get_entry_date_from_document method."""
    from bson.objectid import ObjectId

    test_id = ObjectId()
    document = {"_id": test_id, "some_field": "some_value"}

    result = mongo_db.MongoDBHandler.get_entry_date_from_document(document)

    assert result == test_id.generation_time
    assert result is not None


def test_get_collection(mocker, mongo_handler):
    """Test get_collection method."""
    mock_client = mocker.MagicMock()
    mock_db = mocker.MagicMock()
    mock_collection = mocker.MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    mongo_db.MongoDBHandler.db_client = mock_client

    result = mongo_handler.get_collection("test_collection", "test_db")

    mock_client.__getitem__.assert_called_once_with("test_db")
    mock_db.__getitem__.assert_called_once_with("test_collection")
    assert result == mock_collection


def test_get_collections(mocker, mongo_handler):
    """Test get_collections method."""
    mock_client = mocker.MagicMock()
    mock_db = mocker.MagicMock()
    mock_db.list_collection_names.return_value = ["coll1", "coll2", "fs.files", "fs.chunks"]
    mock_client.__getitem__.return_value = mock_db

    mongo_db.MongoDBHandler.db_client = mock_client

    result = mongo_handler.get_collections("test_db", model_collections_only=False)

    assert result == ["coll1", "coll2", "fs.files", "fs.chunks"]
    assert "test_db" in mongo_handler.list_of_collections


def test_get_collections_model_only(mocker, mongo_handler):
    """Test get_collections with model_collections_only=True."""
    mock_client = mocker.MagicMock()
    mock_db = mocker.MagicMock()
    mock_db.list_collection_names.return_value = ["coll1", "coll2", "fs.files", "fs.chunks"]
    mock_client.__getitem__.return_value = mock_db

    mongo_db.MongoDBHandler.db_client = mock_client

    result = mongo_handler.get_collections("test_db", model_collections_only=True)

    assert result == ["coll1", "coll2"]


def test_list_database_names(mocker, mongo_handler):
    """Test list_database_names method."""
    mock_client = mocker.MagicMock()
    mock_client.list_database_names.return_value = ["db1", "db2", "admin"]

    mongo_db.MongoDBHandler.db_client = mock_client

    result = mongo_handler.list_database_names()

    assert result == ["db1", "db2", "admin"]
    mock_client.list_database_names.assert_called_once()


def test_query_db(mocker, mongo_handler):
    """Test query_db method."""
    mock_collection = mocker.MagicMock()
    mock_collection.find.return_value = [{"param": "value1"}, {"param": "value2"}]

    mocker.patch.object(mongo_handler, "get_collection", return_value=mock_collection)

    query = {"field": "value"}
    result = mongo_handler.query_db(query, "test_collection", "test_db")

    assert len(result) == 2
    assert result[0] == {"param": "value1"}
    mock_collection.find.assert_called_once_with(query)


def test_query_db_no_results(mocker, mongo_handler):
    """Test query_db with no results."""
    mock_collection = mocker.MagicMock()
    mock_collection.find.return_value = []

    mocker.patch.object(mongo_handler, "get_collection", return_value=mock_collection)

    query = {"field": "value"}
    with pytest.raises(ValueError, match="returned zero results"):
        mongo_handler.query_db(query, "test_collection", "test_db")


def test_find_one(mocker, mongo_handler):
    """Test find_one method."""
    mock_collection = mocker.MagicMock()
    mock_collection.find_one.return_value = {"param": "value"}

    mocker.patch.object(mongo_handler, "get_collection", return_value=mock_collection)

    query = {"field": "value"}
    result = mongo_handler.find_one(query, "test_collection", "test_db")

    assert result == {"param": "value"}
    mock_collection.find_one.assert_called_once_with(query)


def test_find_one_no_result(mocker, mongo_handler):
    """Test find_one with no result."""
    mock_collection = mocker.MagicMock()
    mock_collection.find_one.return_value = None

    mocker.patch.object(mongo_handler, "get_collection", return_value=mock_collection)

    query = {"field": "value"}
    result = mongo_handler.find_one(query, "test_collection", "test_db")

    assert result is None


def test_insert_one(mocker, mongo_handler):
    """Test insert_one method."""
    mock_collection = mocker.MagicMock()
    mock_result = mocker.MagicMock()
    mock_collection.insert_one.return_value = mock_result

    mocker.patch.object(mongo_handler, "get_collection", return_value=mock_collection)

    document = {"field": "value"}
    result = mongo_handler.insert_one(document, "test_collection", "test_db")

    assert result == mock_result
    mock_collection.insert_one.assert_called_once_with(document)


def test_get_file_from_db_exists(mocker, mongo_handler):
    """Test get_file_from_db when file exists."""
    mock_gridfs = mocker.patch("simtools.db.mongo_db.gridfs.GridFS")
    mock_fs = mock_gridfs.return_value
    mock_file = mocker.MagicMock()
    mock_fs.exists.return_value = True
    mock_fs.find_one.return_value = mock_file

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    result = mongo_handler.get_file_from_db("test_db", "test_file.dat")

    assert result == mock_file
    mock_fs.exists.assert_called_once_with({"filename": "test_file.dat"})
    mock_fs.find_one.assert_called_once_with({"filename": "test_file.dat"})


def test_get_file_from_db_not_found(mocker, mongo_handler):
    """Test get_file_from_db when file does not exist."""
    mock_gridfs = mocker.patch("simtools.db.mongo_db.gridfs.GridFS")
    mock_fs = mock_gridfs.return_value
    mock_fs.exists.return_value = False

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    with pytest.raises(FileNotFoundError, match="does not exist in the database"):
        mongo_handler.get_file_from_db("test_db", "test_file.dat")


def test_write_file_from_db_to_disk(mocker, tmp_path, mongo_handler):
    """Test write_file_from_db_to_disk."""
    mock_gridfs_bucket = mocker.patch("simtools.db.mongo_db.gridfs.GridFSBucket")
    mock_fs_output = mock_gridfs_bucket.return_value
    mock_file = mocker.MagicMock()
    mock_file.filename = "test_file.dat"

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    mock_open = mocker.patch("builtins.open", mocker.mock_open())

    mongo_handler.write_file_from_db_to_disk("test_db", tmp_path, mock_file)

    mock_open.assert_called_once_with(Path(tmp_path).joinpath("test_file.dat"), "wb")
    mock_fs_output.download_to_stream_by_name.assert_called_once()


def test_get_ecsv_file_as_astropy_table(mocker, mongo_handler):
    """Test get_ecsv_file_as_astropy_table."""
    mock_gridfs_bucket = mocker.patch("simtools.db.mongo_db.gridfs.GridFSBucket")
    fs_instance = mock_gridfs_bucket.return_value

    ecsv_content = (
        b"# %ECSV 0.9\n# ---\n# datatype:\n# - {name: x, datatype: float64}\n# schema: {}\nx\n1.0\n"
    )

    def download_side_effect(filename, buf):
        buf.write(ecsv_content)

    fs_instance.download_to_stream_by_name.side_effect = download_side_effect

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    result = mongo_handler.get_ecsv_file_as_astropy_table("test_file.ecsv", "test_db")

    assert result is not None
    assert len(result) == 1


def test_get_ecsv_file_as_astropy_table_not_found(mocker, mongo_handler):
    """Test get_ecsv_file_as_astropy_table when file not found."""
    mock_gridfs_bucket = mocker.patch("simtools.db.mongo_db.gridfs.GridFSBucket")
    fs_instance = mock_gridfs_bucket.return_value

    import gridfs

    fs_instance.download_to_stream_by_name.side_effect = gridfs.errors.NoFile("no such file")

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    with pytest.raises(FileNotFoundError, match="not found in DB"):
        mongo_handler.get_ecsv_file_as_astropy_table("test_file.ecsv", "test_db")


def test_insert_file_to_db_new_file(mocker, tmp_path, mongo_handler):
    """Test insert_file_to_db with a new file."""
    test_file = tmp_path / "test_file.dat"
    test_file.write_text("test content", encoding="utf-8")

    mock_gridfs = mocker.patch("simtools.db.mongo_db.gridfs.GridFS")
    mock_fs = mock_gridfs.return_value
    mock_fs.exists.return_value = False
    mock_fs.put.return_value = "new_file_id"

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    mocker.patch("simtools.db.mongo_db.ascii_handler.is_utf8_file", return_value=True)

    result = mongo_handler.insert_file_to_db(str(test_file), "test_db")

    assert result == "new_file_id"
    mock_fs.exists.assert_called_once()
    mock_fs.put.assert_called_once()


def test_insert_file_to_db_existing_file(mocker, tmp_path, mongo_handler):
    """Test insert_file_to_db with an existing file."""
    test_file = tmp_path / "test_file.dat"
    test_file.write_text("test content", encoding="utf-8")

    mock_gridfs = mocker.patch("simtools.db.mongo_db.gridfs.GridFS")
    mock_fs = mock_gridfs.return_value
    mock_fs.exists.return_value = True
    mock_file_instance = mocker.MagicMock()
    mock_file_instance._id = "existing_file_id"
    mock_fs.find_one.return_value = mock_file_instance

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    result = mongo_handler.insert_file_to_db(str(test_file), "test_db")

    assert result == "existing_file_id"
    mock_fs.exists.assert_called_once()
    mock_fs.put.assert_not_called()


def test_insert_file_to_db_non_utf8(mocker, tmp_path, mongo_handler):
    """Test insert_file_to_db with non-UTF8 file."""
    test_file = tmp_path / "test_file.dat"
    test_file.write_text("test content", encoding="utf-8")

    mock_gridfs = mocker.patch("simtools.db.mongo_db.gridfs.GridFS")
    mock_fs = mock_gridfs.return_value
    mock_fs.exists.return_value = False

    mock_client = mocker.MagicMock()
    mongo_db.MongoDBHandler.db_client = mock_client

    mocker.patch("simtools.db.mongo_db.ascii_handler.is_utf8_file", return_value=False)

    with pytest.raises(ValueError, match="File is not UTF-8 encoded"):
        mongo_handler.insert_file_to_db(str(test_file), "test_db")


def test_generate_compound_indexes(mocker, mongo_handler):
    """Test generate_compound_indexes method."""
    mock_collection = mocker.MagicMock()
    mocker.patch.object(mongo_handler, "get_collection", return_value=mock_collection)

    mongo_handler.generate_compound_indexes("test_db")

    assert mock_collection.create_index.call_count == 6


def test_generate_compound_indexes_for_databases(mocker, mongo_handler):
    """Test generate_compound_indexes_for_databases method."""
    mock_client = mocker.MagicMock()
    mock_client.list_database_names.return_value = [
        "config",
        "admin",
        "local",
        "test_db",
        "other_db",
    ]
    mongo_db.MongoDBHandler.db_client = mock_client

    mocker.patch.object(mongo_handler, "generate_compound_indexes")

    mongo_handler.generate_compound_indexes_for_databases("test_db", "test_model", "1.0.0")

    mongo_handler.generate_compound_indexes.assert_called_once_with(db_name="test_db")


def test_generate_compound_indexes_for_databases_all(mocker, mongo_handler):
    """Test generate_compound_indexes_for_databases with 'all'."""
    mock_client = mocker.MagicMock()
    mock_client.list_database_names.return_value = [
        "config",
        "admin",
        "local",
        "db1",
        "db2",
    ]
    mongo_db.MongoDBHandler.db_client = mock_client

    mocker.patch.object(mongo_handler, "generate_compound_indexes")

    mongo_handler.generate_compound_indexes_for_databases("all", "test_model", "1.0.0")

    assert mongo_handler.generate_compound_indexes.call_count == 2


def test_generate_compound_indexes_for_databases_not_found(mocker, mongo_handler):
    """Test generate_compound_indexes_for_databases with non-existent database."""
    mock_client = mocker.MagicMock()
    mock_client.list_database_names.return_value = ["config", "admin", "local", "existing_db"]
    mongo_db.MongoDBHandler.db_client = mock_client

    with pytest.raises(ValueError, match="not found"):
        mongo_handler.generate_compound_indexes_for_databases("non_existent", "test_model", "1.0.0")
