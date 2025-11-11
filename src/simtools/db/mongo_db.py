"""MongoDB database handler for direct database operations."""

import io
import logging
import re
from pathlib import Path
from threading import Lock

import gridfs
import jsonschema
from astropy.table import Table
from bson.objectid import ObjectId
from pymongo import MongoClient, monitoring

from simtools.io import ascii_handler

logging.getLogger("pymongo").setLevel(logging.WARNING)


jsonschema_db_dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema#",
    "type": "object",
    "description": "MongoDB configuration",
    "properties": {
        "db_server": {"type": "string", "description": "DB server address"},
        "db_api_port": {
            "type": "integer",
            "minimum": 1,
            "maximum": 65535,
            "default": 27017,
            "description": "Port to use",
        },
        "db_api_user": {"type": "string", "description": "API username"},
        "db_api_pw": {"type": "string", "description": "Password for the API user"},
        "db_api_authentication_database": {
            "type": ["string", "null"],
            "default": "admin",
            "description": "DB with user info (optional)",
        },
        "db_simulation_model": {
            "type": "string",
            "description": "Name of simulation model database",
        },
        "db_simulation_model_version": {
            "type": "string",
            "description": "Version of simulation model database",
        },
    },
    "required": [
        "db_server",
        "db_api_port",
        "db_api_user",
        "db_api_pw",
        "db_simulation_model",
        "db_simulation_model_version",
    ],
}


class IdleConnectionMonitor(monitoring.ConnectionPoolListener):
    """
    A listener to track MongoDB connection pool activity.

    Used to monitor idle connections and log connection events.
    Switched on in debug mode.
    """

    def __init__(self):
        self._logger = logging.getLogger("IdleConnectionMonitor")
        self.open_connections = 0

    def connection_created(self, event):
        """Handle connection creation event."""
        self.open_connections += 1
        self._logger.debug(
            f"MongoDB connection Created: {event.address}. Total in Pool: {self.open_connections}"
        )

    def connection_closed(self, event):
        """Handle connection closure event."""
        self.open_connections -= 1
        self._logger.debug(
            f"MongoDB connection Closed: {event.address}. Reason: {event.reason}. "
            f"Total in Pool: {self.open_connections}"
        )

    def connection_check_out_started(self, event):
        """Handle connection check out started event."""

    def connection_check_out_failed(self, event):
        """Handle connection check out failure event."""

    def connection_checked_out(self, event):
        """Handle connection checked out event."""

    def connection_checked_in(self, event):
        """Handle connection checked in event."""

    def connection_ready(self, event):
        """Handle connection ready event."""

    def pool_created(self, event):
        """Handle connection pool creation event."""

    def pool_ready(self, event):
        """Handle connection pool ready event."""

    def pool_cleared(self, event):
        """Handle connection pool cleared event."""

    def pool_closed(self, event):
        """Handle connection pool closure event."""


class MongoDBHandler:  # pylint: disable=unsubscriptable-object
    """
    MongoDBHandler provides low-level interface to MongoDB operations.

    This class handles direct MongoDB operations including connection management,
    database queries, file operations via GridFS, and index generation.

    Parameters
    ----------
    db_config: dict
        Dictionary with the MongoDB configuration (see jsonschema_db_dict for details).
    """

    db_client: MongoClient = None
    _lock = Lock()
    _logger = logging.getLogger(__name__)

    def __init__(self, db_config=None):
        """Initialize the MongoDBHandler class."""
        self.db_config = MongoDBHandler.validate_db_config(db_config)
        self.list_of_collections = {}

        if self.db_config:
            self._initialize_client(self.db_config)

    @classmethod
    def _initialize_client(cls, db_config):
        """
        Initialize the MongoDB client in a thread-safe manner.

        Only initializes if it hasn't been done yet. Uses double-checked locking
        to ensure thread safety.

        Parameters
        ----------
        db_config: dict
            Dictionary with the MongoDB configuration.
        """
        if cls.db_client is not None:
            return
        with cls._lock:
            if cls.db_client is None:
                try:
                    uri = cls._build_uri(db_config)
                    client_kwargs = {"maxIdleTimeMS": 10000}

                    if cls._logger.isEnabledFor(logging.DEBUG):
                        client_kwargs["event_listeners"] = [IdleConnectionMonitor()]

                    cls.db_client = MongoClient(uri, **client_kwargs)
                    cls._logger.debug("MongoDB client initialized successfully.")
                except Exception as e:
                    cls._logger.error(f"Failed to initialize MongoDB client: {e}")
                    raise

    @staticmethod
    def _build_uri(db_config):
        """
        Build MongoDB URI from configuration.

        Parameters
        ----------
        db_config: dict
            Dictionary with the MongoDB configuration.

        Returns
        -------
        str
            MongoDB connection URI.
        """
        direct_connection = db_config["db_server"] in (
            "localhost",
            "simtools-mongodb",
            "mongodb",
        )
        auth_source = (
            db_config.get("db_api_authentication_database")
            if db_config.get("db_api_authentication_database")
            else "admin"
        )

        username = db_config["db_api_user"]
        password = db_config["db_api_pw"]
        server = db_config["db_server"]
        port = db_config["db_api_port"]

        uri_base = f"mongodb://{username}:{password}@{server}:{port}/"
        params = [f"authSource={auth_source}"]

        if direct_connection:
            params.append("directConnection=true")
        else:
            params.append("ssl=true")
            params.append("tlsAllowInvalidHostnames=true")
            params.append("tlsAllowInvalidCertificates=true")

        return f"{uri_base}?{'&'.join(params)}"

    @staticmethod
    def validate_db_config(db_config):
        """
        Validate the MongoDB configuration.

        Parameters
        ----------
        db_config: dict
            Dictionary with the MongoDB configuration.

        Returns
        -------
        dict or None
            Validated MongoDB configuration or None if no valid config provided.

        Raises
        ------
        ValueError
            If the MongoDB configuration is invalid.
        """
        if db_config is None or all(value is None for value in db_config.values()):
            return None
        try:
            jsonschema.validate(instance=db_config, schema=jsonschema_db_dict)
            return db_config
        except jsonschema.exceptions.ValidationError as err:
            raise ValueError("Invalid MongoDB configuration") from err

    @staticmethod
    def get_db_name(db_name=None, db_simulation_model_version=None, model_name=None):
        """
        Build DB name from configuration.

        Parameters
        ----------
        db_name: str
            Direct database name (if provided, returns this).
        db_simulation_model_version: str
            Version of the simulation model.
        model_name: str
            Name of the simulation model.

        Returns
        -------
        str or None
            Database name.
        """
        if db_name:
            return db_name
        if db_simulation_model_version and model_name:
            return f"{model_name}-{db_simulation_model_version.replace('.', '-')}"
        return None

    def print_connection_info(self, db_name):
        """
        Print the connection information.

        Parameters
        ----------
        db_name: str
            Name of the database.
        """
        if self.db_config:
            self._logger.info(
                f"Connected to MongoDB at {self.db_config['db_server']}:"
                f"{self.db_config['db_api_port']} "
                f"using database: {db_name}"
            )
        else:
            self._logger.info("No MongoDB configuration provided.")

    def is_remote_database(self):
        """
        Check if the database is remote.

        Check for domain pattern like "cta-simpipe-protodb.zeuthen.desy.de"

        Returns
        -------
        bool
            True if the database is remote, False otherwise.
        """
        if self.db_config:
            db_server = self.db_config["db_server"]
            domain_pattern = r"^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$"
            return bool(re.match(domain_pattern, db_server))
        return False

    @staticmethod
    def get_entry_date_from_document(document):
        """
        Extract entry date from a MongoDB document's ObjectId.

        Parameters
        ----------
        document: dict
            MongoDB document with '_id' field.

        Returns
        -------
        datetime.datetime
            The generation time of the document's ObjectId.
        """
        return ObjectId(document["_id"]).generation_time

    def get_collection(self, collection_name, db_name):
        """
        Get a collection from the DB.

        Parameters
        ----------
        collection_name: str
            Name of the collection.
        db_name: str
            Name of the DB.

        Returns
        -------
        pymongo.collection.Collection
            The collection from the DB.
        """
        return MongoDBHandler.db_client[db_name][collection_name]

    def get_collections(self, db_name, model_collections_only=False):
        """
        List of collections in the DB.

        Parameters
        ----------
        db_name: str
            Database name.
        model_collections_only: bool
            If True, only return model collections (i.e. exclude fs.files, fs.chunks)

        Returns
        -------
        list
            List of collection names
        """
        if db_name not in self.list_of_collections:
            self.list_of_collections[db_name] = MongoDBHandler.db_client[
                db_name
            ].list_collection_names()
        collections = self.list_of_collections[db_name]
        if model_collections_only:
            return [collection for collection in collections if not collection.startswith("fs.")]
        return collections

    def list_database_names(self):
        """
        Get list of database names.

        Returns
        -------
        list
            List of database names.
        """
        return MongoDBHandler.db_client.list_database_names()

    def generate_compound_indexes_for_databases(
        self, db_name, db_simulation_model, db_simulation_model_version
    ):
        """
        Generate compound indexes for several databases.

        Parameters
        ----------
        db_name: str
            Name of the database.
        db_simulation_model: str
            Name of the simulation model.
        db_simulation_model_version: str
            Version of the simulation model.

        Raises
        ------
        ValueError
            If the requested database is not found.
        """
        databases = [
            d
            for d in MongoDBHandler.db_client.list_database_names()
            if d not in ("config", "admin", "local")
        ]
        requested = self.get_db_name(
            db_name=db_name,
            db_simulation_model_version=db_simulation_model_version,
            model_name=db_simulation_model,
        )
        if requested != "all" and requested not in databases:
            raise ValueError(
                f"Requested database '{requested}' not found. "
                f"Following databases are available: {', '.join(databases)}"
            )

        databases = databases if requested == "all" else [requested]
        for dbs in databases:
            self._logger.info(f"Generating compound indexes for database: {dbs}")
            self.generate_compound_indexes(db_name=dbs)

    def generate_compound_indexes(self, db_name):
        """
        Generate compound indexes for the MongoDB collections.

        Indexes based on the typical query patterns.

        Parameters
        ----------
        db_name: str
            Name of the database.
        """
        collection_names = [
            "telescopes",
            "sites",
            "configuration_sim_telarray",
            "configuration_corsika",
            "calibration_devices",
        ]
        for collection_name in collection_names:
            db_collection = self.get_collection(collection_name, db_name=db_name)
            db_collection.create_index(
                [("instrument", 1), ("site", 1), ("parameter", 1), ("parameter_version", 1)]
            )
        db_collection = self.get_collection("production_tables", db_name=db_name)
        db_collection.create_index([("collection", 1), ("model_version", 1)])

    def query_db(self, query, collection_name, db_name):
        """
        Query MongoDB and return results as list.

        Parameters
        ----------
        query: dict
            Query to execute.
        collection_name: str
            Collection name.
        db_name: str
            Database name.

        Returns
        -------
        list
            List of documents matching the query.

        Raises
        ------
        ValueError
            if query returned no results.
        """
        collection = self.get_collection(collection_name, db_name=db_name)
        posts = list(collection.find(query))
        if not posts:
            raise ValueError(
                f"The following query for {collection_name} returned zero results: {query} "
            )
        return posts

    def find_one(self, query, collection_name, db_name):
        """
        Query MongoDB and return first result.

        Parameters
        ----------
        query: dict
            Query to execute.
        collection_name: str
            Collection name.
        db_name: str
            Database name.

        Returns
        -------
        dict or None
            First document matching the query or None.
        """
        collection = self.get_collection(collection_name, db_name=db_name)
        return collection.find_one(query)

    def insert_one(self, document, collection_name, db_name):
        """
        Insert a document into a collection.

        Parameters
        ----------
        document: dict
            Document to insert.
        collection_name: str
            Collection name.
        db_name: str
            Database name.

        Returns
        -------
        InsertOneResult
            Result of the insert operation.
        """
        collection = self.get_collection(collection_name, db_name=db_name)
        return collection.insert_one(document)

    def get_file_from_db(self, db_name, file_name):
        """
        Extract a file from MongoDB and return GridFS file instance.

        Parameters
        ----------
        db_name: str
            The name of the DB with files of tabulated data
        file_name: str
            The name of the file requested

        Returns
        -------
        GridOut
            A file instance returned by GridFS find_one

        Raises
        ------
        FileNotFoundError
            If the desired file is not found.
        """
        db = MongoDBHandler.db_client[db_name]
        file_system = gridfs.GridFS(db)
        if file_system.exists({"filename": file_name}):
            return file_system.find_one({"filename": file_name})

        raise FileNotFoundError(f"The file {file_name} does not exist in the database {db_name}")

    def write_file_from_db_to_disk(self, db_name, path, file):
        """
        Extract a file from MongoDB and write it to disk.

        Parameters
        ----------
        db_name: str
            The name of the DB with files of tabulated data
        path: str or Path
            The path to write the file to
        file: GridOut
            A file instance returned by GridFS find_one
        """
        db = MongoDBHandler.db_client[db_name]
        fs_output = gridfs.GridFSBucket(db)
        with open(Path(path).joinpath(file.filename), "wb") as output_file:
            fs_output.download_to_stream_by_name(file.filename, output_file)

    def get_ecsv_file_as_astropy_table(self, file_name, db_name):
        """
        Read contents of an ECSV file from the database and return it as an Astropy Table.

        Files are not written to disk.

        Parameters
        ----------
        file_name: str
            The name of the ECSV file.
        db_name: str
            The name of the database.

        Returns
        -------
        astropy.table.Table
            The contents of the ECSV file as an Astropy Table.
        """
        db = MongoDBHandler.db_client[db_name]
        fs = gridfs.GridFSBucket(db)

        buf = io.BytesIO()
        try:
            fs.download_to_stream_by_name(file_name, buf)
        except gridfs.errors.NoFile as exc:
            raise FileNotFoundError(f"ECSV file '{file_name}' not found in DB.") from exc
        buf.seek(0)
        return Table.read(buf.getvalue().decode("utf-8"), format="ascii.ecsv")

    def insert_file_to_db(self, file_name, db_name, **kwargs):
        """
        Insert a file to the DB.

        Parameters
        ----------
        file_name: str or Path
            The name of the file to insert (full path).
        db_name: str
            The name of the DB
        **kwargs (optional): keyword arguments for file creation.
            The full list of arguments can be found in
            https://www.mongodb.com/docs/manual/core/gridfs/

        Returns
        -------
        file_id: GridOut._id
            If the file exists, return its GridOut._id, otherwise insert the file and return
            its newly created DB GridOut._id.
        """
        db = MongoDBHandler.db_client[db_name]
        file_system = gridfs.GridFS(db)

        kwargs.setdefault("content_type", "ascii/dat")
        kwargs.setdefault("filename", Path(file_name).name)

        if file_system.exists({"filename": kwargs["filename"]}):
            self._logger.warning(
                f"The file {kwargs['filename']} exists in the DB. Returning its ID"
            )
            # _id is a public attribute in GridFS GridOut objects
            # pylint: disable=protected-access
            return file_system.find_one({"filename": kwargs["filename"]})._id

        if not ascii_handler.is_utf8_file(file_name):
            raise ValueError(f"File is not UTF-8 encoded: {file_name}")

        self._logger.debug(f"Writing file to DB: {file_name}")
        with open(file_name, "rb") as data_file:
            return file_system.put(data_file, **kwargs)
