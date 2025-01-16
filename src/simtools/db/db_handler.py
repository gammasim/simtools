"""Module to handle interaction with DB."""

import logging
import re
from pathlib import Path
from threading import Lock

import gridfs
import jsonschema
from bson.objectid import ObjectId
from packaging.version import Version
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import BulkWriteError

from simtools.data_model import validate_data
from simtools.io_operations import io_handler
from simtools.utils import names, value_conversion

__all__ = ["DatabaseHandler"]

logging.getLogger("pymongo").setLevel(logging.WARNING)


# pylint: disable=unsubscriptable-object
# The above comment is because pylint does not know that DatabaseHandler.db_client is subscriptable


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
            "type": "string",
            "default": "admin",
            "description": "DB with user info (optional)",
        },
        "db_simulation_model": {
            "type": "string",
            "description": "Name of simulation model database",
        },
    },
    "required": ["db_server", "db_api_port", "db_api_user", "db_api_pw", "db_simulation_model"],
}


class DatabaseHandler:
    """
    DatabaseHandler provides the interface to the DB.

    Parameters
    ----------
    mongo_db_config: dict
        Dictionary with the MongoDB configuration (see jsonschema_db_dict for details).
    """

    DB_CTA_SIMULATION_MODEL_DESCRIPTIONS = "CTA-Simulation-Model-Descriptions"
    # DB collection with updates field names
    DB_DERIVED_VALUES = "Staging-CTA-Simulation-Model-Derived-Values"

    ALLOWED_FILE_EXTENSIONS = [".dat", ".txt", ".lis", ".cfg", ".yml", ".yaml", ".ecsv"]

    db_client = None
    production_table_cached = {}
    site_parameters_cached = {}
    model_parameters_cached = {}
    model_versions_cached = {}
    corsika_configuration_parameters_cached = {}

    def __init__(self, mongo_db_config=None):
        """Initialize the DatabaseHandler class."""
        self._logger = logging.getLogger(__name__)

        self.mongo_db_config = self._validate_mongo_db_config(mongo_db_config)
        self.io_handler = io_handler.IOHandler()
        self.list_of_collections = {}

        self._set_up_connection()
        self._find_latest_simulation_model_db()

    def _set_up_connection(self):
        """Open the connection to MongoDB."""
        if self.mongo_db_config and DatabaseHandler.db_client is None:
            lock = Lock()
            with lock:
                DatabaseHandler.db_client = self._open_mongo_db()

    def _validate_mongo_db_config(self, mongo_db_config):
        """Validate the MongoDB configuration."""
        if mongo_db_config is None or all(value is None for value in mongo_db_config.values()):
            return None
        try:
            jsonschema.validate(instance=mongo_db_config, schema=jsonschema_db_dict)
            return mongo_db_config
        except jsonschema.exceptions.ValidationError as err:
            raise ValueError("Invalid MongoDB configuration") from err

    def _open_mongo_db(self):
        """
        Open a connection to MongoDB and return the client to read/write to the DB with.

        Returns
        -------
        A PyMongo DB client

        Raises
        ------
        KeyError
            If the DB configuration is invalid
        """
        direct_connection = self.mongo_db_config["db_server"] in (
            "localhost",
            "simtools-mongodb",
        )
        return MongoClient(
            self.mongo_db_config["db_server"],
            port=self.mongo_db_config["db_api_port"],
            username=self.mongo_db_config["db_api_user"],
            password=self.mongo_db_config["db_api_pw"],
            authSource=self.mongo_db_config.get("db_api_authentication_database", "admin"),
            directConnection=direct_connection,
            ssl=not direct_connection,
            tlsallowinvalidhostnames=True,
            tlsallowinvalidcertificates=True,
        )

    def _find_latest_simulation_model_db(self):
        """
        Find the latest released version of the simulation model and update the DB config.

        This is indicated by adding "LATEST" to the name of the simulation model database
        (field "db_simulation_model" in the database configuration dictionary).
        Only released versions are considered, pre-releases are ignored.

        Raises
        ------
        ValueError
            If the "LATEST" version is requested but no versions are found in the DB.

        """
        try:
            db_simulation_model = self.mongo_db_config["db_simulation_model"]
            if not db_simulation_model.endswith("LATEST"):
                return
        except TypeError:  # if db_simulation_model is None
            return

        prefix = db_simulation_model.replace("LATEST", "")
        list_of_db_names = self.db_client.list_database_names()
        filtered_list_of_db_names = [s for s in list_of_db_names if s.startswith(prefix)]
        versioned_strings = []
        version_pattern = re.compile(
            rf"{re.escape(prefix)}v?(\d+)-(\d+)-(\d+)(?:-([a-zA-Z0-9_.]+))?"
        )

        for s in filtered_list_of_db_names:
            match = version_pattern.search(s)
            # A version is considered a pre-release if it contains a '-' character (re group 4)
            if match and match.group(4) is None:
                version_str = match.group(1) + "." + match.group(2) + "." + match.group(3)
                version = Version(version_str)
                versioned_strings.append((s, version))

        if versioned_strings:
            latest_string, _ = max(versioned_strings, key=lambda x: x[1])
            self.mongo_db_config["db_simulation_model"] = latest_string
            self._logger.info(
                f"Updated the DB simulation model to the latest version {latest_string}"
            )
        else:
            raise ValueError("Found LATEST in the DB name but no matching versions found in DB.")

    def get_model_parameter(
        self,
        parameter,
        parameter_version,
        site,
        array_element_name,
        collection,
    ):
        """
        Get a single model parameter (using the parameter version).

        Parameters
        ----------
        parameter: str
            Name of the parameter.
        parameter_version: str
            Version of the parameter.
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        collection: str
            Collection of array element (e.g. telescopes, calibration_devices).

        Returns
        -------
        dict containing the parameter

        """
        query = {
            "parameter_version": parameter_version,
            "parameter": parameter,
        }
        if array_element_name is not None:
            query["instrument"] = array_element_name
        if site is not None:
            query["site"] = site
        return self.read_mongo_db(
            query=query,
            collection_name=collection,
            write_files=False,
        )

    def get_model_parameters(
        self,
        site,
        array_element_name,
        model_version,
        collection,
        allow_missing_array_elements=False,
    ):
        """
        Get model parameters from MongoDB.

        An array element can be e.g., a telescope or a calibration device.
        Always queries parameters for design and for the specified array element (if necessary).

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. LSTN-01, MSTS-design).
        model_version: str
            Version of the model.
        collection: str
            collection of array element (e.g. telescopes, calibration_devices).
        allow_missing_array_elements: bool
            Allow missing array elements in the DB without raising an error.

        Returns
        -------
        dict containing the parameters
        """
        production_table = self.get_production_table_from_mongo_db(collection, model_version)
        design_model = f"{names.get_array_element_type_from_name(array_element_name)}-design"
        array_element_list = (
            [array_element_name]
            if "-design" in array_element_name
            else [design_model, array_element_name]
        )

        pars = {}
        for array_element in array_element_list:  # design model must be read first
            cache_key = self._cache_key(
                names.validate_site_name(site), array_element, model_version, collection
            )
            pars.update(DatabaseHandler.model_parameters_cached.get(cache_key, {}))
            try:
                parameter_version_table = production_table["parameters"][array_element]
            except KeyError as exc:
                if array_element == design_model and not allow_missing_array_elements:
                    self._logger.error(f"Parameters for {array_element} could not be found.")
                    raise exc
                # non-design model not defined (e.g. in collection 'configuration_sim_telarray')
                continue
            pars.update(
                self.read_mongo_db(
                    query=self._get_query_from_parameter_version_table(
                        parameter_version_table, array_element
                    ),
                    collection_name=collection,
                    write_files=False,
                )
            )
            DatabaseHandler.model_parameters_cached[cache_key] = pars

        return pars

    def get_collection(self, db_name, collection_name):
        """
        Get a collection from the DB.

        Parameters
        ----------
        db_name: str
            Name of the DB.
        collection_name: str
            Name of the collection.

        Returns
        -------
        pymongo.collection.Collection
            The collection from the DB.

        """
        db_name = self._get_db_name(db_name)
        return DatabaseHandler.db_client[db_name][collection_name]

    def export_file_db(self, db_name, dest, file_name):
        """
        Get file from the DB and write to disk.

        Parameters
        ----------
        db_name: str
            Name of the DB to search in.
        dest: str or Path
            Location where to write the file to.
        file_name: str
            Name of the file to get.

        Returns
        -------
        file_id: GridOut._id
            the database ID the file was assigned when it was inserted to the DB.

        Raises
        ------
        FileNotFoundError
            If the desired file is not found.

        """
        db_name = self._get_db_name(db_name)

        self._logger.debug(f"Getting {file_name} from {db_name} and writing it to {dest}")
        file_path_instance = self._get_file_mongo_db(db_name, file_name)
        self._write_file_from_mongo_to_disk(db_name, dest, file_path_instance)
        return file_path_instance._id  # pylint: disable=protected-access;

    def export_model_files(self, parameters, dest):
        """
        Export all the files in a model from the DB and write them to disk.

        Parameters
        ----------
        parameters: dict
            Dict of model parameters
        dest: str or Path
            Location where to write the files to.

        Raises
        ------
        FileNotFoundError
            if a file in parameters.values is not found

        """
        if self.mongo_db_config:
            for info in parameters.values():
                if not info or not info.get("file") or info["value"] is None:
                    continue
                if Path(dest).joinpath(info["value"]).exists():
                    continue
                file = self._get_file_mongo_db(self._get_db_name(), info["value"])
                self._write_file_from_mongo_to_disk(self._get_db_name(), dest, file)

    @staticmethod
    def _is_file(value):
        """Verify if a parameter value is a file name."""
        return any(ext in str(value) for ext in DatabaseHandler.ALLOWED_FILE_EXTENSIONS)

    def _get_query_from_parameter_version_table(self, parameter_version_table, array_element_name):
        """Return query based on parameter version table."""
        return {
            "instrument": array_element_name,
            "$or": [
                {"parameter": param, "parameter_version": version}
                for param, version in parameter_version_table.items()
            ],
        }

    def read_mongo_db(
        self,
        query,
        collection_name,
        run_location=None,
        write_files=True,
    ):
        """
        Build and execute query to Read the MongoDB for a specific array element.

        Also writes the files listed in the parameter values into the sim_telarray run location

        Parameters
        ----------
        query: dict
            Dictionary describing the query to execute.
        run_location: Path or str
            The sim_telarray run location to write the tabulated data files into.
        collection_name: str
            The name of the collection to read from.
        write_files: bool
            If true, write the files to the run_location.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned no results or if the collection is not found in the production table.
        """
        db_name = self._get_db_name()
        collection = self.get_collection(db_name, collection_name)
        posts = list(collection.find(query).sort("parameter", ASCENDING))
        if not posts:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        parameters = {}
        for post in posts:
            par_now = post["parameter"]
            parameters[par_now] = post
            parameters[par_now].pop("parameter", None)
            parameters[par_now]["entry_date"] = ObjectId(post["_id"]).generation_time
            if parameters[par_now]["file"] and write_files:
                file = self._get_file_mongo_db(db_name, parameters[par_now]["value"])
                self._write_file_from_mongo_to_disk(db_name, run_location, file)

        return parameters

    def get_site_parameters(self, site, model_version):
        """
        Get site parameters from MongoDB.

        Parameters
        ----------
        site: str
            Site name.
        model_version: str
            Version of the model.

        Returns
        -------
        dict containing the parameters
        """
        site = names.validate_site_name(site)
        production_table = self.get_production_table_from_mongo_db("sites", model_version)
        cache_key = self._cache_key(site, None, production_table.get("model_version"))
        try:
            return DatabaseHandler.site_parameters_cached[cache_key]
        except KeyError:
            pass

        try:
            parameter_query = production_table["parameters"][f"OBS-{site}"]
        except KeyError as exc:
            raise ValueError(f"Site {site} not found in the production table") from exc
        query = {
            "site": site,
            "$or": [
                {"parameter": param, "parameter_version": version}
                for param, version in parameter_query.items()
            ],
        }
        DatabaseHandler.site_parameters_cached[cache_key] = self.read_mongo_db(
            query=query, collection_name="sites", write_files=False
        )
        return DatabaseHandler.site_parameters_cached[cache_key]

    def get_production_table_from_mongo_db(self, collection_name, model_version):
        """
        Get production table from MongoDB.

        Parameters
        ----------
        collection_name: str
            Name of the collection.
        model_version: str
            Version of the model.
        """
        try:
            return DatabaseHandler.production_table_cached[
                self._cache_key(None, None, model_version, collection_name)
            ]
        except KeyError:
            pass

        query = {"model_version": model_version, "collection": collection_name}
        collection = self.get_collection(self._get_db_name(), "production_tables")
        post = collection.find_one(query, sort=[("_id", DESCENDING)])
        if not post:
            raise ValueError(f"The following query returned zero results: {query}")

        return {
            "collection": post["collection"],
            "model_version": post["model_version"],
            "parameters": post["parameters"],
            "entry_date": ObjectId(post["_id"]).generation_time,
        }

    def get_array_elements_of_type(self, array_element_type, model_version, collection):
        """
        Get all array elements of a certain type (e.g. 'LSTN') from a collection in the DB.

        Parameters
        ----------
        array_element_type: str
            Type of the array element (e.g. LSTN, MSTS).
        model_version: str
            Version of the model.
        collection: str
            Which collection to get the array elements from:
            i.e. telescopes, calibration_devices.

        Returns
        -------
        list
            Sorted list of all array element names found in collection
        """
        production_table = self.get_production_table_from_mongo_db(collection, model_version)
        all_array_elements = production_table["parameters"]
        return sorted(
            [
                entry
                for entry in all_array_elements
                if entry.startswith(array_element_type) and "-design" not in entry
            ]
        )

    def get_derived_values(self, array_element_name, model_version):
        """
        Get all derived values from the DB for a specific array element.

        Parameters
        ----------
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        model_version: str
            Version of the model.

        Returns
        -------
        dict containing the parameters

        """
        array_element_name = (
            names.validate_array_element_name(array_element_name) if array_element_name else None
        )

        return self.read_mongo_db(
            DatabaseHandler.DB_DERIVED_VALUES,
            array_element_name,
            model_version,
            run_location=None,
            collection_name="derived_values",
            write_files=False,
        )

    def get_simulation_configuration_parameters(
        self, simulation_software, site, array_element_name, model_version
    ):
        """
        Get simulation configuration parameters from the DB.

        Parameters
        ----------
        simulation_software: str
            Name of the simulation software.
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        model_version: str
            Version of the model.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if simulation_software is not valid.
        """
        if simulation_software == "corsika":
            return self.get_corsika_configuration_parameters(model_version)
        if simulation_software == "simtel":
            return (
                # not all array elements are present in the DB (allow for missing elements)
                self.get_model_parameters(
                    site,
                    array_element_name,
                    model_version,
                    collection="configuration_sim_telarray",
                    allow_missing_array_elements=True,
                )
                if site and array_element_name
                else {}
            )
        raise ValueError(f"Unknown simulation software: {simulation_software}")

    def get_corsika_configuration_parameters(self, model_version):
        """
        Get CORSIKA configuration parameters from the DB.

        Parameters
        ----------
        model_version : str
            Version of the model.

        Returns
        -------
        dict
            Configuration parameters for CORSIKA
        """
        _production_table = self.get_production_table_from_mongo_db(
            "configuration_corsika", model_version
        )
        cache_key = self._cache_key(None, None, _production_table.get("model_version"))

        try:
            return DatabaseHandler.corsika_configuration_parameters_cached[cache_key]
        except KeyError:
            pass

        DatabaseHandler.corsika_configuration_parameters_cached[cache_key] = self.read_mongo_db(
            query=self._get_query_from_parameter_version_table(
                _production_table["parameters"], None
            ),
            collection_name="configuration_corsika",
            write_files=False,
        )
        return DatabaseHandler.corsika_configuration_parameters_cached[cache_key]

    @staticmethod
    def _get_file_mongo_db(db_name, file_name):
        """
        Extract a file from MongoDB and return GridFS file instance.

        Parameters
        ----------
        db_name: str
            the name of the DB with files of tabulated data
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
        db = DatabaseHandler.db_client[db_name]
        file_system = gridfs.GridFS(db)
        if file_system.exists({"filename": file_name}):
            return file_system.find_one({"filename": file_name})

        raise FileNotFoundError(f"The file {file_name} does not exist in the database {db_name}")

    @staticmethod
    def _write_file_from_mongo_to_disk(db_name, path, file):
        """
        Extract a file from MongoDB and write it to disk.

        Parameters
        ----------
        db_name: str
            the name of the DB with files of tabulated data
        path: str or Path
            The path to write the file to
        file: GridOut
            A file instance returned by GridFS find_one
        """
        db = DatabaseHandler.db_client[db_name]
        fs_output = gridfs.GridFSBucket(db)
        with open(Path(path).joinpath(file.filename), "wb") as output_file:
            fs_output.download_to_stream_by_name(file.filename, output_file)

    def copy_array_element(
        self,
        db_name,
        element_to_copy,
        version_to_copy,
        new_array_element_name,
        collection_name="telescopes",
        db_to_copy_to=None,
        collection_to_copy_to=None,
    ):
        """
        Copy a full array element configuration to a new array element name.

        Only a specific version is copied.
        This function should be rarely used and is intended to simplify unit tests.

        Parameters
        ----------
        db_name: str
            the name of the DB to copy from
        element_to_copy: str
            The array element to copy
        version_to_copy: str
            The version of the configuration to copy
        new_array_element_name: str
            The name of the new array element
        collection_name: str
            The name of the collection to copy from.
        db_to_copy_to: str
            The name of the DB to copy to.
        collection_to_copy_to: str
            The name of the collection to copy to.

        Raises
        ------
        BulkWriteError
        """
        db_name = self._get_db_name(db_name)
        if db_to_copy_to is None:
            db_to_copy_to = db_name

        if collection_to_copy_to is None:
            collection_to_copy_to = collection_name

        self._logger.info(
            f"Copying version {version_to_copy} of {element_to_copy} "
            f"to the new array element {new_array_element_name} in the {db_to_copy_to} DB"
        )

        collection = self.get_collection(db_name, collection_name)
        db_entries = []

        query = {
            "instrument": element_to_copy,
            "version": version_to_copy,
        }
        for post in collection.find(query):
            post["instrument"] = new_array_element_name
            post.pop("_id", None)
            db_entries.append(post)

        self._logger.info(f"Creating new array element {new_array_element_name}")
        collection = self.get_collection(db_to_copy_to, collection_to_copy_to)
        try:
            collection.insert_many(db_entries)
        except BulkWriteError as exc:
            raise BulkWriteError(str(exc.details)) from exc

    def add_production_table(self, db_name, production_table):
        """
        Add a production table for a given model version to the DB.

        Parameters
        ----------
        db_name: str
            the name of the DB.
        production_table: dict
            The production table to add to the DB.
        """
        db_name = self._get_db_name(db_name)
        collection = self.get_collection(db_name, "production_tables")
        self._logger.info(f"Adding production for {production_table.get('collection')} to to DB")
        collection.insert_one(production_table)
        self._reset_production_table_cache(
            production_table.get("collection"), production_table.get("model_version")
        )

    def add_new_parameter(
        self,
        db_name,
        par_dict,
        collection_name="telescopes",
        file_prefix=None,
    ):
        """
        Add a parameter dictionary for a specific array element to the DB.

        A new document will be added to the DB, with all fields taken from the input parameters.
        Parameter dictionaries are validated before submission using the corresponding schema.

        Parameters
        ----------
        db_name: str
            the name of the DB
        par_dict: dict
            dictionary with parameter data
        collection_name: str
            The name of the collection to add a parameter to.
        file_prefix: str or Path
            where to find files to upload to the DB
        """
        par_dict = validate_data.DataValidator.validate_model_parameter(par_dict)

        db_name = self._get_db_name(db_name)
        collection = self.get_collection(db_name, collection_name)

        par_dict["value"], _base_unit, _ = value_conversion.get_value_unit_type(
            value=par_dict["value"], unit_str=par_dict.get("unit", None)
        )
        par_dict["unit"] = _base_unit if _base_unit else None

        files_to_add_to_db = set()
        if par_dict["file"] and par_dict["value"]:
            if file_prefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    f"corresponding to the {par_dict['parameter']} parameter, must be provided."
                )
            file_path = Path(file_prefix).joinpath(par_dict["value"])
            files_to_add_to_db.add(f"{file_path}")

        self._logger.info(
            f"Adding a new entry to DB {db_name} and collection {db_name}:\n{par_dict}"
        )
        collection.insert_one(par_dict)

        for file_to_insert_now in files_to_add_to_db:
            self._logger.info(f"Will also add the file {file_to_insert_now} to the DB")
            self.insert_file_to_db(file_to_insert_now, db_name)

        self._reset_parameter_cache(par_dict["site"], par_dict["instrument"], None)

    def _get_db_name(self, db_name=None):
        """
        Return database name. If not provided, return the default database name.

        Parameters
        ----------
        db_name: str
            Database name

        Returns
        -------
        str
            Database name
        """
        return self.mongo_db_config["db_simulation_model"] if db_name is None else db_name

    def insert_file_to_db(self, file_name, db_name=None, **kwargs):
        """
        Insert a file to the DB.

        Parameters
        ----------
        file_name: str or Path
            The name of the file to insert (full path).
        db_name: str
            the name of the DB
        **kwargs (optional): keyword arguments for file creation.
            The full list of arguments can be found in, \
            https://docs.mongodb.com/manual/core/gridfs/#the-files-collection
            mostly these are unnecessary though.

        Returns
        -------
        file_iD: GridOut._id
            If the file exists, return its GridOut._id, otherwise insert the file and return its"
            "newly created DB GridOut._id.

        """
        db_name = self._get_db_name(db_name)
        db = DatabaseHandler.db_client[db_name]
        file_system = gridfs.GridFS(db)

        kwargs.setdefault("content_type", "ascii/dat")
        kwargs.setdefault("filename", Path(file_name).name)

        if file_system.exists({"filename": kwargs["filename"]}):
            self._logger.warning(
                f"The file {kwargs['filename']} exists in the DB. Returning its ID"
            )
            return file_system.find_one(  # pylint: disable=protected-access
                {"filename": kwargs["filename"]}
            )._id
        with open(file_name, "rb") as data_file:
            return file_system.put(data_file, **kwargs)

    def _cache_key(self, site=None, array_element_name=None, model_version=None, collection=None):
        """
        Create a cache key for the parameter cache dictionaries.

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Array element name.
        model_version: str
            Model version.
        collection: str
            DB collection name.

        Returns
        -------
        str
            Cache key.
        """
        return "-".join(
            part for part in [model_version, collection, site, array_element_name] if part
        )

    def _reset_production_table_cache(self, collection_name, model_version):
        """
        Reset the cache for the production tables.

        Parameters
        ----------
        collection_name: str
            Collection name.
        model_version: str
            Model version.
        """
        DatabaseHandler.production_table_cached.pop(
            self._cache_key(model_version=model_version, collection=collection_name), None
        )

    def _reset_parameter_cache(self, site, array_element_name, model_version):
        """
        Reset the cache for the parameters.

        A value of 'None' for any of the parameters will reset the entire cache.

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Array element name.
        model_version: str
            Model version.

        """
        self._logger.debug(f"Resetting cache for {site} {array_element_name} {model_version}")
        if None in [site, array_element_name, model_version]:
            DatabaseHandler.site_parameters_cached.clear()
            DatabaseHandler.model_parameters_cached.clear()
        else:
            _cache_key = self._cache_key(site, array_element_name, model_version)
            DatabaseHandler.site_parameters_cached.pop(_cache_key, None)
            DatabaseHandler.model_parameters_cached.pop(_cache_key, None)

    def get_collections(self, db_name=None, model_collections_only=False):
        """
        List of collections in the DB.

        Parameters
        ----------
        db_name: str
            Database name.

        Returns
        -------
        list
            List of collection names
        model_collections_only: bool
            If True, only return model collections (i.e. exclude fs.files, fs.chunks, metadata)

        """
        db_name = db_name or self._get_db_name()
        if db_name not in self.list_of_collections:
            self.list_of_collections[db_name] = DatabaseHandler.db_client[
                db_name
            ].list_collection_names()
        collections = self.list_of_collections[db_name]
        if model_collections_only:
            return [
                collection
                for collection in collections
                if not collection.startswith("fs.") and collection != "metadata"
            ]
        return collections
