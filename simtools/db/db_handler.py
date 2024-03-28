""" Module to handle interaction with DB. """

import logging
from pathlib import Path
from threading import Lock

import gridfs
import pymongo
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

import simtools.utils.general as gen
from simtools.db import db_from_repo_handler
from simtools.io_operations import io_handler
from simtools.utils import names

__all__ = ["DatabaseHandler"]


# pylint: disable=unsubscriptable-object
# The above comment is because pylint does not know that DatabaseHandler.db_client is subscriptable


class DatabaseHandler:
    """
    DatabaseHandler provides the interface to the DB.

    Parameters
    ----------
    io_handler: IOHandler
        Instance of IOHandler
    mongo_db_config: dict
        Dictionary with the MongoDB configuration with the following entries:
        "db_server" - DB server address
        "db_api_port" - Port to use
        "db_api_user" - API username
        "db_api_pw" - Password for the API user
        "db_api_authentication_database" - DB with user info (optional, default is "admin")
    """

    #    DB_TABULATED_DATA = "CTA-Simulation-Model"
    #    DB_CTA_SIMULATION_MODEL = "CTA-Simulation-Model"
    # DB collection with updates field names
    DB_TABULATED_DATA = "Staging-CTA-Simulation-Model"
    DB_CTA_SIMULATION_MODEL = "Staging-CTA-Simulation-Model"
    DB_CTA_SIMULATION_MODEL_DESCRIPTIONS = "CTA-Simulation-Model-Descriptions"
    DB_REFERENCE_DATA = "Staging-CTA-Reference-Data"
    DB_DERIVED_VALUES = "Staging-CTA-Simulation-Model-Derived-Values"

    ALLOWED_FILE_EXTENSIONS = [".dat", ".txt", ".lis", ".cfg", ".yml", ".yaml", ".ecsv"]

    db_client = None

    def __init__(self, mongo_db_config=None):
        """
        Initialize the DatabaseHandler class.

        Raises
        ------
        KeyError
            if there is non-valid key in the db_config.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialize DatabaseHandler")

        self.mongo_db_config = mongo_db_config
        self.io_handler = io_handler.IOHandler()
        self._available_telescopes = None

        self._set_up_connection()

    def _set_up_connection(self):
        """
        Open the connection to MongoDB.
        """
        if self.mongo_db_config:
            if DatabaseHandler.db_client is None:
                lock = Lock()
                with lock:
                    DatabaseHandler.db_client = self._open_mongo_db()

    def _open_mongo_db(self):
        """
        Open a connection to MongoDB and return the client to read/write to the DB with.

        Returns
        -------
        A PyMongo DB client


        """
        try:
            _db_client = MongoClient(
                self.mongo_db_config["db_server"],
                port=self.mongo_db_config["db_api_port"],
                username=self.mongo_db_config["db_api_user"],
                password=self.mongo_db_config["db_api_pw"],
                authSource=self.mongo_db_config.get("db_api_authentication_database", "admin"),
                ssl=True,
                tlsallowinvalidhostnames=True,
                tlsallowinvalidcertificates=True,
            )
        except KeyError:
            self._logger.error("Invalid setting of DB configuration")
            raise

        return _db_client

    def get_model_parameters(
        self,
        site,
        telescope_model_name,
        model_version,
        only_applicable=False,
    ):
        """
        Get parameters from either MongoDB or simulation model repository for a specific telescope.

        Parameters
        ----------
        site: str
            South or North.
        telescope_model_name: str
            Name of the telescope model (e.g. LST-1, MST-FlashCam-D)
        model_version: str
            Version of the model.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        """

        _site_validated = names.validate_site_name(site)
        _tel_model_name_validated = names.validate_telescope_name(telescope_model_name)

        _model_version = self._convert_version_to_tagged(
            model_version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )
        _version_validated = names.validate_model_version_name(_model_version)

        _pars = self._get_model_parameters_mongo_db(
            DatabaseHandler.DB_CTA_SIMULATION_MODEL,
            _tel_model_name_validated,
            _version_validated,
            only_applicable,
        )

        # update using simulation model repository
        if self.mongo_db_config.get("db_simulation_model_url", None) is not None:
            return db_from_repo_handler.update_model_parameters_from_repo(
                parameters=_pars,
                site=_site_validated,
                telescope_name=_tel_model_name_validated,
                model_version=_version_validated,
                db_simulation_model_url=self.mongo_db_config.get("db_simulation_model_url", None),
            )

        return _pars

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
            self._logger.debug("Exporting model files from MongoDB")
            for info in parameters.values():
                if not info.get("file"):
                    continue
                if Path(dest).joinpath(info["value"]).exists():
                    self._logger.debug(f"File {info['value']} already exists in {dest}")
                    continue
                file = self._get_file_mongo_db(
                    DatabaseHandler.DB_CTA_SIMULATION_MODEL, info["value"]
                )
                self._write_file_from_mongo_to_disk(
                    DatabaseHandler.DB_CTA_SIMULATION_MODEL, dest, file
                )
        if self.mongo_db_config.get("db_simulation_model_url", None) is not None:
            self._logger.warning(
                "Exporting model files from simulation model repository not yet implemented"
            )

    @staticmethod
    def _is_file(value):
        """Verify if a parameter value is a file name."""
        return any(ext in str(value) for ext in DatabaseHandler.ALLOWED_FILE_EXTENSIONS)

    def _get_model_parameters_mongo_db(
        self, db_name, telescope_model_name, model_version, only_applicable=False
    ):
        """
        Get parameters from MongoDB for a specific telescope.

        Parameters
        ----------
        db_name: str
            the name of the DB
        telescope_model_name: str
            Name of the telescope model (e.g. MST-FlashCam-D ...)
        model_version: str
            Version of the model.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        """

        self._logger.debug(f"Tel_name_db: {telescope_model_name}")
        _which_tel_labels = [
            self.get_telescope_db_name(
                telescope_name=telescope_model_name,
                model_version=model_version,
            )
        ]

        _pars = {}
        for _tel in _which_tel_labels:
            self._logger.debug(f"Getting {_tel} parameters from MongoDB")

            _pars.update(
                self.read_mongo_db(
                    db_name,
                    _tel,
                    model_version,
                    run_location=None,
                    write_files=False,
                    only_applicable=only_applicable,
                )
            )

        return _pars

    def read_mongo_db(
        self,
        db_name,
        telescope_model_name,
        model_version,
        run_location,
        collection_name="telescopes",
        write_files=True,
        only_applicable=False,
    ):
        """
        Build and execute query to Read the MongoDB for a specific telescope.
        Also writes the files listed in the parameter values into the sim_telarray run location

        Parameters
        ----------
        db_name: str
            the name of the DB
        telescope_model_name: str
            Name of the telescope model (e.g. MSTN-design ...)
        model_version: str
            Version of the model.
        run_location: Path or str
            The sim_telarray run location to write the tabulated data files into.
        collection_name: str
            The name of the collection to read from.
        write_files: bool
            If true, write the files to the run_location.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned zero results.

        """

        collection = DatabaseHandler.db_client[db_name][collection_name]
        _parameters = {}

        _model_version = self._convert_version_to_tagged(
            model_version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "instrument": telescope_model_name,
            "version": _model_version,
        }

        self._logger.debug(f"Trying the following query: {query}")
        if only_applicable:
            query["applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query):
            par_now = post["parameter"]
            _parameters[par_now] = post
            _parameters[par_now].pop("parameter", None)
            _parameters[par_now].pop("instrument", None)
            _parameters[par_now]["entry_date"] = ObjectId(post["_id"]).generation_time
            if _parameters[par_now]["file"] and write_files:
                file = self._get_file_mongo_db(db_name, _parameters[par_now]["value"])

                self._write_file_from_mongo_to_disk(db_name, run_location, file)

        return _parameters

    def get_site_parameters(
        self,
        site,
        model_version,
        only_applicable=False,
    ):
        """
        Get parameters from either MongoDB or simulation model repository for a specific site.

        Parameters
        ----------
        site: str
            South or North.
        model_version: str
            Version of the model.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        """
        _site = names.validate_site_name(site)
        _version_validated = names.validate_model_version_name(model_version)
        _pars = self._get_site_parameters_mongo_db(
            DatabaseHandler.DB_CTA_SIMULATION_MODEL,
            _site,
            _version_validated,
            only_applicable,
        )
        # update simulation model using repository
        if self.mongo_db_config.get("db_simulation_model_url", None) is not None:
            return db_from_repo_handler.update_site_parameters_from_repo(
                parameters=_pars,
                site=site,
                model_version=_version_validated,
                db_simulation_model_url=self.mongo_db_config.get("db_simulation_model_url", None),
            )
        return _pars

    def _get_site_parameters_mongo_db(self, db_name, site, model_version, only_applicable=False):
        """
        Get parameters from MongoDB for a specific telescope.

        Parameters
        ----------
        db_name: str
            The name of the DB.
        site: str
            South or North.
        model_version: str
            Version of the model.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned zero results.

        """

        _site_validated = names.validate_site_name(site)
        collection = DatabaseHandler.db_client[db_name].sites
        _parameters = {}

        _model_version = self._convert_version_to_tagged(
            model_version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "site": _site_validated,
            "version": _model_version,
        }
        if only_applicable:
            query["applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query):
            par_now = post["parameter"]
            _parameters[par_now] = post
            _parameters[par_now].pop("parameter", None)
            _parameters[par_now].pop("site", None)
            _parameters[par_now]["entry_date"] = ObjectId(post["_id"]).generation_time

        return _parameters

    def get_reference_data(self, site, model_version, only_applicable=False):
        """
        Get parameters from MongoDB for a specific telescope.

        Parameters
        ----------
        site: str
            South or North.
        model_version: str
            Version of the model.
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned zero results.

        """

        _site_validated = names.validate_site_name(site)
        collection = DatabaseHandler.db_client[DatabaseHandler.DB_REFERENCE_DATA].reference_values
        _parameters = {}

        _model_version = self._convert_version_to_tagged(
            names.validate_model_version_name(model_version),
            DatabaseHandler.DB_CTA_SIMULATION_MODEL,
        )

        query = {
            "site": _site_validated,
            "version": _model_version,
        }
        if only_applicable:
            query["applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query):
            par_now = post["parameter"]
            _parameters[par_now] = post
            _parameters[par_now].pop("parameter", None)
            _parameters[par_now].pop("site", None)
            _parameters[par_now]["entry_date"] = ObjectId(post["_id"]).generation_time

        return _parameters

    def get_derived_values(self, site, telescope_model_name, model_version):
        """
        Get a derived value from the DB for a specific telescope.

        Parameters
        ----------
        site: str
            South or North.
        telescope_model_name: str
            Name of the telescope model (e.g. MST-FlashCam-D ...)
        model_version: str
            Version of the model.

        Returns
        -------
        dict containing the parameters

        """

        try:
            _tel_model_name_validated = names.validate_telescope_name(telescope_model_name)
        except AttributeError as exc:
            self._logger.error(
                "Telescope name required to return derived model parameters"
                " (this error might point to a problem with the DB configuration)"
            )
            raise exc
        _model_version = self._convert_version_to_tagged(
            names.validate_model_version_name(model_version),
            DatabaseHandler.DB_CTA_SIMULATION_MODEL,
        )

        self._logger.debug(
            f"Getting derived values for {_tel_model_name_validated}, {site} from the DB"
        )

        _pars = self.read_mongo_db(
            DatabaseHandler.DB_DERIVED_VALUES,
            _tel_model_name_validated,
            _model_version,
            run_location=None,
            collection_name="derived_values",
            write_files=False,
        )

        return _pars

    @staticmethod
    def _get_file_mongo_db(db_name, file_name):
        """
        Extract a file from MongoDB and return GridFS file instance

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
        Extract a file from MongoDB and write it to disk

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

    def copy_telescope(
        self,
        db_name,
        tel_to_copy,
        version_to_copy,
        new_tel_name,
        collection_name="telescopes",
        db_to_copy_to=None,
        collection_to_copy_to=None,
    ):
        """
        Copy a full telescope configuration to a new telescope name.
        Only a specific version is copied.
        (This function should be rarely used, probably only during "construction".)

        Parameters
        ----------
        db_name: str
            the name of the DB to copy from
        tel_to_copy: str
            The telescope to copy
        version_to_copy: str
            The version of the configuration to copy
        new_tel_name: str
            The name of the new telescope
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

        if db_to_copy_to is None:
            db_to_copy_to = db_name

        if collection_to_copy_to is None:
            collection_to_copy_to = collection_name

        self._logger.info(
            f"Copying version {version_to_copy} of {tel_to_copy} "
            f"to the new telescope {new_tel_name} in the {db_to_copy_to} DB"
        )

        collection = DatabaseHandler.db_client[db_name][collection_name]
        db_entries = []

        _version_to_copy = self._convert_version_to_tagged(
            version_to_copy, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "instrument": tel_to_copy,
            "version": _version_to_copy,
        }
        for post in collection.find(query):
            post["instrument"] = new_tel_name
            post.pop("_id", None)
            db_entries.append(post)

        self._logger.info(f"Creating new telescope {new_tel_name}")
        collection = DatabaseHandler.db_client[db_to_copy_to][collection_to_copy_to]
        try:
            collection.insert_many(db_entries)
        except BulkWriteError as exc:
            raise BulkWriteError(str(exc.details)) from exc

    def copy_documents(self, db_name, collection, query, db_to_copy_to, collection_to_copy_to=None):
        """
        Copy the documents matching to "query" to the DB "db_to_copy_to".
        The documents are copied to the same collection as in "db_name".
        (This function should be rarely used, probably only during "construction".)

        Parameters
        ----------
        db_name: str
            the name of the DB to copy from
        collection: str
            the name of the collection to copy from
        query: dict
            A dictionary with a query to search for documents to copy.
            For example, the query below would copy all entries of prod4 version
            from telescope LSTN-01 to "db_to_copy_to".

            .. code-block:: python

                query = {
                    "instrument": "LSTN-01",
                    "version": "prod6",
                }
        db_to_copy_to: str
            The name of the DB to copy to.

        Raises
        ------
        BulkWriteError

        """

        _collection = DatabaseHandler.db_client[db_name][collection]
        if collection_to_copy_to is None:
            collection_to_copy_to = collection
        db_entries = []

        for post in _collection.find(query):
            post.pop("_id", None)
            db_entries.append(post)

        self._logger.info(
            f"Copying documents matching the following query {query}\nto {db_to_copy_to}"
        )
        _collection = DatabaseHandler.db_client[db_to_copy_to][collection_to_copy_to]
        try:
            _collection.insert_many(db_entries)
        except BulkWriteError as exc:
            raise BulkWriteError(str(exc.details)) from exc

    def delete_query(self, db_name, collection, query):
        """
        Delete all entries from the DB which correspond to the provided query.
        (This function should be rarely used, if at all.)

        Parameters
        ----------
        db_name: str
            the name of the DB
        collection: str
            the name of the collection to copy from
        query: dict
            A dictionary listing the fields/values to delete.
            For example, the query below would delete the entire prod6 version
            from telescope LSTN-01.

            .. code-block:: python

                query = {
                    "instrument": "LSTN-01",
                    "version": "prod6",
                }

        """

        _collection = DatabaseHandler.db_client[db_name][collection]

        if "version" in query:
            query["version"] = self._convert_version_to_tagged(
                query["version"], DatabaseHandler.DB_CTA_SIMULATION_MODEL
            )

        self._logger.info(f"Deleting {_collection.count_documents(query)} entries from {db_name}")

        _collection.delete_many(query)

    def update_parameter(
        self,
        db_name,
        telescope,
        version,
        parameter,
        new_value,
        collection_name="telescopes",
        file_prefix=None,
    ):
        """
        Update a parameter value for a specific telescope/version.
        (This function should be rarely used since new values
        should ideally have their own version.)

        Parameters
        ----------
        db_name: str
            the name of the DB
        telescope: str
            Which telescope to update
        version: str
            Which version to update
        parameter: str
            Which parameter to update
        new_value: type identical to the original parameter type
            The new value to set for the parameter
        collection_name: str
            The name of the collection in which to update the parameter.
        file_prefix: str or Path
            where to find files to upload to the DB

        Raises
        ------
        FileNotFoundError
            if file_prefix is None

        """

        collection = DatabaseHandler.db_client[db_name][collection_name]

        _model_version = self._convert_version_to_tagged(
            version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "instrument": telescope,
            "version": _model_version,
            "parameter": parameter,
        }

        par_entry = collection.find_one(query)
        old_value = par_entry["value"]

        self._logger.info(
            f"For telescope {telescope}, version {_model_version}\n"
            f"replacing {parameter} value from {old_value} to {new_value}"
        )

        files_to_add_to_db = set()
        if self._is_file(new_value):
            file = True
            if file_prefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    f"corresponding to the {parameter} parameter, must be provided."
                )
            file_path = Path(file_prefix).joinpath(new_value)
            files_to_add_to_db.add(f"{file_path}")
            self._logger.info(f"Will also add the file {file_path} to the DB")
        else:
            file = False

        query_update = {"$set": {"value": new_value, "file": file}}

        collection.update_one(query, query_update)
        for file_now in files_to_add_to_db:
            self.insert_file_to_db(file_now, db_name)

    def update_parameter_field(
        self,
        db_name,
        version,
        parameter,
        field,
        new_value,
        telescope=None,
        site=None,
        collection_name="telescopes",
    ):
        """
        Update a parameter field value for a specific telescope/version.
        This function only modifies the value of one of the following
        DB entries: Applicable, units, Type, items, minimum, maximum.
        These type of changes should be very rare. However they can
        be done without changing the Object ID of the entry since
        they are generally "harmless".

        Parameters
        ----------
        db_name: str
            the name of the DB
        version: str
            Which version to update
        parameter: str
            Which parameter to update
        field: str
            Field to update (only options are Applicable, units, Type, items, minimum, maximum).
            If the field does not exist, it will be added.
        new_value: type identical to the original field type
            The new value to set to the field given in "field".
        telescope: str
            Which telescope to update, if None then update a site parameter
        site: str, North or South
            Update a site parameter (the telescope argument must be None)
        collection_name: str
            The name of the collection in which to update the parameter.

        Raises
        ------
        ValueError
            if field not in allowed fields

        """

        allowed_fields = ["applicable", "unit", "type", "items", "minimum", "maximum"]
        if field not in allowed_fields:
            raise ValueError(f"The field {field} must be one of {', '.join(allowed_fields)}")

        collection = DatabaseHandler.db_client[db_name][collection_name]

        _model_version = self._convert_version_to_tagged(
            version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "version": _model_version,
            "parameter": parameter,
        }
        if telescope is not None:
            query["instrument"] = telescope
            logger_info = f"instrument {telescope}"
        elif site is not None and site in names.site_names:
            query["site"] = site
            logger_info = f"site {site}"
        else:
            raise ValueError("You need to specify if to update a telescope or a site.")

        par_entry = collection.find_one(query)
        if par_entry is None:
            self._logger.warning(
                f"The query {query} did not return any results. I will not make any changes."
            )
            return

        if field in par_entry:
            old_field_value = par_entry[field]

            if old_field_value == new_value:
                self._logger.warning(
                    f"The value of the field {field} is already {new_value}. No changes necessary"
                )
                return

            self._logger.info(
                f"For {logger_info}, version {_model_version}, parameter {parameter}, "
                f"replacing field {field} value from {old_field_value} to {new_value}"
            )
        else:
            self._logger.info(
                f"For {logger_info}, version {_model_version}, parameter {parameter}, "
                f"the field {field} does not exist, adding it"
            )

        query_update = {"$set": {field: new_value}}

        collection.update_one(query, query_update)

    def add_parameter(
        self,
        db_name,
        telescope,
        parameter,
        new_version,
        new_value,
        collection_name="telescopes",
        file_prefix=None,
    ):
        """
        Add a parameter value for a specific telescope.
        A new document will be added to the DB,
        with all fields taken from the last entry of this parameter to this telescope,
        except the ones changed.

        Parameters
        ----------
        db_name: str
            the name of the DB
        telescope: str
            Which telescope to update
        parameter: str
            Which parameter to add
        new_version: str
            The version of the new parameter value
        new_value: type identical to the original parameter type
            The new value to set for the parameter
        collection_name: str
            The name of the collection to which to add a parameter.
        file_prefix: str or Path
            where to find files to upload to the DB

        Raises
        ------
        FileNotFoundError
            if file_prefix is None

        """

        collection = DatabaseHandler.db_client[db_name][collection_name]

        _new_version = self._convert_version_to_tagged(
            new_version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "instrument": telescope,
            "parameter": parameter,
        }

        par_entry = collection.find(query).sort("_id", pymongo.DESCENDING)[0]
        par_entry["value"] = new_value
        par_entry["version"] = _new_version
        par_entry.pop("_id", None)

        files_to_add_to_db = set()
        if self._is_file(new_value):
            par_entry["file"] = True
            if file_prefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    f"corresponding to the {parameter} parameter, must be provided."
                )
            file_path = Path(file_prefix).joinpath(new_value)
            files_to_add_to_db.add(f"{file_path}")
        else:
            par_entry["file"] = False

        self._logger.info(f"Will add the following entry to DB:\n{par_entry}")

        collection.insert_one(par_entry)
        if len(files_to_add_to_db) > 0:
            self._logger.info(f"Will also add the file {file_path} to the DB")
            self.insert_file_to_db(files_to_add_to_db, db_name)

    def add_new_parameter(
        self,
        db_name,
        version,
        parameter,
        value,
        telescope=None,
        site=None,
        collection_name="telescopes",
        file_prefix=None,
        **kwargs,
    ):
        """
        Add a parameter value for a specific telescope.
        A new document will be added to the DB,
        with all fields taken from the input parameters.

        Parameters
        ----------
        db_name: str
            the name of the DB
        version: str
            The version of the new parameter value
        parameter: str
            Which parameter to add
        value: can be any type, preferably given in kwargs
            The value to set for the new parameter
        telescope: str
            The name of the telescope to add a parameter to
            (only used if collection_name is "telescopes").
        site: str
           South or North, ignored if collection_name is "telescopes".
        collection_name: str
            The name of the collection to add a parameter to.
        file_prefix: str or Path
            where to find files to upload to the DB
        kwargs: dict
            Any additional fields to add to the parameter

        Raises
        ------
        ValueError
            If key to collection_name is not valid. Valid entries are: 'telescopes' and 'sites'.

        """

        collection = DatabaseHandler.db_client[db_name][collection_name]

        db_entry = {}
        if "telescopes" in collection_name:
            db_entry["instrument"] = names.validate_telescope_name(telescope)
        elif "sites" in collection_name:
            db_entry["instrument"] = names.validate_site_name(site)
        else:
            raise ValueError("Can only add new parameters to the sites or telescopes collections")

        db_entry["version"] = version
        db_entry["parameter"] = parameter
        if site is not None:
            db_entry["site"] = names.validate_site_name(site)

        _base_value, _base_unit, _base_type = gen.get_value_unit_type(
            value=value, unit_str=kwargs.get("unit", None)
        )
        db_entry["value"] = _base_value
        if _base_unit is not None:
            db_entry["unit"] = _base_unit
        db_entry["type"] = kwargs["type"] if "type" in kwargs else _base_type

        files_to_add_to_db = set()
        db_entry["file"] = False
        if self._is_file(value):
            db_entry["file"] = True
            if file_prefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    f"corresponding to the {parameter} parameter, must be provided."
                )
            file_path = Path(file_prefix).joinpath(value)
            files_to_add_to_db.add(f"{file_path}")

        kwargs.pop("type", None)
        db_entry.update(kwargs)

        self._logger.info(f"Will add the following entry to DB:\n{db_entry}")

        collection.insert_one(db_entry)
        for file_to_insert_now in files_to_add_to_db:
            self._logger.info(f"Will also add the file {file_to_insert_now} to the DB")
            self.insert_file_to_db(file_to_insert_now, db_name)

    def _convert_version_to_tagged(self, model_version, db_name):
        """Convert to tagged version, if needed."""
        if model_version in ["Released", "Latest"]:
            return self._get_tagged_version(db_name, model_version)

        return model_version

    @staticmethod
    def _get_tagged_version(db_name, version="Released"):
        """
        Get the tag of the "Released" or "Latest" version of the MC Model.
        The "Released" is the latest stable MC Model,
        the latest is the latest tag (not necessarily stable, but can be equivalent to "Released").

        Parameters
        ----------
        db_name: str
            the name of the DB
        version: str
            Can be "Released" or "Latest" (default: "Released").

        Returns
        -------
        str
            The version name in the Simulation DB of the requested tag

        Raises
        ------
        ValueError
            if version not valid. Valid versions are: 'Released' and 'Latest'.

        """

        if version not in ["Released", "Latest"]:
            raise ValueError('The only default versions are "Released" or "Latest"')

        collection = DatabaseHandler.db_client[db_name].metadata
        query = {"Entry": "Simulation-Model-Tags"}

        tags = collection.find(query).sort("_id", pymongo.DESCENDING)[0]

        return tags["Tags"][version]["Value"]

    def insert_file_to_db(self, file_name, db_name=DB_CTA_SIMULATION_MODEL, **kwargs):
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

        db = DatabaseHandler.db_client[db_name]
        file_system = gridfs.GridFS(db)

        if "content_type" not in kwargs:
            kwargs["content_type"] = "ascii/dat"

        if "filename" not in kwargs:
            kwargs["filename"] = Path(file_name).name

        if file_system.exists({"filename": kwargs["filename"]}):
            self._logger.warning(
                f"The file {kwargs['filename']} exists in the DB. Returning its ID"
            )
            return file_system.find_one(  # pylint: disable=protected-access
                {"filename": kwargs["filename"]}
            )._id
        with open(file_name, "rb") as data_file:
            return file_system.put(data_file, **kwargs)

    def get_all_versions(
        self,
        db_name,
        parameter,
        telescope_model_name=None,
        site=None,
        collection_name="telescopes",
    ):
        """
        Get all version entries in the DB of a telescope or site for a specific parameter.

        Parameters
        ----------
        db_name: str
            the name of the DB
        parameter: str
            Which parameter to get the versions of
        telescope_model_name: str
            Which telescope to get the versions of (in case "collection_name" is "telescopes")
        site: str, North or South
            In case "collection_name" is "telescopes", the site is used to build the telescope name.
            In case "collection_name" is "sites",
            this argument sets which site parameter get the versions of
        collection_name: str
            The name of the collection in which to update the parameter.

        Returns
        -------
        all_versions: list
            List of all versions found

        Raises
        ------
        ValueError
            If key to collection_name is not valid. Valid entries are: 'telescopes' and 'sites'.

        """

        collection = DatabaseHandler.db_client[db_name][collection_name]

        query = {
            "parameter": parameter,
        }

        _site_validated = names.validate_site_name(site)
        if collection_name == "telescopes":
            _tel_model_name_validated = names.validate_telescope_name(telescope_model_name)
            query["instrument"] = _tel_model_name_validated
        elif collection_name == "sites":
            query["site"] = _site_validated
        else:
            raise ValueError("Can only get versions of the telescopes and sites collections.")

        _all_versions = [post["version"] for post in collection.find(query)]

        if len(_all_versions) == 0:
            self._logger.warning(f"The query {query} did not return any results. No versions found")

        return _all_versions

    def get_all_available_telescopes(
        self,
        model_version,
        db_name=DB_CTA_SIMULATION_MODEL,
    ):
        """
        Get all available telescope names in the collection "telescopes" in the DB.

        Parameters
        ----------
        db_name: str
            the name of the DB
        model_version: str
            Which version to get the telescopes of

        Returns
        -------
        all_available_telescopes: list
            List of all telescope names found

        """

        collection = DatabaseHandler.db_client[db_name]["telescopes"]

        _model_version = self._convert_version_to_tagged(
            names.validate_model_version_name(model_version),
            DatabaseHandler.DB_CTA_SIMULATION_MODEL,
        )

        query = {
            "version": _model_version,
        }

        _all_available_telescopes = collection.find(query).distinct("instrument")

        return _all_available_telescopes

    def get_telescope_db_name(self, telescope_name, model_version):
        """
        Translate telescope name to the name used in the DB. This is required,
        as not all telescopes are defined in the database yet. In these cases,
        use the "design" telescope.

        Parameters
        ----------
        telescope_name: str
            Name of the telescope model (e.g. MSTN-01)
        model_version: str
            Which version to get the telescopes

        Returns
        -------
        str
            Telescope model name as used in the DB.

        Raises
        ------
        ValueError
            If the telescope name is not found in the database.

        """

        if self._available_telescopes is None:
            self._available_telescopes = self.get_all_available_telescopes(model_version)

        _telescope_name_validated = names.validate_telescope_name(telescope_name)
        if _telescope_name_validated in self._available_telescopes:
            return _telescope_name_validated
        _design_name = (
            f"{names.get_telescope_type_from_telescope_name(_telescope_name_validated)}-design"
        )
        if _design_name in self._available_telescopes:
            return _design_name

        self._logger.error("Telescope %s not found in the database.", telescope_name)
        raise ValueError
