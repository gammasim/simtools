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

logging.getLogger("pymongo").setLevel(logging.WARNING)


# pylint: disable=unsubscriptable-object
# The above comment is because pylint does not know that DatabaseHandler.db_client is subscriptable


class DatabaseHandler:
    """
    DatabaseHandler provides the interface to the DB.

    Parameters
    ----------
    mongo_db_config: dict
        Dictionary with the MongoDB configuration with the following entries:
        "db_server" - DB server address
        "db_api_port" - Port to use
        "db_api_user" - API username
        "db_api_pw" - Password for the API user
        "db_api_authentication_database" - DB with user info (optional, default is "admin")
        "db_simulation_model" - Name of simulation model database
    """

    DB_CTA_SIMULATION_MODEL_DESCRIPTIONS = "CTA-Simulation-Model-Descriptions"
    # DB collection with updates field names
    DB_DERIVED_VALUES = "Staging-CTA-Simulation-Model-Derived-Values"

    ALLOWED_FILE_EXTENSIONS = [".dat", ".txt", ".lis", ".cfg", ".yml", ".yaml", ".ecsv"]

    db_client = None
    site_parameters_cached = {}
    model_parameters_cached = {}
    model_versions_cached = {}
    sim_telarray_configuration_parameters_cached = {}

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
        self._available_array_elements = None
        self.list_of_collections = {}

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
        collection="telescope",
        only_applicable=False,
    ):
        """
        Get parameters from MongoDB or simulation model repository for an array element.
        An array element can be e.g., a telescope or a calibration device.
        Read parameters for design and for the specified array element (if necessary). This allows
        to overwrite design parameters with specific telescope parameters without having to copy
        all model parameters when changing only a few.

        Parameters
        ----------
        site: str
            South or North.
        telescope_model_name: str
            Name of the telescope model (e.g. LSTN-01, MSTS-design)
        model_version: str
            Version of the model.
        collection: str
            collection of array element (e.g. telescopes, calibration_devices)
        only_applicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters

        """

        _site, _telescope_model_name, _model_version = self._validate_model_input(
            site, telescope_model_name, model_version
        )
        telescope_list = self.get_telescope_list_for_db_query(_telescope_model_name, model_version)
        pars = {}
        for telescope in telescope_list:
            _array_elements_cache_key = self._parameter_cache_key(site, telescope, model_version)
            try:
                pars.update(DatabaseHandler.model_parameters_cached[_array_elements_cache_key])
            except KeyError:
                pars.update(
                    self.read_mongo_db(
                        self.mongo_db_config.get("db_simulation_model", None),
                        telescope_model_name=telescope,
                        model_version=_model_version,
                        collection_name=collection,
                        run_location=None,
                        write_files=False,
                        only_applicable=only_applicable,
                    )
                )
                if self.mongo_db_config.get("db_simulation_model_url", None) is not None:
                    pars = db_from_repo_handler.update_model_parameters_from_repo(
                        parameters=pars,
                        site=_site,
                        parameter_collection=collection,
                        telescope_name=telescope,
                        model_version=_model_version,
                        db_simulation_model_url=self.mongo_db_config.get("db_simulation_model_url"),
                    )
            DatabaseHandler.model_parameters_cached[_array_elements_cache_key] = pars

        return pars

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
            self._logger.debug("Exporting model files from MongoDB")
            for info in parameters.values():
                if not info or not info.get("file") or info["value"] is None:
                    continue
                if Path(dest).joinpath(info["value"]).exists():
                    self._logger.debug(f"File {info['value']} already exists in {dest}")
                    continue
                file = self._get_file_mongo_db(self._get_db_name(), info["value"])
                self._write_file_from_mongo_to_disk(self._get_db_name(), dest, file)
        if self.mongo_db_config.get("db_simulation_model_url", None) is not None:
            self._logger.warning(
                "Exporting model files from simulation model repository not yet implemented"
            )

    @staticmethod
    def _is_file(value):
        """Verify if a parameter value is a file name."""
        return any(ext in str(value) for ext in DatabaseHandler.ALLOWED_FILE_EXTENSIONS)

    def read_mongo_db(
        self,
        db_name,
        telescope_model_name,
        model_version,
        run_location,
        collection_name,
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

        query = {
            "instrument": telescope_model_name,
            "version": self.model_version(model_version, db_name),
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
        _site, _, _model_version = self._validate_model_input(site, None, model_version)
        _db_name = self._get_db_name()
        self._logger.debug(
            f"Getting {site} parameters from MongoDB {_db_name}"
            f" {model_version} {only_applicable}"
        )
        _site_cache_key = self._parameter_cache_key(site, None, model_version)
        try:
            return DatabaseHandler.site_parameters_cached[_site_cache_key]
        except KeyError:
            pass

        _pars = self._get_site_parameters_mongo_db(
            _db_name,
            _site,
            _model_version,
            only_applicable,
        )
        # update simulation model using repository
        if self.mongo_db_config.get("db_simulation_model_url", None) is not None:
            _pars = db_from_repo_handler.update_model_parameters_from_repo(
                parameters=_pars,
                site=_site,
                telescope_name=None,
                parameter_collection="site",
                model_version=_model_version,
                db_simulation_model_url=self.mongo_db_config.get("db_simulation_model_url", None),
            )

        DatabaseHandler.site_parameters_cached[_site_cache_key] = _pars
        return DatabaseHandler.site_parameters_cached[_site_cache_key]

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

        collection = DatabaseHandler.db_client[db_name].sites
        _parameters = {}

        query = {
            "site": site,
            "version": model_version,
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
        Get all derived values from the DB for a specific telescope.

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
        _, _telescope_model_name, _model_version = self._validate_model_input(
            site, telescope_model_name, model_version
        )

        return self.read_mongo_db(
            DatabaseHandler.DB_DERIVED_VALUES,
            _telescope_model_name,
            _model_version,
            run_location=None,
            collection_name="derived_values",
            write_files=False,
        )

    def get_sim_telarray_configuration_parameters(self, site, telescope_model_name, model_version):
        """
        Get sim_telarray configuration parameters from the DB for a specific telescope.

        """

        _, _telescope_model_name, _model_version = self._validate_model_input(
            site, telescope_model_name, model_version
        )
        _array_elements_cache_key = self._parameter_cache_key(
            site, telescope_model_name, model_version
        )
        try:
            return DatabaseHandler.sim_telarray_configuration_parameters_cached[
                _array_elements_cache_key
            ]
        except KeyError:
            pass
        pars = {}
        try:
            pars = self.read_mongo_db(
                self._get_db_name(),
                _telescope_model_name,
                _model_version,
                run_location=None,
                collection_name="configuration_sim_telarray",
                write_files=False,
            )
        except ValueError:
            pars = self.read_mongo_db(
                self._get_db_name(),
                names.get_telescope_type_from_telescope_name(_telescope_model_name) + "-design",
                _model_version,
                run_location=None,
                collection_name="configuration_sim_telarray",
                write_files=False,
            )
        DatabaseHandler.sim_telarray_configuration_parameters_cached[_array_elements_cache_key] = (
            pars
        )
        return pars

    def _validate_model_input(self, site, telescope_model_name, model_version):
        """
        Validate input for model parameter queries.

        site: str
            South or North.
        telescope_model_name: str
            Name of the telescope model (e.g. LSTN-01, MSTS-design)
        model_version: str
            Version of the model.

        """
        return (
            names.validate_site_name(site),
            names.validate_telescope_name(telescope_model_name) if telescope_model_name else None,
            self.model_version(model_version),
        )

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

        db_name = self._get_db_name(db_name)
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

        _version_to_copy = self.model_version(version_to_copy)

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
        db_name = self._get_db_name(db_name)

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
            query["version"] = self.model_version(query["version"])

        self._logger.info(f"Deleting {_collection.count_documents(query)} entries from {db_name}")

        _collection.delete_many(query)

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

        db_name = self._get_db_name(db_name)
        allowed_fields = ["applicable", "unit", "type", "items", "minimum", "maximum"]
        if field not in allowed_fields:
            raise ValueError(f"The field {field} must be one of {', '.join(allowed_fields)}")

        collection = DatabaseHandler.db_client[db_name][collection_name]
        _model_version = self.model_version(version, db_name)

        query = {
            "version": _model_version,
            "parameter": parameter,
        }
        if telescope is not None:
            query["instrument"] = telescope
            logger_info = f"instrument {telescope}"
        elif site is not None and site in names.site_names():
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

        self._reset_parameter_cache(
            site=site if site is not None else names.get_site_from_telescope_name(telescope),
            telescope=telescope,
            model_version=_model_version,
        )

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

        db_name = self._get_db_name(db_name)
        collection = DatabaseHandler.db_client[db_name][collection_name]

        db_entry = {}
        if "telescopes" in collection_name or "configuration_sim_telarray" in collection_name:
            db_entry["instrument"] = names.validate_telescope_name(telescope)
        elif "sites" in collection_name:
            db_entry["instrument"] = names.validate_site_name(site)
        elif "calibration_devices" in collection_name:
            db_entry["instrument"] = names.validate_telescope_name(telescope)
        else:
            raise ValueError(
                "Can only add new parameters to the sites, \
                telescopes or calibration_devices collections"
            )

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

        self._reset_parameter_cache(site, telescope, version, db_name)

    def add_tagged_version(
        self,
        tags,
        db_name=None,
    ):
        """
        Set the tag of the "Released" or "Latest" version of the MC Model.

        Parameters
        ----------
        released_version: str
            The version name to set as "Released"
        tags: dict
            The version tags consisting of tag name and version name.
        db_name: str
            Database name

        """
        collection = DatabaseHandler.db_client[self._get_db_name(db_name)]["metadata"]
        self._logger.debug(f"Adding tags {tags} to DB {self._get_db_name(db_name)}")
        collection.insert_one({"Entry": "Simulation-Model-Tags", "Tags": tags})

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

    def model_version(self, version="Released", db_name=None):
        """
        Return the model version for the requested tag.

        Resolve the "Released" or "Latest" tag to the actual version name.

        Parameters
        ----------
        version : str
            Model version.
        db_name : str
            Database name.

        Returns
        -------
        str
            Model version.

        Raises
        ------
        ValueError
            if version not valid. Valid versions are: 'Released' and 'Latest'.

        """

        _all_versions = self.get_all_versions()
        if version in _all_versions:
            return version

        collection = DatabaseHandler.db_client[self._get_db_name(db_name)].metadata
        query = {"Entry": "Simulation-Model-Tags"}

        tags = collection.find(query).sort("_id", pymongo.DESCENDING)[0]
        # case insensitive search
        for key in tags["Tags"]:
            if version.lower() == key.lower():
                return tags["Tags"][key]["Value"]

        self._logger.error(
            f"Invalid model version {version} in DB {self._get_db_name(db_name)} "
            f"(allowed are {_all_versions})"
        )
        raise ValueError

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
        parameter=None,
        telescope_model_name=None,
        site=None,
        collection="telescopes",
    ):
        """
        Get all version entries in the DB of collection and/or a specific parameter.

        Parameters
        ----------
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

        _cache_key = f"model_versions_{self._get_db_name()}-{collection}"

        query = {}
        if parameter is not None:
            query["parameter"] = parameter
            _cache_key = f"{_cache_key}-{parameter}"
        if collection == "telescopes" and telescope_model_name is not None:
            query["instrument"] = names.validate_telescope_name(telescope_model_name)
            _cache_key = f"{_cache_key}-{query['instrument']}"
        elif collection == "sites" and site is not None:
            query["site"] = names.validate_site_name(site)
            _cache_key = f"{_cache_key}-{query['site']}"

        if _cache_key not in DatabaseHandler.model_versions_cached:
            self._logger.debug(f"Getting all versions from {_cache_key}")
            db_collection = DatabaseHandler.db_client[self._get_db_name()][collection]
            DatabaseHandler.model_versions_cached[_cache_key] = list(
                set(post["version"] for post in db_collection.find(query))
            )
            if len(DatabaseHandler.model_versions_cached[_cache_key]) == 0:
                self._logger.warning(
                    f"The query {query} did not return any results. No versions found"
                )

        return DatabaseHandler.model_versions_cached[_cache_key]

    def get_all_available_array_elements(self, model_version, collection, db_name=None):
        """
        Get all available array element names in the specified collection in the DB.

        Parameters
        ----------
        db_name: str
            the name of the DB
        model_version: str
            Which version to get the array elements of
        collection: str
            Which collection to get the array elements from:
            i.e. telescopes, calibration_devices
        Returns
        -------
        _available_array_elements: list
            List of all array element names found in collection

        """
        db_name = self._get_db_name(db_name)
        collection = DatabaseHandler.db_client[db_name][collection]

        query = {"version": self.model_version(model_version)}
        try:
            _all_available_array_elements = collection.find(query).distinct("instrument")
        except ValueError as exc:
            raise ValueError(f"Query for collection name {collection} not implemented.") from exc

        return _all_available_array_elements

    def get_telescope_list_for_db_query(self, telescope_model_name, model_version):
        """
        Returns a list of telescope names to be used for querying the database.
        The first entry of the list is the design telescope (if it exists in the DB),
        followed by the actual telescope model name.

        Parameters
        ----------
        telescope_model_name: str
            Name of the telescope model (e.g. MSTN-01).
        model_version: str
            Model version.

        Returns
        -------
        list
            List of telescope model names as used in the DB.

        """

        if "-design" in telescope_model_name:
            return [telescope_model_name]
        try:
            return [
                self.get_telescope_db_name(
                    names.get_telescope_type_from_telescope_name(telescope_model_name) + "-design",
                    model_version,
                ),
                self.get_telescope_db_name(telescope_model_name, model_version),
            ]
        except ValueError:  # e.g., no design model defined for this telescope type
            return [
                self.get_telescope_db_name(telescope_model_name, model_version),
            ]

    def get_telescope_db_name(self, telescope_name, model_version, collection="telescopes"):
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
        collection: str
            collection of array element (e.g. telescopes, calibration_devices)

        Returns
        -------
        str
            Telescope model name as used in the DB.

        Raises
        ------
        ValueError
            If the telescope name is not found in the database.

        """

        if self._available_array_elements is None:
            self._available_array_elements = self.get_all_available_array_elements(
                model_version, collection
            )
        _telescope_name_validated = names.validate_telescope_name(telescope_name)
        if _telescope_name_validated in self._available_array_elements:
            return _telescope_name_validated
        _design_name = (
            f"{names.get_telescope_type_from_telescope_name(_telescope_name_validated)}-design"
        )
        if _design_name in self._available_array_elements:
            return _design_name

        raise ValueError

    def _parameter_cache_key(self, site, telescope, model_version, db_name=None):
        """
        Create a cache key for the parameter cache dictionaries.

        """
        _model_version = self.model_version(model_version, db_name=db_name)

        if telescope is None:
            return f"{site}-{_model_version}"
        return f"{site}-{telescope}-{_model_version}"

    def _reset_parameter_cache(self, site, telescope, model_version, db_name=None):
        """
        Reset the cache for the parameters.

        """
        self._logger.debug(f"Resetting cache for {site} {telescope} {model_version}")
        _cache_key = self._parameter_cache_key(site, telescope, model_version, db_name)
        DatabaseHandler.site_parameters_cached.pop(_cache_key, None)
        DatabaseHandler.model_parameters_cached.pop(_cache_key, None)

    def get_collections(self, db_name=None):
        """
        List of collections in the DB.

        Returns
        -------
        list
            List of collection names

        """
        db_name = self._get_db_name() if db_name is None else db_name
        if db_name not in self.list_of_collections:
            self.list_of_collections[db_name] = DatabaseHandler.db_client[
                db_name
            ].list_collection_names()
        return self.list_of_collections[db_name]
