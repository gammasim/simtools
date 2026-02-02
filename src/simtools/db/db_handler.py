"""Module to handle interaction with DB."""

import logging
from collections import defaultdict
from pathlib import Path

from simtools import settings
from simtools.data_model import validate_data
from simtools.db.mongo_db import MongoDBHandler
from simtools.io import io_handler
from simtools.simtel import simtel_table_reader
from simtools.utils import names, value_conversion
from simtools.version import resolve_version_to_latest_patch


class DatabaseHandler:
    """
    DatabaseHandler provides the interface to the DB.

    Note the two types of version variables used in this class:

    - db_simulation_model_version (from db_config): version of the simulation model database
    - model_version (from production_tables): version of the model contained in the database
    """

    ALLOWED_FILE_EXTENSIONS = [".dat", ".txt", ".lis", ".cfg", ".yml", ".yaml", ".ecsv"]

    production_table_cached = {}
    model_parameters_cached = {}
    model_versions_cached = {}

    def __init__(self):
        """Initialize the DatabaseHandler class."""
        self._logger = logging.getLogger(__name__)

        self.db_config = (
            MongoDBHandler.validate_db_config(dict(settings.config.db_config))
            if settings.config.db_config
            else None
        )
        self.io_handler = io_handler.IOHandler()
        self.mongo_db_handler = MongoDBHandler(self.db_config) if self.db_config else None

        self.db_name = (
            MongoDBHandler.get_db_name(
                db_simulation_model_version=self.db_config.get("db_simulation_model_version"),
                model_name=self.db_config.get("db_simulation_model"),
            )
            if self.db_config
            else None
        )

    def is_configured(self):
        """
        Check if the DatabaseHandler is configured.

        Returns
        -------
        bool
            True if the DatabaseHandler is configured, False otherwise.
        """
        return self.mongo_db_handler is not None

    def get_db_name(self, db_name=None, db_simulation_model_version=None, model_name=None):
        """Build DB name from configuration."""
        if db_name:
            return db_name
        if db_simulation_model_version and model_name:
            return MongoDBHandler.get_db_name(
                db_simulation_model_version=db_simulation_model_version,
                model_name=model_name,
            )
        if not (db_simulation_model_version or model_name):
            return self.db_name
        return None

    def print_connection_info(self):
        """Print the connection information."""
        if self.mongo_db_handler:
            self.mongo_db_handler.print_connection_info(self.db_name)
        else:
            self._logger.info("No database defined.")

    def is_remote_database(self):
        """
        Check if the database is remote.

        Check for domain pattern like "cta-simpipe-protodb.zeuthen.desy.de"

        Returns
        -------
        bool
            True if the database is remote, False otherwise.
        """
        return bool(self.mongo_db_handler and self.mongo_db_handler.is_remote_database())

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
        """
        self.mongo_db_handler.generate_compound_indexes_for_databases(
            db_name, db_simulation_model, db_simulation_model_version
        )

    def get_model_parameter(
        self,
        parameter,
        site,
        array_element_name,
        parameter_version=None,
        model_version=None,
    ):
        """
        Get a single model parameter using model or parameter version.

        Note that this function should not be called in a loop for many parameters,
        as it each call queries the database.

        Parameters
        ----------
        parameter: str
            Name of the parameter.
        site: str
            Site name.
        array_element_name: str
            Name of the array element model.
        parameter_version: str
            Version of the parameter.
        model_version: str
            Version of the model.

        Returns
        -------
        dict containing the parameter

        """
        collection_name = names.get_collection_name_from_parameter_name(parameter)
        if model_version:
            if isinstance(model_version, list):
                raise ValueError(
                    "Only one model version can be passed to get_model_parameter, not a list."
                )
            model_version = resolve_version_to_latest_patch(
                model_version, self.get_model_versions(collection_name)
            )
            production_table = self.read_production_table_from_db(collection_name, model_version)
            array_element_list = self._get_array_element_list(
                array_element_name, site, production_table, collection_name
            )
            for array_element in reversed(array_element_list):
                parameter_version = (
                    production_table["parameters"].get(array_element, {}).get(parameter)
                )
                if parameter_version:
                    array_element_name = array_element
                    break

        query = {
            "parameter_version": parameter_version,
            "parameter": parameter,
        }
        if array_element_name:
            query["instrument"] = array_element_name
        if site:
            query["site"] = site
        return self._read_db(query=query, collection_name=collection_name)

    def get_model_parameters(self, site, array_element_name, collection, model_version):
        """
        Get model parameters using the model version.

        Queries parameters for design and for the specified array element (if necessary).

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. LSTN-01, MSTx-FlashCam, ILLN-01).
        model_version: str, list
            Version(s) of the model.
        collection: str
            Collection of array element (e.g. telescopes, calibration_devices).

        Returns
        -------
        dict containing the parameters
        """
        pars = {}
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection)
        )
        production_table = self.read_production_table_from_db(collection, model_version)
        array_element_list = self._get_array_element_list(
            array_element_name, site, production_table, collection
        )
        for array_element in array_element_list:
            pars.update(
                self._get_parameter_for_model_version(
                    array_element, model_version, site, collection, production_table
                )
            )
        return pars

    def get_model_parameters_for_all_model_versions(self, site, array_element_name, collection):
        """
        Get model parameters for all model versions.

        Parameters
        ----------
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. LSTN-01, MSTx-FlashCam, ILLN-01).
        collection: str
            Collection of array element (e.g. telescopes, calibration_devices).

        Returns
        -------
        dict containing the parameters with model version as first key
        """
        pars = defaultdict(dict)
        for _model_version in self.get_model_versions(collection):
            try:
                parameter_data = self.get_model_parameters(
                    site, array_element_name, collection, _model_version
                )
                pars[_model_version].update(parameter_data)
            except KeyError:
                self._logger.debug(
                    f"Skipping model version {_model_version} - {array_element_name} not found"
                )
                continue
        return pars

    def _get_parameter_for_model_version(
        self, array_element, model_version, site, collection, production_table
    ):
        """
        Get parameters for a specific model version and array element.

        Uses caching wherever possible.
        """
        cache_key, cache_dict = self._read_cache(
            DatabaseHandler.model_parameters_cached,
            names.validate_site_name(site) if site else None,
            array_element,
            model_version,
            collection,
        )
        if cache_dict:
            return cache_dict

        try:
            parameter_version_table = production_table["parameters"][array_element]
        except KeyError:  # allow missing array elements (parameter dict is checked later)
            return {}
        DatabaseHandler.model_parameters_cached[cache_key] = self._read_db(
            query=self._get_query_from_parameter_version_table(
                parameter_version_table, array_element, site
            ),
            collection_name=collection,
        )
        return DatabaseHandler.model_parameters_cached[cache_key]

    def get_collection(self, collection_name, db_name=None):
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
        db_name = db_name or self.db_name
        return self.mongo_db_handler.get_collection(collection_name, db_name)

    def get_collections(self, db_name=None, model_collections_only=False):
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
        return self.mongo_db_handler.get_collections(
            db_name or self.db_name, model_collections_only
        )

    def export_model_file(
        self,
        parameter,
        site,
        array_element_name,
        model_version=None,
        parameter_version=None,
        export_file_as_table=False,
    ):
        """
        Export single model file from the DB identified by the parameter name.

        The parameter can be identified by model or parameter version.
        Files can be exported as astropy tables (ecsv format).

        Parameters
        ----------
        parameter: str
            Name of the parameter.
        site: str
            Site name.
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        parameter_version: str
            Version of the parameter.
        model_version: str
            Version of the model.
        export_file_as_table: bool
            If True, export the file as an astropy table (ecsv format).

        Returns
        -------
        astropy.table.Table or None
            If export_file_as_table is True
        """
        parameters = self.get_model_parameter(
            parameter,
            site,
            array_element_name,
            parameter_version=parameter_version,
            model_version=model_version,
        )
        self.export_model_files(parameters=parameters, dest=self.io_handler.get_output_directory())
        if export_file_as_table:
            return simtel_table_reader.read_simtel_table(
                parameter,
                self.io_handler.get_output_directory().joinpath(parameters[parameter]["value"]),
            )
        return None

    def export_model_files(self, parameters=None, file_names=None, dest=None, db_name=None):
        """
        Export models files from the DB to given directory.

        The files to be exported can be specified by file_name or retrieved from the database
        using the parameters dictionary.

        Parameters
        ----------
        parameters: dict
            Dict of model parameters
        file_names: list, str
            List (or string) of file names to export
        dest: str or Path
            Location where to write the files to.

        Returns
        -------
        file_id: dict of GridOut._id
            Dict of database IDs of files.
        """
        db_name = db_name or self.db_name

        if file_names:
            file_names = [file_names] if not isinstance(file_names, list) else file_names
        elif parameters:
            file_names = [
                info["value"]
                for info in parameters.values()
                if info and info.get("file") and info["value"] is not None
            ]

        instance_ids = {}
        for file_name in file_names:
            if Path(dest).joinpath(file_name).exists():
                instance_ids[file_name] = "file exists"
            else:
                file_path_instance = self.mongo_db_handler.get_file_from_db(db_name, file_name)
                self._write_file_from_db_to_disk(db_name, dest, file_path_instance)
                instance_ids[file_name] = file_path_instance._id  # pylint: disable=protected-access
        return instance_ids

    def _get_query_from_parameter_version_table(
        self, parameter_version_table, array_element_name, site
    ):
        """Return query based on parameter version table."""
        query_dict = {
            "$or": [
                {"parameter": param, "parameter_version": version}
                for param, version in parameter_version_table.items()
            ],
        }
        # 'xSTX-design' is a placeholder to ignore 'instrument' field in query.
        if array_element_name and array_element_name != "xSTx-design":
            query_dict["instrument"] = array_element_name
        if site:
            query_dict["site"] = site
        return query_dict

    def _read_db(self, query, collection_name):
        """
        Query MongoDB.

        Parameters
        ----------
        query: dict
            Query to execute.
        collection_name: str
            Collection name.

        Returns
        -------
        dict containing the parameters

        Raises
        ------
        ValueError
            if query returned no results.
        """
        posts = self.mongo_db_handler.query_db(query, collection_name, self.db_name)
        parameters = {}
        for post in posts:
            par_now = post["parameter"]
            parameters[par_now] = post
            parameters[par_now]["entry_date"] = self.mongo_db_handler.get_entry_date_from_document(
                post
            )
        return {k: parameters[k] for k in sorted(parameters)}

    def read_production_table_from_db(self, collection_name, model_version):
        """
        Read production table for the given collection from MongoDB.

        Parameters
        ----------
        collection_name: str
            Name of the collection.
        model_version: str
            Version of the model.

        Raises
        ------
        ValueError
            if query returned no results.
        """
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection_name)
        )
        try:
            return DatabaseHandler.production_table_cached[
                self._cache_key(None, None, model_version, collection_name)
            ]
        except KeyError:
            pass

        query = {"model_version": model_version, "collection": collection_name}
        post = self.mongo_db_handler.find_one(query, "production_tables", self.db_name)
        if not post:
            raise ValueError(f"The following query returned zero results: {query}")

        return {
            "collection": post["collection"],
            "model_version": post["model_version"],
            "parameters": post["parameters"],
            "design_model": post.get("design_model", {}),
            "entry_date": self.mongo_db_handler.get_entry_date_from_document(post),
        }

    def get_model_versions(self, collection_name="telescopes"):
        """
        Get list of model versions from the DB with caching.

        Parameters
        ----------
        collection_name: str
            Name of the collection.

        Returns
        -------
        list
            List of model versions
        """
        if collection_name not in DatabaseHandler.model_versions_cached:
            collection = self.get_collection("production_tables", db_name=self.db_name)
            DatabaseHandler.model_versions_cached[collection_name] = sorted(
                {post["model_version"] for post in collection.find({"collection": collection_name})}
            )
        return DatabaseHandler.model_versions_cached[collection_name]

    def get_array_elements(self, model_version, collection="telescopes"):
        """
        Get list array elements for a given model version and collection from the DB.

        Parameters
        ----------
        model_version: str
            Version of the model.
        collection: str
            Which collection to get the array elements from:
            i.e. telescopes, calibration_devices.

        Returns
        -------
        list
            Sorted list of all array elements found in collection
        """
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection)
        )
        production_table = self.read_production_table_from_db(collection, model_version)
        return sorted([entry for entry in production_table["parameters"] if "-design" not in entry])

    def get_design_model(self, model_version, array_element_name, collection="telescopes"):
        """
        Get the design model used for a given array element and a given model version.

        Parameters
        ----------
        model_version: str
            Version of the model.
        array_element_name: str
            Name of the array element model (e.g. MSTN, SSTS).
        collection: str
            Which collection to get the array elements from:
            i.e. telescopes, calibration_devices.

        Returns
        -------
        str
            Design model for a given array element.
        """
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection)
        )
        production_table = self.read_production_table_from_db(collection, model_version)
        try:
            return production_table["design_model"][array_element_name]
        except KeyError:
            # for eg. array_element_name == 'LSTN-design' returns 'LSTN-design'
            return array_element_name

    def get_array_elements_of_type(self, array_element_type, model_version, collection):
        """
        Get array elements of a certain type (e.g. 'LSTN') for a DB collection.

        Does not return 'design' models.

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
        model_version = resolve_version_to_latest_patch(
            model_version, self.get_model_versions(collection)
        )
        production_table = self.read_production_table_from_db(collection, model_version)
        all_array_elements = production_table["parameters"]
        return sorted(
            [
                entry
                for entry in all_array_elements
                if entry.startswith(array_element_type) and "-design" not in entry
            ]
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
            return self.get_model_parameters(
                None,
                None,
                model_version=model_version,
                collection="configuration_corsika",
            )
        if simulation_software == "sim_telarray":
            return (
                self.get_model_parameters(
                    site,
                    array_element_name,
                    model_version=model_version,
                    collection="configuration_sim_telarray",
                )
                if site and array_element_name
                else {}
            )
        raise ValueError(f"Unknown simulation software: {simulation_software}")

    def _write_file_from_db_to_disk(self, db_name, path, file):
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
        self.mongo_db_handler.write_file_from_db_to_disk(db_name, path, file)

    def get_ecsv_file_as_astropy_table(self, file_name, db_name=None):
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
        return self.mongo_db_handler.get_ecsv_file_as_astropy_table(
            file_name, db_name or self.db_name
        )

    def add_production_table(self, production_table, db_name=None):
        """
        Add a production table to the DB.

        Parameters
        ----------
        production_table: dict
            The production table to add to the DB.
        db_name: str
            the name of the DB.
        """
        self._logger.debug(f"Adding production for {production_table.get('collection')} to the DB")
        self.mongo_db_handler.insert_one(
            production_table, "production_tables", db_name or self.db_name
        )
        DatabaseHandler.production_table_cached.clear()
        DatabaseHandler.model_versions_cached.clear()

    def add_new_parameter(
        self,
        par_dict,
        db_name=None,
        collection_name="telescopes",
        file_prefix=None,
    ):
        """
        Add a new parameter dictionary to the DB.

        A new document will be added to the DB, with all fields taken from the input parameters.
        Parameter dictionaries are validated before submission using the corresponding schema.

        Parameters
        ----------
        par_dict: dict
            dictionary with parameter data
        db_name: str
            the name of the DB
        collection_name: str
            The name of the collection to add a parameter to.
        file_prefix: str or Path
            where to find files to upload to the DB
        """
        par_dict = validate_data.DataValidator.validate_model_parameter(par_dict)

        db_name = db_name or self.db_name

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

        self._logger.debug(
            f"Adding a new entry to DB {db_name} and collection {collection_name}:\n{par_dict}"
        )
        self.mongo_db_handler.insert_one(par_dict, collection_name, db_name)

        for file_to_insert_now in files_to_add_to_db:
            self._logger.debug(f"Will also add the file {file_to_insert_now} to the DB")
            self.insert_file_to_db(file_to_insert_now, db_name)

        self._reset_parameter_cache()

    def insert_file_to_db(self, file_name, db_name=None):
        """
        Insert a file to the DB.

        Parameters
        ----------
        file_name: str or Path
            The name of the file to insert (full path).
        db_name: str
            the name of the DB

        Returns
        -------
        file_id: GridOut._id
            If the file exists, return its GridOut._id, otherwise insert the file and return
            its newly created DB GridOut._id.
        """
        return self.mongo_db_handler.insert_file_to_db(file_name, db_name or self.db_name)

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

    def _read_cache(
        self, cache_dict, site=None, array_element_name=None, model_version=None, collection=None
    ):
        """
        Read parameters from cache.

        Parameters
        ----------
        cache_dict: dict
            Cache dictionary.
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
        cache_key = self._cache_key(site, array_element_name, model_version, collection)
        try:
            return cache_key, cache_dict[cache_key]
        except KeyError:
            return cache_key, None

    def _reset_parameter_cache(self):
        """Reset the cache for the parameters."""
        DatabaseHandler.model_parameters_cached.clear()
        DatabaseHandler.model_versions_cached.clear()

    def _get_array_element_list(self, array_element_name, site, production_table, collection):
        """
        Return list of array elements for DB queries (add design model if needed).

        Design model is added if found in the production table.

        Parameters
        ----------
        array_element_name: str
            Name of the array element.
        site: str
            Site name.
        production_table: dict
            Production table.
        collection: str
            collection of array element (e.g. telescopes, calibration_devices).

        Returns
        -------
        list
            List of array elements
        """
        if collection == "configuration_corsika":
            return ["xSTx-design"]  # placeholder to ignore 'instrument' field in query.
        if collection == "sites":
            return [f"OBS-{site}"]
        if names.is_design_type(array_element_name):
            return [array_element_name]
        if collection == "configuration_sim_telarray":
            # get design model from 'telescope' or 'calibration_device' production tables
            production_table = self.read_production_table_from_db(
                names.get_collection_name_from_array_element_name(array_element_name),
                production_table["model_version"],
            )
        try:
            return [
                production_table["design_model"][array_element_name],
                array_element_name,
            ]
        except KeyError as exc:
            # simplified model definitions when e.g. adding new telescopes without design model
            if settings.config.args.get("ignore_missing_design_model", False):
                element_type = names.get_array_element_type_from_name(array_element_name)
                return [array_element_name, f"{element_type}-01", f"{element_type}-design"]
            raise KeyError(
                f"Failed generated array element list for db query for {array_element_name}"
            ) from exc
