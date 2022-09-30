""" Module to handle interaction with DB. """

import logging
from pathlib import Path
from threading import Lock

import gridfs
import pymongo
import yaml
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

import simtools.config as cfg
from simtools.util import names
from simtools.util.model import getTelescopeClass

__all__ = ["DatabaseHandler"]


class DatabaseHandler:
    """
    DatabaseHandler provides the interface to the DB.

    Attributes
    ----------
    mode: str
        Yaml or MongoDB, only these two options are allowed.

    Methods
    -------
    getModelParameters()
        Get the model parameters of a specific telescope with a specific version.
    getSiteParameters()
        Get the site parameters of a specific version of a site.
    copyTelescope()
        Copy a full telescope configuration of a specific version to a new telescope name.
    deleteQuery()
        Delete all entries from the DB which correspond to the provided query.
    updateParameter()
        Update a parameter value for a specific telescope/version.
    updateParameterField()
        Update a parameter field other than value for a specific telescope/version.
    addParameter()
        Add a parameter value for a specific telescope.
    addNewParamete()
        Add a new parameter for a specific telescope.
    insertFileToDB()
        Insert a file to the DB.
    insertFilesToDB()
        Insert a list of files to the DB.
    exportFileDB()
        Get a file from the DB and write it to disk.
    """

    # TODO move into config file?
    DB_TABULATED_DATA = "CTA-Simulation-Model"
    DB_CTA_SIMULATION_MODEL = "CTA-Simulation-Model"
    DB_CTA_SIMULATION_MODEL_DESCRIPTIONS = "CTA-Simulation-Model-Descriptions"
    DB_REFERENCE_DATA = "CTA-Reference-Data"
    DB_DERIVED_VALUES = "CTA-Simulation-Model-Derived-Values"

    ALLOWED_FILE_EXTENSIONS = [".dat", ".txt", ".lis", ".cfg", ".yml", ".ecsv"]

    dbClient = None

    def __init__(self):
        """
        Initialize the DatabaseHandler class.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Initialize DatabaseHandler")

        if cfg.get("useMongoDB"):
            if DatabaseHandler.dbClient is None:
                with Lock():
                    self.dbDetails = self._readDetailsMongoDB()
                    DatabaseHandler.dbClient = self._openMongoDB()

    # END of _init_

    @staticmethod
    def _readDetailsMongoDB():
        """
        Read the MongoDB details (server, user, pass, etc.) from an external file.

        Returns
        -------
        dbDetails: dict
            Dictionary containing the DB details.
        """

        dbDetails = dict()
        dbDetailsFile = cfg.get("mongoDBConfigFile")
        with open(dbDetailsFile, "r") as stream:
            dbDetails = yaml.safe_load(stream)

        return dbDetails

    def _openMongoDB(self):
        """
        Open a connection to MongoDB and return the client to read/write to the DB with.

        Returns
        -------
        A PyMongo DB client
        """
        _dbClient = MongoClient(
            self.dbDetails["mongodbServer"],
            port=self.dbDetails["dbPort"],
            username=self.dbDetails["userDB"],
            password=self.dbDetails["passDB"],
            authSource=self.dbDetails["authenticationDatabase"],
            ssl=True,
            tlsallowinvalidhostnames=True,
            tlsallowinvalidcertificates=True,
        )

        return _dbClient

    @staticmethod
    def _getTelescopeModelNameForDB(site, telescopeModelName):
        """Make telescope name as the DB needs from site and telescopeModelName."""
        return site + "-" + telescopeModelName

    def getModelParameters(
        self,
        site,
        telescopeModelName,
        modelVersion,
        onlyApplicable=False,
    ):
        """
        Get parameters from either MongoDB or Yaml DB for a specific telescope.

        Parameters
        ----------
        site: str
            South or North.
        telescopeModelName: str
            Name of the telescope model (e.g. LST-1, MST-FlashCam-D)
        modelVersion: str
            Version of the model.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """

        _siteValidated = names.validateSiteName(site)
        _telModelNameValidated = names.validateTelescopeModelName(telescopeModelName)

        if cfg.get("useMongoDB"):

            # Only MongoDB supports tagged version
            _modelVersion = self._convertVersionToTagged(
                modelVersion, DatabaseHandler.DB_CTA_SIMULATION_MODEL
            )
            _versionValidated = names.validateModelVersionName(_modelVersion)

            _pars = self._getModelParametersMongoDB(
                DatabaseHandler.DB_CTA_SIMULATION_MODEL,
                _siteValidated,
                _telModelNameValidated,
                _versionValidated,
                onlyApplicable,
            )
            return _pars
        else:
            _versionValidated = names.validateModelVersionName(modelVersion)

            return self._getModelParametersYaml(
                _siteValidated,
                _telModelNameValidated,
                _versionValidated,
                onlyApplicable,
            )

    def exportFileDB(self, dbName, dest, fileName):
        """
        Get a file from the DB and write it to disk.

        Parameters
        ----------
        dbName: str
            Name of the DB to search in.
        dest: str or Path
            Location where to write the file to.
        fileName: str
            Name of the file to get.
        """

        self._logger.debug(f"Getting {fileName} and writing it to {dest}")
        file = self._getFileMongoDB(dbName, fileName)
        self._writeFileFromMongoToDisk(dbName, dest, file)

    def exportModelFiles(self, parameters, dest):
        """
        Export all the files in a model from the DB (Mongo or yaml) and write them to disk.

        Parameters
        ----------
        parameters: dict
            Dict of model parameters
        dest: str or Path
            Location where to write the files to.
        """

        if cfg.get("useMongoDB"):
            self._logger.debug("Exporting model files from MongoDB")
            for info in parameters.values():
                if not info["File"]:
                    continue
                file = self._getFileMongoDB(DatabaseHandler.DB_CTA_SIMULATION_MODEL, info["Value"])
                self._writeFileFromMongoToDisk(DatabaseHandler.DB_CTA_SIMULATION_MODEL, dest, file)
        else:
            self._logger.debug("Exporting model files from local model file directories")
            for value in parameters.values():

                if not self._isFile(value):
                    continue
                self._writeModelFileYaml(value, dest, noFileOk=True)

    @staticmethod
    def _isFile(value):
        """Verify if a parameter value is a file name."""
        return any(ext in str(value) for ext in DatabaseHandler.ALLOWED_FILE_EXTENSIONS)

    def _writeModelFileYaml(self, fileName, destDir, noFileOk=False):
        """
        Find the fileName in the model files location and write a copy
        at the destDir directory.

        Parameters
        ----------
        fileName: str or Path
            File name to be found and copied.
        destDir: str or Path
            Path of the directory where the file will be written.
        """

        destFile = Path(destDir).joinpath(fileName)
        try:
            file = cfg.findFile(fileName, cfg.get("modelFilesLocations"))
        except FileNotFoundError:
            if noFileOk:
                self._logger.debug("File {} not found but noFileOk".format(fileName))
                return
            else:
                raise

        destFile.write_text(file.read_text())

    def _getModelParametersYaml(self, site, telescopeModelName, modelVersion, onlyApplicable=False):
        """
        Get parameters from DB for one specific type.

        Parameters
        ----------
        site: str
            North or South.
        telescopeModelName: str
            Telescope model name (e.g MST-FlashCam-D ...).
        modelVersion: str
            Version of the model.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """

        _telClass = getTelescopeClass(telescopeModelName)
        _telNameConverted = names.convertTelescopeModelNameToYaml(telescopeModelName)

        if _telClass == "MST":
            # MST-FlashCam or MST-NectarCam
            _whichTelLabels = [_telNameConverted, "MST-optics"]
        elif _telClass == "SST":
            # SST = SST-Camera + SST-Structure
            _whichTelLabels = ["SST-Camera", "SST-Structure"]
        else:
            _whichTelLabels = [_telNameConverted]

        # Selecting version and applicable (if on)
        _pars = dict()
        for _tel in _whichTelLabels:
            _allPars = self._getAllModelParametersYaml(_tel)

            # If _tel is a structure, only the applicable parameters will be collected, always.
            # The default ones will be covered by the camera parameters.
            _selectOnlyApplicable = onlyApplicable or (_tel in ["MST-optics", "SST-Structure"])

            for parNameIn, parInfo in _allPars.items():

                if not parInfo["Applicable"] and _selectOnlyApplicable:
                    continue

                if modelVersion not in parInfo:
                    continue

                _pars[parNameIn] = parInfo[modelVersion]

        return _pars

    def _getModelParametersMongoDB(
        self, dbName, site, telescopeModelName, modelVersion, onlyApplicable=False
    ):
        """
        Get parameters from MongoDB for a specific telescope.

        Parameters
        ----------
        dbName: str
            the name of the DB
        site: str
            South or North.
        telescopeModelName: str
            Name of the telescope model (e.g. MST-FlashCam-D ...)
        modelVersion: str
            Version of the model.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """

        _siteValidated = names.validateSiteName(site)
        _telNameDB = self._getTelescopeModelNameForDB(_siteValidated, telescopeModelName)
        _telClass = getTelescopeClass(telescopeModelName)

        self._logger.debug("TelNameDB: {}".format(_telNameDB))
        self._logger.debug("TelClass: {}".format(_telClass))

        if _telClass == "MST":
            # MST-FlashCam or MST-NectarCam
            _whichTelLabels = ["{}-MST-Structure-D".format(_siteValidated), _telNameDB]
        elif _telClass == "SST":
            # SST = SST-Camera + SST-Structure
            _whichTelLabels = [
                "{}-SST-Camera-D".format(_siteValidated),
                "{}-SST-Structure-D".format(_siteValidated),
            ]
        else:
            _whichTelLabels = [_telNameDB]

        # Selecting version and applicable (if on)
        _pars = dict()
        for _tel in _whichTelLabels:
            self._logger.debug("Getting {} parameters from MongoDB".format(_tel))

            # If tel is a structure, only applicable pars will be collected, always.
            # The default ones will be covered by the camera pars.
            _selectOnlyApplicable = onlyApplicable or (
                _tel
                in [
                    "{}-MST-Structure-D".format(_siteValidated),
                    "{}-SST-Structure-D".format(_siteValidated),
                ]
            )

            _pars.update(
                self.readMongoDB(
                    dbName,
                    _tel,
                    modelVersion,
                    runLocation=None,
                    writeFiles=False,
                    onlyApplicable=_selectOnlyApplicable,
                )
            )

        return _pars

    def readMongoDB(
        self,
        dbName,
        telescopeModelNameDB,
        modelVersion,
        runLocation,
        collectionName="telescopes",
        writeFiles=True,
        onlyApplicable=False,
    ):
        """
        Build and execute query to Read the MongoDB for a specific telescope.
        Also writes the files listed in the parameter values into the sim_telarray run location

        Parameters
        ----------
        dbName: str
            the name of the DB
        telescopeModelNameDB: str
            Name of the telescope model (e.g. MST-FlashCam-D ...)
        modelVersion: str
            Version of the model.
        runLocation: Path or str
            The sim_telarray run location to write the tabulated data files into.
        collectionName: str
            The name of the collection to read from (default is "telescopes")
        writeFiles: bool
            If true, write the files to the runLocation.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """

        collection = DatabaseHandler.dbClient[dbName][collectionName]
        _parameters = dict()

        _modelVersion = self._convertVersionToTagged(
            modelVersion, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "Telescope": telescopeModelNameDB,
            "Version": _modelVersion,
        }

        self._logger.debug("Trying the following query: {}".format(query))
        if onlyApplicable:
            query["Applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query):
            parNow = post["Parameter"]
            _parameters[parNow] = post
            _parameters[parNow].pop("Parameter", None)
            _parameters[parNow].pop("Telescope", None)
            _parameters[parNow]["entryDate"] = ObjectId(post["_id"]).generation_time
            if _parameters[parNow]["File"] and writeFiles:
                file = self._getFileMongoDB(dbName, _parameters[parNow]["Value"])

                self._writeFileFromMongoToDisk(dbName, runLocation, file)

        return _parameters

    def _getAllModelParametersYaml(self, telescopeNameYaml):
        """
        Get all parameters from Yaml DB for one specific type.
        No selection is applied.

        Parameters
        ----------
        telescopeNameYaml: str
            Telescope name as required by the yaml files.

        Returns
        -------
        dict containing the parameters
        """

        _fileNameDB = "parValues-{}.yml".format(telescopeNameYaml)
        _yamlFile = cfg.findFile(_fileNameDB, cfg.get("modelFilesLocations"))
        self._logger.debug("Reading DB file {}".format(_yamlFile))
        with open(_yamlFile, "r") as stream:
            _allPars = yaml.safe_load(stream)
        return _allPars

    def getSiteParameters(
        self,
        site,
        modelVersion,
        onlyApplicable=False,
    ):
        """
        Get parameters from either MongoDB or Yaml DB for a specific site.

        Parameters
        ----------
        site: str
            South or North.
        modelVersion: str
            Version of the model.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """
        _site = names.validateSiteName(site)
        _modelVersion = names.validateModelVersionName(modelVersion)

        if cfg.get("useMongoDB"):
            _pars = self._getSiteParametersMongoDB(
                DatabaseHandler.DB_CTA_SIMULATION_MODEL,
                _site,
                _modelVersion,
                onlyApplicable,
            )
            return _pars
        else:
            return self._getSiteParametersYaml(_site, _modelVersion, onlyApplicable)

    def _getSiteParametersYaml(self, site, modelVersion, onlyApplicable=False):
        """
        Get parameters from DB for a specific type.

        Parameters
        ----------
        site: str
            North or South.
        modelVersion: str
            Version of the model.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """

        siteYaml = "lapalma" if site == "North" else "paranal"

        yamlFile = cfg.findFile("parValues-Sites.yml", cfg.get("modelFilesLocations"))
        self._logger.info("Reading DB file {}".format(yamlFile))
        with open(yamlFile, "r") as stream:
            _allParsVersions = yaml.safe_load(stream)

        _pars = dict()
        for parName, parInfo in _allParsVersions.items():

            if not parInfo["Applicable"] and onlyApplicable:
                continue
            if siteYaml in parName:
                parNameIn = "_".join(parName.split("_")[1:])

                _pars[parNameIn] = parInfo[modelVersion]

        return _pars

    def _getSiteParametersMongoDB(self, dbName, site, modelVersion, onlyApplicable=False):
        """
        Get parameters from MongoDB for a specific telescope.

        Parameters
        ----------
        dbName: str
            The name of the DB.
        site: str
            South or North.
        modelVersion: str
            Version of the model.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """

        _siteValidated = names.validateSiteName(site)
        collection = DatabaseHandler.dbClient[dbName].sites
        _parameters = dict()

        _modelVersion = self._convertVersionToTagged(
            modelVersion, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "Site": _siteValidated,
            "Version": _modelVersion,
        }
        if onlyApplicable:
            query["Applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query):
            parNow = post["Parameter"]
            _parameters[parNow] = post
            _parameters[parNow].pop("Parameter", None)
            _parameters[parNow].pop("Site", None)
            _parameters[parNow]["entryDate"] = ObjectId(post["_id"]).generation_time

        return _parameters

    def getReferenceData(self, site, modelVersion, onlyApplicable=False):
        """
        Get parameters from MongoDB for a specific telescope.

        Parameters
        ----------
        dbName: str
            The name of the DB.
        site: str
            South or North.
        modelVersion: str
            Version of the model.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        """

        _siteValidated = names.validateSiteName(site)
        collection = DatabaseHandler.dbClient[DatabaseHandler.DB_REFERENCE_DATA].reference_values
        _parameters = dict()

        _modelVersion = self._convertVersionToTagged(
            names.validateModelVersionName(modelVersion), DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "Site": _siteValidated,
            "Version": _modelVersion,
        }
        if onlyApplicable:
            query["Applicable"] = True
        if collection.count_documents(query) < 1:
            raise ValueError(
                "The following query returned zero results! Check the input data and rerun.\n",
                query,
            )
        for post in collection.find(query):
            parNow = post["Parameter"]
            _parameters[parNow] = post
            _parameters[parNow].pop("Parameter", None)
            _parameters[parNow].pop("Site", None)
            _parameters[parNow]["entryDate"] = ObjectId(post["_id"]).generation_time

        return _parameters

    def getDerivedValues(self, site, telescopeModelName, modelVersion):
        """
        Get a derived value from the DB for a specific telescope.

        Parameters
        ----------
        site: str
            South or North.
        telescopeModelName: str
            Name of the telescope model (e.g. MST-FlashCam-D ...)
        modelVersion: str
            Version of the model.

        Returns
        -------
        dict containing the parameters
        """

        _siteValidated = names.validateSiteName(site)
        _telModelNameValidated = names.validateTelescopeModelName(telescopeModelName)
        _telNameDB = self._getTelescopeModelNameForDB(_siteValidated, _telModelNameValidated)
        _modelVersion = self._convertVersionToTagged(
            names.validateModelVersionName(modelVersion), DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        self._logger.debug("Getting derived values for {} from the DB".format(_telNameDB))

        _pars = self.readMongoDB(
            DatabaseHandler.DB_DERIVED_VALUES,
            _telNameDB,
            _modelVersion,
            runLocation=None,
            collectionName="derived_values",
            writeFiles=False,
        )

        return _pars

    @staticmethod
    def _getFileMongoDB(dbName, fileName):
        """
        Extract a file from MongoDB and return GridFS file instance

        Parameters
        ----------
        dbName: str
            the name of the DB with files of tabulated data
        fileName: str
            The name of the file requested

        Returns
        -------
        GridOut
            A file instance returned by GridFS find_one
        """

        db = DatabaseHandler.dbClient[dbName]
        fileSystem = gridfs.GridFS(db)
        if fileSystem.exists({"filename": fileName}):
            return fileSystem.find_one({"filename": fileName})
        else:
            raise FileNotFoundError(
                "The file {} does not exist in the database {}".format(fileName, dbName)
            )

    @staticmethod
    def _writeFileFromMongoToDisk(dbName, path, file):
        """
        Extract a file from MongoDB and write it to disk

        Parameters
        ----------
        dbName: str
            the name of the DB with files of tabulated data
        path: str or Path
            The path to write the file to
        file: GridOut
            A file instance returned by GridFS find_one
        """

        db = DatabaseHandler.dbClient[dbName]
        fsOutput = gridfs.GridFSBucket(db)
        with open(Path(path).joinpath(file.filename), "wb") as outputFile:
            fsOutput.download_to_stream_by_name(file.filename, outputFile)

        return

    def copyTelescope(
        self,
        dbName,
        telToCopy,
        versionToCopy,
        newTelName,
        collectionName="telescopes",
        dbToCopyTo=None,
        collectionToCopyTo=None,
    ):
        """
        Copy a full telescope configuration to a new telescope name.
        Only a specific version is copied.
        (This function should be rarely used, probably only during "construction".)

        Parameters
        ----------
        dbName: str
            the name of the DB to copy from
        telToCopy: str
            The telescope to copy
        versionToCopy: str
            The version of the configuration to copy
        newTelName: str
            The name of the new telescope
        collectionName: str
            The name of the collection to copy from (default is "telescopes")
        dbToCopyTo: str
            The name of the DB to copy to (default is the same as dbName)
        collectionToCopyTo: str
            The name of the collection to copy to (default is the same as collection)
        """

        if dbToCopyTo is None:
            dbToCopyTo = dbName

        if collectionToCopyTo is None:
            collectionToCopyTo = collectionName

        self._logger.info(
            "Copying version {} of {} to the new telescope {} in the {} DB".format(
                versionToCopy, telToCopy, newTelName, dbToCopyTo
            )
        )

        collection = DatabaseHandler.dbClient[dbName][collectionName]
        dbEntries = list()

        _versionToCopy = self._convertVersionToTagged(
            versionToCopy, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "Telescope": telToCopy,
            "Version": _versionToCopy,
        }
        for post in collection.find(query):
            post["Telescope"] = newTelName
            post.pop("_id", None)
            dbEntries.append(post)

        self._logger.info("Creating new telescope {}".format(newTelName))
        collection = DatabaseHandler.dbClient[dbToCopyTo][collectionToCopyTo]
        try:
            collection.insert_many(dbEntries)
        except BulkWriteError as exc:
            raise exc(exc.details)

        return

    def copyDocuments(self, dbName, collection, query, dbToCopyTo, collectionToCopyTo=None):
        """
        Copy the documents matching to "query" to the DB "dbToCopyTo".
        The documents are copied to the same collection as in "dbName".
        (This function should be rarely used, probably only during "construction".)

        Parameters
        ----------
        dbName: str
            the name of the DB to copy from
        collection: str
            the name of the collection to copy from
        query: dict
            A dictionary with a query to search for documents to copy.
            For example,
            query = {
                "Telescope": "North-LST-1",
                "Version": "prod4",
            }
            would copy all entries of prod4 version from telescope North-LST-1 to "dbToCopyTo".
        dbToCopyTo: str
            The name of the DB to copy to.
        """

        _collection = DatabaseHandler.dbClient[dbName][collection]
        if collectionToCopyTo is None:
            collectionToCopyTo = collection
        dbEntries = list()

        for post in _collection.find(query):
            post.pop("_id", None)
            dbEntries.append(post)

        self._logger.info(
            "Copying documents matching the following query {}\nto {}".format(query, dbToCopyTo)
        )
        _collection = DatabaseHandler.dbClient[dbToCopyTo][collectionToCopyTo]
        try:
            _collection.insert_many(dbEntries)
        except BulkWriteError as exc:
            raise exc(exc.details)

        return

    def deleteQuery(self, dbName, collection, query):
        """
        Delete all entries from the DB which correspond to the provided query.
        (This function should be rarely used, if at all.)

        Parameters
        ----------
        dbName: str
            the name of the DB
        query: dict
            A dictionary listing the fields/values to delete.
            For example,
            query = {
                "Telescope": "North-LST-1",
                "Version": "prod4",
            }
            would delete the entire prod4 version from telescope North-LST-1.
        """

        _collection = DatabaseHandler.dbClient[dbName][collection]

        if "Version" in query:
            query["Version"] = self._convertVersionToTagged(
                query["Version"], DatabaseHandler.DB_CTA_SIMULATION_MODEL
            )

        self._logger.info(
            "Deleting {} entries from {}".format(
                _collection.count_documents(query),
                dbName,
            )
        )

        _collection.delete_many(query)

        return

    def updateParameter(
        self,
        dbName,
        telescope,
        version,
        parameter,
        newValue,
        collectionName="telescopes",
        filePrefix=None,
    ):
        """
        Update a parameter value for a specific telescope/version.
        (This function should be rarely used since new values
         should ideally have their own version.)

        Parameters
        ----------
        dbName: str
            the name of the DB
        telescope: str
            Which telescope to update
        version: str
            Which version to update
        parameter: str
            Which parameter to update
        newValue: type identical to the original parameter type
            The new value to set for the parameter
        collectionName: str
            The name of the collection in which to update the parameter (default is "telescopes")
        filePrefix: str or Path
            where to find files to upload to the DB
        """

        collection = DatabaseHandler.dbClient[dbName][collectionName]

        _modelVersion = self._convertVersionToTagged(
            version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "Telescope": telescope,
            "Version": _modelVersion,
            "Parameter": parameter,
        }

        parEntry = collection.find_one(query)
        oldValue = parEntry["Value"]

        self._logger.info(
            "For telescope {}, version {}\nreplacing {} value from {} to {}".format(
                telescope, _modelVersion, parameter, oldValue, newValue
            )
        )

        filesToAddToDB = set()
        if self._isFile(newValue):
            file = True
            if filePrefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    "corresponding to the {} parameter, must be provided."
                ).format(parameter)
            filePath = Path(filePrefix).joinpath(newValue)
            filesToAddToDB.add("{}".format(filePath))
            self._logger.info("Will also add the file {} to the DB".format(filePath))
        else:
            file = False

        queryUpdate = {"$set": {"Value": newValue, "File": file}}

        collection.update_one(query, queryUpdate)
        self.insertFilesToDB(filesToAddToDB, dbName)

        return

    def updateParameterField(
        self,
        dbName,
        version,
        parameter,
        field,
        newValue,
        telescope=None,
        site=None,
        collectionName="telescopes",
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
        dbName: str
            the name of the DB
        version: str
            Which version to update
        parameter: str
            Which parameter to update
        field: str
            Field to update (only options are Applicable, units, Type, items, minimum, maximum).
            If the field does not exist, it will be added.
        newValue: type identical to the original field type
            The new value to set to the field given in "field".
        telescope: str
            Which telescope to update, if None then update a site parameter
        site: str, North or South
            Update a site parameter (the telescope argument must be None)
        collectionName: str
            The name of the collection in which to update the parameter (default is "telescopes")
        """

        allowed_fields = ["Applicable", "units", "Type", "items", "minimum", "maximum"]
        if field not in allowed_fields:
            raise ValueError(f"The field to change must be one of {', '.join(allowed_fields)}")

        collection = DatabaseHandler.dbClient[dbName][collectionName]

        _modelVersion = self._convertVersionToTagged(
            version, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "Version": _modelVersion,
            "Parameter": parameter,
        }
        if telescope is not None:
            query["Telescope"] = telescope
            loggerInfo = f"telescope {telescope}"
        elif site is not None and site in ["North", "South"]:
            query["Site"] = site
            loggerInfo = f"site {site}"
        else:
            raise ValueError("You need to specifiy if to update a telescope or a site.")

        parEntry = collection.find_one(query)
        if parEntry is None:
            self._logger.warning(
                "The query {} did not return any results. I will not make any changes.".format(
                    query
                )
            )
            return

        if field in parEntry:
            oldFieldValue = parEntry[field]

            if oldFieldValue == newValue:
                self._logger.warning(
                    f"The value of the field {field} is already {newValue}. No changes necessary"
                )
                return
            else:
                self._logger.info(
                    f"For {loggerInfo}, version {_modelVersion}, parameter {parameter}, "
                    f"replacing field {field} value from {oldFieldValue} to {newValue}"
                )
        else:
            self._logger.info(
                f"For {loggerInfo}, version {_modelVersion}, parameter {parameter}, "
                f"the field {field} does not exist, adding it"
            )

        queryUpdate = {"$set": {field: newValue}}

        collection.update_one(query, queryUpdate)

        return

    def addParameter(
        self,
        dbName,
        telescope,
        parameter,
        newVersion,
        newValue,
        collectionName="telescopes",
        filePrefix=None,
    ):
        """
        Add a parameter value for a specific telescope.
        A new document will be added to the DB,
        with all fields taken from the last entry of this parameter to this telescope,
        except the ones changed.

        Parameters
        ----------
        dbName: str
            the name of the DB
        telescope: str
            Which telescope to update
        parameter: str
            Which parameter to add
        newVersion: str
            The version of the new parameter value
        newValue: type identical to the original parameter type
            The new value to set for the parameter
        collectionName: str
            The name of the collection to which to add a parameter (default is "telescopes")
        filePrefix: str or Path
            where to find files to upload to the DB
        """

        collection = DatabaseHandler.dbClient[dbName][collectionName]

        _newVersion = self._convertVersionToTagged(
            newVersion, DatabaseHandler.DB_CTA_SIMULATION_MODEL
        )

        query = {
            "Telescope": telescope,
            "Parameter": parameter,
        }

        parEntry = collection.find(query).sort("_id", pymongo.DESCENDING)[0]
        parEntry["Value"] = newValue
        parEntry["Version"] = _newVersion
        parEntry.pop("_id", None)

        filesToAddToDB = set()
        if self._isFile(newValue):
            parEntry["File"] = True
            if filePrefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    "corresponding to the {} parameter, must be provided."
                ).format(parameter)
            filePath = Path(filePrefix).joinpath(newValue)
            filesToAddToDB.add("{}".format(filePath))
        else:
            parEntry["File"] = False

        self._logger.info("Will add the following entry to DB:\n{}".format(parEntry))

        collection.insert_one(parEntry)
        if len(filesToAddToDB) > 0:
            self._logger.info("Will also add the file {} to the DB".format(filePath))
            self.insertFilesToDB(filesToAddToDB, dbName)

        return

    def addNewParameter(
        self,
        dbName,
        telescope,
        version,
        parameter,
        value,
        site=None,
        collectionName="telescopes",
        filePrefix=None,
        **kwargs,
    ):
        """
        Add a parameter value for a specific telescope.
        A new document will be added to the DB,
        with all fields taken from the input parameters.

        Parameters
        ----------
        dbName: str
            the name of the DB
        telescope: str
            The name of the telescope to add a parameter to.
        parameter: str
            Which parameter to add
        version: str
            The version of the new parameter value
        value: can be any type, preferably given in kwargs
            The value to set for the new parameter
        site: str
           South or North, ignored if collectionName is "telescopes".
        collectionName: str
            The name of the collection to add a parameter to (default is "telescopes")
        filePrefix: str or Path
            where to find files to upload to the DB
        kwargs: dict
            Any additional fields to add to the parameter
        """

        collection = DatabaseHandler.dbClient[dbName][collectionName]

        dbEntry = dict()
        if "telescopes" in collectionName:
            dbEntry["Telescope"] = names.validateTelescopeNameDB(telescope)
        elif "sites" in collectionName:
            dbEntry["Site"] = names.validateSiteName(site)
        else:
            raise ValueError("Can only add new parameters to the sites or telescopes collections")

        dbEntry["Version"] = version
        dbEntry["Parameter"] = parameter
        dbEntry["Value"] = value
        dbEntry["Type"] = kwargs["Type"] if "Type" in kwargs else str(type(value))

        filesToAddToDB = set()
        dbEntry["File"] = False
        if self._isFile(value):
            dbEntry["File"] = True
            if filePrefix is None:
                raise FileNotFoundError(
                    "The location of the file to upload, "
                    "corresponding to the {} parameter, must be provided."
                ).format(parameter)
            filePath = Path(filePrefix).joinpath(value)
            filesToAddToDB.add("{}".format(filePath))

        kwargs.pop("Type", None)
        dbEntry.update(kwargs)

        self._logger.info("Will add the following entry to DB:\n{}".format(dbEntry))

        collection.insert_one(dbEntry)
        if len(filesToAddToDB) > 0:
            self._logger.info("Will also add the file {} to the DB".format(filePath))
            self.insertFilesToDB(filesToAddToDB, dbName)

        return

    def _convertVersionToTagged(self, modelVersion, dbName):
        """Convert to tagged version, if needed."""
        if modelVersion in ["Current", "Latest"]:
            return self._getTaggedVersion(dbName, modelVersion)
        else:
            return modelVersion

    @staticmethod
    def _getTaggedVersion(dbName, version="Current"):
        """
        Get the tag of the "Current" or "Latest" version of the MC Model.
        The "Current" is the latest stable MC Model,
        the latest is the latest tag (not necessarily stable, but can be equivalent to "Current").

        Parameters
        ----------
        dbName: str
            the name of the DB
        version: str
            Can be "Current" or "Latest" (default: "Current").

        Returns
        -------
        str
            The version name in the Simulation DB of the requested tag
        """

        if version not in ["Current", "Latest"]:
            raise ValueError('The only default versions are "Current" or "Latest"')

        collection = DatabaseHandler.dbClient[dbName].metadata
        query = {"Entry": "Simulation-Model-Tags"}

        tags = collection.find(query).sort("_id", pymongo.DESCENDING)[0]

        return tags["Tags"][version]["Value"]

    @staticmethod
    def insertFileToDB(file, dbName=DB_CTA_SIMULATION_MODEL, **kwargs):
        """
        Insert a file to the DB.

        Parameters
        ----------
        dbName: str
            the name of the DB
        file: str or Path
            The name of the file to insert (full path).
        **kwargs (optional): keyword arguments for file creation.
            The full list of arguments can be found in, \
            https://docs.mongodb.com/manual/core/gridfs/#the-files-collection
            mostly these are unnecessary though.

        Returns
        -------
        file_id: gridfs "_id"
            If the file exists, returns the "_id" of that one, otherwise creates a new one.
        """

        db = DatabaseHandler.dbClient[dbName]
        fileSystem = gridfs.GridFS(db)

        if "content_type" not in kwargs:
            kwargs["content_type"] = "ascii/dat"
        if "filename" not in kwargs:
            kwargs["filename"] = Path(file).name

        if fileSystem.exists({"filename": kwargs["filename"]}):
            return fileSystem.find_one({"filename": kwargs["filename"]})

        with open(file, "rb") as dataFile:
            file_id = fileSystem.put(dataFile, **kwargs)

        return file_id

    def insertFilesToDB(self, filesToAddToDB, dbName=DB_CTA_SIMULATION_MODEL):
        """
        Insert a list of files to the DB.

        Parameters
        ----------
        dbName: str
            the name of the DB
        filesToAddToDB: list of strings or Paths
            Each entry in the list is the name of the file to insert (full path).
        """

        for fileNow in filesToAddToDB:
            kwargs = {"content_type": "ascii/dat", "filename": Path(fileNow).name}
            self.insertFileToDB(fileNow, dbName, **kwargs)

        return

    def getAllVersions(
        self,
        dbName,
        parameter,
        telescopeModelName=None,
        site=None,
        collectionName="telescopes",
    ):
        """
        Get all version entries in the DB of a telescope or site for a specific parameter.

        Parameters
        ----------
        dbName: str
            the name of the DB
        parameter: str
            Which parameter to get the versions of
        telescopeModelName: str
            Which telescope to get the versions of (in case "collectionName" is "telescopes")
        site: str, North or South
            In case "collectionName" is "telescopes", the site is used to build the telescope name.
            In case "collectionName" is "sites",
            this argument sets which site parameter get the versions of
        collectionName: str
            The name of the collection in which to update the parameter (default is "telescopes")

        Returns
        -------
        allVersions: list
            List of all versions found
        """

        collection = DatabaseHandler.dbClient[dbName][collectionName]

        query = {
            "Parameter": parameter,
        }

        _siteValidated = names.validateSiteName(site)
        if collectionName == "telescopes":
            _telModelNameValidated = names.validateTelescopeModelName(telescopeModelName)
            _telNameDB = self._getTelescopeModelNameForDB(_siteValidated, _telModelNameValidated)
            query["Telescope"] = _telNameDB
        elif collectionName == "sites":
            query["Site"] = _siteValidated
        else:
            raise ValueError("Can only get versions of the telescopes and sites collections.")

        _allVersions = list()
        for post in collection.find(query):
            _allVersions.append(post["Version"])

        if len(_allVersions) == 0:
            self._logger.warning(f"The query {query} did not return any results. No versions found")

        return _allVersions
