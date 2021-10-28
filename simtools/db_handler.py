''' Module to handle interaction with DB. '''

import logging
import yaml
import shlex
import subprocess
import time
import atexit
import getpass
from pathlib import Path
from bson.objectid import ObjectId
from threading import Lock

import pymongo
import gridfs
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

import simtools.config as cfg
from simtools.util import names
from simtools.util.model import getTelescopeClass

__all__ = ['DatabaseHandler']


class DatabaseHandler:
    '''
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
    addParameter()
        Add a parameter value for a specific telescope.
    addNewParamete()
        Add a new parameter for a specific telescope.
    insertFileToDB()
        Insert a file to the DB.
    insertFilesToDB()
        Insert a list of files to the DB.
    '''

    # TODO move into config file?
    DB_TABULATED_DATA = 'CTA-Simulation-Model'
    DB_CTA_SIMULATION_MODEL = 'CTA-Simulation-Model'
    DB_CTA_SIMULATION_MODEL_DESCRIPTIONS = 'CTA-Simulation-Model-Descriptions'

    dbClient = None
    tunnel = None

    def __init__(self):
        '''
        Initialize the DatabaseHandler class.
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Initialize DatabaseHandler')

        if cfg.get('useMongoDB'):
            if DatabaseHandler.dbClient is None or DatabaseHandler.tunnel is None:
                with Lock():
                    self.dbDetails = self._readDetailsMongoDB()
                    DatabaseHandler.dbClient, DatabaseHandler.tunnel = self._openMongoDB()

    # END of _init_

    def _readDetailsMongoDB(self):
        '''
        Read the MongoDB details (server, user, pass, etc.) from an external file.

        Returns
        -------
        dbDetails: dict
            Dictionary containing the DB details.
        '''

        dbDetails = dict()
        dbDetailsFile = cfg.get('mongoDBConfigFile')
        with open(dbDetailsFile, 'r') as stream:
            dbDetails = yaml.load(stream, Loader=yaml.FullLoader)

        return dbDetails

    def _openMongoDB(self):
        '''
        Open a connection to MongoDB and return the client to read/write to the DB with.

        Returns
        -------
        A PyMongo DB client and the tunnel process handle
        '''

        user = getpass.getuser()
        if 'userDESY' in self.dbDetails:
            user = self.dbDetails['userDESY']

        # Start tunnel
        _tunnel = self._createTunnel(
            localport=self.dbDetails['localport'],
            remoteport=self.dbDetails['remoteport'],
            user=user,
            mongodbServer=self.dbDetails['mongodbServer'],
            tunnelServer=self.dbDetails['tunnelServer']
        )
        atexit.register(self._closeSSHTunnel, [_tunnel])

        userDB = self.dbDetails['userDB']
        dbServer = 'localhost'
        _dbClient = MongoClient(
            dbServer,
            port=self.dbDetails['dbPort'],
            username=userDB,
            password=self.dbDetails['passDB'],
            authSource=self.dbDetails['authenticationDatabase'],
            ssl=True,
            tlsallowinvalidhostnames=True,
            tlsallowinvalidcertificates=True
        )

        return _dbClient, _tunnel

    def _createTunnel(self, localport, remoteport, user, mongodbServer, tunnelServer):
        '''
        Create SSH Tunnels for database connection.

        Parameters
        ----------
        localport: int
            The local port to connect to the DB through (for MongoDB, usually 27018)
        remoteport: int
            The port on the server to connect to the DB through (for MongoDB, usually 27017)
        user: str
            User name to connect with.
        tunnelServer: str
            The server to run the tunnel through (should be warp).

        Returns
        -------
        Tunnel process handle.
        '''

        tunnelCmd = (
            'ssh -4 -N -L {localport}:{mongodbServer}:{remoteport} {user}@{tunnelServer}'.format(
                localport=localport,
                remoteport=remoteport,
                user=user,
                mongodbServer=mongodbServer,
                tunnelServer=tunnelServer
            )
        )

        args = shlex.split(tunnelCmd)
        _tunnel = subprocess.Popen(args)

        time.sleep(2)  # Give it a couple seconds to finish setting up

        # return the tunnel so you can kill it before you stop
        # the program - else the connection will persist
        # after the script ends
        return _tunnel

    def _closeSSHTunnel(self, tunnels):
        '''
        Close SSH tunnels given in the process handles "tunnels"

        Parameters
        ----------
        tunnels: a tunnel process handle (or a list of those)
        '''

        self._logger.info('Closing SSH tunnel(s)')
        if not isinstance(tunnels, list):
            tunnels = [tunnels]

        for _tunnel in tunnels:
            _tunnel.kill()

        return

    def _getTelescopeModelNameForDB(self, site, telescopeModelName):
        ''' Make telescope name as the DB needs from site and telescopeModelName. '''
        return site + '-' + telescopeModelName

    def getModelParameters(
        self,
        site,
        telescopeModelName,
        modelVersion,
        onlyApplicable=False,
    ):
        '''
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
        '''

        _siteValidated = names.validateSiteName(site)
        _telModelNameValidated = names.validateTelescopeModelName(telescopeModelName)

        if cfg.get('useMongoDB'):

            # Only MongoDB suppports tagged version
            _modelVersion = self._convertVersionToTagged(
                modelVersion,
                DatabaseHandler.DB_CTA_SIMULATION_MODEL
            )
            _versionValidated = names.validateModelVersionName(_modelVersion)

            _pars = self._getModelParametersMongoDB(
                DatabaseHandler.DB_CTA_SIMULATION_MODEL,
                _siteValidated,
                _telModelNameValidated,
                _versionValidated,
                onlyApplicable
            )
            return _pars
        else:
            _versionValidated = names.validateModelVersionName(modelVersion)

            return self._getModelParametersYaml(
                _siteValidated,
                _telModelNameValidated,
                _versionValidated,
                onlyApplicable
            )

    def exportModelFiles(self, parameters, dest):

        if cfg.get('useMongoDB'):
            self._logger.debug('Exporting model files from MongoDB')
            for par, info in parameters.items():
                if not info['File']:
                    continue
                file = self._getFileMongoDB(
                    DatabaseHandler.DB_CTA_SIMULATION_MODEL,
                    info['Value']
                )
                self._writeFileFromMongoToDisk(DatabaseHandler.DB_CTA_SIMULATION_MODEL, dest, file)
        else:
            self._logger.debug('Exporting model files from local model file directories')
            for par, value in parameters.items():

                if not self._isFile(value):
                    continue
                self._writeModelFileYaml(value, dest, noFileOk=True)

    @staticmethod
    def _isFile(value):
        ''' Vefiry if a parameter value is a file name. '''
        return any(ext in str(value) for ext in ['.dat', '.txt', '.lis'])

    def _writeModelFileYaml(self, fileName, destDir, noFileOk=False):
        '''
        Find the fileName in the model files location and write a copy
        at the destDir directory.

        Parameters
        ----------
        fileName: str or Path
            File name to be found and copied.
        destDir: str or Path
            Path of the directory where the file will be written.
        '''

        destFile = Path(destDir).joinpath(fileName)
        try:
            file = cfg.findFile(fileName, cfg.get('modelFilesLocations'))
        except FileNotFoundError:
            if noFileOk:
                self._logger.debug('File {} not found but noFileOk'.format(fileName))
                return
            else:
                raise

        destFile.write_text(file.read_text())

    def _getModelParametersYaml(self, site, telescopeModelName, modelVersion, onlyApplicable=False):
        '''
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
        '''

        _telClass = getTelescopeClass(telescopeModelName)
        _telNameConverted = names.convertTelescopeModelNameToYaml(telescopeModelName)

        if _telClass == 'MST':
            # MST-FlashCam or MST-NectarCam
            _whichTelLabels = [_telNameConverted, 'MST-optics']
        elif _telClass == 'SST':
            # SST = SST-Camera + SST-Structure
            _whichTelLabels = ['SST-Camera', 'SST-Structure']
        else:
            _whichTelLabels = [_telNameConverted]

        # Selecting version and applicable (if on)
        _pars = dict()
        for _tel in _whichTelLabels:
            _allPars = self._getAllModelParametersYaml(_tel)

            # If _tel is a structure, only the applicable parameters will be collected, always.
            # The default ones will be covered by the camera parameters.
            _selectOnlyApplicable = onlyApplicable or (_tel in ['MST-optics', 'SST-Structure'])

            for parNameIn, parInfo in _allPars.items():

                if not parInfo['Applicable'] and _selectOnlyApplicable:
                    continue

                if modelVersion not in parInfo:
                    continue

                _pars[parNameIn] = parInfo[modelVersion]

        return _pars

    def _getModelParametersMongoDB(
        self,
        dbName,
        site,
        telescopeModelName,
        modelVersion,
        onlyApplicable=False
    ):
        '''
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
        '''

        _telNameDB = self._getTelescopeModelNameForDB(site, telescopeModelName)
        _telClass = getTelescopeClass(telescopeModelName)

        self._logger.debug('TelNameDB: {}'.format(_telNameDB))
        self._logger.debug('TelClass: {}'.format(_telClass))

        if _telClass == 'MST':
            # MST-FlashCam or MST-NectarCam
            _whichTelLabels = [_telNameDB, '{}-MST-Structure-D'.format(site)]
        elif _telClass == 'SST':
            # SST = SST-Camera + SST-Structure
            _whichTelLabels = ['{}-SST-Camera-D'.format(site), '{}-SST-Structure-D'.format(site)]
        else:
            _whichTelLabels = [_telNameDB]

        # Selecting version and applicable (if on)
        _pars = dict()
        for _tel in _whichTelLabels:

            # If tel is a struture, only applicable pars will be collected, always.
            # The default ones will be covered by the camera pars.
            _selectOnlyApplicable = onlyApplicable or (_tel in [
                '{}-MST-Structure-D'.format(site),
                '{}-SST-Structure-D'.format(site)
            ])

            _pars.update(self.readMongoDB(
                dbName,
                _tel,
                modelVersion,
                runLocation=None,
                writeFiles=False,
                onlyApplicable=_selectOnlyApplicable
            ))

        return _pars

    def readMongoDB(
        self,
        dbName,
        telescopeModelNameDB,
        modelVersion,
        runLocation,
        writeFiles=True,
        onlyApplicable=False
    ):
        '''
        Build and execute query to Read the MongoDB for a specific telescope.
        Also writes the files listed in the parameter values into the sim_telarray run location

        Parameters
        ----------
        dbName: str
            the name of the DB
        site: str
            South or North.
        telescopeModelNameDB: str
            Name of the telescope model (e.g. MST-FlashCam-D ...)
        modelVersion: str
            Version of the model.
        runLocation: Path or str
            The sim_telarray run location to write the tabulated data files into.
        writeFiles: bool
            If true, write the files to the runLocation.
        onlyApplicable: bool
            If True, only applicable parameters will be read.

        Returns
        -------
        dict containing the parameters
        '''

        collection = DatabaseHandler.dbClient[dbName]['telescopes']
        _parameters = dict()

        _modelVersion = self._convertVersionToTagged(modelVersion, dbName)

        query = {
            'Telescope': telescopeModelNameDB,
            'Version': _modelVersion,
        }

        self._logger.debug('Trying the following query: {}'.format(query))
        if onlyApplicable:
            query['Applicable'] = onlyApplicable
        if collection.count_documents(query) < 1:
            raise ValueError(
                'The following query returned zero results! Check the input data and rerun.\n',
                query
            )
        for post in collection.find(query):
            parNow = post['Parameter']
            _parameters[parNow] = post
            _parameters[parNow].pop('Parameter', None)
            _parameters[parNow].pop('Telescope', None)
            _parameters[parNow]['entryDate'] = ObjectId(post['_id']).generation_time
            if _parameters[parNow]['File'] and writeFiles:
                file = self._getFileMongoDB(
                    dbName,
                    _parameters[parNow]['Value']
                )

                self._writeFileFromMongoToDisk(dbName, runLocation, file)

        return _parameters

    def _getAllModelParametersYaml(self, telescopeNameYaml):
        '''
        Get all parameters from Yaml DB for one specific type.
        No selection is applied.

        Parameters
        ----------
        telescopeNameYaml: str
            Telescope name as required by the yaml files.

        Returns
        -------
        dict containing the parameters
        '''

        _fileNameDB = 'parValues-{}.yml'.format(telescopeNameYaml)
        _yamlFile = cfg.findFile(
            _fileNameDB,
            cfg.get('modelFilesLocations')
        )
        self._logger.debug('Reading DB file {}'.format(_yamlFile))
        with open(_yamlFile, 'r') as stream:
            _allPars = yaml.load(stream, Loader=yaml.FullLoader)
        return _allPars

    def getSiteParameters(
        self,
        site,
        modelVersion,
        onlyApplicable=False,
    ):
        '''
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
        '''
        _site = names.validateSiteName(site)
        _modelVersion = names.validateModelVersionName(modelVersion)

        if cfg.get('useMongoDB'):
            _pars = self._getSiteParametersMongoDB(
                DatabaseHandler.DB_CTA_SIMULATION_MODEL,
                _site,
                _modelVersion,
                onlyApplicable
            )
            return _pars
        else:
            return self._getSiteParametersYaml(_site, _modelVersion, onlyApplicable)

    def _getSiteParametersYaml(self, site, modelVersion, onlyApplicable=False):
        '''
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
        '''

        siteYaml = 'lapalma' if site == 'North' else 'paranal'

        yamlFile = cfg.findFile('parValues-Sites.yml', cfg.get('modelFilesLocations'))
        self._logger.info('Reading DB file {}'.format(yamlFile))
        with open(yamlFile, 'r') as stream:
            _allParsVersions = yaml.load(stream, Loader=yaml.FullLoader)

        _pars = dict()
        for parName, parInfo in _allParsVersions.items():

            if not parInfo['Applicable'] and onlyApplicable:
                continue
            if siteYaml in parName:
                parNameIn = '_'.join(parName.split('_')[1:])

                _pars[parNameIn] = parInfo[modelVersion]

        return _pars

    def _getSiteParametersMongoDB(
        self,
        dbName,
        site,
        modelVersion,
        onlyApplicable=False
    ):
        '''
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
        '''

        collection = DatabaseHandler.dbClient[dbName].sites
        _parameters = dict()

        _modelVersion = self._convertVersionToTagged(modelVersion, dbName)

        query = {
            'Site': site,
            'Version': _modelVersion,
        }
        if onlyApplicable:
            query['Applicable'] = onlyApplicable
        if collection.count_documents(query) < 1:
            raise ValueError(
                'The following query returned zero results! Check the input data and rerun.\n',
                query
            )
        for post in collection.find(query):
            parNow = post['Parameter']
            _parameters[parNow] = post
            _parameters[parNow].pop('Parameter', None)
            _parameters[parNow].pop('Site', None)
            _parameters[parNow]['entryDate'] = ObjectId(post['_id']).generation_time

        return _parameters

    def _getFileMongoDB(self, dbName, fileName):
        '''
        Extract a file from MongoDB and write it to disk

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
        '''

        db = DatabaseHandler.dbClient[dbName]
        fileSystem = gridfs.GridFS(db)
        if fileSystem.exists({'filename': fileName}):
            return fileSystem.find_one({'filename': fileName})
        else:
            raise FileNotFoundError(
                'The file {} does not exist in the database {}'.format(fileName, dbName)
            )

    def _writeFileFromMongoToDisk(self, dbName, path, file):
        '''
        Extract a file from MongoDB and write it to disk

        Parameters
        ----------
        dbName: str
            the name of the DB with files of tabulated data
        path: str or Path
            The path to write the file to
        file: GridOut
            A file instance returned by GridFS find_one
        '''

        db = DatabaseHandler.dbClient[dbName]
        fsOutput = gridfs.GridFSBucket(db)
        with open(Path(path).joinpath(file.filename), 'wb') as outputFile:
            fsOutput.download_to_stream_by_name(file.filename, outputFile)

        return

    def copyTelescope(self, dbName, telToCopy, versionToCopy, newTelName, dbToCopyTo=None):
        '''
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
        dbToCopyTo: str
            The name of the DB to copy to (default is the same as dbName)
        '''

        if dbToCopyTo is None:
            dbToCopyTo = dbName

        self._logger.info('Copying version {} of {} to the new telescope {} in the {} DB'.format(
            versionToCopy,
            telToCopy,
            newTelName,
            dbToCopyTo
        ))

        collection = DatabaseHandler.dbClient[dbName].telescopes
        dbEntries = list()

        _versionToCopy = self._convertVersionToTagged(versionToCopy, dbName)

        query = {
            'Telescope': telToCopy,
            'Version': _versionToCopy,
        }
        for post in collection.find(query):
            post['Telescope'] = newTelName
            post.pop('_id', None)
            dbEntries.append(post)

        self._logger.info('Creating new telescope {}'.format(newTelName))
        collection = DatabaseHandler.dbClient[dbToCopyTo].telescopes
        try:
            collection.insert_many(dbEntries)
        except BulkWriteError as exc:
            raise exc(exc.details)

        return

    def copyDocuments(self, dbName, collection, query, dbToCopyTo):
        '''
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
                'Telescope': 'North-LST-1',
                'Version': 'prod4',
            }
            would copy all entries of prod4 version from telescope North-LST-1 to "dbToCopyTo".
        dbToCopyTo: str
            The name of the DB to copy to.
        '''

        _collection = DatabaseHandler.dbClient[dbName][collection]
        dbEntries = list()

        for post in _collection.find(query):
            post.pop('_id', None)
            dbEntries.append(post)

        self._logger.info(
            'Copying documents matching the following query {}\nto {}'.format(query, dbToCopyTo)
        )
        _collection = DatabaseHandler.dbClient[dbToCopyTo][collection]
        try:
            _collection.insert_many(dbEntries)
        except BulkWriteError as exc:
            raise exc(exc.details)

        return

    def deleteQuery(self, dbName, collection, query):
        '''
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
                'Telescope': 'North-LST-1',
                'Version': 'prod4',
            }
            would delete the entire prod4 version from telescope North-LST-1.
        '''

        _collection = DatabaseHandler.dbClient[dbName][collection]

        if 'Version' in query:
            query['Version'] = self._convertVersionToTagged(query['Version'], dbName)

        self._logger.info('Deleting {} entries from {}'.format(
            _collection.count_documents(query),
            dbName,
        ))

        _collection.delete_many(query)

        return

    def updateParameter(self, dbName, telescope, version, parameter, newValue, filePrefix=None):
        '''
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
        filePrefix: str or Path
            where to find files to upload to the DB
        '''

        collection = DatabaseHandler.dbClient[dbName].telescopes

        _modelVersion = self._convertVersionToTagged(version, dbName)

        query = {
            'Telescope': telescope,
            'Version': _modelVersion,
            'Parameter': parameter,
        }

        parEntry = collection.find_one(query)
        oldValue = parEntry['Value']

        self._logger.info('For telescope {}, version {}\nreplacing {} value from {} to {}'.format(
            telescope,
            _modelVersion,
            parameter,
            oldValue,
            newValue
        ))

        filesToAddToDB = set()
        if self._isFile(newValue):
            file = True
            if filePrefix is None:
                raise FileNotFoundError(
                    'The location of the file to upload, '
                    'corresponding to the {} parameter, must be provided.'
                ).format(parameter)
            filePath = Path(filePrefix).joinpath(newValue)
            filesToAddToDB.add('{}'.format(filePath))
            self._logger.info(
                'Will also add the file {} to the DB'.format(filePath)
            )
        else:
            file = False

        queryUpdate = {'$set': {'Value': newValue, 'File': file}}

        collection.update_one(query, queryUpdate)
        self.insertFilesToDB(dbName, filesToAddToDB)

        return

    def addParameter(self, dbName, telescope, parameter, newVersion, newValue):
        '''
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
        '''

        collection = DatabaseHandler.dbClient[dbName].telescopes

        _newVersion = self._convertVersionToTagged(newVersion, dbName)

        query = {
            'Telescope': telescope,
            'Parameter': parameter,
        }

        parEntry = collection.find(query).sort('_id', pymongo.DESCENDING)[0]
        parEntry['Value'] = newValue
        parEntry['Version'] = _newVersion
        parEntry.pop('_id', None)

        self._logger.info('Will add the following entry to DB\n', parEntry)

        collection.insert_one(parEntry)

        return

    def addNewParameter(self, dbName, telescope, version, parameter, value, **kwargs):
        '''
        Add a parameter value for a specific telescope.
        A new document will be added to the DB,
        with all fields taken from the input parameters.

        Parameters
        ----------
        dbName: str
            the name of the DB
        telescope: str
            The name of the telescope to add a parameter to.
            Assumed to be a valid name!
        parameter: str
            Which parameter to add
        version: str
            The version of the new parameter value
        value: can be any type, preferably given in kwargs
            The value to set for the new parameter
        kwargs: dict
            Any additional fields to add to the parameter
        '''

        collection = DatabaseHandler.dbClient[dbName].telescopes

        dbEntry = dict()
        dbEntry['Telescope'] = telescope
        dbEntry['Version'] = version
        dbEntry['Parameter'] = parameter
        dbEntry['Value'] = value
        dbEntry['Type'] = kwargs['Type'] if 'Type' in kwargs else str(type(value))
        dbEntry['File'] = False
        if self._isFile(value):
            dbEntry['File'] = True

        kwargs.pop('Type', None)
        dbEntry.update(kwargs)

        self._logger.info('Will add the following entry to DB\n', dbEntry)

        collection.insert_one(dbEntry)

        return

    def _convertVersionToTagged(self, modelVersion, dbName):
        ''' Convert to tagged version, if needed. '''
        if modelVersion in ['Current', 'Latest']:
            return self._getTaggedVersion(dbName, modelVersion)
        else:
            return modelVersion

    def _getTaggedVersion(self, dbName, version='Current'):
        '''
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
        '''

        if version not in ['Current', 'Latest']:
            raise ValueError('The only default versions are "Current" or "Latest"')

        collection = DatabaseHandler.dbClient[dbName].metadata
        query = {'Entry': 'Simulation-Model-Tags'}

        tags = collection.find(query).sort('_id', pymongo.DESCENDING)[0]

        return tags['Tags'][version]['Value']

    def insertFileToDB(self, file, dbName=DatabaseHandler.DB_CTA_SIMULATION_MODEL, **kwargs):
        '''
        Insert a file to the DB.

        Parameters
        ----------
        dbName: str
            the name of the DB
        file: str or Path
            The name of the file to insert (full path).

        Returns
        -------
        file_id: gridfs "_id"
            If the file exists, returns the "_id" of that one, otherwise creates a new one.
        '''

        db = DatabaseHandler.dbClient[dbName]
        fileSystem = gridfs.GridFS(db)
        if fileSystem.exists({'filename': kwargs['filename']}):
            return fileSystem.find_one({'filename': kwargs['filename']})

        if 'content_type' not in kwargs:
            kwargs['content_type'] = 'ascii/dat'
        if 'filename' not in kwargs:
            kwargs['filename'] = Path(file).name

        with open(file, 'rb') as dataFile:
            file_id = fileSystem.put(dataFile, **kwargs)

        return file_id

    def insertFilesToDB(self, filesToAddToDB, dbName=DatabaseHandler.DB_CTA_SIMULATION_MODEL):
        '''
        Insert a list of files to the DB.

        Parameters
        ----------
        dbName: str
            the name of the DB
        filesToAddToDB: list of strings or Paths
            Each entry in the list is the name of the file to insert (full path).
        '''

        for fileNow in filesToAddToDB:
            kwargs = {'content_type': 'ascii/dat', 'filename': Path(fileNow).name}
            insertFileToDB(dbName, fileNow, **kwargs)

        return
