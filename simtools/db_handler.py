''' Module to handle interaction with DB. '''

import logging
import datetime
import yaml
import shlex
import subprocess
import time
import atexit
import getpass
from pathlib import Path

import gridfs
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

import simtools.config as cfg
from simtools.util import names
from simtools.util.model import validateModelParameter, getTelescopeSize

__all__ = ['getArrayDB']


logger = logging.getLogger(__name__)

DB_TABULATED_DATA = 'CTA-Simulation-Model'
DB_CTA_SIMULATION_MODEL = 'CTA-Simulation-Model'
DB_CTA_SIMULATION_MODEL_DESCRIPTIONS = 'CTA-Simulation-Model-Descriptions'


def createTunnel(localport, remoteport, user, mongodbServer, tunnelServer):
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

    tunnelCmd = 'ssh -N -L {localport}:{mongodbServer}:{remoteport} {user}@{tunnelServer}'.format(
        localport=localport,
        remoteport=remoteport,
        user=user,
        mongodbServer=mongodbServer,
        tunnelServer=tunnelServer
    )

    args = shlex.split(tunnelCmd)
    tunnel = subprocess.Popen(args)

    time.sleep(2)  # Give it a couple seconds to finish setting up

    # return the tunnel so you can kill it before you stop
    # the program - else the connection will persist
    # after the script ends
    return tunnel


def closeSSHTunnel(tunnels):
    '''
    Close SSH tunnels given in the process handles "tunnels"

    Parameters
    ----------
    tunnels: a tunnel process handle (or a list of those)
    '''

    logger.info('Closing SSH tunnel(s)')
    if not isinstance(tunnels, list):
        tunnels = [tunnels]

    for tunnel in tunnels:
        tunnel.kill()

    return


def openMongoDB():
    '''
    Open a connection to MongoDB and return the client to read/write to the DB with.

    Returns
    -------
    A PyMongo DB client and the tunnel process handle
    '''

    dbDetailsFile = cfg.get('mongoDBConfigFile')
    dbDetails = readDetailsDB(dbDetailsFile)

    user = getpass.getuser()

    # Start tunnel
    tunnel = createTunnel(
        localport=dbDetails['localport'],
        remoteport=dbDetails['remoteport'],
        user=user,
        mongodbServer=dbDetails['mongodbServer'],
        tunnelServer=dbDetails['tunnelServer']
    )
    atexit.register(closeSSHTunnel, [tunnel])

    userDB = dbDetails['userDB']
    dbServer = 'localhost'
    dbClient = MongoClient(
        dbServer,
        port=dbDetails['dbPort'],
        username=userDB,
        password=dbDetails['passDB'],
        authSource=dbDetails['authenticationDatabase'],
        ssl=True,
        tlsallowinvalidhostnames=True,
        tlsallowinvalidcertificates=True
    )

    return dbClient, tunnel


def getArrayDB(databaseLocation):
    '''
    Get array db info as a dict.

    Parameters
    ----------
    databaseLocation: str or Path

    Returns
    -------
    dict
    '''

    file = Path(databaseLocation).joinpath('arrays').joinpath('arrays.yml')
    out = dict()
    with open(file, 'r') as stream:
        out = yaml.load(stream, Loader=yaml.FullLoader)
    return out


def readDetailsDB(dbDetailsFile):
    '''
    Get a dict with db details (name, ports, password).

    Parameters
    ----------
    dbDetailsFile: str or Path

    Returns
    -------
    dict
    '''

    dbDetails = dict()
    with open(dbDetailsFile, 'r') as stream:
        dbDetails = yaml.load(stream, Loader=yaml.FullLoader)
    return dbDetails


def getModelParameters(telescopeType, version, onlyApplicable=False, runPath='./play/datFiles/'):

    if cfg.get('useMongoDB'):
        # TODO - This is probably not efficient to open a new connection
        # every time we want to read the parameters of a single telescope.
        # Change this to keep the connection open.
        # Probably would be easier to do if db_handler is a class.
        dbClient, tunnel = openMongoDB()
        _pars = getModelParametersMongoDB(dbClient, telescopeType, version, onlyApplicable)
        atexit.unregister(closeSSHTunnel)
        closeSSHTunnel([tunnel])
        return _pars
    else:
        return getModelParametersYaml(telescopeType, version, onlyApplicable)   


def getModelParametersYaml(telescopeType, version, onlyApplicable=False):
    '''
    Get parameters from DB for one specific type.

    Parameters
    ----------
    telescopeType: str
    version: str
        Version of the model.
    onlyApplicable: bool
        If True, only applicable parameters will be selected.

    Returns
    -------
    dict containing the parameters
    '''

    _telTypeValidated = names.validateName(telescopeType, names.allTelescopeTypeNames)
    _versionValidated = names.validateName(version, names.allModelVersionNames)

    if getTelescopeSize(_telTypeValidated) == 'MST':
        # MST-FlashCam or MST-NectarCam
        _whichTelLabels = [_telTypeValidated, 'MST-optics']
    elif _telTypeValidated == 'SST':
        # SST = SST-Camera + SST-Structure
        _whichTelLabels = ['SST-Camera', 'SST-Structure']
    else:
        _whichTelLabels = [_telTypeValidated]

    # Selecting version and applicable (if on)
    _pars = dict()
    for _tel in _whichTelLabels:
        _allPars = getAllModelParametersYaml(_tel, _versionValidated)

        # If tel is a struture, only applicable pars will be collected, always.
        # The default ones will be covered by the camera pars.
        _selectOnlyApplicable = onlyApplicable or (_tel in ['MST-optics', 'SST-Structure'])

        for parNameIn, parInfo in _allPars.items():

            if not parInfo['Applicable'] and _selectOnlyApplicable:
                continue

            _pars[parNameIn] = parInfo[_versionValidated]

    return _pars


def getModelParametersMongoDB(
    dbClient,
    telescopeType,
    version,
    runLocation,
    onlyApplicable=False
):
    '''
    Get parameters from MongoDB for a specific telescope.

    Parameters
    ----------
    dbClient: a MongoDB client provided by openMongoDB
    telescopeType: str
    version: str
        Version of the model.
    runLocation: Path or str
        The sim_telarray run location to write the tabulated data files into.
    onlyApplicable: bool
        If True, only applicable parameters will be read.

    Returns
    -------
    dict containing the parameters
    '''

    # TODO the naming is a mess at the moment, need to fix it everywhere consistently.
    _telTypeValidated = names.validateName(telescopeType, names.allTelescopeTypeNames)
    _versionValidated = names.validateName(version, names.allModelVersionNames)

    site = _telTypeValidated.split('-')[0]
    if 'MST' in _telTypeValidated:
        # MST-FlashCam or MST-NectarCam
        _whichTelLabels = [_telTypeValidated, '{}-MST-Structure-D'.format(site)]
    elif 'SST' in _telTypeValidated:
        # SST = SST-Camera + SST-Structure
        _whichTelLabels = ['{}-SST-Camera-D'.format(site), '{}-SST-Structure'.format(site)]
    else:
        _whichTelLabels = [_telTypeValidated]

    # Selecting version and applicable (if on)
    _pars = dict()
    for _tel in _whichTelLabels:

        # If tel is a struture, only applicable pars will be collected, always.
        # The default ones will be covered by the camera pars.
        _selectOnlyApplicable = onlyApplicable or (_tel in [
            '{}-MST-Structure-D'.format(site),
            '{}-SST-Structure-D'.format(site)
        ])

        _pars.update(readMongoDB(
            dbClient,
            _tel,
            _versionValidated,
            runLocation,
            _selectOnlyApplicable
        ))

    return _pars


def readMongoDB(
    dbClient,
    telescopeType,
    version,
    runLocation,
    onlyApplicable=False
):
    '''
    Build and execute query to Read the MongoDB for a specific telescope.
    Also writes the files listed in the parameter values into the sim_telarray run location

    Parameters
    ----------
    dbClient: a MongoDB client provided by openMongoDB
    telescopeType: str
    version: str
        Version of the model.
    runLocation: Path or str
        The sim_telarray run location to write the tabulated data files into.
    onlyApplicable: bool
        If True, only applicable parameters will be read.

    Returns
    -------
    dict containing the parameters
    '''

    posts = dbClient[DB_CTA_SIMULATION_MODEL].posts
    _parameters = dict()

    query = {
        'Telescope': telescopeType,
        'Version': version,
    }
    if onlyApplicable:
        query['Applicable'] = onlyApplicable
    for i_entry, post in enumerate(posts.find(query)):
        parNow = post['Parameter']
        _parameters[parNow] = post
        _parameters[parNow].pop('_id', None)
        _parameters[parNow].pop('Parameter', None)
        _parameters[parNow].pop('Telescope', None)
        if _parameters[parNow]['File']:
            file = getFileMongoDB(
                dbClient,
                DB_TABULATED_DATA,
                _parameters[parNow]['Value']
            )

            writeFileFromMongoToDisk(dbClient, DB_TABULATED_DATA, runLocation, file)

    return _parameters


def getAllModelParametersYaml(telescopeType, version):
    '''
    Get all parameters from Yaml DB for one specific type.
    No selection is applied.

    Parameters
    ----------
    telescopeType: str
    version: str
        Version of the model.

    Returns
    -------
    dict containing the parameters
    '''
    _fileNameDB = 'parValues-{}.yml'.format(telescopeType)
    _yamlFile = cfg.findFile(
        _fileNameDB,
        cfg.get('modelFilesLocations')
    )
    logger.debug('Reading DB file {}'.format(_yamlFile))
    with open(_yamlFile, 'r') as stream:
        _allPars = yaml.load(stream, Loader=yaml.FullLoader)
    return _allPars


def writeModelFile(fileName, destDir):
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
    file = Path(getModelFile(fileName))
    destFile.write_text(file.read_text())


def getModelFile(fileName):
    '''
    Find file in model files locations and return its full path.

    Parameters
    ----------
    fileName: str
        File name to be found.

    Returns
    -------
    Path
        Full path of the file.
    '''

    file = cfg.findFile(fileName, cfg.get('modelFilesLocations'))
    return file


def getFileMongoDB(dbClient, dbName, fileName):
    '''
    Extract a file from MongoDB and write it to disk

    Parameters
    ----------
    dbClient: a MongoDB client provided by openMongoDB
    dbName: str
        the name of the file DB
    fileName: str
        The name of the file requested

    Returns
    -------
    GridOut
        A file instance returned by GridFS find_one

    '''

    db = dbClient[dbName]
    fileSystem = gridfs.GridFS(db)
    if fileSystem.exists({'filename': fileName}):
        return fileSystem.find_one({'filename': fileName})
    else:
        raise FileNotFoundError(
            'The file {} does not exist in the database {}'.format(fileName, dbName)
        )


def writeFileFromMongoToDisk(dbClient, dbName, path, file):
    '''
    Extract a file from MongoDB and write it to disk

    Parameters
    ----------
    dbClient: a MongoDB client provided by openMongoDB
    dbName: str
        the name of the file DB
    path: str or Path
        The path to write the file to
    file: GridOut
        A file instance returned by GridFS find_one

    '''

    db = dbClient[dbName]
    fsOutput = gridfs.GridFSBucket(db)
    with open(Path(path).joinpath(file.filename), 'wb') as outputFile:
        fsOutput.download_to_stream_by_name(file.filename, outputFile)

    return
