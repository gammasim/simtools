'''
Module containing TelescopeModel class.
TelescopeModel is an representation of the MC model for single telescopes.

Author: Raul R Prado
'''

import logging
import yaml
from pathlib import Path

from simtools.util import names
from simtools.util.model import getTelescopeSize
from simtools.model.model_parameters import MODEL_PARS
import simtools.config as cfg
import simtools.io_handler as io

__all__ = ['TelescopeModel']

logger = logging.getLogger(__name__)


class TelescopeModel:
    '''
    TelescopeModel is an abstract representation of the MC model at the telescope level. It contains
    the list of parameters and useful methods to handle it.

    Attributes
    ----------
    telescopeType: str
        Telescope type for the base set of parameters (ex. SST-2M-ASTRI, LST, ...).
    site: str
        Paranal or LaPalma.
    version: str
        Version of the model (ex. prod4).
    label: str
        Instance label.

    Methods
    -------
    @classmethod
    fromConfigFile(configFileName, telescopeType, site, label=None, filesLocation=None)
        Create a TelescopeModel from a sim_telarray cfg file.
    getParameter(parName):
        Get an existing parameter of the model.
    addParameters(**kwargs)
        Add new parameters to the model.
    changeParameters(**kwargs)
        Change the value of existing parameters to the model.
    removeParameters(**args)
        Remove parameters from the model.
    exportConfigFile()
        Export config file for sim_telarray.
    getConfigFile()
        Get the path to the config file for sim_telarray.
    '''
    def __init__(
        self,
        telescopeType,
        site,
        version='default',
        label=None,
        modelFilesLocations=None,
        filesLocation=None,
        readFromDB=True
    ):
        '''
        TelescopeModel.

        Parameters
        ----------
        telescopeType: str
            Telescope type for the base set of parameters (ex. SST-2M-ASTRI, LST, ...).
        site: str
            Paranal or LaPalma.
        version: str, optional
            Version of the model (ex. prod4) (Default = default).
        label: str, optional
            Instance label. Important for output file naming.
        modelFilesLocation: str (or Path), optional
            Location of the MC model files. If not given, it will be taken from the config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        readFromDB: bool, optional
            If True, parameters will be loaded from the DB at the init level. Default = True.
        '''
        logger.debug('Init TelescopeModel')

        self.version = names.validateName(version, names.allModelVersionNames)
        self.telescopeType = names.validateName(telescopeType, names.allTelescopeTypeNames)
        self.site = names.validateName(site, names.allSiteNames)
        self.label = label

        self._modelFilesLocations = cfg.collectConfigArg('modelFilesLocations', modelFilesLocations)
        self._filesLocation = cfg.collectConfigArg('outputLocation', filesLocation)

        self._parameters = dict()

        if readFromDB:
            self._loadParametersFromDB()

        self._setConfigFileDirectory()
        self._isConfigFileUpdated = False
        self._areMirrorParametersLoaded = False

    @property
    def numberOfMirrors(self):
        if not self._areMirrorParametersLoaded:
            self._loadMirrorParameters()
        return self._numberOfMirrors

    @property
    def mirrorFocalLength(self):
        if not self._areMirrorParametersLoaded:
            self._loadMirrorParameters
        return self._mirrorFocalLength

    @property
    def mirrorDiameter(self):
        if not self._areMirrorParametersLoaded:
            self._loadMirrorParameters
        return self._mirrorDiameter

    @property
    def mirrorShape(self):
        if not self._areMirrorParametersLoaded:
            self._loadMirrorParameters
        return self._mirrorShape

    @classmethod
    def fromConfigFile(
        cls,
        configFileName,
        telescopeType,
        site,
        label=None,
        modelFilesLocations=None,
        filesLocation=None
    ):
        '''
        Create a TelescopeModel from a sim_telarray config file.

        Notes
        -----
        TODO: dealing with ifdef/indef etc. By now it just keeps the last version of the parameters
        in the file.

        Parameters
        ----------
        configFileName: str or Path
            Path to the input config file.
        telescopeType: str
            Telescope type for the base set of parameters (ex. SST-2M-ASTRI, LST, ...).
        site: str
            Paranal or LaPalma.
        label: str, optional
            Instance label. Important for output file naming.
        modelFilesLocation: str (or Path), optional
            Location of the MC model files. If not given, it will be taken from the config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.

        Returns
        -------
        Instance of the TelescopeModel class.
        '''
        parameters = dict()
        tel = cls(
            telescopeType=telescopeType,
            site=site,
            label=label,
            modelFilesLocations=modelFilesLocations,
            filesLocation=filesLocation,
            readFromDB=False
        )

        def _processLine(words):
            '''
            Process a line of the input config file that contains a parameter.

            Parameters
            ----------
            words: list of str
                List of str from the split of a line from the file.

            Returns
            -------
            (parName, parValue)
            '''
            iComment = len(words)  # Index of any possible comment
            for w in words:
                if '%' in w:
                    iComment = words.index(w)
                    break
            words = words[0:iComment]  # Removing comment
            parName = words[0].replace('=', '')
            parValue = ''
            for w in words[1:]:
                w = w.replace('=', '')
                w = w.replace(',', ' ')
                parValue += w + ' '
            return parName, parValue

        with open(configFileName, 'r') as file:
            for line in file:
                words = line.split()
                if len(words) == 0:
                    continue
                elif '%' in words[0] or 'echo' in words:
                    continue
                elif '#' not in line and len(words) > 0:
                    par, value = _processLine(words)
                    par, value = tel._validateParameter(par, value)
                    parameters[par] = value

        tel.addParameters(**parameters)
        return tel

    def _setConfigFileDirectory(self):
        ''' Define the variable _configFileDirectory and create directories, if needed '''
        self._configFileDirectory = io.getModelOutputDirectory(self._filesLocation, self.label)
        if not self._configFileDirectory.exists():
            self._configFileDirectory.mkdir(parents=True, exist_ok=True)
            logger.info('Creating directory {}'.format(self._configFileDirectory))
        return

    def _loadParametersFromDB(self):
        ''' Read parameters from DB and store them in _parameters. '''

        def _readParsFromOneType(telescopeType):
            '''
            Read parameters from DB for one specific type (telescopeTYpe, site ...).

            Parameters
            ----------
            telescopeType: str

            Returns
            -------
            dict containing the parameters
            '''
            fileNameDB = 'parValues-{}.yml'.format(telescopeType)
            yamlFile = cfg.findFile(fileNameDB, self._modelFilesLocations)
            logger.debug('Reading DB file {}'.format(yamlFile))
            with open(yamlFile, 'r') as stream:
                pars = yaml.load(stream, Loader=yaml.FullLoader)
            return pars

        def _collectAplicablePars(pars):
            '''
            Collect only Applicable parameters from pars and store them into _parameters

            Parameters
            ----------
            pars: dict
                Dict with the whole list of parameters of a certain telescope type.
            '''
            for parNameIn, parInfo in pars.items():
                if parInfo['Applicable']:
                    parNameOut, parValueOut = self._validateParameter(
                        parNameIn,
                        parInfo[self.version]
                    )
                    self._parameters[parNameOut] = parValueOut

        parametersDB = _readParsFromOneType(telescopeType=self.telescopeType)
        _collectAplicablePars(parametersDB)

        if getTelescopeSize(self.telescopeType) == 'MST':
            logger.debug('Telescope is MST type - reading optics parameters')
            parametersDB = _readParsFromOneType(telescopeType='MST-optics')
            _collectAplicablePars(parametersDB)

        print(self._parameters)

        # Site: Two site parameters need to be read: atmospheric_transmission and altitude
        logger.debug('Reading site parameters from DB')

        def _getSiteParameter(site, parName):
            ''' Get the value of parName for a given site '''
            yamlFile = cfg.findFile('parValues-Sites.yml', self._modelFilesLocations)
            logger.info('Reading DB file {}'.format(yamlFile))
            with open(yamlFile, 'r') as stream:
                allPars = yaml.load(stream, Loader=yaml.FullLoader)
                for par in allPars:
                    if parName in par and self.site.lower() in par:
                        return allPars[par][self.version]
            logger.warning('Parameter {} not found for site {}'.format(parName, site))
            return None

        for siteParName in ['atmospheric_transmission', 'altitude']:
            siteParValue = _getSiteParameter(self.site, siteParName)
            parName, parValue = self._validateParameter(siteParName, siteParValue)
            self._parameters[parName] = parValue
    # END _loadParametersFromDB

    def _validateParameter(self, parNameIn, parValueIn):
        '''
        Validate model parameter based on the dict MODEL_PARS.

        Parameters
        ----------
        parNameIn: str
            Name of the parameter to be validated.
        parValueIn: str
            Value of the parameter to be validated.

        Returns
        -------
        (parName, parValue) after validated. parValueIn is converted to the proper type if that
        information is available in MODEL_PARS
        '''
        logger.debug('Validating parameter {}'.format(parNameIn))
        for parNameModel in MODEL_PARS.keys():
            if parNameIn == parNameModel or parNameIn in MODEL_PARS[parNameModel]['names']:
                parType = MODEL_PARS[parNameModel]['type']
                return parNameModel, parType(parValueIn)
        return parNameIn, parValueIn

    def getParameter(self, parName):
        '''
        Get an EXISTING parameter of the model.

        Parameters
        ----------
        parName: str
            Name of the parameter.

        Returns
        -------
        Value of the parameter

        Raises
        ------
        ValueError
            If parName does not match any parameter in _parameters.
        '''
        if parName in self._parameters:
            return self._parameters[parName]
        else:
            msg = 'Parameter {} was not found in the model'.format(parName)
            logger.error(msg)
            raise ValueError(msg)

    def addParameters(self, **kwargs):
        '''
        Add a NEW parameters to the model.

        Parameters
        ----------
        **kwargs:
            Parameters should be passed as parameterName=value.

        Raises
        ------
        ValueError
            If an existing parameter is tried to be set added.
        '''
        for par, value in kwargs.items():
            if par in self._parameters.keys():
                msg = 'Parameter {} already in the model, use changeParameter instead'.format(par)
                logger.error(msg)
                raise ValueError(msg)
            else:
                logger.info('Adding {}={} to the model'.format(par, value))
                self._parameters[par] = str(value)
        self._isConfigFileUpdated = False

    def changeParameters(self, **kwargs):
        '''
        Change the value of EXISTING parameters to the model.

        Parameters
        ----------
        **kwargs
            Parameters should be passed as parameterName=value.

        Raises
        ------
        ValueError
            If the parameter to be changed does not exist.
        '''
        for par, value in kwargs.items():
            if par not in self._parameters.keys():
                msg = 'Parameter {} not in the model, use addParameters instead'.format(par)
                logger.error(msg)
                raise ValueError(msg)
            else:
                self._parameters[par] = value
        self._isConfigFileUpdated = False

    def removeParameters(self, *args):
        '''
        Remove a parameter from the model.

        Parameters
        ----------
        args
            Each parameter to be removed has to be passed as args.

        Raises
        ------
        ValueError
            If the parameter to be removed is not on the model.
        '''
        for par in args:
            if par in self._parameters.keys():
                logger.info('Removing parameter {}'.format(par))
                del self._parameters[par]
            else:
                msg = 'Could not remove parameter {} because it does not exist'.format(par)
                logger.error(msg)
                raise ValueError(msg)
        self._isConfigFileUpdated = False

    def exportConfigFile(self):
        ''' Export the config file used by sim_telarray. '''

        # Setting file name and the location
        configFileName = names.simtelConfigFileName(
            self.version,
            self.site,
            self.telescopeType,
            self.label
        )
        self._configFilePath = self._configFileDirectory.joinpath(configFileName)

        # Writing parameters to the file
        logger.info('Writing config file - {}'.format(self._configFilePath))
        with open(self._configFilePath, 'w') as file:
            header = (
                '%{}\n'.format(99 * '=')
                + '% Configuration file for:\n'
                + '% TelescopeType: {}\n'.format(self.telescopeType)
                + '% Site: {}\n'.format(self.site)
                + '% Label: {}\n'.format(self.label) if self.label is not None else ''
                + '%{}\n'.format(99 * '=')
            )

            file.write(header)
            for par in self._parameters:
                value = self._parameters[par]
                file.write('{} = {}\n'.format(par, value))

        self._isConfigFileUpdated = True
    # END exportConfigFile

    def getConfigFile(self):
        '''
        Get the path of the config file for sim_telarray.
        The config file is produced if the file is not updated.

        Returns
        -------
        Path of the exported config file for sim_telarray.
        '''
        if not self._isConfigFileUpdated:
            self.exportConfigFile()
        return self._configFilePath

    def getTelescopeTransmissionParameters(self):
        '''
        Get tel. transmission pars as a list of floats.
        Importante for RayTracing analysis.

        Returns
        -------
        list of floats
            List of 4 parameters that decsribe the tel. transmission vs off-axis.
        '''
        pars = list()
        for p in self.getParameter('telescope_transmission').split():
            pars.append(float(p))
        return pars

    def exportSingleMirrorListFile(self, mirrorNumber, setFocalLengthToZero):
        '''
        Export a mirror list file with a single mirror in it.

        Parameters
        ----------
        mirrorNumber: int
            Number index of the mirror.
        setFocalLengthToZero: bool
            Set the focal length to zero if True.
        '''
        if mirrorNumber > self.numberOfMirrors:
            logging.error('mirrorNumber > numberOfMirrors')

        fileName = names.simtelSingleMirrorListFileName(
            self.version,
            self.site,
            self.telescopeType,
            mirrorNumber,
            self.label
        )
        if '_singleMirrorListFilePath' not in self.__dict__:
            self._singleMirrorListFilePaths = dict()
        self._singleMirrorListFilePaths[mirrorNumber] = self._configFileDirectory.joinpath(fileName)

        mirrorListFileName = self._parameters['mirror_list']
        mirrorListFile = cfg.findFile(mirrorListFileName, self._modelFilesLocations)
        with open(mirrorListFile, 'r') as file:
            mirrorCounter = 0
            for line in file:
                if '#' in line:
                    continue
                mirrorCounter += 1
                if mirrorCounter == mirrorNumber:
                    line = line.split()
                    thisMirrorDiameter = float(line[2])
                    thisMirrorFocalLength = float(line[3])
                    thisMirrorShape = int(line[4])
                    break

        with open(self._singleMirrorListFilePaths[mirrorNumber], 'w') as file:
            file.write('# Column 1: X pos. [cm] (North/Down)\n')
            file.write('# Column 2: Y pos. [cm] (West/Right from camera)\n')
            file.write('# Column 3: flat-to-flat diameter [cm]\n')
            file.write(
                '# Column 4: focal length [cm], typically zero = adapting in sim_telarray.\n'
            )
            file.write(
                '# Column 5: shape type: 0=circular, 1=hex. with flat side parallel to y, '
                '2=square, 3=other hex. (default: 0)\n'
            )
            file.write(
                '# Column 6: Z pos (height above dish backplane) [cm], typ. omitted (or zero)'
                ' to adapt to dish shape settings.\n'
            )
            file.write('#\n')
            file.write('0. 0. {} {} {} 0.\n'.format(
                thisMirrorDiameter,
                thisMirrorFocalLength if not setFocalLengthToZero else 0,
                thisMirrorShape
            ))
    # END of exportSingleMirrorListFile

    def getSingleMirrorListFile(self, mirrorNumber, setFocalLengthToZero=False):
        ''' Get the path to the single mirror list file.'''
        self.exportSingleMirrorListFile(mirrorNumber, setFocalLengthToZero)
        return self._singleMirrorListFilePaths[mirrorNumber]

    def _loadMirrorParameters(self):
        mirrorListFileName = self._parameters['mirror_list']
        mirrorListFile = cfg.findFile(mirrorListFileName, self._modelFilesLocations)
        collectGeoPars = True
        mirrorCounter = 0
        with open(mirrorListFile, 'r') as file:
            for line in file:
                if '#' in line:
                    continue
                if collectGeoPars:
                    line = line.split()
                    self._mirrorDiameter = float(line[2])
                    self._mirrorFocalLength = float(line[3])
                    self._mirrorShape = int(line[4])
                    collectGeoPars = False
                mirrorCounter += 1
        self._numberOfMirrors = mirrorCounter
        if 'mirror_focal_length' in self._parameters:
            self._mirrorFocalLength = self._parameters['mirror_focal_length']
        self._areMirrorParametersLoaded = True
        return
