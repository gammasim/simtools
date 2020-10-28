import logging
import yaml
import copy
from pathlib import Path

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.util import names
from simtools.util.model import validateModelParameter
from simtools.model.mirrors import Mirrors
from simtools.model.camera import Camera
from simtools.model.model_parameters import MODEL_PARS


__all__ = ['TelescopeModel']


class TelescopeModel:
    '''
    TelescopeModel is an abstract representation of the MC model at the telescope level. It contains
    the list of parameters and useful methods to handle it.

    Attributes
    ----------
    telescopeName: str
        Telescope name for the base set of parameters (ex. North-LST-1, ...).
    version: str
        Version of the model (ex. prod4).
    label: str
        Instance label.
    mirrors: Mirrors
        Instance of the Mirrors class created with the mirror list of the model.
    camera: Camera
        Instance of the Camera class created with the camera config file of the model.

    Methods
    -------
    fromConfigFile(configFileName, telescopeName, label=None, filesLocation=None)
        Create a TelescopeModel from a sim_telarray cfg file.
    hasParameter(parName):
        Verify if parameter is in the model.
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
        telescopeName,
        version='Current',
        label=None,
        modelFilesLocations=None,
        filesLocation=None,
        readFromDB=True,
        logger=__name__
    ):
        '''
        TelescopeModel.

        Parameters
        ----------
        telescopeName: str
            Telescope name for the base set of parameters (ex. North-LST-1, ...).
        version: str, optional
            Version of the model (ex. prod4) (default: "Current").
        label: str, optional
            Instance label. Important for output file naming.
        modelFilesLocation: str (or Path), optional
            Location of the MC model files. If not given, it will be taken from the config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        readFromDB: bool, optional
            If True, parameters will be loaded from the DB at the init level. Default = True.
        logger: str
            Logger name to use in this instance
        '''
        self._logger = logging.getLogger(logger)
        self._logger.debug('Init TelescopeModel')

        self.version = names.validateModelVersionName(version)
        self.telescopeName = names.validateTelescopeName(telescopeName)
        self.label = label

        self._modelFilesLocations = cfg.getConfigArg('modelFilesLocations', modelFilesLocations)
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        self._parameters = dict()

        if readFromDB:
            self._loadParametersFromDB()

        self._setConfigFileDirectory()
        self._isConfigFileUpdated = False

    @property
    def mirrors(self):
        if '_mirrors' not in self.__dict__:
            self._loadMirrors()
        return self._mirrors

    @property
    def camera(self):
        if '_camera' not in self.__dict__:
            self._loadCamera()
        return self._camera

    @classmethod
    def fromConfigFile(
        cls,
        configFileName,
        telescopeName,
        label=None,
        modelFilesLocations=None,
        filesLocation=None,
        logger=__name__
    ):
        '''
        Create a TelescopeModel from a sim_telarray config file.

        Note
        ----
        Todo: Dealing with ifdef/indef etc. By now it just keeps the last version of the parameters
        in the file.

        Parameters
        ----------
        configFileName: str or Path
            Path to the input config file.
        telescopeName: str
            Telescope name for the base set of parameters (ex. North-LST-1, ...).
        label: str, optional
            Instance label. Important for output file naming.
        modelFilesLocation: str (or Path), optional
            Location of the MC model files. If not given, it will be taken from the config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        logger: str
            Logger name to use in this instance

        Returns
        -------
        Instance of the TelescopeModel class.
        '''
        parameters = dict()
        tel = cls(
            telescopeName=telescopeName,
            label=label,
            modelFilesLocations=modelFilesLocations,
            filesLocation=filesLocation,
            readFromDB=False,
            logger=logger
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
                    par, value = validateModelParameter(par, value)
                    parameters[par] = value

        tel.addParameters(**parameters)
        return tel

    def _setConfigFileDirectory(self):
        ''' Define the variable _configFileDirectory and create directories, if needed '''
        self._configFileDirectory = io.getModelOutputDirectory(self._filesLocation, self.label)
        if not self._configFileDirectory.exists():
            self._configFileDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.info('Creating directory {}'.format(self._configFileDirectory))
        return

    def _loadParametersFromDB(self):
        ''' Read parameters from DB and store them in _parameters. '''

        self._logger.debug('Reading telescope parameters from DB')

        self._setConfigFileDirectory()
        db = db_handler.DatabaseHandler(self._logger.name)
        self._parameters = db.getModelParameters(
            self.telescopeName,
            self.version,
            self._configFileDirectory,
            onlyApplicable=True
        )

        self._logger.debug('Reading site parameters from DB')

        site = names.getSiteFromTelescopeName(self.telescopeName)
        _sitePars = db.getSiteParameters(
            site,
            self.version,
            self._configFileDirectory,
            onlyApplicable=True
        )

        # Only the following are read by sim_telarray, the others produce an error.
        # TODO - Should we find a better solution for this?
        for _parNow in _sitePars.copy():
            if _parNow not in ['atmospheric_transmission', 'altitude']:
                _sitePars.pop(_parNow, None)

        self._parameters.update(_sitePars)

        # Removing all info but the value
        # TODO - change to use the full dict instead.
        if cfg.get('useMongoDB'):
            _pars = dict()
            for key, value in self._parameters.items():
                _pars[key] = value['Value']
            self._parameters = copy.copy(_pars)

        # The following cannot be read by sim_telarray
        # TODO - Should we find a better solution for this?
        parsToRemove = [
            'pixel_shape',
            'pixel_diameter',
            'lightguide_efficiency_angle_file',
            'lightguide_efficiency_wavelength_file'
        ]
        for _parNow in parsToRemove:
            if _parNow in self._parameters:
                self._parameters.pop(_parNow, None)

    # END _loadParametersFromDB

    def hasParameter(self, parName):
        '''
        Verify if the parameter is in the model.

        Parameters
        ----------
        parName: str
            Name of the parameter.

        Returns
        -------
        bool
        '''
        return parName in self._parameters.keys()

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
            self._logger.error(msg)
            raise ValueError(msg)

    def addParameters(self, **kwargs):
        '''
        Add a NEW parameters to the model.
        This function does not modify the DB, it affects only the current instance.

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
                self._logger.error(msg)
                raise ValueError(msg)
            else:
                self._logger.info('Adding {}={} to the model'.format(par, value))
                self._parameters[par] = str(value)
        self._isConfigFileUpdated = False

    def changeParameters(self, **kwargs):
        '''
        Change the value of EXISTING parameters to the model.
        This function does not modify the DB, it affects only the current instance.

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
                self._logger.error(msg)
                raise ValueError(msg)
            else:
                if type(self._parameters[par]) != type(value):
                    self._logger.warning('Value type differs from the current one')
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
                self._logger.info('Removing parameter {}'.format(par))
                del self._parameters[par]
            else:
                msg = 'Could not remove parameter {} because it does not exist'.format(par)
                self._logger.error(msg)
                raise ValueError(msg)
        self._isConfigFileUpdated = False

    def exportConfigFile(self):
        ''' Export the config file used by sim_telarray. '''

        # Setting file name and the location
        configFileName = names.simtelConfigFileName(
            self.version,
            self.telescopeName,
            self.label
        )
        self._configFilePath = self._configFileDirectory.joinpath(configFileName)

        # Writing parameters to the file
        self._logger.info('Writing config file - {}'.format(self._configFilePath))
        with open(self._configFilePath, 'w') as file:
            header = (
                '%{}\n'.format(99 * '=')
                + '% Configuration file for:\n'
                + '% TelescopeName: {}\n'.format(self.telescopeName)
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
        if mirrorNumber > self.mirrors.numberOfMirrors:
            logging.error('mirrorNumber > numberOfMirrors')
            return None

        fileName = names.simtelSingleMirrorListFileName(
            self.version,
            self.telescopeName,
            mirrorNumber,
            self.label
        )
        if '_singleMirrorListFilePath' not in self.__dict__:
            self._singleMirrorListFilePaths = dict()
        self._singleMirrorListFilePaths[mirrorNumber] = self._configFileDirectory.joinpath(fileName)

        __, __, diameter, flen, shape = self.mirrors.getSingleMirrorParameters(mirrorNumber)

        print(self._singleMirrorListFilePaths[mirrorNumber])
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
                diameter,
                flen if not setFocalLengthToZero else 0,
                shape
            ))
    # END of exportSingleMirrorListFile

    def getSingleMirrorListFile(self, mirrorNumber, setFocalLengthToZero=False):
        ''' Get the path to the single mirror list file.'''
        self.exportSingleMirrorListFile(mirrorNumber, setFocalLengthToZero)
        return self._singleMirrorListFilePaths[mirrorNumber]

    def _loadMirrors(self):
        mirrorListFileName = self._parameters['mirror_list']
        try:
            mirrorListFile = cfg.findFile(mirrorListFileName, self._configFileDirectory)
        except:
            mirrorListFile = cfg.findFile(mirrorListFileName, self._modelFilesLocations)
        self._mirrors = Mirrors(mirrorListFile, logger=self._logger.name)
        return

    def _loadCamera(self):
        cameraConfigFile = self._parameters['camera_config_file']
        focalLength = self._parameters['effective_focal_length']
        if focalLength == 0.:
            self._logger.warning('Using focal_length because effective_focal_length is 0.')
            focalLength = self._parameters['focal_length']
        try:
            cameraConfigFilePath = cfg.findFile(cameraConfigFile, self._configFileDirectory)
        except:
            cameraConfigFilePath = cfg.findFile(cameraConfigFile, self._modelFilesLocations)

        self._camera = Camera(
            telescopeName=self.telescopeName,
            cameraConfigFile=cameraConfigFilePath,
            focalLength=focalLength,
            logger=self._logger.name
        )
        return

    def isASTRI(self):
        '''
        Check if telescope is an ASTRI type.

        Returns
        ----------
        bool:
            True if telescope  is a ASTRI, False otherwise.
        '''
        return self.telescopeName in ['SST-2M-ASTRI', 'SST', 'South-SST-D']

    def isFile2D(self, par):
        '''
        Check if the file referenced by par is a 2D table.

        Parameters
        ----------
        par: str
            Name of the parameter.

        Returns
        -------
        bool:
            True if the file is a 2D map type, False otherwise.
        '''
        if not self.hasParameter(par):
            logging.error('Parameter {} does not exist'.format(par))
            return False

        fileName = self.getParameter(par)
        file = cfg.findFile(fileName)
        with open(file, 'r') as f:
            is2D = '@RPOL@' in f.read()
        return is2D
