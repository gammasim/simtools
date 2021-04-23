import logging
import shutil

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.util import names
from simtools.util.model import validateModelParameter
from simtools.model.mirrors import Mirrors
from simtools.model.camera import Camera
from simtools.simtel.simtel_config_writer import SimtelConfigWriter


__all__ = ['TelescopeModel']


class InvalidParameter(Exception):
    pass


class TelescopeModel:
    '''
    TelescopeModel represents the MC model of an individual telescope. \
    It contains the list of parameters that can be read from the DB. \
    A set of methods are available to manipulate parameters (changing, adding, removing etc). \


    Attributes
    ----------
    site: str
        North or South.
    name: str
        Telescope name for the base set of parameters (ex. LST-1, ...).
    modelVersion: str
        Version of the model (ex. prod4).
    label: str
        Instance label.
    mirrors: Mirrors
        Mirrors object created from the mirror list of the model.
    camera: Camera
        Camera object created from the camera config file of the model.
    extraLabel: str
        Extra label to be used in case of multiple telescope configurations (e.g. by ArrayModel).

    Methods
    -------
    fromConfigFile(configFileName, telescopeName, label=None, filesLocation=None)
        Create a TelescopeModel from a sim_telarray cfg file.
    setExtraLabel(extraLabel)
        Set an extra label for the name of the config file.
    hasParameter(parName):
        Verify if parameter is in the model.
    getParameter(parName):
        Get an existing parameter of the model.
    addParameter(parName, value)
        Add new parameters to the model.
    changeParameter(parName, value)
        Change the value of existing parameters to the model.
    changeMultipleParameters(**pars)
        Change the value of existing parameters to the model.
    removeParameters(*args)
        Remove parameters from the model.
    exportConfigFile()
        Export config file for sim_telarray.
    getConfigFile()
        Get the path to the config file for sim_telarray.
    '''

    def __init__(
        self,
        site,
        telescopeModelName,
        modelVersion='Current',
        label=None,
        modelFilesLocations=None,
        filesLocation=None,
        readFromDB=True
    ):
        '''
        TelescopeModel.

        Parameters
        ----------
        site: str
            South or North.
        telescopeModelName: str
            Telescope name (ex. LST-1, ...).
        modelVersion: str, optional
            Version of the model (ex. prod4) (default='Current').
        label: str, optional
            Instance label. Important for output file naming.
        modelFilesLocation: str (or Path), optional
            Location of the MC model files. If not given, it will be taken from the \
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be \
            taken from the config.yml file.
        readFromDB: bool, optional
            If True, parameters will be loaded from the DB at the init level. (default=True).
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init TelescopeModel')

        self.site = names.validateSiteName(site)
        self.name = names.validateTelescopeName(telescopeModelName)
        self.modelVersion = names.validateModelVersionName(modelVersion)
        self.label = label
        self._extraLabel = None

        self._modelFilesLocations = cfg.getConfigArg('modelFilesLocations', modelFilesLocations)
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        self._parameters = dict()

        if readFromDB:
            self._loadParametersFromDB()

        self._setConfigFileDirectoryAndName()
        self._isConfigFileUpdated = False

    @property
    def mirrors(self):
        if not hasattr(self, '_mirrors'):
            self._loadMirrors()
        return self._mirrors

    @property
    def camera(self):
        if not hasattr(self, '_camera'):
            self._loadCamera()
        return self._camera

    @property
    def extraLabel(self):
        return self._extraLabel if self._extraLabel is not None else ''

    @classmethod
    def fromConfigFile(
        cls,
        configFileName,
        site,
        telescopeModelName,
        label=None,
        modelFilesLocations=None,
        filesLocation=None
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
        site: str
            South or North.
        telescopeModelName: str
            Telescope model name for the base set of parameters (ex. LST-1, ...).
        label: str, optional
            Instance label. Important for output file naming.
        modelFilesLocation: str (or Path), optional
            Location of the MC model files. If not given, it will be taken from the config.yml
            file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.

        Returns
        -------
        Instance of the TelescopeModel class.
        '''
        parameters = dict()
        tel = cls(
            site=site,
            telescopeModelName=telescopeModelName,
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
                    par, value = validateModelParameter(par, value)
                    parameters[par] = value

        for par, value in parameters.items():
            tel.addParameter(par, value)
        return tel
    # End of fromConfigFile

    def setExtraLabel(self, extraLabel):
        '''
        Set an extra label for the name of the config file.

        Notes
        -----
        The config file directory name is not affected by the extra label. \
        Only the file name is changed. This is important for the ArrayModel \
        class to export multiple config files in the same directory.

        Parameters
        ----------
        extraLabel: str
            Extra label to be appended to the original label.
        '''
        self._extraLabel = extraLabel
        self._setConfigFileDirectoryAndName()

    def _setConfigFileDirectoryAndName(self):
        ''' Define the variable _configFileDirectory and create directories, if needed '''
        self._configFileDirectory = io.getModelOutputDirectory(self._filesLocation, self.label)
        if not self._configFileDirectory.exists():
            self._configFileDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.debug('Creating directory {}'.format(self._configFileDirectory))

        # Setting file name and the location
        configFileName = names.simtelTelescopeConfigFileName(
            self.site,
            self.name,
            self.modelVersion,
            self.label,
            self._extraLabel
        )
        self._configFilePath = self._configFileDirectory.joinpath(configFileName)

    def _loadParametersFromDB(self):
        ''' Read parameters from DB and store them in _parameters. '''

        self._logger.debug('Reading telescope parameters from DB')

        self._setConfigFileDirectoryAndName()
        db = db_handler.DatabaseHandler()
        self._parameters = db.getModelParameters(
            self.site,
            self.name,
            self.modelVersion,
            self._configFileDirectory,
            onlyApplicable=True
        )

        self._logger.debug('Reading site parameters from DB')
        _sitePars = db.getSiteParameters(
            self.site,
            self.modelVersion,
            self._configFileDirectory,
            onlyApplicable=True
        )
        # UPDATE: SimtelConfigWriter delas with which parameters should be written or not.
        # _parameters here can contain all the parameters from the DB.
        self._parameters.update(_sitePars)
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
        Get an existing parameter of the model.

        Parameters
        ----------
        parName: str
            Name of the parameter.

        Raises
        ------
        InvalidParameter
            If parName does not match any parameter in this model.

        Returns
        -------
        Value of the parameter
        '''
        try:
            return self._parameters[parName]
        except KeyError:
            msg = 'Parameter {} was not found in the model'.format(parName)
            self._logger.error(msg)
            raise InvalidParameter(msg)

    def getParameterValue(self, parName):
        '''
        Get the value of an existing parameter of the model.

        Parameters
        ----------
        parName: str
            Name of the parameter.

        Raises
        ------
        InvalidParameter
            If parName does not match any parameter in this model.

        Returns
        -------
        Value of the parameter
        '''
        parInfo = self.getParameter(parName)
        return parInfo['Value']

    def addParameter(self, parName, value, isFile=False, isAplicable=True):
        '''
        Add a new parameters to the model. \
        This function does not modify the DB, it affects only the current instance. \

        Parameters
        ----------
        parName: str
            Name of the parameter.
        value:
            Value of the parameter.
        isFile: bool
            Indicates whether the new parameter is a file or not.
        isAplicable: bool
            Indicates whether the new parameter is aplicable or not.

        Raises
        ------
        InvalidParameter
            If an existing parameter is tried to be added.
        '''
        if parName in self._parameters.keys():
            msg = 'Parameter {} already in the model, use changeParameter instead'.format(parName)
            self._logger.error(msg)
            raise InvalidParameter(msg)
        else:
            self._logger.info('Adding {}={} to the model'.format(parName, value))
            self._parameters[parName] = dict()
            self._parameters[parName]['Value'] = value
            self._parameters[parName]['Type'] = type(value)
            self._parameters[parName]['Applicable'] = isAplicable
            self._parameters[parName]['File'] = isFile

        self._isConfigFileUpdated = False

    def changeParameter(self, parName, value):
        '''
        Change the value of an existing parameter to the model. \
        This function does not modify the DB, it affects only the current instance.

        Parameters
        ----------
        parName: str
            Name of the parameter.
        value:
            Value of the parameter.

        Raises
        ------
        InvalidParameter
            If the parameter to be changed does not exist in this model.
        '''
        if parName not in self._parameters.keys():
            msg = 'Parameter {} not in the model, use addParameters instead'.format(parName)
            self._logger.error(msg)
            raise InvalidParameter(msg)
        else:
            # TODO: fix this in order to use the type from the DB directly.
            if not isinstance(value, type(self._parameters[parName]['Value'])):
                self._logger.warning('Value type differs from the current one')
            self._parameters[parName]['Value'] = value
            self._logger.debug('Changing parameter {}'.format(parName))
        self._isConfigFileUpdated = False

    def changeMultipleParameters(self, **kwargs):
        '''
        Change the value of multiple existing parameters in the model. \
        This function does not modify the DB, it affects only the current instance.

        Parameters
        ----------
        **kwargs
            Parameters should be passed as parameterName=value.

        Raises
        ------
        InvalidParameter
            If at least one of the parameters to be changed does not exist in this model.
        '''
        for par, value in kwargs.items():
            self.changeParameter(par, value)
        self._isConfigFileUpdated = False

    def removeParameters(self, *args):
        '''
        Remove a set of parameters from the model.

        Parameters
        ----------
        args
            Each parameter to be removed has to be passed as args.

        Raises
        ------
        InvalidParameter
            If at least one of the parameter to be removed is not in this model.
        '''
        for par in args:
            if par in self._parameters.keys():
                self._logger.info('Removing parameter {}'.format(par))
                del self._parameters[par]
            else:
                msg = 'Could not remove parameter {} because it does not exist'.format(par)
                self._logger.error(msg)
                raise InvalidParameter(msg)
        self._isConfigFileUpdated = False

    def addParameterFile(self, filePath):
        '''
        Add a file to the config file directory.

        Parameters
        ----------
        filePath: str
            Path of the file to be added to the config file directory.
        '''
        shutil.copy(filePath, self._configFileDirectory)
        return

    def exportConfigFile(self):
        ''' Export the config file used by sim_telarray. '''

        # Using SimtelConfigWriter to write the config file.
        self._loadSimtelConfigWriter()
        self.simtelConfigWriter.writeTelescopeConfigFile(
            configFilePath=self._configFilePath,
            parameters=self._parameters
        )

    def getConfigFile(self, noExport=False):
        '''
        Get the path of the config file for sim_telarray. \
        The config file is produced if the file is not updated.

        Parameters
        ----------
        noExport: bool
            Turn it on if you do not want the file to be exported.

        Returns
        -------
        Path of the exported config file for sim_telarray.
        '''
        if not self._isConfigFileUpdated and not noExport:
            self.exportConfigFile()
        return self._configFilePath

    def getTelescopeTransmissionParameters(self):
        '''
        Get tel. transmission pars as a list of floats.

        Returns
        -------
        list of floats
            List of 4 parameters that decsribe the tel. transmission vs off-axis.
        '''
        return [float(v) for v in self.getParameterValue('telescope_transmission').split()]

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
            self.site,
            self.name,
            self.modelVersion,
            mirrorNumber,
            self.label
        )
        if not hasattr(self, '_singleMirrorListFilePath'):
            self._singleMirrorListFilePaths = dict()
        self._singleMirrorListFilePaths[mirrorNumber] = self._configFileDirectory.joinpath(fileName)

        # Using SimtelConfigWriter
        self._loadSimtelConfigWriter()
        self.simtelConfigWriter.writeSingleMirrorListFile(
            mirrorNumber,
            self.mirrors,
            self._singleMirrorListFilePaths[mirrorNumber],
            setFocalLengthToZero
        )
    # END of exportSingleMirrorListFile

    def getSingleMirrorListFile(self, mirrorNumber, setFocalLengthToZero=False):
        '''
        Get the path to the single mirror list file.

        Parameters
        ----------
        mirrorNumber: int
            Mirror number.
        setFocalLengthToZero: bool
            Flag to set the focal length to zero.

        Returns
        -------
        Path
            Path of the single mirror list file.
        '''
        self.exportSingleMirrorListFile(mirrorNumber, setFocalLengthToZero)
        return self._singleMirrorListFilePaths[mirrorNumber]

    def _loadMirrors(self):
        ''' Load the attribute mirrors by creating a Mirrors object iwth the mirror list file. '''
        mirrorListFileName = self._parameters['mirror_list']['Value']
        try:
            mirrorListFile = cfg.findFile(mirrorListFileName, self._configFileDirectory)
        except FileNotFoundError:
            mirrorListFile = cfg.findFile(mirrorListFileName, self._modelFilesLocations)
            self._logger.warning(
                'MirrorListFile was not found in the config directory - '
                'Using the one found in the modelFilesLocations'
            )
        self._mirrors = Mirrors(mirrorListFile)

    def _loadCamera(self):
        ''' Loading camera attribute by creating a Camera object with the camera config file. '''
        cameraConfigFile = self.getParameterValue('camera_config_file')
        focalLength = self.getParameterValue('effective_focal_length')
        if focalLength == 0.:
            self._logger.warning('Using focal_length because effective_focal_length is 0.')
            focalLength = self.getParameterValue('focal_length')
        try:
            cameraConfigFilePath = cfg.findFile(cameraConfigFile, self._configFileDirectory)
        except FileNotFoundError:
            self._logger.warning(
                'CameraConfigFile was not found in the config directory - '
                'Using the one found in the modelFilesLocations'
            )
            cameraConfigFilePath = cfg.findFile(cameraConfigFile, self._modelFilesLocations)

        self._camera = Camera(
            telescopeName=self.telescopeName,
            cameraConfigFile=cameraConfigFilePath,
            focalLength=focalLength
        )

    def _loadSimtelConfigWriter(self):
        if not hasattr(self, 'simtelConfigWriter'):
            self.simtelConfigWriter = SimtelConfigWriter(
                site=self.site,
                telescopeModelName=self.name,
                modelVersion=self.modelVersion,
                label=self.label
            )

    def isASTRI(self):
        '''
        Check if telescope is an ASTRI type.

        Returns
        ----------
        bool:
            True if telescope  is a ASTRI, False otherwise.
        '''
        return self.name in ['SST-2M-ASTRI', 'SST', 'SST-D']

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

        fileName = self.getParameterValue(par)
        file = cfg.findFile(fileName)
        with open(file, 'r') as f:
            is2D = '@RPOL@' in f.read()
        return is2D
