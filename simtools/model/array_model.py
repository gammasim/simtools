import logging
from copy import copy

import simtools.config as cfg
import simtools.io_handler as io
from simtools import db_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.layout.layout_array import LayoutArray
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ['ArrayModel']


class InvalidArrayConfigData(Exception):
    pass


class ArrayModel:
    '''
    ArrayModel is an abstract representation of the MC model at the array level.
    It contains the list of TelescopeModel's and a LayoutArray.

    Attributes
    ----------
    site: str
        North or South.
    layoutName: str
        Name of the layout.
    layout: LayoutArray
        Instance of LayoutArray.
    modelVersion: str
        Version of the model (ex. prod4).
    label: str
        Instance label.
    numberOfTelescopes: int
        Number of telescopes in the ArrayModel.

    Methods
    -------
    printTelescopeList()
        Print out the list of telescopes for quick inspection.
    exportSimtelTelescopeConfigFiles()
        Export sim_telarray config files for all the telescopes
        into the output model directory.
    exportSimtelArrayConfigFile()
        Export sim_telarray config file for the array into the output model
        directory.
    exportAllSimtelConfigFiles()
        Export sim_telarray config file for the array and for each individual telescope
        into the output model directory.
    getArrayConfigFile()
        Get the path to the config file for sim_telarray.
    '''

    SITE_PARS_TO_WRITE_IN_CONFIG = ['altitude', 'atmospheric_transmission']

    def __init__(
        self,
        label=None,
        arrayConfigFile=None,
        arrayConfigData=None,
        modelFilesLocations=None,
        filesLocation=None
    ):
        '''
        ArrayModel.

        Parameters
        ----------
        arrayConfigFile: str
            Path to a yaml file with the array config data.
        arrayConfigData: dict
            Dict with the array config data.
        version: str, optional
            Version of the model (ex. prod4) (default: "Current").
        label: str, optional
            Instance label. Important for output file naming.
        modelFilesLocation: str (or Path), optional
            Location of the MC model files. If not given, it will be taken from the
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init ArrayModel')

        self.label = label
        self.site = None
        self.layout = None
        self.layoutName = None
        self.modelVersion = None

        self._modelFilesLocations = cfg.getConfigArg('modelFilesLocations', modelFilesLocations)
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        arrayConfigData = collectDataFromYamlOrDict(arrayConfigFile, arrayConfigData)
        self._loadArrayData(arrayConfigData)

        self._setConfigFileDirectory()

        self._buildArrayModel()
        # End of init

    @property
    def numberOfTelescopes(self):
        return self.layout.getNumberOfTelescopes()

    def _loadArrayData(self, arrayConfigData):
        ''' Load parameters from arrayData.

        Parameters
        ----------
        arrayConfigData: dict
        '''
        # Validating arrayConfigData
        # Keys 'site', 'layoutName' and 'default' are mandatory.
        # 'default' must have 'LST', 'MST' and 'SST' (for South site) keys.
        self._validateArrayData(arrayConfigData)

        # Site
        self.site = names.validateSiteName(arrayConfigData['site'])

        # Grabbing layout name and building LayoutArray
        self.layoutName = names.validateLayoutArrayName(arrayConfigData['layoutName'])
        self.layout = LayoutArray.fromLayoutArrayName(
            self.site + '-' + self.layoutName,
            label=self.label
        )

        # Model version
        if 'modelVersion' not in arrayConfigData.keys() or arrayConfigData['modelVersion'] is None:
            self._logger.warning('modelVersion not given in arrayConfigData - using current')
            self.modelVersion = 'current'
        else:
            self.modelVersion = names.validateModelVersionName(arrayConfigData['modelVersion'])

        # Removing keys that were stored in attributes and keepig the remaining as a dict
        self._arrayConfigData = {
            k: v for (k, v) in arrayConfigData.items()
            if k not in ['site', 'layoutName', 'modelVersion']
        }
    # End of _loadArrayData

    def _validateArrayData(self, arrayConfigData):
        '''
        Validate arrayData by checking the existence of the relevant keys.
        Searching for the keys: 'site', 'array' and 'default'.
        '''

        def runOverPars(pars, data, parent=None):
            '''Run over pars and validate it. '''
            allKeys = data.keys() if parent is None else data[parent].keys()
            for pp in pars:
                if pp not in allKeys:
                    key = pp if parent is None else parent + '.' + pp
                    msg = (
                        'Key {} was not found in arrayConfigData '.format(key)
                        + '- impossible to build array model'
                    )
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)

        runOverPars(['site', 'layoutName', 'default'], arrayConfigData)
        # End of _validateArrayData

    def _setConfigFileDirectory(self):
        ''' Define the variable _configFileDirectory and create directories, if needed '''
        self._configFileDirectory = io.getModelOutputDirectory(self._filesLocation, self.label)
        if not self._configFileDirectory.exists():
            self._configFileDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.info('Creating directory {}'.format(self._configFileDirectory))
        return

    def _buildArrayModel(self):
        '''
        Build the site parameters and the list of telescope models,
        including reading the parameters from the DB.
        '''

        # Getting site parameters from DB
        db = db_handler.DatabaseHandler()
        self._siteParameters = db.getSiteParameters(
            self.site,
            self.modelVersion,
            self._configFileDirectory,
            onlyApplicable=True
        )

        # Building telescope models
        self._telescopeModel = list()  # List of telescope models
        _allTelescopeModelNames = list()  # List of telescope names without repetition
        _allParsToChange = dict()
        for tel in self.layout:
            telSize = tel.getTelescopeSize()

            # Collecting telescope name and pars to change from arrayConfigData
            telModelName, parsToChange = self._getSingleTelescopeInfoFromArrayConfig(
                tel.name,
                telSize
            )
            if len(parsToChange) > 0:
                _allParsToChange[tel.name] = parsToChange

            self._logger.debug('TelModelName: {}'.format(telModelName))

            # Building the basic models - no pars to change yet
            if telModelName not in _allTelescopeModelNames:
                # First time a telescope name is built
                _allTelescopeModelNames.append(telModelName)
                telModel = TelescopeModel(
                    telescopeName=telModelName,
                    version=self.modelVersion,
                    label=self.label,
                    modelFilesLocations=self._modelFilesLocations,
                    filesLocation=self._filesLocation
                )
            else:
                # Telescope name already exists.
                # Finding the TelescopeModel and copying it.
                for tel in self._telescopeModel:
                    if tel.telescopeName != telModelName:
                        continue
                    self._logger.debug(
                        'Copying tel model {} already loaded from DB'.format(tel.telescopeName)
                    )
                    telModel = copy(tel)
                    break

            self._telescopeModel.append(telModel)

        # Checking whether the size of the telescope list and the layout match
        if len(self._telescopeModel) != len(self.layout):
            self._logger.warning(
                'Number of telescopes in the list of telescope models does '
                'not match the number of telescopes in the LayoutArray - something is wrong!'
            )

        # Changing parameters, if there are any in allParsToChange
        if len(_allParsToChange) > 0:
            for telData, telModel in zip(self.layout, self._telescopeModel):
                if telData.name not in _allParsToChange.keys():
                    continue
                self._logger.debug('Changing {} pars of a {}: {}, ...'.format(
                    len(_allParsToChange[telData.name]),
                    telData.name,
                    *_allParsToChange[telData.name]
                ))
                telModel.changeParameters(**_allParsToChange[telData.name])
                telModel.setExtraLabel(telData.name)
    # End of _buildArrayModel

    def _getSingleTelescopeInfoFromArrayConfig(self, telName, telSize):
        '''
        arrayConfigData contains the default telescope models for each
        telescope size and the list of specific telescopes.
        For each case, the data can be given only as a name or
        as a dict with 'name' and parameters to change.
        This function has to identify these two cases and collect
        the telescope name and the dict with the parameters to change.

        Parameters
        ----------
        telName: str
            Name of the telescope at the layout level (L-01, M-05, ...).
        telSize: str
            LST, MST or SST.
        '''

        def _proccessSingleTelescope(data):
            '''
            Parameters
            ----------
            data: dict or str
                Piece of the arrayConfigData for one specific telescope.
            '''
            if isinstance(data, dict):
                # Case 0: data is dict
                if 'name' not in data.keys():
                    msg = 'ArrayConfig has no name for a telescope'
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)
                telName = self.site + '-' + telSize + '-' + data['name']
                parsToChange = {k: v for (k, v) in data.items() if k != 'name'}
                self._logger.debug(
                    'Grabbing tel data as dict - '
                    + 'name: {}, '.format(telName)
                    + '{} pars to change'.format(len(parsToChange))
                )
                return telName, parsToChange
            elif isinstance(data, str):
                # Case 1: data is string (only name)
                telName = self.site + '-' + telSize + '-' + data
                return telName, dict()
            else:
                # Case 2: data has a wrong type
                msg = 'ArrayConfig has wrong input for a telescope'
                self._logger.error(msg)
                raise InvalidArrayConfigData(msg)
        # End of _proccessData

        if telName in self._arrayConfigData.keys():
            # Specific info for this telescope
            return _proccessSingleTelescope(self._arrayConfigData[telName])
        else:
            # Checking if default option exists in arrayConfigData
            notContainsDefaultKey = (
                'default' not in self._arrayConfigData.keys()
                or telSize not in self._arrayConfigData['default'].keys()
            )

            if notContainsDefaultKey:
                msg = (
                    'default option was not given in arrayConfigData '
                    + 'for the telescope {}'.format(telName)
                )
                self._logger.error(msg)
                raise InvalidArrayConfigData(msg)

            # Grabbing the default option
            return _proccessSingleTelescope(self._arrayConfigData['default'][telSize])
    # End of _getSingleTelescopeInfoFromArrayConfig

    def printTelescopeList(self):
        ''' Print out the list of telescopes for quick inspection. '''
        for telData, telModel in zip(self.layout, self._telescopeModel):
            print('Name: {}\t Model: {}'.format(telData.name, telModel.telescopeName))

    def exportSimtelTelescopeConfigFiles(self):
        '''
        Export sim_telarray config files for all the telescopes
        into the output model directory.
        '''
        exportedModels = list()
        for telModel in self._telescopeModel:
            name = (
                telModel.telescopeName
                + ('_' + telModel.extraLabel if telModel.extraLabel != '' else '')
            )
            if name not in exportedModels:
                self._logger.debug('Exporting config file for tel {}'.format(name))
                telModel.exportConfigFile()
                exportedModels.append(name)
            else:
                self._logger.debug('Config file for tel {} already exists - skipping'.format(name))

    def exportSimtelArrayConfigFile(self):
        '''
        Export sim_telarray config file for the array into the output model
        directory.
        '''

        # Setting file name and the location
        configFileName = names.simtelArrayConfigFileName(
            self.layoutName,
            self.site,
            self.modelVersion,
            self.label
        )
        self._configFilePath = self._configFileDirectory.joinpath(configFileName)

        # Writing parameters to the file
        self._logger.info('Writing array config file - {}'.format(self._configFilePath))
        with open(self._configFilePath, 'w') as file:
            header = (
                '%{}\n'.format(50 * '=')
                + '% ARRAY CONFIGURATION FILE\n'
                + '% Site: {}\n'.format(self.site)
                + '% LayoutName: {}\n'.format(self.layoutName)
                + '% ModelVersion: {}\n'.format(self.modelVersion)
                + ('% Label: {}\n'.format(self.label) if self.label is not None else '')
                + '%{}\n\n'.format(50 * '=')
            )
            file.write(header)

            # Be carefull with the formating - simtel is sensitive
            tab = ' ' * 3

            file.write('#ifndef TELESCOPE\n')
            file.write('# define TELESCOPE 0\n')
            file.write('#endif\n\n')

            # TELESCOPE 0 - global parameters
            file.write('#if TELESCOPE == 0\n')
            file.write(tab + 'echo *****************************\n')
            file.write(tab + 'echo Site: {}\n'.format(self.site))
            file.write(tab + 'echo LayoutName: {}\n'.format(self.layoutName))
            file.write(tab + 'echo ModelVersion: {}\n'.format(self.modelVersion))
            file.write(tab + 'echo *****************************\n\n')

            # Writing site parameters
            file.write(tab + '% Site parameters\n')
            for par in self._siteParameters:
                if par not in self.SITE_PARS_TO_WRITE_IN_CONFIG:
                    continue
                value = self._siteParameters[par]['Value']
                file.write(tab + '{} = {}\n'.format(par, value))
            file.write('\n')

            # Writing common parameters
            file.write(tab + '% Common parameters\n')
            self._writeCommonParameters(file)
            file.write('\n')

            # Maximum telescopes
            file.write(tab + 'maximum_telescopes = {}\n\n'.format(self.numberOfTelescopes))

            # Default telescope - 0th tel in telescope list
            telConfigFile = (
                self._telescopeModel[0].getConfigFile(noExport=True).name
            )
            file.write('# include <{}>\n\n'.format(telConfigFile))

            # Looping over telescopes - from 1 to ...
            for count, telModel in enumerate(self._telescopeModel):
                telConfigFile = telModel.getConfigFile(noExport=True).name
                file.write('%{}\n'.format(self.layout[count].name))
                file.write('#elif TELESCOPE == {}\n\n'.format(count + 1))
                file.write('# include <{}>\n\n'.format(telConfigFile))
            file.write('#endif \n\n')
    # END exportSimtelArrayConfigFile

    def _writeCommonParameters(self, file):
        # Common parameters taken from CTA-PROD4-common.cfg
        # TODO: Store these somewhere else
        self._logger.warning('Common parameters are hardcoded!')
        COMMON_PARS = {
            'trigger_telescopes': 1,
            'array_trigger': 'none',
            'trigger_telescopes': 2,
            'only_triggered_telescopes': 1,
            'array_window': 1000,
            'output_format': 1,
            'mirror_list': 'none',
            'telescope_random_angle': 0.,
            'telescope_random_error': 0.,
            'convergent_depth': 0,
            'iobuf_maximum': 1000000000,
            'iobuf_output_maximum': 400000000,
            'multiplicity_offset': -0.5,
            'discriminator_pulse_shape': 'none',
            'discriminator_amplitude': 0.,
            'discriminator_threshold': 99999.,
            'fadc_noise': 0.,
            'asum_threshold': 0.,
            'asum_shaping_file': 'none',
            'asum_offset': 0.0,
            'dsum_threshold': 0,
            'fadc_pulse_shape': 'none',
            'fadc_amplitude': 0.,
            'fadc_pedestal': 100.,
            'fadc_max_signal': 4095,
            'fadc_max_sum': 16777215,
            'store_photoelectrons': 30,
            'pulse_analysis': -30,
            'sum_before_peak': 3,
            'sum_after_peak': 4
        }

        for par, value in COMMON_PARS.items():
            file.write('   {} = {}\n'.format(par, value))
    # End of writeCommonParameters

    def exportAllSimtelConfigFiles(self):
        '''
        Export sim_telarray config file for the array and for each individual telescope
        into the output model directory.
        '''
        self.exportSimtelTelescopeConfigFiles()
        self.exportSimtelArrayConfigFile()

    def getConfigFile(self):
        '''
        Get the path of the array config file for sim_telarray.
        The config file is produced if the file is not updated.

        Returns
        -------
        Path of the exported config file for sim_telarray.
        '''
        self.exportAllSimtelConfigFiles()
        return self._configFilePath
