import logging
from copy import copy

from simtools import db_handler, io_handler
from simtools.layout.layout_array import LayoutArray
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.util import names
from simtools.util.general import collect_data_from_yaml_or_dict

__all__ = ["ArrayModel"]


class InvalidArrayConfigData(Exception):
    pass


class ArrayModel:
    """
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
    number_of_telescopes: int
        Number of telescopes in the ArrayModel.

    Methods
    -------
    print_telescope_list()
        Print out the list of telescopes for quick inspection.
    export_simtel_telescope_config_files()
        Export sim_telarray config files for all the telescopes
        into the output model directory.
    export_simtel_array_config_file()
        Export sim_telarray config file for the array into the output model
        directory.
    export_all_simtel_config_files()
        Export sim_telarray config file for the array and for each individual telescope
        into the output model directory.
    get_config_file()
        Get the path to the config file for sim_telarray.
    get_config_directory()
        Get the path of the array config directory for sim_telarray.
    """

    def __init__(
        self,
        mongoDBConfig,
        label=None,
        arrayConfigFile=None,
        arrayConfigData=None,
    ):
        """
        ArrayModel.

        Parameters
        ----------
        mongoDBConfig: dict
            MongoDB configuration.
        arrayConfigFile: str
            Path to a yaml file with the array config data.
        arrayConfigData: dict
            Dict with the array config data.
        label: str, optional
            Instance label. Important for output file naming.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init ArrayModel")

        self.label = label
        self.site = None
        self.layout = None
        self.layoutName = None
        self.modelVersion = None

        self.io_handler = io_handler.IOHandler()

        arrayConfigData = collect_data_from_yaml_or_dict(arrayConfigFile, arrayConfigData)
        self._load_array_data(arrayConfigData)

        self._set_config_file_directory()

        self._build_array_model(mongoDBConfig)

        self._telescopeModelFilesExported = False
        self._arrayModelFileExported = False

    @property
    def number_of_telescopes(self):
        return self.layout.get_number_of_telescopes()

    def _load_array_data(self, arrayConfigData):
        """Load parameters from arrayData.

        Parameters
        ----------
        arrayConfigData: dict
        """
        # Validating arrayConfigData
        # Keys 'site', 'layoutName' and 'default' are mandatory.
        # 'default' must have 'LST', 'MST' and 'SST' (for South site) keys.
        self._validate_array_data(arrayConfigData)

        # Site
        self.site = names.validate_site_name(arrayConfigData["site"])

        # Grabbing layout name and building LayoutArray
        self.layoutName = names.validate_layout_array_name(arrayConfigData["layoutName"])
        self.layout = LayoutArray.from_layout_array_name(
            self.site + "-" + self.layoutName,
            label=self.label,
        )

        # Model version
        if "modelVersion" not in arrayConfigData.keys() or arrayConfigData["modelVersion"] is None:
            self._logger.warning("modelVersion not given in arrayConfigData - using current")
            self.modelVersion = "current"
        else:
            self.modelVersion = names.validate_model_version_name(arrayConfigData["modelVersion"])

        # Removing keys that were stored in attributes and keeping the remaining as a dict
        self._arrayConfigData = {
            k: v
            for (k, v) in arrayConfigData.items()
            if k not in ["site", "layoutName", "modelVersion"]
        }

    def _validate_array_data(self, arrayConfigData):
        """
        Validate arrayData by checking the existence of the relevant keys.
        Searching for the keys: 'site', 'array' and 'default'.
        """

        def runOverPars(pars, data, parent=None):
            """Run over pars and validate it."""
            allKeys = data.keys() if parent is None else data[parent].keys()
            for pp in pars:
                if pp not in allKeys:
                    key = pp if parent is None else parent + "." + pp
                    msg = (
                        "Key {} was not found in arrayConfigData ".format(key)
                        + "- impossible to build array model"
                    )
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)

        runOverPars(["site", "layoutName", "default"], arrayConfigData)

    def _set_config_file_directory(self):
        """Define the variable _configFileDirectory and create directories, if needed"""
        self._configFileDirectory = self.io_handler.get_output_directory(self.label, "model")
        if not self._configFileDirectory.exists():
            self._configFileDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.info("Creating directory {}".format(self._configFileDirectory))
        return

    def _build_array_model(self, mongoDBConfig):
        """
        Build the site parameters and the list of telescope models,
        including reading the parameters from the DB.

        Parameters
        ----------
        mongoDBConfig: str
            MongoDB configuration.

        """

        # Getting site parameters from DB
        db = db_handler.DatabaseHandler(mongoDBConfig=mongoDBConfig)
        self._siteParameters = db.get_site_parameters(
            self.site, self.modelVersion, onlyApplicable=True
        )

        # Building telescope models
        self._telescopeModel = list()  # List of telescope models
        _allTelescopeModelNames = list()  # List of telescope names without repetition
        _allParsToChange = dict()
        for tel in self.layout:
            telSize = self.layout.get_telescope_type(tel.name)

            # Collecting telescope name and pars to change from arrayConfigData
            telModelName, parsToChange = self._get_single_telescope_info_from_array_config(
                tel.name, telSize
            )
            if len(parsToChange) > 0:
                _allParsToChange[tel.name] = parsToChange

            self._logger.debug("TelModelName: {}".format(telModelName))

            # Building the basic models - no pars to change yet
            if telModelName not in _allTelescopeModelNames:
                # First time a telescope name is built
                _allTelescopeModelNames.append(telModelName)
                telModel = TelescopeModel(
                    site=self.site,
                    telescopeModelName=telModelName,
                    modelVersion=self.modelVersion,
                    label=self.label,
                    mongoDBConfig=mongoDBConfig,
                )
            else:
                # Telescope name already exists.
                # Finding the TelescopeModel and copying it.
                for tel in self._telescopeModel:
                    if tel.name != telModelName:
                        continue
                    self._logger.debug(
                        "Copying tel model {} already loaded from DB".format(tel.name)
                    )
                    telModel = copy(tel)
                    break

            self._telescopeModel.append(telModel)

        # Checking whether the size of the telescope list and the layout match
        if len(self._telescopeModel) != len(self.layout):
            self._logger.warning(
                "Number of telescopes in the list of telescope models does "
                "not match the number of telescopes in the LayoutArray - something is wrong!"
            )

        # Changing parameters, if there are any in allParsToChange
        if len(_allParsToChange) > 0:
            for telData, telModel in zip(self.layout, self._telescopeModel):
                if telData.name not in _allParsToChange:
                    continue
                self._logger.debug(
                    "Changing {} pars of a {}: {}, ...".format(
                        len(_allParsToChange[telData.name]),
                        telData.name,
                        *_allParsToChange[telData.name]
                    )
                )
                telModel.change_multiple_parameters(**_allParsToChange[telData.name])
                telModel.set_extra_label(telData.name)

    def _get_single_telescope_info_from_array_config(self, telName, telSize):
        """
        arrayConfigData contains the default telescope models for each
        telescope size and the list of specific telescopes.
        For each case, the data can be given only as a name or
        as a dict with 'name' and parameters to change.
        This function has to identify these two cases and collect
        the telescope name and the dict with the parameters to change.

        Parameters
        ----------
        telName: str
            Name of the telescope at the layout level (LST-01, MST-05, ...).
        telSize: str
            LST, MST or SST.
        """

        def _proccessSingleTelescope(data):
            """
            Parameters
            ----------
            data: dict or str
                Piece of the arrayConfigData for one specific telescope.
            """
            if isinstance(data, dict):
                # Case 0: data is dict
                if "name" not in data.keys():
                    msg = "ArrayConfig has no name for a telescope"
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)
                telName = telSize + "-" + data["name"]
                parsToChange = {k: v for (k, v) in data.items() if k != "name"}
                self._logger.debug(
                    "Grabbing tel data as dict - "
                    + "name: {}, ".format(telName)
                    + "{} pars to change".format(len(parsToChange))
                )
                return telName, parsToChange
            elif isinstance(data, str):
                # Case 1: data is string (only name)
                telName = telSize + "-" + data
                return telName, dict()
            else:
                # Case 2: data has a wrong type
                msg = "ArrayConfig has wrong input for a telescope"
                self._logger.error(msg)
                raise InvalidArrayConfigData(msg)

        if telName in self._arrayConfigData.keys():
            # Specific info for this telescope
            return _proccessSingleTelescope(self._arrayConfigData[telName])
        else:
            # Checking if default option exists in arrayConfigData
            notContainsDefaultKey = (
                "default" not in self._arrayConfigData.keys()
                or telSize not in self._arrayConfigData["default"].keys()
            )

            if notContainsDefaultKey:
                msg = (
                    "default option was not given in arrayConfigData "
                    + "for the telescope {}".format(telName)
                )
                self._logger.error(msg)
                raise InvalidArrayConfigData(msg)

            # Grabbing the default option
            return _proccessSingleTelescope(self._arrayConfigData["default"][telSize])

    def print_telescope_list(self):
        """Print out the list of telescopes for quick inspection."""
        for telData, telModel in zip(self.layout, self._telescopeModel):
            print("Name: {}\t Model: {}".format(telData.name, telModel.name))

    def export_simtel_telescope_config_files(self):
        """
        Export sim_telarray config files for all the telescopes
        into the output model directory.
        """
        exportedModels = list()
        for telModel in self._telescopeModel:
            name = telModel.name + (
                "_" + telModel.extra_label if telModel.extra_label != "" else ""
            )
            if name not in exportedModels:
                self._logger.debug("Exporting config file for tel {}".format(name))
                telModel.export_config_file()
                exportedModels.append(name)
            else:
                self._logger.debug("Config file for tel {} already exists - skipping".format(name))

        self._telescopeModelFilesExported = True

    def export_simtel_array_config_file(self):
        """
        Export sim_telarray config file for the array into the output model
        directory.
        """

        # Setting file name and the location
        configFileName = names.simtel_array_config_file_name(
            self.layoutName, self.site, self.modelVersion, self.label
        )
        self._configFilePath = self._configFileDirectory.joinpath(configFileName)

        # Writing parameters to the file
        self._logger.info("Writing array config file into {}".format(self._configFilePath))
        simtelWriter = SimtelConfigWriter(
            site=self.site,
            layoutName=self.layoutName,
            modelVersion=self.modelVersion,
            label=self.label,
        )
        simtelWriter.write_array_config_file(
            configFilePath=self._configFilePath,
            layout=self.layout,
            telescopeModel=self._telescopeModel,
            siteParameters=self._siteParameters,
        )
        self._arrayModelFileExported = True

    def export_all_simtel_config_files(self):
        """
        Export sim_telarray config file for the array and for each individual telescope
        into the output model directory.
        """
        if not self._telescopeModelFilesExported:
            self.export_simtel_telescope_config_files()
        if not self._arrayModelFileExported:
            self.export_simtel_array_config_file()

    def get_config_file(self):
        """
        Get the path of the array config file for sim_telarray.
        The config file is produced if the file is not updated.

        Returns
        -------
        Path of the exported config file for sim_telarray.
        """
        self.export_all_simtel_config_files()
        return self._configFilePath

    def get_config_directory(self):
        """
        Get the path of the array config directory for sim_telarray.

        Returns
        -------
        Path of the config directory path for sim_telarray.
        """
        return self._configFileDirectory
