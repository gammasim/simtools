import logging
import shutil
from copy import copy

import astropy.io.ascii
import numpy as np
from astropy.table import Table

import simtools.util.general as gen
from simtools import db_handler, io_handler
from simtools.model.camera import Camera
from simtools.model.mirrors import Mirrors
from simtools.simtel.simtel_config_writer import SimtelConfigWriter
from simtools.util import names
from simtools.util.model import validate_model_parameter

__all__ = ["TelescopeModel"]


class InvalidParameter(Exception):
    pass


class TelescopeModel:
    """
    TelescopeModel represents the MC model of an individual telescope. \
    It contains the list of parameters that can be read from the DB. \
    A set of methods are available to manipulate parameters (changing, adding, removing etc). \


    Attributes
    ----------
    site: str
        North or South.
    name: str
        Telescope name for the base set of parameters (e.g., LST-1, ...).
    modelVersion: str
        Version of the model (e.g., prod5).
    label: str
        Instance label.
    mirrors: Mirrors
        Mirrors object created from the mirror list of the model.
    camera: Camera
        Camera object created from the camera config file of the model.
    reference_data: Reference data
        Dictionary with reference data parameters (e.g., NSB reference value)
    extra_label: str
        Extra label to be used in case of multiple telescope configurations (e.g., by ArrayModel).

    Methods
    -------
    from_config_file(configFileName, telescopeModelName, label=None)
        Create a TelescopeModel from a sim_telarray cfg file.
    set_extra_label(extra_label)
        Set an extra label for the name of the config file.
    has_parameter(parName)
        Verify if parameter is in the model.
    get_parameter(parName)
        Get an existing parameter of the model.
    add_parameter(parName, value)
        Add new parameters to the model.
    change_parameter(parName, value)
        Change the value of existing parameters to the model.
    change_multiple_parameters(**pars)
        Change the value of existing parameters to the model.
    remove_parameters(*args)
        Remove parameters from the model.
    print_parameters()
        Print parameters and their values for debugging purposes.
    export_config_file()
        Export config file for sim_telarray.
    get_config_file()
        Get the path to the config file for sim_telarray.
    """

    def __init__(
        self,
        site,
        telescopeModelName,
        mongoDBConfig,
        modelVersion="Current",
        label=None,
    ):
        """
        TelescopeModel.

        Parameters
        ----------
        site: str
            South or North.
        telescopeModelName: str
            Telescope name (ex. LST-1, ...).
        mongoDBConfig: dict
            MongoDB configuration.
        modelVersion: str, optional
            Version of the model (ex. prod4) (default='Current').
        label: str, optional
            Instance label. Important for output file naming.

        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init TelescopeModel")

        self.site = names.validate_site_name(site)
        self.name = names.validate_telescope_model_name(telescopeModelName)
        self.modelVersion = names.validate_model_version_name(modelVersion)
        self.label = label
        self._extra_label = None

        self.io_handler = io_handler.IOHandler()
        self.mongoDBConfig = mongoDBConfig

        self._parameters = dict()

        self._load_parameters_from_db()

        self._set_config_file_directory_and_name()
        self._isConfigFileUpToDate = False
        self._isExportedModelFilesUpToDate = False

    @property
    def mirrors(self):
        if not hasattr(self, "_mirrors"):
            self._load_mirrors()
        return self._mirrors

    @property
    def camera(self):
        if not hasattr(self, "_camera"):
            self._load_camera()
        return self._camera

    @property
    def reference_data(self):
        if not hasattr(self, "_reference_data"):
            self._load_reference_data()
        return self._reference_data

    @property
    def derived(self):
        if not hasattr(self, "_derived"):
            self._load_derived_values()
            self.export_derived_files()
        return self._derived

    @property
    def extra_label(self):
        return self._extra_label if self._extra_label is not None else ""

    @classmethod
    def from_config_file(
        cls,
        configFileName,
        site,
        telescopeModelName,
        label=None,
    ):
        """
        Create a TelescopeModel from a sim_telarray config file.

        Notes
        -----
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

        Returns
        -------
        Instance of the TelescopeModel class.
        """
        parameters = dict()
        tel = cls(
            site=site,
            telescopeModelName=telescopeModelName,
            mongoDBConfig=None,
            label=label,
        )

        def _processLine(words):
            """
            Process a line of the input config file that contains a parameter.

            Parameters
            ----------
            words: list of str
                List of str from the split of a line from the file.

            Returns
            -------
            (parName, parValue)
            """
            iComment = len(words)  # Index of any possible comment
            for w in words:
                if "%" in w:
                    iComment = words.index(w)
                    break
            words = words[0:iComment]  # Removing comment
            parName = words[0].replace("=", "")
            parValue = ""
            for w in words[1:]:
                w = w.replace("=", "")
                w = w.replace(",", " ")
                parValue += w + " "
            parValue = parValue.rstrip().lstrip()  # Removing trailing spaces (left and right)
            return parName, parValue

        with open(configFileName, "r") as file:
            for line in file:
                words = line.split()
                if len(words) == 0:
                    continue
                elif "%" in words[0] or "echo" in words:
                    continue
                elif "#" not in line and len(words) > 0:
                    par, value = _processLine(words)
                    par, value = validate_model_parameter(par, value)
                    parameters[par] = value

        for par, value in parameters.items():
            tel.add_parameter(par, value)

        tel._isExportedModelFilesUpToDate = True
        return tel

    def set_extra_label(self, extra_label):
        """
        Set an extra label for the name of the config file.

        Notes
        -----
        The config file directory name is not affected by the extra label. \
        Only the file name is changed. This is important for the ArrayModel \
        class to export multiple config files in the same directory.

        Parameters
        ----------
        extra_label: str
            Extra label to be appended to the original label.
        """
        self._extra_label = extra_label
        self._set_config_file_directory_and_name()

    def _set_config_file_directory_and_name(self):
        """Define the variable _configFileDirectory and create directories, if needed"""

        self._configFileDirectory = self.io_handler.get_output_directory(self.label, "model")

        # Setting file name and the location
        configFileName = names.simtel_telescope_config_file_name(
            self.site, self.name, self.modelVersion, self.label, self._extra_label
        )
        self._configFilePath = self._configFileDirectory.joinpath(configFileName)

    def _load_parameters_from_db(self):
        """Read parameters from DB and store them in _parameters."""

        if self.mongoDBConfig is None:
            return

        self._logger.debug("Reading telescope parameters from DB")

        self._set_config_file_directory_and_name()
        db = db_handler.DatabaseHandler(mongoDBConfig=self.mongoDBConfig)
        self._parameters = db.get_model_parameters(
            self.site, self.name, self.modelVersion, onlyApplicable=True
        )

        self._logger.debug("Reading site parameters from DB")
        _sitePars = db.get_site_parameters(self.site, self.modelVersion, onlyApplicable=True)
        self._parameters.update(_sitePars)

    def has_parameter(self, parName):
        """
        Verify if the parameter is in the model.

        Parameters
        ----------
        parName: str
            Name of the parameter.

        Returns
        -------
        bool
        """
        return parName in self._parameters

    def get_parameter(self, parName):
        """
        Get an existing parameter of the model, including derived parameters.

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
        """
        try:
            return self._parameters[parName]
        except KeyError:
            pass  # search in the derived parameters
        try:
            return self.derived[parName]
        except KeyError:
            msg = f"Parameter {parName} was not found in the model"
            self._logger.error(msg)
            raise InvalidParameter(msg)

    def get_parameter_value(self, parName):
        """
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
        """
        parInfo = self.get_parameter(parName)
        return parInfo["Value"]

    def add_parameter(self, parName, value, isFile=False, isAplicable=True):
        """
        Add a new parameters to the model. \
        This function does not modify the DB, it affects only the current instance.

        Parameters
        ----------
        parName: str
            Name of the parameter.
        value:
            Value of the parameter.
        isFile: bool
            Indicates whether the new parameter is a file or not.
        isAplicable: bool
            Indicates whether the new parameter is applicable or not.

        Raises
        ------
        InvalidParameter
            If an existing parameter is tried to be added.
        """
        if parName in self._parameters:
            msg = "Parameter {} already in the model, use change_parameter instead".format(parName)
            self._logger.error(msg)
            raise InvalidParameter(msg)
        else:
            self._logger.info("Adding {}={} to the model".format(parName, value))
            self._parameters[parName] = dict()
            self._parameters[parName]["Value"] = value
            self._parameters[parName]["Type"] = type(value)
            self._parameters[parName]["Applicable"] = isAplicable
            self._parameters[parName]["File"] = isFile

        self._isConfigFileUpToDate = False
        if isFile:
            self._isExportedModelFilesUpToDate = False

    def change_parameter(self, parName, value):
        """
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
        """
        if parName not in self._parameters:
            msg = "Parameter {} not in the model, use add_parameters instead".format(parName)
            self._logger.error(msg)
            raise InvalidParameter(msg)
        else:
            # TODO: fix this in order to use the type from the DB directly.
            if not isinstance(value, type(self._parameters[parName]["Value"])):
                self._logger.warning(f"Value type of {parName} differs from the current one")
            self._parameters[parName]["Value"] = value
            self._logger.debug("Changing parameter {}".format(parName))

            # In case parameter is a file, the model files will be outdated
            if self._parameters[parName]["File"]:
                self._isExportedModelFilesUpToDate = False

        self._isConfigFileUpToDate = False

    def change_multiple_parameters(self, **kwargs):
        """
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
        """
        for par, value in kwargs.items():
            if par in self._parameters:
                self.change_parameter(par, value)
            else:
                self.add_parameter(par, value)

        self._isConfigFileUpToDate = False

    def remove_parameters(self, *args):
        """
        Remove a set of parameters from the model.

        Parameters
        ----------
        args
            Each parameter to be removed has to be passed as args.

        Raises
        ------
        InvalidParameter
            If at least one of the parameter to be removed is not in this model.
        """
        for par in args:
            if par in self._parameters:
                self._logger.info("Removing parameter {}".format(par))
                del self._parameters[par]
            else:
                msg = "Could not remove parameter {} because it does not exist".format(par)
                self._logger.error(msg)
                raise InvalidParameter(msg)
        self._isConfigFileUpToDate = False

    def add_parameter_file(self, parName, filePath):
        """
        Add a file to the config file directory.

        Parameters
        ----------
        parName: str
            Name of the parameter.
        filePath: str
            Path of the file to be added to the config file directory.
        """
        if not hasattr(self, "_addedParameterFiles"):
            self._addedParameterFiles = list()
        self._addedParameterFiles.append(parName)
        shutil.copy(filePath, self._configFileDirectory)

    def export_model_files(self):
        """Exports the model files into the config file directory."""
        db = db_handler.DatabaseHandler(mongoDBConfig=self.mongoDBConfig)

        # Removing parameter files added manually (which are not in DB)
        parsFromDB = copy(self._parameters)
        if hasattr(self, "_addedParameterFiles"):
            for par in self._addedParameterFiles:
                parsFromDB.pop(par)

        db.export_model_files(parsFromDB, self._configFileDirectory)
        self._isExportedModelFilesUpToDate = True

    def print_parameters(self):
        """Print parameters and their values for debugging purposes."""
        for par, info in self._parameters.items():
            print("{} = {}".format(par, info["Value"]))

    def export_config_file(self):
        """Export the config file used by sim_telarray."""

        # Exporting model file
        if not self._isExportedModelFilesUpToDate:
            self.export_model_files()

        # Using SimtelConfigWriter to write the config file.
        self._load_simtel_config_writer()
        self.simtelConfigWriter.write_telescope_config_file(
            configFilePath=self._configFilePath, parameters=self._parameters
        )

    def export_derived_files(self):
        """
        Write to disk a file from the derived values DB.
        """

        db = db_handler.DatabaseHandler(mongoDBConfig=self.mongoDBConfig)
        for parNow in self.derived:
            if self.derived[parNow]["File"]:
                db.export_file_db(
                    dbName=db.DB_DERIVED_VALUES,
                    dest=self.io_handler.get_output_directory(self.label, "derived"),
                    fileName=self.derived[parNow]["Value"],
                )

    def get_config_file(self, noExport=False):
        """
        Get the path of the config file for sim_telarray. \
        The config file is produced if the file is not updated.

        Parameters
        ----------
        noExport: bool
            Turn it on if you do not want the file to be exported.

        Returns
        -------
        Path of the exported config file for sim_telarray.
        """
        if not self._isConfigFileUpToDate and not noExport:
            self.export_config_file()
        return self._configFilePath

    def get_config_directory(self):
        """
        Get the path where all the configuration files for sim_telarray are written to.

        Returns
        -------
        Path where all the configuration files for sim_telarray are written to.
        """
        return self._configFileDirectory

    def get_derived_directory(self):
        """
        Get the path where all the files with derived values for are written to.

        Returns
        -------
        Path where all the files with derived values are written to.
        """
        return self._configFileDirectory.parents[0].joinpath("derived")

    def get_telescope_transmission_parameters(self):
        """
        Get tel. transmission pars as a list of floats.

        Returns
        -------
        list of floats
            List of 4 parameters that describe the tel. transmission vs off-axis.
        """

        telescopeTransmission = self.get_parameter_value("telescope_transmission")
        if isinstance(telescopeTransmission, str):
            return [float(v) for v in self.get_parameter_value("telescope_transmission").split()]
        else:
            return [float(telescopeTransmission), 0, 0, 0]

    def export_single_mirror_list_file(self, mirrorNumber, setFocalLengthToZero):
        """
        Export a mirror list file with a single mirror in it.

        Parameters
        ----------
        mirrorNumber: int
            Number index of the mirror.
        setFocalLengthToZero: bool
            Set the focal length to zero if True.
        """
        if mirrorNumber > self.mirrors.numberOfMirrors:
            logging.error("mirrorNumber > numberOfMirrors")
            return None

        fileName = names.simtel_single_mirror_list_file_name(
            self.site, self.name, self.modelVersion, mirrorNumber, self.label
        )
        if not hasattr(self, "_singleMirrorListFilePath"):
            self._singleMirrorListFilePaths = dict()
        self._singleMirrorListFilePaths[mirrorNumber] = self._configFileDirectory.joinpath(fileName)

        # Using SimtelConfigWriter
        self._load_simtel_config_writer()
        self.simtelConfigWriter.write_single_mirror_list_file(
            mirrorNumber,
            self.mirrors,
            self._singleMirrorListFilePaths[mirrorNumber],
            setFocalLengthToZero,
        )

    # END of export_single_mirror_list_file

    def get_single_mirror_list_file(self, mirrorNumber, setFocalLengthToZero=False):
        """
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
        """
        self.export_single_mirror_list_file(mirrorNumber, setFocalLengthToZero)
        return self._singleMirrorListFilePaths[mirrorNumber]

    def _load_mirrors(self):
        """Load the attribute mirrors by creating a Mirrors object with the mirror list file."""
        mirrorListFileName = self._parameters["mirror_list"]["Value"]
        try:
            mirrorListFile = gen.find_file(mirrorListFileName, self._configFileDirectory)
        except FileNotFoundError:
            mirrorListFile = gen.find_file(mirrorListFileName, self.io_handler.model_path)
            self._logger.warning(
                "MirrorListFile was not found in the config directory - "
                "Using the one found in the model_path"
            )
        self._mirrors = Mirrors(mirrorListFile)

    def _load_reference_data(self):
        """Load the reference data for this telescope from the DB."""
        self._logger.debug("Reading reference data from DB")
        db = db_handler.DatabaseHandler(mongoDBConfig=self.mongoDBConfig)
        self._reference_data = db.get_reference_data(
            self.site, self.modelVersion, onlyApplicable=True
        )

    def _load_derived_values(self):
        """Load the derived values for this telescope from the DB."""
        self._logger.debug("Reading derived data from DB")
        db = db_handler.DatabaseHandler(mongoDBConfig=self.mongoDBConfig)
        self._derived = db.get_derived_values(
            self.site,
            self.name,
            self.modelVersion,
        )

    def _load_camera(self):
        """Loading camera attribute by creating a Camera object with the camera config file."""
        cameraConfigFile = self.get_parameter_value("camera_config_file")
        focalLength = self.get_parameter_value("effective_focal_length")
        if focalLength == 0.0:
            self._logger.warning("Using focal_length because effective_focal_length is 0.")
            focalLength = self.get_parameter_value("focal_length")
        try:
            cameraConfigFilePath = gen.find_file(cameraConfigFile, self._configFileDirectory)
        except FileNotFoundError:
            self._logger.warning(
                "CameraConfigFile was not found in the config directory - "
                "Using the one found in the model_path"
            )
            cameraConfigFilePath = gen.find_file(cameraConfigFile, self.io_handler.model_path)

        self._camera = Camera(
            telescopeModelName=self.name,
            cameraConfigFile=cameraConfigFilePath,
            focalLength=focalLength,
        )

    def _load_simtel_config_writer(self):
        if not hasattr(self, "simtelConfigWriter"):
            self.simtelConfigWriter = SimtelConfigWriter(
                site=self.site,
                telescopeModelName=self.name,
                modelVersion=self.modelVersion,
                label=self.label,
            )

    def is_file_2D(self, par):
        """
        Check if the file referenced by par is a 2D table.

        Parameters
        ----------
        par: str
            Name of the parameter.

        Returns
        -------
        bool:
            True if the file is a 2D map type, False otherwise.
        """
        if not self.has_parameter(par):
            logging.error("Parameter {} does not exist".format(par))
            return False

        fileName = self.get_parameter_value(par)
        file = self.get_config_directory().joinpath(fileName)
        with open(file, "r") as f:
            is2D = "@RPOL@" in f.read()
        return is2D

    def read_two_dim_wavelength_angle(self, fileName):
        """
        Read a two dimensional distribution of wavelngth and angle (z-axis can be anything).
        Return a dictionary with three arrays,
        wavelength, angles, z (can be transmission, reflectivity, etc.)

        Parameters
        ----------
        fileName: str or Path
            File assumed to be in the model directory

        Returns
        -------
        dict:
            dict of three arrays, wavelength, degrees, z
        """

        _file = self.get_config_directory().joinpath(fileName)
        with open(_file, "r") as f:
            for i_line, line in enumerate(f):
                if line.startswith("ANGLE"):
                    degrees = np.array(line.strip().split("=")[1].split(), dtype=np.float16)
                    break  # The rest can be read with np.loadtxt

        _data = np.loadtxt(_file, skiprows=i_line + 1)

        return {
            "Wavelength": _data[:, 0],
            "Angle": degrees,
            "z": np.array(_data[:, 1:]).T,
        }

    def get_on_axis_eff_optical_area(self):
        """
        Return the on-axis effective optical area (derived previously for this telescope).
        """

        rayTracingData = astropy.io.ascii.read(
            self.get_derived_directory().joinpath(self.derived["ray_tracing"]["Value"])
        )
        if not np.isclose(rayTracingData["Off-axis angle"][0], 0):
            self._logger.error(
                f"No value for the on-axis effective optical area exists."
                f" The minumum off-axis angle is {rayTracingData['Off-axis angle'][0]}"
            )
            raise ValueError
        return rayTracingData["eff_area"][0]

    def read_incidence_angle_distribution(self, incidenceAngleDistFile):
        """
        Read the incidence angle distrubution from a file

        Parameters
        ----------
        incidenceAngleDistFile: str
            File name of the incidence angle distribution

        Returns
        -------
        incidenceAngleDist: Astropy table
            Astropy table with the incidence angle distribution
        """

        incidenceAngleDist = astropy.io.ascii.read(
            self.get_derived_directory().joinpath(incidenceAngleDistFile)
        )
        return incidenceAngleDist

    @staticmethod
    def calc_average_curve(curves, incidenceAngleDist):
        """
        Calculate an average curve from a set of curves, using as weights
        the distribution of incidence angles provided in incidenceAngleDist

        Parameters
        ----------
        curves: dict
            dict of with 3 "columns", Wavelength, Angle and z
            The dictionary represents a two dimensional distribution of wavelengths and angles
            with the z value being e.g., reflectivity, transmission, etc.
        incidenceAngleDist: Astropy table
            Astropy table with the incidence angle distribution
            The assumed columns are "Incidence angle" and "Fraction".

        Returns
        -------
        averageCurve: Astropy Table
            Table with the averaged curve
        """

        weights = list()
        for angleNow in curves["Angle"]:
            weights.append(
                incidenceAngleDist["Fraction"][
                    np.nanargmin(np.abs(angleNow - incidenceAngleDist["Incidence angle"].value))
                ]
            )

        averageCurve = Table(
            [curves["Wavelength"], np.average(curves["z"], weights=weights, axis=0)],
            names=("Wavelength", "z"),
        )

        return averageCurve

    def export_table_to_model_directory(self, fileName, table):
        """
        Write out a file with the provided table to the model directory.

        Parameters
        ----------
        fileName: str
            File name to write to
        table: Astropy Table
            Table with the values to write to the file

        Returns
        -------
        Path:
            Path to the file exported.
        """

        fileToWriteTo = self._configFileDirectory.joinpath(fileName)
        table.write(fileToWriteTo, format="ascii.commented_header", overwrite=True)
        return fileToWriteTo.absolute()
