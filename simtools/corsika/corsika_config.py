import copy
import logging

import astropy.units as u
import numpy as np
from astropy.io.misc import yaml

import simtools.util.general as gen
from simtools import io_handler
from simtools.layout.layout_array import LayoutArray
from simtools.util import names
from simtools.util.general import collect_data_from_yaml_or_dict

__all__ = [
    "CorsikaConfig",
    "MissingRequiredInputInCorsikaConfigData",
    "InvalidCorsikaInput",
]


class MissingRequiredInputInCorsikaConfigData(Exception):
    pass


class InvalidCorsikaInput(Exception):
    pass


class CorsikaConfig:
    """
    CorsikaConfig deals with configuration for running CORSIKA. \
    User parameters must be given by the corsikaConfigData or \
    corsikaConfigFile arguments. An example of corsikaConfigData follows \
    below.

    .. code-block:: python

        corsikaConfigData = {
            'primary': 'proton',
            'nshow': 10000,
            'nrun': 1,
            'zenith': 20 * u.deg,
            'viewcone': 5 * u.deg,
            'erange': [10 * u.GeV, 100 * u.TeV],
            'eslope': -2,
            'phi': 0 * u.deg,
            'cscat': [10, 1500 * u.m, 0]
        }

    The remaining CORSIKA parameters can be set as a yaml file, using the argument \
    corsikaParametersFile. When not given, corsikaParameters will be loaded \
    from data/corsika/corsika_parameters.yml.

    Attributes
    ----------
    site: str
        North or South.
    layoutName: str
        Name of the layout.
    layout: LayoutArray
        LayoutArray object containing the telescope positions.
    label: str
        Instance label.
    primary: str
        Name of the primary particle (e.g gamma, proton ...).

    Methods
    -------
    set_user_parameters(corsikaConfigData)
        Set user parameters from a dict.
    get_user_parameter(parName)
        Get the value of a user parameter.
    print_user_parameters()
        Print user parameters for inspection.
    get_output_file_name(runNumber)
        Get the name of the CORSIKA output file for a certain run.
    export_input_file()
        Create and export CORSIKA input file.
    get_input_file()
        Get the full path of the CORSIKA input file.
    get_input_file_name_for_run(runNumber)
        Get the CORSIKA input file for one specific run.
        This is the input file after being pre-processed by sim_telarray (pfp).
    """

    def __init__(
        self,
        site,
        layoutName,
        label=None,
        corsikaConfigData=None,
        corsikaConfigFile=None,
        corsikaParametersFile=None,
    ):
        """
        CorsikaConfig init.

        Parameters
        ----------
        site: str
            South or North.
        layoutName: str
            Name of the layout.
        layout: LayoutArray
            Instance of LayoutArray.
        label: str
            Instance label.
        corsikaConfigData: dict
            Dict with CORSIKA config data.
        corsikaConfigFile: str or Path
            Path to yaml file containing CORSIKA config data.
        corsikaParametersFile: str or Path
            Path to yaml file containing CORSIKA parameters.
        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaConfig")

        self.label = label
        self.site = names.validate_site_name(site)

        self.io_handler = io_handler.IOHandler()

        # Grabbing layout name and building LayoutArray
        self.layoutName = names.validate_layout_array_name(layoutName)
        self.layout = LayoutArray.from_layout_array_name(
            self.site + "-" + self.layoutName, label=self.label
        )

        # Load parameters
        self._load_corsika_parameters_file(corsikaParametersFile)

        corsikaConfigData = collect_data_from_yaml_or_dict(corsikaConfigFile, corsikaConfigData)
        self.set_user_parameters(corsikaConfigData)
        self._is_fileUpdated = False

    def __repr__(self):
        text = "<class {}> (site={}, layout={}, label={})".format(
            self.__class__.__name__, self.site, self.layoutName, self.label
        )
        return text

    def _load_corsika_parameters_file(self, filename):
        """
        Load CORSIKA parameters from a given file (filename not None),
        or from the default parameter file provided in the
        data directory (filename is None).
        """
        if filename is not None:
            # User provided file.
            self._corsikaParametersFile = filename
        else:
            # Default file from data directory.
            self._corsikaParametersFile = self.io_handler.get_input_data_file(
                "corsika", "corsika_parameters.yml"
            )
        self._logger.debug(
            "Loading CORSIKA parameters from file {}".format(self._corsikaParametersFile)
        )
        with open(self._corsikaParametersFile, "r") as f:
            self._corsikaParameters = yaml.load(f)

    def set_user_parameters(self, corsikaConfigData):
        """
        Set user parameters from a dict.

        Parameters
        ----------
        corsikaConfigData: dict
            Contains the user parameters. Ex.

            .. code-block:: python

                corsikaConfigData = {
                    'primary': 'proton',
                    'nshow': 10000,
                    'nrun': 1,
                    'zenith': 20 * u.deg,
                    'viewcone': 5 * u.deg,
                    'erange': [10 * u.GeV, 100 * u.TeV],
                    'eslope': -2,
                    'phi': 0 * u.deg,
                    'cscat': [10, 1500 * u.m, 0]
                }

        Raises
        ------
        InvalidCorsikaInput
            If any parameter given as input has wrong len, unit or
            an invalid name.
        MissingRequiredInputInCorsikaConfigData
            If any required user parameter is missing.
        """
        self._logger.debug("Setting user parameters from corsikaConfigData")
        self._userParameters = dict()

        userPars = self._corsikaParameters["USER_PARAMETERS"]

        # Collecting all parameters given as arguments
        for keyArgs, valueArgs in corsikaConfigData.items():
            # Looping over USER_PARAMETERS and searching for a match
            isIdentified = False
            for parName, parInfo in userPars.items():
                if keyArgs.upper() != parName and keyArgs.upper() not in parInfo["names"]:
                    continue
                # Matched parameter
                validatedValueArgs = self._validate_and_convert_argument(
                    parName, parInfo, valueArgs
                )
                self._userParameters[parName] = validatedValueArgs
                isIdentified = True

            # Raising error for an unidentified input.
            if not isIdentified:
                msg = "Argument {} cannot be identified.".format(keyArgs)
                self._logger.error(msg)
                raise InvalidCorsikaInput(msg)

        # Checking for parameters with default option
        # If it is not given, filling it with the default value
        for parName, parInfo in userPars.items():
            if parName in self._userParameters:
                continue
            elif "default" in parInfo.keys():
                validatedValue = self._validate_and_convert_argument(
                    parName, parInfo, parInfo["default"]
                )
                self._userParameters[parName] = validatedValue
            else:
                msg = "Required parameters {} was not given (there may be more).".format(parName)
                self._logger.error(msg)
                raise MissingRequiredInputInCorsikaConfigData(msg)

        # Converting AZM to CORSIKA reference (PHIP)
        phip = 180.0 - self._userParameters["AZM"][0]
        phip = phip + 360.0 if phip < 0.0 else phip
        phip = phip - 360.0 if phip >= 360.0 else phip
        self._userParameters["PHIP"] = [phip, phip]

        self._is_fileUpdated = False

    def _validate_and_convert_argument(self, parName, parInfo, valueArgsIn):
        """
        Validate input user parameter and convert it to the right units, if needed.
        Returns the validated arguments in a list.
        """

        # Turning valueArgs into a list, if it is not.
        valueArgs = gen.copy_as_list(valueArgsIn)

        if len(valueArgs) == 1 and parName in ["THETAP", "AZM"]:
            # Fixing single value zenith or azimuth angle.
            # THETAP and AZM should be written as a 2 values range in the CORSIKA input file
            valueArgs = valueArgs * 2
        elif len(valueArgs) == 1 and parName == "VIEWCONE":
            # Fixing single value viewcone.
            # VIEWCONE should be written as a 2 values range in the CORSIKA input file
            valueArgs = [0 * parInfo["unit"][0], valueArgs[0]]
        elif parName == "PRMPAR":
            valueArgs = self._convert_primary_input_and_store_primary_name(valueArgs)

        if len(valueArgs) != parInfo["len"]:
            msg = "CORSIKA input entry with wrong len: {}".format(parName)
            self._logger.error(msg)
            raise InvalidCorsikaInput(msg)

        if "unit" not in parInfo.keys():
            return valueArgs
        else:
            # Turning parInfo['unit'] into a list, if it is not.
            parUnit = gen.copy_as_list(parInfo["unit"])

            # Checking units and converting them, if needed.
            valueArgsWithUnits = list()
            for arg, unit in zip(valueArgs, parUnit):
                if unit is None:
                    valueArgsWithUnits.append(arg)
                    continue

                if isinstance(arg, str):
                    arg = u.quantity.Quantity(arg)

                if not isinstance(arg, u.quantity.Quantity):
                    msg = "CORSIKA input given without unit: {}".format(parName)
                    self._logger.error(msg)
                    raise InvalidCorsikaInput(msg)
                elif not arg.unit.is_equivalent(unit):
                    msg = "CORSIKA input given with wrong unit: {}".format(parName)
                    self._logger.error(msg)
                    raise InvalidCorsikaInput(msg)
                else:
                    valueArgsWithUnits.append(arg.to(unit).value)

            return valueArgsWithUnits

    def _convert_primary_input_and_store_primary_name(self, value):
        """
        Convert a primary name into the proper CORSIKA particle ID and store \
        its name in self.primary attribute.

        Parameters
        ----------
        value: str
            Input primary name (e.g gamma, proton ...)

        Raises
        ------
        InvalidPrimary
            If the input name is not found.

        Returns
        -------
        int
            Respective number of the given primary.
        """
        for primName, primInfo in self._corsikaParameters["PRIMARIES"].items():
            if value[0].upper() == primName or value[0].upper() in primInfo["names"]:
                self.primary = primName.lower()
                return [primInfo["number"]]
        msg = "Primary not valid: {}".format(value)
        self._logger.error(msg)
        raise InvalidCorsikaInput(msg)

    def get_user_parameter(self, parName):
        """
        Get the value of a user parameter.

        Parameters
        ----------
        parName: str
            Name of the parameter as used in the CORSIKA input file (e.g. PRMPAR, THETAP ...)

        Raises
        ------
        KeyError
            When parName is not a valid parameter name.

        Returns
        -------
        list
            Value(s) of the parameter.
        """
        try:
            parValue = self._userParameters[parName.upper()]
        except KeyError:
            self._logger.warning("Parameter {} is not a user parameter".format(parName))
            raise

        return parValue if len(parValue) > 1 else parValue[0]

    def print_user_parameters(self):
        """Print user parameters for inspection."""
        for par, value in self._userParameters.items():
            print("{} = {}".format(par, value))

    def get_output_file_name(self, runNumber):
        """
        Get the name of the CORSIKA output file for a certain run.

        Parameters
        ----------
        runNumber: int
            Run number.

        Returns
        -------
        str:
            Name of the output CORSIKA file.
        """
        return names.corsika_output_file_name(
            runNumber,
            self.primary,
            self.layoutName,
            self.site,
            self._userParameters["THETAP"][0],
            self._userParameters["AZM"][0],
            self.label,
        )

    def export_input_file(self):
        """Create and export CORSIKA input file."""
        self._set_output_file_and_directory()
        self._logger.debug("Exporting CORSIKA input file to {}".format(self._configFilePath))

        def _getTextSingleLine(pars):
            text = ""
            for par, values in pars.items():
                line = par + " "
                for v in values:
                    line += str(v) + " "
                line += "\n"
                text += line
            return text

        def _getTextMultipleLines(pars):
            text = ""
            for par, valueList in pars.items():
                for value in valueList:
                    newPars = {par: value}
                    text += _getTextSingleLine(newPars)
            return text

        with open(self._configFilePath, "w") as file:
            file.write("\n* [ RUN PARAMETERS ]\n")
            # Removing AZM entry first
            _userParsTemp = copy.copy(self._userParameters)
            _userParsTemp.pop("AZM")
            textParameters = _getTextSingleLine(_userParsTemp)
            file.write(textParameters)

            file.write("\n* [ SITE PARAMETERS ]\n")
            textSiteParameters = _getTextSingleLine(
                self._corsikaParameters["SITE_PARAMETERS"][self.site]
            )
            file.write(textSiteParameters)

            # Defining the IACT variables for the output file name
            file.write("\n")
            file.write("IACT setenv PRMNAME {}\n".format(self.primary))
            file.write("IACT setenv ZA {}\n".format(int(self._userParameters["THETAP"][0])))
            file.write("IACT setenv AZM {}\n".format(int(self._userParameters["AZM"][0])))

            file.write("\n* [ SEEDS ]\n")
            self._write_seeds(file)

            file.write("\n* [ TELESCOPES ]\n")
            telescopeListText = self.layout.get_corsika_input_list()
            file.write(telescopeListText)

            file.write("\n* [ INTERACTION FLAGS ]\n")
            textInteractionFlags = _getTextSingleLine(self._corsikaParameters["INTERACTION_FLAGS"])
            file.write(textInteractionFlags)

            file.write("\n* [ CHERENKOV EMISSION PARAMETERS ]\n")
            textCherenkov = _getTextSingleLine(
                self._corsikaParameters["CHERENKOV_EMISSION_PARAMETERS"]
            )
            file.write(textCherenkov)

            file.write("\n* [ DEBUGGING OUTPUT PARAMETERS ]\n")
            textDebugging = _getTextSingleLine(
                self._corsikaParameters["DEBUGGING_OUTPUT_PARAMETERS"]
            )
            file.write(textDebugging)

            file.write("\n* [ OUTUPUT FILE ]\n")
            file.write("TELFIL {}\n".format(self._outputGenericFileName))

            file.write("\n* [ IACT TUNING PARAMETERS ]\n")
            textIact = _getTextMultipleLines(self._corsikaParameters["IACT_TUNING_PARAMETERS"])
            file.write(textIact)

            file.write("\nEXIT")

        self._is_fileUpdated = True

    def _set_output_file_and_directory(self):
        configFileName = names.corsika_config_file_name(
            arrayName=self.layoutName,
            site=self.site,
            primary=self.primary,
            zenith=self._userParameters["THETAP"],
            viewCone=self._userParameters["VIEWCONE"],
            label=self.label,
        )
        fileDirectory = self.io_handler.get_output_directory(label=self.label, dirType="corsika")
        fileDirectory.mkdir(parents=True, exist_ok=True)
        self._logger.info("Creating directory {}, if needed.".format(fileDirectory))
        self._configFilePath = fileDirectory.joinpath(configFileName)

        self._outputGenericFileName = names.corsika_output_generic_file_name(
            arrayName=self.layoutName, site=self.site, label=self.label
        )

    def _write_seeds(self, file):
        """
        Generate and write seeds in the CORSIKA input file.

        Parameters
        ----------
        file: stream
            File where the telescope positions will be written.
        """
        randomSeed = self._userParameters["PRMPAR"][0] + self._userParameters["RUNNR"][0]
        rng = np.random.default_rng(randomSeed)
        corsikaSeeds = [int(rng.uniform(0, 1e7)) for i in range(4)]

        for s in corsikaSeeds:
            file.write("SEED {} 0 0\n".format(s))

    def get_input_file(self):
        """
        Get the full path of the CORSIKA input file.

        Returns
        -------
        Path:
            Full path of the CORSIKA input file.
        """
        if not self._is_fileUpdated:
            self.export_input_file()
        return self._configFilePath

    def get_input_file_name_for_run(self, runNumber):
        """
        Get the CORSIKA input file for one specific run.
        This is the input file after being pre-processed by sim_telarray (pfp).

        Parameters
        ----------
        runNumber: int
            Run number.

        Returns
        -------
        str:
            Name of the input file.
        """
        return names.corsika_config_tmp_file_name(
            arrayName=self.layoutName,
            site=self.site,
            primary=self.primary,
            zenith=self._userParameters["THETAP"],
            viewCone=self._userParameters["VIEWCONE"],
            run=runNumber,
            label=self.label,
        )
