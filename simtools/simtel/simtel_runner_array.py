import logging
from pathlib import Path

import simtools.io_handler as io_handler
import simtools.util.general as gen
from simtools.simtel.simtel_runner import InvalidOutputFile, SimtelRunner
from simtools.util import names

__all__ = ["SimtelRunnerArray"]


class SimtelRunnerArray(SimtelRunner):
    """
    SimtelRunnerArray is the interface with sim_telarray to perform array simulations.

    Configurable parameters:
        simtelDataDirectory:
            len: 1
            default: null
            unit: null
        primary:
            len: 1
            unit: null
        zenithAngle:
            len: 1
            unit: deg
            default: 20 deg
        azimuthAngle:
            len: 1
            unit: deg
            default: 0 deg

    Attributes
    ----------
    label: str, optional
        Instance label.
    arrayModel: ArrayModel
        Instance of the ArrayModel class.
    config: namedtuple
        Contains the configurable parameters (zenithAngle).

    Methods
    -------
    getRunScript(self, test=False, inputFile=None, runNumber=None)
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.
    run(test=False, force=False, inputFile=None, runNumber=None)
        Run sim_telarray. test=True will make it faster and force=True will remove existing files
        and run again.
    """

    def __init__(
        self,
        arrayModel,
        label=None,
        simtelSourcePath=None,
        configData=None,
        configFile=None,
    ):
        """
        SimtelRunnerArray.

        Parameters
        ----------
        arrayModel: str
            Instance of TelescopeModel class.
        label: str, optional
            Instance label. Important for output file naming.
        simtelSourcePath: str (or Path), optional
            Location of sim_telarray installation.
        configData: dict.
            Dict containing the configurable parameters.
        configFile: str or Path
            Path of the yaml file containing the configurable parameters.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SimtelRunnerArray")

        super().__init__(label=label, simtelSourcePath=simtelSourcePath)

        self.arrayModel = self._validateArrayModel(arrayModel)
        self.label = label if label is not None else self.arrayModel.label

        self.io_handler = io_handler.IOHandler()

        self._baseDirectory = self.io_handler.getOutputDirectory(self.label, "array-simulator")

        # Loading configData
        _configDataIn = gen.collectDataFromYamlOrDict(configFile, configData)
        _parameterFile = self.io_handler.getInputDataFile(
            "parameters", "simtel-runner-array_parameters.yml"
        )
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_configDataIn, _parameters)

        self._loadSimtelDataDirectories()

    def _loadSimtelDataDirectories(self):
        """
        Create sim_telarray output directories for data, log and input.

        If simtelDataDirectory is not given as a configurable parameter,
        the standard directory of simtools output (simtools-output) will
        be used. A sub directory simtel-data will be created and sub directories for
        log and data will be created inside it.
        """

        if self.config.simtelDataDirectory is None:
            # Default config value
            simtelBaseDir = self._baseDirectory
        else:
            simtelBaseDir = Path(self.config.simtelDataDirectory)

        simtelBaseDir = simtelBaseDir.joinpath("simtel-data")
        simtelBaseDir = simtelBaseDir.joinpath(self.arrayModel.site)
        simtelBaseDir = simtelBaseDir.joinpath(self.config.primary)
        simtelBaseDir = simtelBaseDir.absolute()

        self._simtelDataDir = simtelBaseDir.joinpath("data")
        self._simtelDataDir.mkdir(parents=True, exist_ok=True)
        self._simtelLogDir = simtelBaseDir.joinpath("log")
        self._simtelLogDir.mkdir(parents=True, exist_ok=True)

    def getLogFile(self, runNumber):
        """Get full path of the simtel log file for a given run."""
        fileName = names.simtelLogFileName(
            run=runNumber,
            primary=self.config.primary,
            arrayName=self.arrayModel.layoutName,
            site=self.arrayModel.site,
            zenith=self.config.zenithAngle,
            azimuth=self.config.azimuthAngle,
            label=self.label,
        )
        return self._simtelLogDir.joinpath(fileName)

    def getSubLogFile(self, runNumber, mode="out"):
        """
        Get the full path of the submission log file.

        Parameters
        ----------
        runNumber: int
            Run number.
        mode: str
            out or err

        Raises
        ------
        ValueError
            If runNumber is not valid (not an unsigned int).

        Returns
        -------
        Path:
            Full path of the run log file.
        """

        fileName = names.simtelSubLogFileName(
            run=runNumber,
            primary=self.config.primary,
            arrayName=self.arrayModel.layoutName,
            site=self.arrayModel.site,
            zenith=self.config.zenithAngle,
            azimuth=self.config.azimuthAngle,
            label=self.label,
            mode=mode,
        )
        return self._simtelLogDir.joinpath(fileName)

    def getHistogramFile(self, runNumber):
        """Get full path of the simtel histogram file for a given run."""
        fileName = names.simtelHistogramFileName(
            run=runNumber,
            primary=self.config.primary,
            arrayName=self.arrayModel.layoutName,
            site=self.arrayModel.site,
            zenith=self.config.zenithAngle,
            azimuth=self.config.azimuthAngle,
            label=self.label,
        )
        return self._simtelDataDir.joinpath(fileName)

    def getOutputFile(self, runNumber):
        """Get full path of the simtel output file for a given run."""
        fileName = names.simtelOutputFileName(
            run=runNumber,
            primary=self.config.primary,
            arrayName=self.arrayModel.layoutName,
            site=self.arrayModel.site,
            zenith=self.config.zenithAngle,
            azimuth=self.config.azimuthAngle,
            label=self.label,
        )
        return self._simtelDataDir.joinpath(fileName)

    def hasSubLogFile(self, runNumber, mode="out"):
        """
        Checks that the sub run log file for this run number
        is a valid file on disk

        Parameters
        ----------
        runNumber: int
            Run number.

        """

        runSubFile = self.getSubLogFile(runNumber=runNumber, mode=mode)
        return Path(runSubFile).is_file()

    def getResources(self, runNumber):
        """
        Reading run time from last line of submission log file.

        Parameters
        ----------
        runNumber: int
            Run number.

        Returns
        -------
        dict
            run time of job in seconds

        """

        subLogFile = self.getSubLogFile(runNumber=runNumber, mode="out")

        self._logger.debug("Reading resources from {}".format(subLogFile))

        _resources = {}

        _resources["runtime"] = None
        with open(subLogFile, "r") as file:
            for line in reversed(list(file)):
                if "RUNTIME" in line:
                    _resources["runtime"] = int(line.split()[1])
                    break

        if _resources["runtime"] is None:
            self._logger.debug("RUNTIME was not found in run log file")

        return _resources

    def _shallRun(self, runNumber=None):
        """Tells if simulations should be run again based on the existence of output files."""
        return not self.getOutputFile(runNumber).exists()

    def _makeRunCommand(self, inputFile, runNumber=1):
        """
        Builds and returns the command to run simtel_array.

        Attributes
        ----------
        inputFile: str
            Full path of the input CORSIKA file
        runNumber: int
            run number

        """

        self._logFile = self.getLogFile(runNumber)
        histogramFile = self.getHistogramFile(runNumber)
        outputFile = self.getOutputFile(runNumber)

        # Array
        command = str(self._simtelSourcePath.joinpath("sim_telarray/bin/sim_telarray"))
        command += " -c {}".format(self.arrayModel.getConfigFile())
        command += " -I{}".format(self.arrayModel.getConfigDirectory())
        command += super()._configOption("telescope_theta", self.config.zenithAngle)
        command += super()._configOption("telescope_phi", self.config.azimuthAngle)
        command += super()._configOption("power_law", "2.5")
        command += super()._configOption("histogram_file", histogramFile)
        command += super()._configOption("output_file", outputFile)
        command += super()._configOption("random_state", "auto")
        command += super()._configOption("show", "all")
        command += " " + str(inputFile)
        command += " > " + str(self._logFile) + " 2>&1"

        return command

    # END of makeRunCommand

    def _checkRunResult(self, runNumber):
        # Checking run
        if not self.getOutputFile(runNumber).exists():
            msg = "sim_telarray output file does not exist."
            self._logger.error(msg)
            raise InvalidOutputFile(msg)
        else:
            self._logger.debug("Everything looks fine with the sim_telarray output file.")
