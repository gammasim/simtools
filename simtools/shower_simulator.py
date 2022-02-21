import logging
import os
import numpy as np
from pathlib import Path
from copy import copy

import simtools.config as cfg
import simtools.io_handler as io
from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ["ShowerSimulator"]


class MissingRequiredEntryInShowerConfig(Exception):
    pass


class InvalidRunsToSimulate(Exception):
    pass


class ShowerSimulator:
    """
    ShowerSimulator is responsible for managing simulation of showers. \
    It interfaces with simulation software-specific packages, like CORSIKA.

    The configuration is set as a dict showerConfigData or a yaml \
    file showerConfigFile. An example of showerConfigData can be found \
    below.

    .. code-block:: python

    self.showerConfigData = {
        'dataDirectory': '.',
        'site': 'South',
        'layoutName': 'Prod5',
        'runRange': [1, 100],
        'nshow': 10,
        'primary': 'gamma',
        'erange': [100 * u.GeV, 1 * u.TeV],
        'eslope': -2,
        'zenith': 20 * u.deg,
        'azimuth': 0 * u.deg,
        'viewcone': 0 * u.deg,
        'cscat': [10, 1500 * u.m, 0]
    }


    Attributes
    ----------
    site: str
        North or South.
    layoutName: str
        Name of the layout.
    label: str
        Instance label.

    Methods
    -------
    getRunScriptFile(runNumber)
        Get the full path of the run script file for a given run number.
    getRunLogFile(runNumber)
        Get the full path of the run log file.
    getCorsikaLogFile(runNumber)
        Get the full path of the CORSIKA log file.
    getCorsikaOutputFile(runNumber)
        Get the full path of the CORSIKA output file.
    """

    def __init__(
        self,
        label=None,
        filesLocation=None,
        simtelSourcePath=None,
        showerConfigData=None,
        showerConfigFile=None,
    ):
        """
        ShowerSimulator init.

        Parameters
        ----------
        label: str
            Instance label.
        filesLocation: str or Path.
            Location of the output files. If not given, it will be set from \
            the config.yml file.
        simtelSourcePath: str or Path
            Location of source of the sim_telarray/CORSIKA package.
        showerConfigData: dict
            Dict with shower config data.
        showerConfigFile: str or Path
            Path to yaml file containing shower config data.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")

        self.label = label

        self._simtelSourcePath = Path(cfg.getConfigArg("simtelPath", simtelSourcePath))
        self._filesLocation = cfg.getConfigArg("outputLocation", filesLocation)
        self._outputDirectory = io.getCorsikaOutputDirectory(
            self._filesLocation, self.label
        )
        self._outputDirectory.mkdir(parents=True, exist_ok=True)
        self._logger.debug(
            "Output directory {} - creating it, if needed.".format(
                self._outputDirectory
            )
        )

        showerConfigData = collectDataFromYamlOrDict(showerConfigFile, showerConfigData)
        self._loadShowerConfigData(showerConfigData)
        self._setCorsikaRunner()

    # End of init

    def _loadShowerConfigData(self, showerConfigData):
        """Validate showerConfigData and store the relevant data in variables."""

        # Copying showerConfigData to corsikaConfigData
        # Few keys will be removed before passing it to CorsikaRunner
        self._corsikaConfigData = copy(showerConfigData)

        # Storing site and layoutName entries in attributes.
        try:
            self.site = names.validateSiteName(showerConfigData["site"])
            self.layoutName = names.validateLayoutArrayName(
                showerConfigData["layoutName"]
            )
            self._corsikaConfigData.pop("site")
            self._corsikaConfigData.pop("layoutName")
            dataDir = self._corsikaConfigData.pop("dataDirectory", None)
            self._corsikaConfigData["corsikaDataDirectory"] = dataDir
        except KeyError:
            msg = "site and/or layoutName were not given in showerConfig"
            self._logger.error(msg)
            raise MissingRequiredEntryInShowerConfig(msg)

        # Grabbing runList and runRange
        runList = showerConfigData.get("runList", None)
        runRange = showerConfigData.get("runRange", None)
        # Validating and merging runList and runRange, if needed.
        self.runs = self._validateRunListAndRange(runList, runRange)

        # Removing runs key from corsikaConfigData
        self._corsikaConfigData.pop("runList", None)
        self._corsikaConfigData.pop("runRange", None)

        # Searching for corsikaParametersFile in showerConfig
        self._corsikaParametersFile = showerConfigData.get(
            "corsikaParametersFile", None
        )
        self._corsikaConfigData.pop("corsikaParametersFile", None)

    def _setCorsikaRunner(self):
        """Creating a CorsikaRunner and setting it to self._corsikaRunner."""
        self._corsikaRunner = CorsikaRunner(
            site=self.site,
            layoutName=self.layoutName,
            label=self.label,
            filesLocation=self._filesLocation,
            simtelSourcePath=self._simtelSourcePath,
            corsikaParametersFile=self._corsikaParametersFile,
            corsikaConfigData=self._corsikaConfigData,
        )

    def run(self, runList=None, runRange=None):
        """
        Run simulation.

        Parameters
        ----------
        runList: list
            List of run numbers to be simulated.
        runRange: list
            List of len 2 with the limits of the range for runs to be simulated.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        """
        runsToSimulate = self._getRunsToSimulate(runList, runRange)
        self._logger.info("Running scripts for {} runs".format(len(runsToSimulate)))

        self._logger.info("Starting running scripts")
        for run in runsToSimulate:
            runScript = self._corsikaRunner.getRunScriptFile(runNumber=run)
            self._logger.info("Run {} - Running script {}".format(run, runScript))
            os.system(runScript)

    def submit(
        self, runList=None, runRange=None, submitCommand=None, extraCommands=None
    ):
        """
        Submit a run script as a job. The submit command can be given by \
        submitCommand or it will be taken from the config.yml file.

        Parameters
        ----------
        runList: list
            List of run numbers to be simulated.
        runRange: list
            List of len 2 with the limits of the range for runs to be simulated.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        """

        subCmd = (
            submitCommand if submitCommand is not None else cfg.get("submissionCommand")
        )
        self._logger.info("Submission command: {}".format(subCmd))

        runsToSimulate = self._getRunsToSimulate(runList, runRange)
        self._logger.info(
            "Submitting run scripts for {} runs".format(len(runsToSimulate))
        )

        self._logger.info("Starting submission")
        for run in runsToSimulate:
            runScript = self._corsikaRunner.getRunScriptFile(
                runNumber=run, extraCommands=extraCommands
            )
<<<<<<< HEAD

            thisSubCmd = copy(subCmd)

            # Checking for log files in sub command and replacing them
            if 'log_out' in subCmd:
                logOutFile = self._corsikaRunner.getSubLogFile(runNumber=run, mode='out')
                thisSubCmd = thisSubCmd.replace('log_out', str(logOutFile))

            if 'log_err' in subCmd:
                logErrFile = self._corsikaRunner.getSubLogFile(runNumber=run, mode='err')
                thisSubCmd = thisSubCmd.replace('log_err', str(logErrFile))

            self._logger.info('Run {} - Submitting script {}'.format(run, runScript))

            shellCommand = thisSubCmd + ' ' + str(runScript)
=======
            self._logger.info("Run {} - Submitting script {}".format(run, runScript))

            shellCommand = subCmd + " " + str(runScript)
>>>>>>> master
            self._logger.debug(shellCommand)
            os.system(shellCommand)

    def makeResourcesReport(self):
        runtime = list()
        nEvents = None
        for run in self.runs:
            if self._corsikaRunner.hasSubLogFile(runNumber=run):
                nEvents, thisRuntime = self._corsikaRunner.getResources(runNumber=run)
                runtime.append(thisRuntime)

        secToHour = 1 / (60 * 60)
        meanRuntime = np.mean(runtime) * secToHour

        resources = dict()
        resources['#events/run'] = nEvents
        resources['Runtime/run [hrs]'] = meanRuntime
        resources['Runtime/1000 events [hrs]'] = meanRuntime * 1000 / nEvents
        return resources

    def printResourcesReport(self):
        resources = self.makeResourcesReport()
        print('-----------------------------')
        print('Computing Resources Report - Showers')
        for key, value in resources.items():
            print('{} = {:.2f}'.format(key, value))
        print('-----------------------------')

    def _getRunsToSimulate(self, runList, runRange):
        """Process runList and runRange and return the validated list of runs."""
        if runList is None and runRange is None:
            if self.runs is None:
                msg = (
                    "Runs to simulate were not given as arguments nor "
                    + "in showerConfigData - aborting"
                )
                self._logger.error(msg)
                raise InvalidRunsToSimulate(msg)
            else:
                return self.runs
        else:
            return self._validateRunListAndRange(runList, runRange)

    def _validateRunListAndRange(self, runList, runRange):
        """
        Validate runList and runRange and return the list of runs. \
        If both arguments are given, they will be merged into a single list.
        """
        if runList is None and runRange is None:
            self._logger.debug("Nothing to validate - runList and runRange not given.")
            return None

        validatedRuns = list()
        if runList is not None:
            if not all(isinstance(r, int) for r in runList):
                msg = "runList must contain only integers."
                self._logger.error(msg)
                raise InvalidRunsToSimulate(msg)
            else:
                self._logger.debug("runList: {}".format(runList))
                validatedRuns = list(runList)

        if runRange is not None:
            if not all(isinstance(r, int) for r in runRange) or len(runRange) != 2:
                msg = "runRange must contain two integers only."
                self._logger.error(msg)
                raise InvalidRunsToSimulate(msg)
            else:
                runRange = np.arange(runRange[0], runRange[1] + 1)
                self._logger.debug("runRange: {}".format(runRange))
                validatedRuns.extend(list(runRange))

        validatedRunsUnique = set(validatedRuns)
        return list(validatedRunsUnique)

    def getListOfOutputFiles(self, runList=None, runRange=None):
        """
        Get list of output files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.

        Returns
        -------
        list
            List with the full path of all the output files.
        """
        self._logger.info("Getting list of output files")
        return self._getListOfFiles(runList=runList, runRange=runRange, which="output")

    def printListOfOutputFiles(self, runList=None, runRange=None):
        """
        Get list of output files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        """
        self._logger.info("Printing list of output files")
        self._printListOfFiles(runList=runList, runRange=runRange, which="output")

    def getListOfLogFiles(self, runList=None, runRange=None):
        """
        Get list of log files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.

        Returns
        -------
        list
            List with the full path of all the log files.
        """
        self._logger.info("Getting list of log files")
        return self._getListOfFiles(runList=runList, runRange=runRange, which="log")

    def printListOfLogFiles(self, runList=None, runRange=None):
        """
        Print list of log files.

        Parameters
        ----------
        runList: list
            List of run numbers.
        runRange: list
            List of len 2 with the limits of the range of the run numbers.

        Raises
        ------
        InvalidRunsToSimulate
            If runs in runList or runRange are invalid.
        """
        self._logger.info("Printing list of log files")
        self._printListOfFiles(runList=runList, runRange=runRange, which="log")

    def _getListOfFiles(self, which, runList, runRange):
        runsToList = self._getRunsToSimulate(runList=runList, runRange=runRange)

        outputFiles = list()
        for run in runsToList:
            if which == "output":
                file = self._corsikaRunner.getCorsikaOutputFile(runNumber=run)
            elif which == "log":
                file = self._corsikaRunner.getCorsikaLogFile(runNumber=run)
            else:
                self._logger.error("Invalid type of files - log or output")
                return None
            outputFiles.append(str(file))

        return outputFiles

    def _printListOfFiles(self, which, runList, runRange):
        files = self._getListOfFiles(runList=runList, runRange=runRange, which=which)
        for f in files:
            print(f)


# End of ShowerSimulator
