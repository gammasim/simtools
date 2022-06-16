import logging
import os
from pathlib import Path
from copy import copy

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen
from simtools.corsika.corsika_config import (
    CorsikaConfig,
    MissingRequiredInputInCorsikaConfigData,
)
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ["CorsikaRunner"]


class MissingRequiredEntryInCorsikaConfig(Exception):
    pass


class CorsikaRunner:
    """
    CorsikaRunner is responsible for running the CORSIKA, through the
    corsika_autoinputs program provided by the sim_telarray package. \
    It provides shell scripts to be run externally or by \
    the module shower_simulator. Same instance can be used to \
    generate scripts for any given run number.

    It uses CorsikaConfig to manage the CORSIKA configuration. \
    User parameters must be given by the corsikaConfigData or \
    corsikaConfigFile arguments. An example of corsikaConfigData follows \
    below.

    .. code-block:: python

    corsikaConfigData = {
        'corsikaDataDirectory': .
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

    The CORSIKA output directory must be set by the corsikaDataDirectory entry. \
    The following directories will be created to store the output data, logs and input \
    file:

    {corsikaDataDirectory}/$site/$primary/data
    {corsikaDataDirectory}/$site/$primary/log
    {corsikaDataDirectory}/$site/$primary/inputs

    Attributes
    ----------
    site: str
        North or South.
    layoutName: str
        Name of the layout.
    label: str
        Instance label.
    corsikaConfig: CorsikaConfig
        CorsikaConfig object.

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
        site,
        layoutName,
        label=None,
        keepSeeds=False,
        filesLocation=None,
        simtelSourcePath=None,
        corsikaParametersFile=None,
        corsikaConfigData=None,
        corsikaConfigFile=None,
    ):
        """
        CorsikaRunner init.

        Parameters
        ----------
        site: str
            South or North.
        layoutName: str
            Name of the layout.
        label: str
            Instance label.
        keepSeeds: bool
            If True, seeds generated by CorsikaConfig, based on the \
            run number and the primary particle will be used. \
            If False, random seeds will be defined automatically by sim_telarray.
        filesLocation: str or Path.
            Location of the output files. If not given, it will be set from \
            the config.yml file.
        simtelSourcePath: str or Path
            Location of source of the sim_telarray/CORSIKA package.
        corsikaConfigData: dict
            Dict with CORSIKA config data.
        corsikaConfigFile: str or Path
            Path to yaml file containing CORSIKA config data.
        corsikaParametersFile: str or Path
            Path to yaml file containing CORSIKA parameters.
        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaRunner")

        self.label = label
        self.site = names.validateSiteName(site)
        self.layoutName = names.validateLayoutArrayName(layoutName)

        self._keepSeeds = keepSeeds

        self._simtelSourcePath = Path(cfg.getConfigArg("simtelPath", simtelSourcePath))
        self._filesLocation = cfg.getConfigArg("outputLocation", filesLocation)
        self._outputDirectory = io.getCorsikaOutputDirectory(
            self._filesLocation, self.label
        )
        self._outputDirectory.mkdir(parents=True, exist_ok=True)
        self._logger.debug(
            "Creating output dir {}, if needed,".format(self._outputDirectory)
        )

        corsikaConfigData = collectDataFromYamlOrDict(
            corsikaConfigFile, corsikaConfigData
        )
        self._loadCorsikaConfigData(corsikaConfigData)

        self._loadCorsikaDataDirectories()

    def _loadCorsikaConfigData(self, corsikaConfigData):
        """Reads corsikaConfigData, creates corsikaConfig and corsikaInputFile."""

        corsikaDataDirectoryFromConfig = corsikaConfigData.get(
            "corsikaDataDirectory", None
        )
        if corsikaDataDirectoryFromConfig is None:
            # corsikaDataDirectory not given (or None).
            msg = (
                "corsikaDataDirectory not given in corsikaConfig "
                "- default output directory will be set."
            )
            self._logger.warning(msg)
            self._corsikaDataDirectory = self._outputDirectory
        else:
            # corsikaDataDirectory given and not None.
            self._corsikaDataDirectory = Path(corsikaDataDirectoryFromConfig)

        self._corsikaDataDirectory = self._corsikaDataDirectory.joinpath("corsika-data")

        # Copying corsikaConfigData and removing corsikaDataDirectory
        # (it does not go to CorsikaConfig)
        self._corsikaConfigData = copy(corsikaConfigData)
        self._corsikaConfigData.pop("corsikaDataDirectory", None)

        # Creating corsikaConfig - this will also validate the input given
        # in corsikaConfigData
        try:
            self.corsikaConfig = CorsikaConfig(
                site=self.site,
                label=self.label,
                layoutName=self.layoutName,
                corsikaConfigData=self._corsikaConfigData,
            )
            # CORSIKA input file used as template for all runs
            self._corsikaInputFile = self.corsikaConfig.getInputFile()
        except MissingRequiredInputInCorsikaConfigData:
            msg = "corsikaConfigData is missing required entries."
            self._logger.error(msg)
            raise

    # End of _loadcorsikaConfigData

    def _loadCorsikaDataDirectories(self):
        """Create CORSIKA directories for data, log and input."""
        corsikaBaseDir = self._corsikaDataDirectory.joinpath(self.site)
        corsikaBaseDir = corsikaBaseDir.joinpath(self.corsikaConfig.primary)
        corsikaBaseDir = corsikaBaseDir.absolute()

        self._corsikaDataDir = corsikaBaseDir.joinpath("data")
        self._corsikaDataDir.mkdir(parents=True, exist_ok=True)
        self._corsikaInputDir = corsikaBaseDir.joinpath("input")
        self._corsikaInputDir.mkdir(parents=True, exist_ok=True)
        self._corsikaLogDir = corsikaBaseDir.joinpath("log")
        self._corsikaLogDir.mkdir(parents=True, exist_ok=True)

    def getRunScriptFile(self, runNumber=None, extraCommands=None):
        """
        Get the full path of the run script file for a given run number.

        Parameters
        ----------
        runNumber: int
            Run number.

        Returns
        -------
        Path:
            Full path of the run script file.
        """
        runNumber = self._validateRunNumber(runNumber)

        # Setting script file name
        scriptFileName = names.corsikaRunScriptFileName(
            arrayName=self.layoutName,
            site=self.site,
            primary=self.corsikaConfig.primary,
            run=runNumber,
            label=self.label,
        )
        scriptFileDir = self._outputDirectory.joinpath("scripts")
        scriptFileDir.mkdir(parents=True, exist_ok=True)
        scriptFilePath = scriptFileDir.joinpath(scriptFileName)

        # CORSIKA input file for a specific run, created by the preprocessor pfp
        corsikaInputTmpName = self.corsikaConfig.getInputFileNameForRun(runNumber)
        corsikaInputTmpFile = self._corsikaInputDir.joinpath(corsikaInputTmpName)

        pfpCommand = self._getPfpCommand(runNumber, corsikaInputTmpFile)
        autoinputsCommand = self._getAutoinputsCommand(runNumber, corsikaInputTmpFile)

        extraCommands = self._getExtraCommands(extraCommands)
        self._logger.debug(
            "Extra commands to be added to the run script {}".format(extraCommands)
        )

        with open(scriptFilePath, "w") as file:
            if extraCommands is not None:
                file.write("# Writing extras\n")
                for line in extraCommands:
                    file.write("{}\n".format(line))
                file.write("# End of extras\n\n")

            file.write("export CORSIKA_DATA={}\n".format(self._corsikaDataDir))
            file.write("# Creating CORSIKA_DATA\n")
            file.write("mkdir -p {}\n".format(self._corsikaDataDir))
            file.write("\n")
            file.write("cd {} || exit 2\n".format(self._corsikaDataDir))
            file.write("\n")
            file.write("# Running pfp\n")
            file.write(pfpCommand)
            file.write("\n")
            file.write("# Running corsika_autoinputs\n")
            file.write(autoinputsCommand)

        # Changing permissions
        os.system("chmod ug+x {}".format(scriptFilePath))

        return scriptFilePath

    # End of getRunScriptFile

    def _getPfpCommand(self, runNumber, inputTmpFile):
        """Get pfp pre-processor command."""
        cmd = self._simtelSourcePath.joinpath("sim_telarray/bin/pfp")
        cmd = str(cmd) + " -V -DWITHOUT_MULTIPIPE - < {}".format(self._corsikaInputFile)
        cmd += " > {}\n".format(inputTmpFile)
        return cmd

    def _getAutoinputsCommand(self, runNumber, inputTmpFile):
        """Get autoinputs command."""
        corsikaBinPath = self._simtelSourcePath.joinpath("corsika-run/corsika")

        logFile = self.getRunLogFile(runNumber)

        cmd = self._simtelSourcePath.joinpath("sim_telarray/bin/corsika_autoinputs")
        cmd = str(cmd) + " --run {}".format(corsikaBinPath)
        cmd += " -R {}".format(runNumber)
        cmd += " -p {}".format(self._corsikaDataDir)
        if self._keepSeeds:
            cmd += " --keep-seeds"
        cmd += " {} > {} 2>&1".format(inputTmpFile, logFile)
        cmd + " || exit 3\n"
        return cmd

    @staticmethod
    def _getExtraCommands(extra):
        """
        Get extra commands by combining the one given as argument and
        what is given in config.yml
        """
        extra = gen.copyAsList(extra) if extra is not None else list()

        extraFromConfig = cfg.get("extraCommands")
        extraFromConfig = (
            gen.copyAsList(extraFromConfig) if extraFromConfig is not None else list()
        )

        extra.extend(extraFromConfig)
        return extra

    def getRunLogFile(self, runNumber=None):
        """
        Get the full path of the run log file.

        Parameters
        ----------
        runNumber: int
            Run number.

        Raises
        ------
        ValueError
            If runNumber is not valid (not an unsigned int).

        Returns
        -------
        Path:
            Full path of the run log file.
        """
        runNumber = self._validateRunNumber(runNumber)
        logFileName = names.corsikaRunLogFileName(
            site=self.site, run=runNumber, arrayName=self.layoutName, label=self.label
        )
        return self._corsikaLogDir.joinpath(logFileName)

    def getCorsikaLogFile(self, runNumber=None):
        """
        Get the full path of the CORSIKA log file.

        Parameters
        ----------
        runNumber: int
            Run number.

        Raises
        ------
        ValueError
            If runNumber is not valid (not an unsigned int).

        Returns
        -------
        Path:
            Full path of the CORSIKA log file.
        """
        runNumber = self._validateRunNumber(runNumber)
        runDir = self._getRunDirectory(runNumber)
        return self._corsikaDataDir.joinpath(runDir).joinpath(
            "run{}.log".format(runNumber)
        )

    def getCorsikaOutputFile(self, runNumber=None):
        """
        Get the full path of the CORSIKA output file.

        Parameters
        ----------
        runNumber: int
            Run number.

        Raises
        ------
        ValueError
            If runNumber is not valid (not an unsigned int).

        Returns
        -------
        Path:
            Full path of the CORSIKA output file.
        """
        runNumber = self._validateRunNumber(runNumber)
        corsikaFileName = self.corsikaConfig.getOutputFileName(runNumber)
        runDir = self._getRunDirectory(runNumber)
        return self._corsikaDataDir.joinpath(runDir).joinpath(corsikaFileName)

    @staticmethod
    def _getRunDirectory(runNumber):
        """Get run directory created by sim_telarray (ex. run000014)."""
        nn = str(runNumber)
        return "run" + nn.zfill(6)

    def _validateRunNumber(self, runNumber):
        """
        Returns the run number from corsikaConfig in case runNumber is None,
        Raise ValueError if runNumber is not valid (< 1) or returns runNumber if
        it is a valid value.
        """
        if runNumber is None:
            return self.corsikaConfig.getUserParameter("RUNNR")
        elif not float(runNumber).is_integer() or runNumber < 1:
            msg = "Invalid type of run number ({}) - it must be an uint.".format(
                runNumber
            )
            self._logger.error(msg)
            raise ValueError(msg)
        else:
            return runNumber
