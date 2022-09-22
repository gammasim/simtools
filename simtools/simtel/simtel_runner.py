import logging
import os
from pathlib import Path

import simtools.config as cfg
import simtools.util.general as gen
from simtools.model.array_model import ArrayModel
from simtools.model.telescope_model import TelescopeModel

__all__ = ["SimtelRunner"]


class SimtelExecutionError(Exception):
    pass


class InvalidOutputFile(Exception):
    pass


class SimtelRunner:
    """
    SimtelRunner is the base class of the sim_telarray interfaces.

    Attributes
    ----------
    label: str, optional
        Instance label.

    Methods
    -------
    getRunScript(self, test=False, inputFile=None, runNumber=None)
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.
    run(test=False, force=False, input=None)
        Run sim_telarray. test=True will make it faster and force=True will remove existing files
        and run again.
    """

    def __init__(
        self,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
    ):
        """
        SimtelRunner.

        Parameters
        ----------
        label: str, optional
            Instance label. Important for output file naming.
        simtelSourcePath: str (or Path), optional
            Location of sim_telarray installation. If not given, it will be taken from the
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        """
        self._logger = logging.getLogger(__name__)

        self._simtelSourcePath = Path(cfg.getConfigArg("simtelPath", simtelSourcePath))
        self.label = label

        # File location
        self._filesLocation = cfg.getConfigArg("outputLocation", filesLocation)

        self.RUNS_PER_SET = 1

    def __repr__(self):
        return "SimtelRunner(label={})\n".format(self.label)

    def _validateTelescopeModel(self, tel):
        """Validate TelescopeModel"""
        if isinstance(tel, TelescopeModel):
            self._logger.debug("TelescopeModel is valid")
            return tel
        else:
            msg = "Invalid TelescopeModel"
            self._logger.error(msg)
            raise ValueError(msg)

    def _validateArrayModel(self, array):
        """Validate ArrayModel"""
        if isinstance(array, ArrayModel):
            self._logger.debug("ArrayModel is valid")
            return array
        else:
            msg = "Invalid ArrayModel"
            self._logger.error(msg)
            raise ValueError(msg)

    def getRunScript(self, test=False, inputFile=None, runNumber=None, extraCommands=None):
        """
        Builds and returns the full path of the bash run script containing
        the sim_telarray command.

        Parameters
        ----------
        test: bool
            Test flag for faster execution.
        inputFile: str or Path
            Full path of the input CORSIKA file.
        runNumber: int
            Run number.
        extraCommands: str
            Additional commands for running simulations given in config.yml

        Returns
        -------
        Path
            Full path of the run script.
        """
        self._logger.debug("Creating run bash script")

        self._scriptDir = self._baseDirectory.joinpath("scripts")
        self._scriptDir.mkdir(parents=True, exist_ok=True)
        self._scriptFile = self._scriptDir.joinpath(
            "run{}-simtel".format(runNumber if runNumber is not None else "")
        )
        self._logger.debug("Run bash script - {}".format(self._scriptFile))

        extraCommands = self._getExtraCommands(extraCommands)
        self._logger.debug("Extra commands to be added to the run script {}".format(extraCommands))

        command = self._makeRunCommand(inputFile=inputFile, runNumber=runNumber)
        with self._scriptFile.open("w") as file:
            # TODO: header
            file.write("#!/usr/bin/bash\n\n")

            # Setting SECONDS variable to measure runtime
            file.write("\nSECONDS=0\n")

            if extraCommands is not None:
                file.write("# Writing extras\n")
                for line in extraCommands:
                    file.write("{}\n".format(line))
                file.write("# End of extras\n\n")

            N = 1 if test else self.RUNS_PER_SET
            for _ in range(N):
                file.write("{}\n\n".format(command))

            # Printing out runtime
            file.write('\necho "RUNTIME: $SECONDS"\n')

        os.system("chmod ug+x {}".format(self._scriptFile))
        return self._scriptFile

    def run(self, test=False, force=False, inputFile=None, runNumber=None):
        """
        Basic sim_telarray run method.

        Parameters
        ----------
        test: bool
            If True, make simulations faster.
        force: bool
            If True, remove possible existing output files and run again.
        """
        self._logger.debug("Running sim_telarray")

        if not hasattr(self, "_makeRunCommand"):
            msg = "run method cannot be executed without the _makeRunCommand method"
            self._logger.error(msg)
            raise RuntimeError(msg)

        if not self._shallRun(runNumber) and not force:
            self._logger.info("Skipping because output exists and force = False")
            return

        command = self._makeRunCommand(inputFile=inputFile, runNumber=runNumber)

        if test:
            self._logger.info("Running (test) with command:{}".format(command))
            os.system(command)
        else:
            self._logger.info("Running ({}x) with command:{}".format(self.RUNS_PER_SET, command))
            os.system(command)

            for _ in range(self.RUNS_PER_SET - 1):
                os.system(command)

        # TODO: fix the fact any ray tracing simulations are failing and
        # uncomment this
        # if self._simtelFailed(sysOutput):
        #     self._raiseSimtelError()

        self._checkRunResult(runNumber=runNumber)

    @staticmethod
    def _simtelFailed(sysOutput):
        return sysOutput != "0"

    def _raiseSimtelError(self):
        """
        Raise sim_telarray execution error. Final 10 lines from the log file
        are collected and printed.
        """
        if hasattr(self, "_logFile"):
            logLines = gen.collectFinalLines(self._logFile, 10)
            msg = (
                "Simtel Error - See below the relevant part of the simtel log file.\n"
                + "===== from simtel log file ======\n"
                + logLines
                + "================================="
            )
        else:
            msg = "Simtel log file does not exist."

        self._logger.error(msg)
        raise SimtelExecutionError(msg)

    def _shallRun(self, runNumber=None):
        self._logger.debug(
            "shallRun is being called from the base class - returning False -"
            + "it should be implemented in the sub class"
        )
        return False

    @staticmethod
    def _getExtraCommands(extra):
        """
        Get extra commands by combining the one given as argument and
        what is given in config.yml
        """
        extra = gen.copyAsList(extra) if extra is not None else list()

        extraFromConfig = cfg.get("extraCommands")
        extraFromConfig = gen.copyAsList(extraFromConfig) if extraFromConfig is not None else list()

        extra.extend(extraFromConfig)
        return extra

    @staticmethod
    def _configOption(par, value=None):
        """Util function for building sim_telarray command."""
        c = " -C {}".format(par)
        c += "={}".format(value) if value is not None else ""
        return c
