import logging
import os
import re
from collections import defaultdict
from copy import copy
from pathlib import Path

import astropy.units as u
import numpy as np

import simtools.config as cfg
import simtools.io_handler as io
import simtools.util.general as gen
from simtools.corsika.corsika_runner import CorsikaRunner
from simtools.model.array_model import ArrayModel
from simtools.simtel.simtel_histograms import SimtelHistograms
from simtools.simtel.simtel_runner_array import SimtelRunnerArray
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ["Simulator"]


class MissingRequiredEntryInArrayConfig(Exception):
    pass


class Simulator:
    """
    Simulator is responsible for managing simulation of showers and array \
    of telescopes. It interfaces with simulation software-specific packages, \
    like CORSIKA and sim_telarray.

    The configuration is set as a dict configData or a yaml \
    file configFile. An example of configData can be found \
    below, for CORSIKA and sim_telarray, respectively.

    .. code-block:: python

    configData = {
        'dataDirectory': '(..)/data',
        'primary': 'gamma',
        'zenith': 20 * u.deg,
        'azimuth': 0 * u.deg,
        'viewcone': 0 * u.deg,
        # ArrayModel
        'site': 'North',
        'layoutName': '1LST',
        'modelVersion': 'Prod5',
        'default': {
            'LST': '1'
        },
        'M-01': 'FlashCam-D'
    }

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
    label: str
        Instance label.
    config: NamedTuple
        Configurable parameters.
    arrayModel: ArrayModel
        Instance of ArrayModel.
    site: str
        North or South.
    layoutName: str
        Name of the layout.

    Methods
    -------

    run(inputFileList):
        Run simulation.
    submit(inputFileList, submitCommand=None, extraCommands=None, test=False):
        Submit a run script as a job. The submit command can be given by submitCommand \
        or it will be taken from the config.yml file.
    printHistograms():
        Print histograms and save a pdf file.
    getRunScriptFile(runNumber)
        Get the full path of the run script file for a given run number.
    getRunLogFile(runNumber)
        Get the full path of the run log file.
    getCorsikaLogFile(runNumber)
        Get the full path of the CORSIKA log file.
    getCorsikaOutputFile(runNumber)
        Get the full path of the CORSIKA output file.
    getListOfOutputFiles():
        Get list of output files.
    getListOfInputFiles():
        Get list of input files.
    getListOfLogFiles():
        Get list of log files.
    printListOfOutputFiles():
        Print list of output files.
    printListOfInputFiles():
        Print list of output files.
    printListOfLogFiles():
        Print list of log files.
    """

    def __init__(
        self,
        label=None,
        filesLocation=None,
        simtelSourcePath=None,
        configData=None,
        configFile=None,
        SimulationSoftware=None,
    ):
        """
        ArraySimulator init.

        Parameters
        ----------
        label: str
            Instance label.
        filesLocation: str or Path.
            Location of the output files. If not given, it will be set from \
            the config.yml file.
        simtelSourcePath: str or Path
            Location of source of the sim_telarray/CORSIKA package.
        configData: dict
            Dict with array model configuration data.
        configFile: str or Path
            Path to yaml file containing configurable data.
        SimulationSoftware: str
            String
        """

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init ArraySimulator")

        self.label = label

        self._simtelSourcePath = Path(cfg.getConfigArg("simtelPath", simtelSourcePath))
        self._filesLocation = cfg.getConfigArg("outputLocation", filesLocation)

        # File location
        self._baseDirectory = io.getArraySimulatorOutputDirectory(self._filesLocation, self.label)
        self._outputDirectory.mkdir(parents=True, exist_ok=True)
        self._logger.debug(
            "Output directory {} - creating it, if needed.".format(self._outputDirectory)
        )

        configData = gen.collectDataFromYamlOrDict(configFile, configData)
        self._loadArrayConfigData(configData)
        self._setSimtelRunner()
        # Storing list of files
        self._results = defaultdict(list)
