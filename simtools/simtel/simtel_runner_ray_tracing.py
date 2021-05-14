import logging
import os
from pathlib import Path

import astropy.units as u

import simtools.io_handler as io
import simtools.config as cfg
import simtools.util.general as gen
from simtools.util import names
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_runner import SimtelRunner

__all__ = ['SimtelRunnerRayTracing']


class SimtelRunnerRayTracing(SimtelRunner):
    '''
    SimtelRunner is the interface with sim_telarray.

    Configurable parameters:
        zenithAngle:
            len: 1
            unit: deg
            default: 20 deg
        offAxisAngle:
            len: 1
            unit: deg
            default: 0 deg
        sourceDistance:
            len: 1
            unit: km
            default: 10 km
        useRandomFocalLength:
            len: 1
            default: False
        mirrorNumber:
            len: 1
            default: 1

    Attributes
    ----------
    mode: str
        RayTracing, Trigger, etc.
    telescopeModel: TelescopeModel
        Instance of the TelescopeModel class.
    label: str, optional
        Instance label.
    config: namedtuple
        Contains the configurable parameters (zenithAngle).

    Methods
    -------
    run(test=False, force=False)
        Run sim_telarray. test=True will make it faster and force=True will remove existing files
        and run again.
    '''

    def __init__(
        self,
        telescopeModel,
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
        configData=None,
        configFile=None
    ):
        '''
        SimtelRunner.

        Parameters
        ----------
        mode: str
            RayTracing, Trigger, ...
        telescopeModel: str
            Instance of TelescopeModel class.
        label: str, optional
            Instance label. Important for output file naming.
        simtelSourcePath: str (or Path), optional
            Location of sim_telarray installation. If not given, it will be taken from the
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
        configData: dict.
            Dict containing the configurable parameters.
        configFile: str or Path
            Path of the yaml file containing the configurable parameters.
        '''
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init SimtelRunnerRayTracing')

        super()._init_(label=label, simtelSourcePath=simtelSourcePath, filesLocation=filesLocation)

        self.telescopeModel = self._validateTelescopeModel(telescopeModel)

        # File location
        self._baseDirectory = io.getOutputDirectory(
            self._filesLocation,
            self.label,
            'ray-tracing'
        )
        self._baseDirectory.mkdir(parents=True, exist_ok=True)

        # RayTracing - default parameters
        self._repNumber = 0
        self.RUNS_PER_SET = 1 if self._isSingleMirrorMode() else 20  # const
        self.PHOTONS_PER_RUN = 10000  # const

        # Loading configData
        _configDataIn = gen.collectDataFromYamlOrDict(configFile, configData)
        _parameterFile = io.getDataFile('parameters', 'simtel-runner-ray-tracing_parameters.yml')
        _parameters = gen.collectDataFromYamlOrDict(_parameterFile, None)
        self.config = gen.validateConfigData(_configDataIn, _parameters)
