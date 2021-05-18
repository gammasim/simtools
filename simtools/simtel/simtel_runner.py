import logging
import os
from pathlib import Path

import astropy.units as u

import simtools.io_handler as io
import simtools.config as cfg
import simtools.util.general as gen
from simtools.util import names
from simtools.model.telescope_model import TelescopeModel
from simtools.model.array_model import ArrayModel

__all__ = ['SimtelRunner']


class SimtelExecutionError(Exception):
    pass


class SimtelRunner:
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
        label=None,
        simtelSourcePath=None,
        filesLocation=None,
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

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self.label = label

        # File location
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)

        self.RUNS_PER_SET = 1

    def __repr__(self):
        return 'SimtelRunner(label={})\n'.format(self.label)

    def _validateTelescopeModel(self, tel):
        ''' Validate TelescopeModel '''
        if isinstance(tel, TelescopeModel):
            self._logger.debug('TelescopeModel is valid')
            return tel
        else:
            msg = 'Invalid TelescopeModel'
            self._logger.error(msg)
            raise ValueError(msg)

    def _validateArrayModel(self, array):
        ''' Validate TelescopeModel '''
        if isinstance(array, ArrayModel):
            self._logger.debug('ArrayModel is valid')
            return array
        else:
            msg = 'Invalid ArrayModel'
            self._logger.error(msg)
            raise ValueError(msg)

    def run(self, test=False, force=False):
        '''
        Run sim_telarray.

        Parameters
        ----------
        test: bool
            If True, make simulations faster.
        force: bool
            If True, remove possible existing output files and run again.
        '''
        self._logger.debug('Running sim_telarray')
        # write all the important parameters

        if not self._shallRun() and not force:
            self._logger.debug('Skipping because output exists and force = False')
            return

        command = self._makeRunCommand()

        if test:
            self._logger.info('Running (test) with command:{}'.format(command))
            sysOutput = os.system(command)
        else:
            self._logger.info('Running ({}x) with command:{}'.format(self.RUNS_PER_SET, command))
            sysOutput = os.system(command)
            for _ in range(self.RUNS_PER_SET - 1):
                os.system(command)

        # if self._simtelFailed(sysOutput):
        #     self._raiseSimtelError()

        self._checkRunResult()

    def _simtelFailed(self, sysOutput):
        return sysOutput != '0'

    def _raiseSimtelError(self):
        if hasattr(self, '_logFile'):
            logLines = gen.collectFinalLines(self._logFile, 10)
            msg = (
                'Simtel Error - See below the relevant part of the simtel log file.\n'
                + '===== from simtel log file ======\n'
                + logLines
                + '================================='
            )
        else:
            msg = 'Simtel log file does not exist'

        self._logger.error(msg)
        raise SimtelExecutionError()

    def _shallRun(self):
        self._logger.debug(
            'shallRun is being called from the base class - returning False -'
            + 'it should be implemented in the sub class'
        )
        return False
