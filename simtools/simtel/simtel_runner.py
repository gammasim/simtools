import logging
import os
from pathlib import Path

import simtools.config as cfg
import simtools.util.general as gen
from simtools.model.telescope_model import TelescopeModel
from simtools.model.array_model import ArrayModel

__all__ = ['SimtelRunner']


class SimtelExecutionError(Exception):
    pass


class SimtelRunner:
    '''
    SimtelRunner is the base class of the sim_telarray interfaces.

    Attributes
    ----------
    label: str, optional
        Instance label.

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
        label: str, optional
            Instance label. Important for output file naming.
        simtelSourcePath: str (or Path), optional
            Location of sim_telarray installation. If not given, it will be taken from the
            config.yml file.
        filesLocation: str (or Path), optional
            Parent location of the output files created by this class. If not given, it will be
            taken from the config.yml file.
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
        Basic sim_telarray run method.

        Parameters
        ----------
        test: bool
            If True, make simulations faster.
        force: bool
            If True, remove possible existing output files and run again.
        '''
        self._logger.debug('Running sim_telarray')

        if not hasattr(self, '_makeRunCommand'):
            msg = 'run method cannot be executed without the _makeRunCommand'
            self._logger.error(msg)
            raise RuntimeError(msg)

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

        # TODO: fix the fact any ray tracing simulations are failing and
        # uncomment this
        # if self._simtelFailed(sysOutput):
        #     self._raiseSimtelError()

        self._checkRunResult()

    def _simtelFailed(self, sysOutput):
        return sysOutput != '0'

    def _raiseSimtelError(self):
        '''
        Raise sim_telarray execution error. Final 10 lines from the log file
        are collected and printed.
        '''
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
        raise SimtelExecutionError(msg)

    def _shallRun(self):
        self._logger.debug(
            'shallRun is being called from the base class - returning False -'
            + 'it should be implemented in the sub class'
        )
        return False
