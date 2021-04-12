#!/usr/bin/python3

import logging
import os
from pathlib import Path
from copy import copy

import simtools.config as cfg
import simtools.io_handler as io
from simtools.corsika.corsika_config import CorsikaConfig, MissingRequiredInputInCorsikaConfigData
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ['CorsikaRunner']


class MissingRequiredEntryInShowerConfig(Exception):
    pass


class CorsikaRunner:
    '''
    CorsikaRunner class.

    Methods
    -------
    setParameters(**kwargs)
    exportFile()
    getFile()
    '''

    def __init__(
        self,
        site,
        layoutName,
        label=None,
        filesLocation=None,
        simtelSourcePath=None,
        corsikaParametersFile=None,
        showerConfigData=None,
        showerConfigFile=None
    ):
        '''
        CorsikaRunner init.

        Parameters
        ----------
        site: str
            Paranal or LaPalma
        layoutName: str
            Name of the layout.
        layout: LayoutArray
            Instance of LayoutArray.
        label: str
            Instance label.
        filesLocation: str or Path.
            Main location of the output file.
        randomSeeds: bool
            If True, seeds will be set randomly. If False, seeds will be defined based on the run
            number.
        **kwargs
            Set of parameters for the corsika config.
        '''

        self._logger = logging.getLogger(__name__)
        self._logger.debug('Init CorsikaRunner')

        self.label = label
        self.site = names.validateSiteName(site)
        self.layoutName = names.validateLayoutArrayName(layoutName)

        self._simtelSourcePath = Path(cfg.getConfigArg('simtelPath', simtelSourcePath))
        self._filesLocation = cfg.getConfigArg('outputLocation', filesLocation)
        self._outputDirectory = io.getCorsikaOutputDirectory(self._filesLocation, self.label)
        if not self._outputDirectory.exists():
            self._outputDirectory.mkdir(parents=True, exist_ok=True)
            self._logger.debug('Creating directory {}'.format(self._outputDirectory))

        showerConfigData = collectDataFromYamlOrDict(showerConfigFile, showerConfigData)
        self._loadShowerConfigData(showerConfigData)

        self._loadCorsikaDataDirectories()

    def _loadShowerConfigData(self, showerConfigData):

        if 'corsikaDataDirectory' not in showerConfigData.keys():
            msg = 'corsikaDataDirectory not given in showerConfig'
            self._logger.error(msg)
            raise MissingRequiredEntryInShowerConfig(msg)
        else:
            self._corsikaDataDirectory = Path(showerConfigData['corsikaDataDirectory'])
            self._showerConfigData = copy(showerConfigData)
            self._showerConfigData.pop('corsikaDataDirectory')

        # Validating showerConfigData by using it to create a CorsikaConfig
        try:
            self.corsikaConfig = CorsikaConfig(
                site=self.site,
                label=self.label,
                layoutName=self.layoutName,
                corsikaConfigData=self._showerConfigData
            )
            # CORSIKA input file used as template for all runs
            self.corsikaInput = self.corsikaConfig.getInputFile()
        except MissingRequiredInputInCorsikaConfigData:
            msg = 'showerConfigData is missing required entries.'
            self._logger.error(msg)
            raise
    # End of _loadShowerConfigData

    def _loadCorsikaDataDirectories(self):
        corsikaBaseDir = self._corsikaDataDirectory.joinpath(self.site)
        corsikaBaseDir = corsikaBaseDir.joinpath(self.corsikaConfig.primary)
        corsikaBaseDir = corsikaBaseDir.absolute()

        self._corsikaDataDir = corsikaBaseDir.joinpath('data')
        self._corsikaDataDir.mkdir(parents=True, exist_ok=True)
        self._corsikaInputDir = corsikaBaseDir.joinpath('input')
        self._corsikaInputDir.mkdir(parents=True, exist_ok=True)
        self._corsikaLogDir = corsikaBaseDir.joinpath('log')
        self._corsikaLogDir.mkdir(parents=True, exist_ok=True)

    def getRunScriptFile(self, runNumber):
        # Setting script file name
        scriptFileName = names.corsikaRunScriptFileName(
            arrayName=self.layoutName,
            site=self.site,
            run=runNumber,
            label=self.label
        )
        scriptFileDir = self._outputDirectory.joinpath('scripts')
        scriptFileDir.mkdir(parents=True, exist_ok=True)
        scriptFilePath = scriptFileDir.joinpath(scriptFileName)

        # CORSIKA input file for a specific run, created by the preprocessor pfp
        corsikaInputTmpName = self.corsikaConfig.getInputTmpFileName(runNumber)
        corsikaInputTmpFile = self._corsikaInputDir.joinpath(corsikaInputTmpName)

        pfpCommand = self._getPfpCommand(runNumber, corsikaInputTmpFile)
        autoinputsCommand = self._getAutoinputsCommand(runNumber, corsikaInputTmpFile)

        with open(scriptFilePath, 'w') as file:
            file.write('export CORSIKA_DATA={}\n'.format(self._corsikaDataDir))
            file.write('# Creating CORSIKA_DATA\n')
            file.write('mkdir -p {}\n'.format(self._corsikaDataDir))
            file.write('\n')
            file.write('cd {} || exit 2\n'.format(self._corsikaDataDir))
            file.write('\n')
            file.write('# Running pfp\n')
            file.write(pfpCommand)
            file.write('\n')
            file.write('# Running corsika_autoinputs\n')
            file.write(autoinputsCommand)

        # Changing permissions
        os.system('chmod ug+x {}'.format(scriptFilePath))

        return scriptFilePath

    def _getPfpCommand(self, runNumber, inputTmpFile):
        cmd = self._simtelSourcePath.joinpath('sim_telarray/bin/pfp')
        cmd = str(cmd) + ' -V -DWITHOUT_MULTIPIPE - < {}'.format(self.corsikaInput)
        cmd += ' > {}\n'.format(inputTmpFile)
        return cmd

    def _getAutoinputsCommand(self, runNumber, inputTmpFile):
        corsikaBinPath = self._simtelSourcePath.joinpath('corsika-run/corsika')

        logFile = self.getRunLogFile(runNumber)

        cmd = self._simtelSourcePath.joinpath('sim_telarray/bin/corsika_autoinputs')
        cmd = str(cmd) + ' --run {}'.format(corsikaBinPath)
        cmd += ' -R {}'.format(runNumber)
        cmd += ' -p {}'.format(self._corsikaDataDir)
        cmd += ' {} > {} 2>&1'.format(inputTmpFile, logFile)
        cmd + ' || exit 3\n'
        return cmd

    def getRunLogFile(self, runNumber):
        logFileName = names.getCorsikaRunLogFileName(
            site=self.site,
            run=runNumber,
            arrayName=self.layoutName,
            label=self.label
        )
        return self._corsikaLogDir.joinpath(logFileName)
