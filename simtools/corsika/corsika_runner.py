#!/usr/bin/python3

import logging
import random
from pathlib import Path
from copy import copy

from astropy.io.misc import yaml

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
        except MissingRequiredInputInCorsikaConfigData:
            msg = 'showerConfigData is missing required entries.'
            self._logger.error(msg)
            raise
    # End of _loadShowerConfigData

    def getRunScriptFile(self, run):
        # Setting script file name
        scriptFileName = names.corsikaRunScriptFileName(
            arrayName=self.layoutName,
            site=self.site,
            run=run,
            label=self.label
        )
        scriptFilePath = self._outputDirectory.joinpath(scriptFileName)

        self._loadCorsikaDataDirectories()

        # Exporting corsika input file
        self.corsikaConfig.exportInputFile()

        with open(scriptFilePath, 'w') as file:
            file.write('export CORSIKA_DATA={}'.format(self._corsikaDataDir))

        return scriptFilePath

    def _loadCorsikaDataDirectories(self):
        if '_corsikaDataDir' in self.__dict__:
            return

        corsikaBaseDir = self._corsikaDataDirectory.joinpath(self.site)
        corsikaBaseDir = corsikaBaseDir.joinpath(self.corsikaConfig.primary)
        corsikaBaseDir = corsikaBaseDir.absolute()

        self._corsikaDataDir = corsikaBaseDir.joinpath('data')
        self._corsikaInputDir = corsikaBaseDir.joinpath('input')
        self._corsikaLogDir = corsikaBaseDir.joinpath('log')
