#!/usr/bin/python3

""" This module contains the TelescopeModel class

Todo:
    * fromConfigFile - deal with ifdef's in simtel cfg files
"""

import logging
import yaml
from pathlib import Path

from ctamclib.util import names
from ctamclib.model_parameters import MODEL_PARS

__all__ = ['TelescopeModel']


def whichTelescopeSize(telescopeType):
    """ Return the telescope size (SST, MST or LST) for a given telescopeType.

    Args:
        telescopeType (str): ex SST-2M-ASTRI, LST, ...

    Returns:
        str: 'SST', 'MST' or 'LST'

    """

    if 'SST' in telescopeType:
        return 'SST'
    elif 'MST' in telescopeType:
        return 'MST'
    elif 'LST' in telescopeType:
        return 'LST'
    else:
        logging.error('Invalid telescopeType {}'.format(telescopeType))


class TelescopeModel:
    """MC Model handler at the telescope level.

    TelescopeModel is an abstract representation of the MC model
    at the telescope level. It contains the list of parameters and
    useful methods to handle it.

    Attributes:
        yamlDBPath (str): path of the yaml database containing the model parameters.
        label (str): instance label to avoid conflict between files.
        version (str): MC model version (ex. prod4).
        telescopeType (str): telescope type for the base set of parameters (ex. SST-2M-ASTRI,
            LST, ...)
        site (str): Paranal(south)/LaPalma(north).
        filesLocation (str): location for created files (ex. simtel cfg file). If None,
            pwd will used.

    """

    def __init__(self, telescopeType, site, yamlDBPath=None, version='default', label=None,
                 filesLocation=None, readFromDB=True):
        """TelescopeModel __init__.

        Args:
            telescopeType (str): telescope type for the base set of parameters (ex. SST-2M-ASTRI,
                LST, ...)
            site (str): Paranal(south)/LaPalma(north).
            yamlDBPath (str): path of the yaml database containing the model parameters.
            version (str): MC model version (ex. prod4).
            label (str): instance label to avoid conflict between files.
            filesLocation (str): location for created files (ex. simtel cfg file). If None,
                pwd will used.
            readFromDB (bool): if True, parameters will be read from DB at the initizalition.
                It must be True most of the cases. It must be False specially if classmethod
                fromConfigFile is used.

        """
        logging.info('Init TelescopeModel')

        self.yamlDBPath = yamlDBPath
        self.label = label
        self._version = None
        self.version = version
        self._telescopeType = None
        self.telescopeType = telescopeType
        self._site = None
        self.site = site
        self._parameters = dict()

        self.filesLocation = Path.cwd() if filesLocation is None else Path(filesLocation)

        if readFromDB:
            self.loadParametersFromDB()

        self._isFileExported = False

    @property
    def version(self):
        """str: model version after validation."""
        return self._version

    @version.setter
    def version(self, value):
        self._version = names.validateName(value, names.allModelVersionNames)

    @property
    def telescopeType(self):
        """str: telescope type after validation."""
        return self._telescopeType

    @telescopeType.setter
    def telescopeType(self, value):
        self._telescopeType = names.validateName(value, names.allTelescopeTypeNames)

    @property
    def site(self):
        """str: site after validation."""
        return self._site

    @site.setter
    def site(self, value):
        self._site = names.validateName(value, names.allSiteNames)

    @classmethod
    def fromConfigFile(cls, telescopeType, site, label, configFileName):
        """ Create a TelescopeModel from a sim_telarray config file.

            Todo:
                - dealing with ifdef/indef etc. By now it just keeps the last version of the
                    parameters in the file.

            Args:
                telescopeType (str):
                site (str):
                label (str):
                configFileName (str): path to the input config file

            Return:
                Instance of TelescopeModel class

        """
        parameters = dict()
        tel = cls(telescopeType=telescopeType, site=site, label=label, readFromDB=False)

        def processLine(words):
            iComment = len(words)
            for w in words:
                if '%' in w:
                    iComment = words.index(w)
                    break
            words = words[0:iComment]
            par = words[0].replace('=', '')
            value = ''
            for w in words[1:]:
                w = w.replace('=', '')
                w = w.replace(',', ' ')
                value += w + ' '
            return par, value

        with open(configFileName, 'r') as file:
            for line in file:
                words = line.split()
                if len(words) == 0:
                    continue
                elif '%' in words[0] or 'echo' in words:
                    continue
                elif '#' not in line and len(words) > 0:
                    par, value = processLine(words)
                    par, value = tel.validateParameter(par, value)
                    parameters[par] = value

        tel.addParameters(**parameters)
        return tel

    def loadParametersFromDB(self):
        """ Read parameters from DB and store it in _parameters (dict).

        """

        def readParsFromOneType(yamlDBPath, telescopeType, parametersDB):
            ''' Read parameters as a dict and concatenate it to parametersDB
                Implementation was needed to concatenate the optics parameters
                to the MST models
            '''
            fileNameDB = '{}/parValues-{}.yml'.format(yamlDBPath, telescopeType)
            logging.info('Reading DB file {}'.format(fileNameDB))
            with open(fileNameDB, 'r') as stream:
                pars = yaml.load(stream)
            parametersDB.update(pars)

        parametersDB = dict()
        readParsFromOneType(self.yamlDBPath, self.telescopeType, parametersDB)
        if whichTelescopeSize(self.telescopeType) == 'MST':
            readParsFromOneType(self.yamlDBPath, 'MST-optics', parametersDB)

        for parNameIn in parametersDB:
            if parametersDB[parNameIn]['Applicable']:
                parName, parValue = self.validateParameter(
                    parNameIn,
                    parametersDB[parNameIn][self._version]
                )
                self._parameters[parName] = parValue

        # Site
        # Two site parameters need to be read:
        # atmospheric_transmission and altitude
        logging.debug('Reading site parameters from DB')

        def getSiteParameter(yamlDBPath, site, parName):
            ''' Get the value of parName for a given site '''
            fileNameDB = '{}/parValues-Sites.yml'.format(yamlDBPath)
            logging.info('Reading DB file {}'.format(fileNameDB))
            with open(fileNameDB, 'r') as stream:
                allPars = yaml.load(stream)
                for par in allPars:
                    if parName in par and self.site.lower() in par:
                        return allPars[par][self._version]

        for siteParName in ['atmospheric_transmission', 'altitude']:
            siteParValue = getSiteParameter(self.yamlDBPath, self.site, siteParName)
            parName, parValue = self.validateParameter(siteParName, siteParValue)
            self._parameters[parName] = parValue

    # end loadParametersFromDB

    def validateParameter(self, parNameIn, parValueIn):
        logging.debug('Validating parameter {}'.format(parNameIn))
        for parNameModel in MODEL_PARS.keys():
            if parNameIn == parNameModel or parNameIn in MODEL_PARS[parNameModel]['names']:
                parType = MODEL_PARS[parNameModel]['type']
                return parNameModel, parType(parValueIn)
        return parNameIn, parValueIn

    def getParameter(self, parName):
        ''' Return an EXISTING parameter of the model '''
        if parName in self._parameters:
            return self._parameters[parName]
        else:
            logging.error('Parameter {} was not found in the model'.format(parName))
            return ''

    def addParameters(self, **kwargs):
        ''' Add a NEW parameters to the model.
            kwargs are used, parameters should be passed as
            parameterName=value
        '''
        for par in kwargs.keys():
            if par in self._parameters.keys():
                logging.error(
                    'Parameter {} already in the model, use changeParameter instead'.format(par)
                )
            else:
                logging.info('Adding {}={} to the model'.format(par, kwargs[par]))
                self._parameters[par] = str(kwargs[par])

    def changeParameters(self, **kwargs):
        ''' Change the value of EXISTING parameters to the model '''
        for par in kwargs.keys():
            if par not in self._parameters.keys():
                logging.error(
                    'Parameter {} not in the model, use addParameters instead'.format(par)
                )
            else:
                self._parameters[par] = kwargs[par]

    def removeParameters(self, *args):
        ''' Remove a parameter from the model '''
        for par in args:
            if par in self._parameters.keys():
                logging.info('Removing parameter {}'.format(par))
                del self._parameters[par]
            else:
                logging.error(
                    'Could not remove parameter {} because it does not exist'.format(par)
                )

    def exportConfigFile(self):
        ''' Export the config file used by sim_telarray.
            loc gives the location that the file should be created.
            If loc is not given, a directory $pwd/SimtelFiles/cfg is created and used
        '''

        # Setting file name and the location
        configFileName = 'CTA-{}-{}-{}'.format(self._version, self.site, self.telescopeType)
        configFileName += '_{}'.format(self.label) if self.label is not None else ''
        configFileName += '.cfg'

        configFileDirectory = self.filesLocation.joinpath('CTAMCFiles/cfg')

        if not configFileDirectory.exists():
            configFileDirectory.mkdir(parents=True, exist_ok=True)
            logging.info('Creating directory {}'.format(configFileDirectory))
        self._configFilePath = configFileDirectory.joinpath(configFileName)

        # Writing parameters to the file
        logging.info('Writing config file - {}'.format(self._configFilePath))
        with open(self._configFilePath, 'w') as file:
            header = (
                '%{}\n'.format(99*'=') +
                '% Configuration file for:\n' +
                '% TelescopeType: {}\n'.format(self._telescopeType) +
                '% Site: {}\n'.format(self.site) +
                '% Label: {}\n'.format(self.label) +
                '%{}\n'.format(99*'=')
            )

            file.write(header)
            for par in self._parameters:
                value = self._parameters[par]
                file.write('{} = {}\n'.format(par, value))

        self._isFileExported = True
    # end exportConfigFile

    def getConfigFile(self):
        ''' Return config file
            Export it first if it was not exported yet
        '''
        if not self._isFileExported:
            self.exportConfigFile()
        return self._configFilePath
