#!/usr/bin/python3

import logging
import yaml

from pathlib import Path
from ctamclib.util import names


def whichTelescopeSize(telescopeType):
    ''' Return the telescope size (SST, MST or LST)
        for a given telescopeType
    '''
    if 'SST' in telescopeType:
        return 'SST'
    elif 'MST' in telescopeType:
        return 'MST'
    elif 'LST' in telescopeType:
        return 'LST'
    else:
        logging.error('Invalid telescopeType {}'.format(telescopeType))


class TelescopeModel:
    def __init__(
        self,
        yamlDBPath,
        telescopeType,
        site,
        version='default',
        label=None,
        filesLocation=None
    ):
        ''' The arguments for this class are ONLY suppose to be given through the constructor
            (where they are safely validated) and they are NOT suppose to be changed afterwards.
        '''
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

        self.loadParametersFromDB()

        self._isFileExported = False

    # Properties: version, telescopeType and site
    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        self._version = names.validateName(value, names.allModelVersionNames)

    @property
    def telescopeType(self):
        return self._telescopeType

    @telescopeType.setter
    def telescopeType(self, value):
        self._telescopeType = names.validateName(value, names.allTelescopeTypeNames)

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        self._site = names.validateName(value, names.allSiteNames)

    @classmethod
    def fromConfigFile(cls, telescopeType, site, label, configFileName):
        ''' Create a TelescopeModel from a sim_telarray config file '''
        pass

    def loadParametersFromDB(self):
        ''' Parameters from DB are stored in the _parameters dict'''

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

        for par in parametersDB:
            if parametersDB[par]['Applicable']:
                self._parameters[par] = parametersDB[par][self._version]

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

        for sitePar in ['atmospheric_transmission', 'altitude']:
            self._parameters[sitePar] = getSiteParameter(
                self.yamlDBPath, self.site, sitePar
            )
    # end loadParametersFromDB

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
