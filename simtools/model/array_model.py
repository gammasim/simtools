import logging

from simtools.layout.layout_array import LayoutArray
from simtools.util import names
from simtools.util.general import collectDataFromYamlOrDict

__all__ = ['ArrayModel']


class InvalidArrayConfigData(Exception):
    pass


class ArrayModel:
    def __init__(
        self,
        label=None,
        arrayConfigFile=None,
        arrayConfigData=None,
        logger=__name__
    ):
        self._logger = logging.getLogger(logger)
        self._logger.debug('Init ArrayModel')

        arrayConfigData = collectDataFromYamlOrDict(arrayConfigFile, arrayConfigData)
        self._loadArrayData(arrayConfigData)

        self.layout.printTelescopeList()

        self._buildArrayModel()
        # End of init

    def _loadArrayData(self, arrayConfigData):
        ''' Loading parameters from arrayData '''
        # Validating arrayConfigData
        self._validateArrayData(arrayConfigData)

        # Site
        self.site = names.validateSiteName(arrayConfigData['site'])

        # Layout name
        self.layoutName = names.validateArrayName(arrayConfigData['arrayName'])
        self.layout = LayoutArray.fromLayoutArrayName(self.site + '-' + self.layoutName)

        self._arrayConfigData = arrayConfigData

    def _validateArrayData(self, arrayConfigData):
        ''' Validate arrayData by checking the existence of the relevant keys.'''

        def runOverPars(pars, data, parent=None):
            allKeys = data.keys() if parent is None else data[parent].keys()
            for pp in pars:
                if pp not in allKeys:
                    key = pp if parent is None else parent + '.' + pp
                    msg = (
                        'Key {} was not found in arrayConfigData '.format(key)
                        + '- impossible to build array model'
                    )
                    self._logger.error(msg)
                    raise InvalidArrayConfigData(msg)

        runOverPars(['site', 'arrayName', 'default'], arrayConfigData)
        runOverPars(['LST', 'MST', 'SST'], arrayConfigData, parent='default')
        # End of _validateArrayData

    def _buildArrayModel(self):

        for tel in self.layout:
            print(tel)


        pass

    def exportCorsikaInputFile():
        pass

    def exportSimtelConfigFiles():
        pass

    def exportArrayConfigFile():
        pass
