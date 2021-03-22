import logging

from simtools.model.telescope_model import TelescopeModel
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

        self.label = label

        arrayConfigData = collectDataFromYamlOrDict(arrayConfigFile, arrayConfigData)
        self._loadArrayData(arrayConfigData)

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

        # Model version
        if 'modelVersion' not in arrayConfigData.keys() or arrayConfigData['modelVersion'] is None:
            self._logger.warning('modelVersion not given in arrayConfigData - using current')
            self.modelVersion = 'current'
        else:
            self.modelVersion = arrayConfigData['modelVersion']

        self._arrayConfigData = {
            k: v for (k, v) in arrayConfigData.items()
            if k not in ['site', 'arrayName', 'modelVersion']
        }

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
        runOverPars(['LST', 'MST'], arrayConfigData, parent='default')
        if names.validateSiteName(arrayConfigData['site']) == 'South':
            runOverPars(['SST'], arrayConfigData, parent='default')
        # End of _validateArrayData

    def _buildArrayModel(self):

        self._telescopeModel = list()
        for tel in self.layout:
            telSize = tel.getTelescopeSize()
            print(telSize)
            if tel.name in self._arrayConfigData.keys():
                telModelName = self.site + '-' + telSize + '-' + self._arrayConfigData[tel.name]
            else:
                telModelName = (
                    self.site + '-' + telSize + '-' + self._arrayConfigData['default'][telSize]
                )
            print(tel.name + ', ' + telModelName)

            telModel = TelescopeModel(
                telescopeName=telModelName,
                version=self.modelVersion,
                label=self.label,
                logger=self._logger.name
            )
            self._telescopeModel.append(telModel)

        assert len(self._telescopeModel) == len(self.layout)

    def exportCorsikaInputFile():
        pass

    def exportSimtelConfigFiles():
        pass

    def exportArrayConfigFile():
        pass
