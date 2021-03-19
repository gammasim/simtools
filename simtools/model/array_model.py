#!/usr/bin/python3

""" This module contains the ArrayModel class

    Todo: everything

"""


__all__ = ['ArrayModel']


class ArrayModel:
    def __init__(
        self,
        label,
        arrayConfigFile=None,
        arrayConfigData=None,
        logger=__name__
    ):

        self.layout = None

        if arrayConfigData is not None and arrayConfigFile is not None:
            self._logger.warning(
                'Both arrayConfigData and arrayConfigFile were given '
                '- arrayConfigData will be used'
            )
            # read file and load config
            pass
        elif arrayConfigFile is not None:
            # read file and load config
            pass
        elif arrayConfigData is not None:
            self._loadArrayData(arrayConfigData)
        else:
            self._logger.error('No arrayConfigData was given - aborting')

    def _loadArrayData(arrayConfigData):
        pass

    def exportCorsikaInputFile():
        pass

    def exportSimtelConfigFiles():
        pass

    def exportArrayConfigFile():
        pass
