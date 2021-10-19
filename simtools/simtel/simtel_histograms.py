import logging


__all__ = ['SimtelHistograms']


class SimtelHistograms:
    '''
    '''

    def __init__(
        self,
        histogramFiles
    ):
        '''
        '''
        self._logger = logging.getLogger(__name__)

        self.histogramFiles = histogramFiles

    def plotAndSaveFigures(self, figName):
        pass
