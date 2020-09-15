import logging

import simtools.io_handler as io

__all__ = ['Mirrors']


class InvalidMirrorListFile(Exception):
    pass


class Mirrors:
    '''
    Mirrors class, created from a mirror list file.

    Attributes
    ----------
    mirrors: dict
        A dictionary with the mirror positions [cm], diameters, focel length and shape.
    shape: int
        Single shape code (0=circular, 1=hex. with flat side parallel to y, 2=square,
        3=other hex.)
    diameter: float
        Single diameter in cm.
    numberOfMirrors: int
        Number of mirrors.

    Methods
    -------
    readMirrorList(mirrorListFile)
        Read the mirror list and store the data.
    plotMirrorLayout()
        Plot the mirror layout (to be implemented).
    '''

    def __init__(self, mirrorListFile, logger=__name__):
        '''
        Mirrors.

        Parameters
        ----------
        mirrorListFile: str
            The sim_telarray file name.
        logger: str
            Logger name to use in this instance
        '''
        self._logger = logging.getLogger(logger)
        self._logger.debug('Mirrors Init')

        self._mirrorListFile = mirrorListFile
        self._readMirrorList()

    def _readMirrorList(self):
        '''
        Read the mirror list in sim_telarray format and store the data.

        Raises
        ------
        InvalidMirrorListFile
            If number of mirrors is 0.
        '''
        self._mirrors = dict()
        self._mirrors['number'] = list()
        self._mirrors['posX'] = list()
        self._mirrors['posY'] = list()
        self._mirrors['diameter'] = list()
        self._mirrors['flen'] = list()
        self._mirrors['shape'] = list()

        mirrorCounter = 0
        collectGeoPars = True
        with open(self._mirrorListFile, 'r') as file:
            for line in file:
                line = line.split()
                if '#' in line[0] or '$' in line[0]:
                    continue
                if collectGeoPars:
                    self.diameter = float(line[2])
                    self.shape = int(line[4])
                    collectGeoPars = False
                    self._logger.debug('Shape = {}'.format(self.shape))
                    self._logger.debug('Diameter = {}'.format(self.diameter))

                self._mirrors['number'].append(mirrorCounter)
                self._mirrors['posX'].append(float(line[0]))
                self._mirrors['posY'].append(float(line[1]))
                self._mirrors['diameter'].append(float(line[2]))
                self._mirrors['flen'].append(float(line[3]))
                self._mirrors['shape'].append(float(line[4]))
                mirrorCounter += 1
        self.numberOfMirrors = mirrorCounter
        if self.numberOfMirrors == 0:
            msg = 'Problem reading mirror list file'
            self._logger.error(msg)
            raise InvalidMirrorListFile()

    def getSingleMirrorParameters(self, number):
        '''
        Get parameters for a single mirror given by number.

        Parameters
        ----------
        number: int
            Mirror number of desired parameters.

        Returns
        -------
        (posX, posY, diameter, flen, shape)
        '''
        if number > self.numberOfMirrors - 1:
            self._logger.error('Mirror number is out range')
            return None
        return (
            self._mirrors['posX'][number],
            self._mirrors['posY'][number],
            self._mirrors['diameter'][number],
            self._mirrors['flen'][number],
            self._mirrors['shape'][number]
        )

    def plotMirrorLayout(self):
        '''
        Plot the mirror layout.
        '''
        pass
