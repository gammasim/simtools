import logging

import simtools.io_handler as io

__all__ = ['Mirrors']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InvalidMirrorListFile(Exception):
    pass


class Mirrors:
    '''
    '''

    def __init__(self, mirrorListFile):
        '''
        Camera class, defining pixel layout including rotation, finding neighbour pixels,
        calculating FoV and plotting the camera.

        Parameters
        ----------
        mirrorListFile: string
                    The sim_telarray file name.

        Attributes
        ----------
        mirrors: dict
            A dictionary with the pixel positions, the camera rotation angle,
            the pixel shape, the pixel diameter, the pixel IDs and their "on" status.
        shape: int
            Array of neighbour indices in a list for each pixel.
        diameter: float
            Array of edge pixel indice
        numberOfMirrors: int
            Array of edge pixel indice

        Methods
        -------
        readMirrorList(mirrorListFile)
            Read the pixel layout from the camera config file,
            assumed to be in a sim_telarray format.
        plotMirrorLayout()
            Plot the pixel layout for an observer facing the camera.
            Including in the plot edge pixels, off pixels, pixel ID for the first 50 pixels,
            coordinate systems, FOV, focal length and the average edge radius.
        '''

        self._mirrorListFile = mirrorListFile
        self._readMirrorList()

    def _readMirrorList(self):
        '''
        Read the pixel layout from the camera config file, assumed to be in a sim_telarray format.

        Parameters
        ----------
        cameraConfigFile: string
            The sim_telarray file name.

        Returns
        -------
        dict: pixels
            A dictionary with the pixel positions, the camera rotation angle,
            the pixel shape, the pixel diameter, the pixel IDs and their "on" status.

        Notes
        -----
        The pixel shape can be hexagonal (denoted as 1 or 3) or a square (denoted as 2).
        The hexagonal shapes differ in their orientation, where those denoted as 3 are rotated
        clockwise by 30 degrees with respect to those denoted as 1.
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
                if '#' in line[0]:
                    continue
                if collectGeoPars:
                    self.diameter = float(line[2])
                    self.shape = int(line[4])
                    collectGeoPars = False
                    logger.debug('Shape = {}'.format(self.shape))
                    logger.debug('Diameter = {}'.format(self.diameter))

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
            logger.error(msg)
            raise InvalidMirrorListFile()

    def getSingleMirrorParameters(self, number):
        '''
        '''
        if number > self.numberOfMirrors - 1:
            logger.error('Mirror number is out range')
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
        Plot the pixel layout for an observer facing the camera.
        Including in the plot edge pixels, off pixels, pixel ID for the first 50 pixels,
        coordinate systems, FOV, focal length and the average edge radius.

        Returns
        -------
        plt: pyplot.plt instance with the pixel layout

        '''
        pass
