from astropy.table import Table
import logging

__all__ = ["Mirrors"]


class InvalidMirrorListFile(Exception):
    pass


class Mirrors:
    """
    Mirrors class, created from a mirror list file.

    Attributes
    ----------
    mirrors: dict
        A dictionary with the mirror positions [cm], diameters, focal length and shape.
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
    """

    def __init__(self, mirrorListFile):
        """
        Mirrors.

        Parameters
        ----------
        mirrorListFile: str
            mirror list in sim_telarray or ecsv format (with
            panel focal length only)
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Mirrors Init")

        self._mirrors = dict()
        self.diameter = None
        self.shape = None
        self.numberOfMirrors = 0

        self._mirrorListFile = mirrorListFile
        self._readMirrorList()

    def _readMirrorList(self):
        """
        Read the mirror lists from disk and store the data

        Allow reading of mirro lists in sim_telarray and ecsv
        format

        """

        if str(self._mirrorListFile).find('ecsv') > 0:
            self._readMirrorList_from_ecsv()
        else:
            self._readMirrorList_from_sim_telarray()

    def _readMirrorList_from_ecsv(self):
        """
        Read the mirror list in ecsv format and store the data.

        Raises
        ------

        """

        # TODO - temporary hard wired geopars - should come from DB
        self.diameter = 120
        self.shape = 1
        self._logger.debug("Shape = {}".format(self.shape))
        self._logger.debug("Diameter = {}".format(self.diameter))

        _mirror_table = Table.read(self._mirrorListFile, format='ascii.ecsv')
        self._logger.debug("Reading mirror properties from {}".format(
            self._mirrorListFile))
        try:
            self._mirrors["flen"] = list(
                _mirror_table['mirror_panel_radius'].to('cm').value/2.)
            self.numberOfMirrors = len(self._mirrors["flen"])
            self._mirrors["number"] = list(range(self.numberOfMirrors))
            self._mirrors["posX"] = [0.]*self.numberOfMirrors
            self._mirrors["posY"] = [0.]*self.numberOfMirrors
            self._mirrors["diameter"] = [self.diameter]*self.numberOfMirrors
            self._mirrors["shape"] = [self.shape]*self.numberOfMirrors
        except KeyError:
            self._logger.debug("Missing column for mirror panel focal length (flen) in {}".format(
                self._mirrorListFile))

        if self.numberOfMirrors == 0:
            msg = "Problem reading mirror list file"
            self._logger.error(msg)
            raise InvalidMirrorListFile()

    def _readMirrorList_from_sim_telarray(self):
        """
        Read the mirror list in sim_telarray format and store the data.

        Raises
        ------
        InvalidMirrorListFile
            If number of mirrors is 0.
        """

        self._mirrors["number"] = list()
        self._mirrors["posX"] = list()
        self._mirrors["posY"] = list()
        self._mirrors["diameter"] = list()
        self._mirrors["flen"] = list()
        self._mirrors["shape"] = list()

        mirrorCounter = 0
        collectGeoPars = True
        with open(self._mirrorListFile, "r") as file:
            for line in file:
                line = line.split()
                if "#" in line[0] or "$" in line[0]:
                    continue
                if collectGeoPars:
                    self.diameter = float(line[2])
                    self.shape = int(line[4])
                    collectGeoPars = False
                    self._logger.debug("Shape = {}".format(self.shape))
                    self._logger.debug("Diameter = {}".format(self.diameter))

                self._mirrors["number"].append(mirrorCounter)
                self._mirrors["posX"].append(float(line[0]))
                self._mirrors["posY"].append(float(line[1]))
                self._mirrors["diameter"].append(float(line[2]))
                self._mirrors["flen"].append(float(line[3]))
                self._mirrors["shape"].append(float(line[4]))
                mirrorCounter += 1
        self.numberOfMirrors = mirrorCounter
        if self.numberOfMirrors == 0:
            msg = "Problem reading mirror list file"
            self._logger.error(msg)
            raise InvalidMirrorListFile()

    def getSingleMirrorParameters(self, number):
        """
        Get parameters for a single mirror given by number.

        Parameters
        ----------
        number: int
            Mirror number of desired parameters.

        Returns
        -------
        (posX, posY, diameter, flen, shape)
        """
        if number > self.numberOfMirrors - 1:
            self._logger.error("Mirror number is out range")
            return None
        return (
            self._mirrors["posX"][number],
            self._mirrors["posY"][number],
            self._mirrors["diameter"][number],
            self._mirrors["flen"][number],
            self._mirrors["shape"][number],
        )

    def plotMirrorLayout(self):
        """
        Plot the mirror layout.

        TODO
        """
        pass
