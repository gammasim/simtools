import logging

import astropy.units as u
from astropy.table import Table

__all__ = ["InvalidMirrorListFile", "Mirrors"]


class InvalidMirrorListFile(Exception):
    """Exception for invalid mirror list file."""


class Mirrors:
    """
    Mirrors class, created from a mirror list file.

    Parameters
    ----------
    mirror_list_file: str
        mirror list in sim_telarray or ecsv format (with panel focal length only).
    """

    def __init__(self, mirror_list_file):
        """
        Initialize Mirrors.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Mirrors Init")

        self._mirrors = {}
        self.diameter = None
        self.shape = None
        self.number_of_mirrors = 0

        self._mirror_list_file = mirror_list_file
        self._read_mirror_list()

    def _read_mirror_list(self):
        """
        Read the mirror lists from disk and store the data. Allow reading of mirror lists in \
        sim_telarray and ecsv format
        """

        if str(self._mirror_list_file).find("ecsv") > 0:
            self._read_mirror_list_from_ecsv()
        else:
            self._read_mirror_list_from_sim_telarray()

    def _read_mirror_list_from_ecsv(self):
        """
        Read the mirror list in ecsv format and store the data.

        Raises
        ------
        InvalidMirrorListFile
            If number of mirrors is 0.
        """
        # Getting mirror parameters from mirror list. Diameter and shape are identical between mirrors.

        _mirror_table = Table.read(self._mirror_list_file, format="ascii.ecsv")
        self._logger.debug(f"Reading mirror properties from {self._mirror_list_file}")

        self.shape = u.Quantity(_mirror_table['shape_code'])[0].value
        self.diameter = u.Quantity(_mirror_table['diameter'])[0].value
        self._logger.debug(f"Shape = {self.shape}")
        self._logger.debug(f"Diameter = {self.diameter}")

        try:
            self._mirrors["flen"] = list(_mirror_table["focal_length"].to("cm").value / 2.0)
            self.number_of_mirrors = len(self._mirrors["flen"])
            self._mirrors["mirror_number"] = list(_mirror_table["mirror_number"].value)
            self._mirrors["pos_x"] = list(_mirror_table["x_pos"].to("cm").value)
            self._mirrors["pos_y"] = list(_mirror_table["y_pos"].to("cm").value)
            self._mirrors["diameter"] = list(_mirror_table["diameter"].to("cm").value)
            self._mirrors["shape"] = list(_mirror_table["shape_code"].value)
        except KeyError:
            self._logger.debug(
                f"Missing column for mirror panel focal length (flen) in {self._mirror_list_file}"
            )

        if self.number_of_mirrors == 0:
            msg = "Problem reading mirror list file"
            self._logger.error(msg)
            raise InvalidMirrorListFile()

    def _read_mirror_list_from_sim_telarray(self):
        """
        Read the mirror list in sim_telarray format and store the data.

        Raises
        ------
        InvalidMirrorListFile
            If number of mirrors is 0.
        """

        self._mirrors["number"] = []
        self._mirrors["pos_x"] = []
        self._mirrors["pos_y"] = []
        self._mirrors["diameter"] = []
        self._mirrors["flen"] = []
        self._mirrors["shape"] = []

        mirror_counter = 0
        collect_geo_pars = True
        with open(self._mirror_list_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.split()
                if "#" in line[0] or "$" in line[0]:
                    continue
                if collect_geo_pars:
                    self.diameter = float(line[2])
                    self.shape = int(line[4])
                    collect_geo_pars = False
                    self._logger.debug(f"Shape = {self.shape}")
                    self._logger.debug(f"Diameter = {self.diameter}")

                self._mirrors["number"].append(mirror_counter)
                self._mirrors["pos_x"].append(float(line[0]))
                self._mirrors["pos_y"].append(float(line[1]))
                self._mirrors["diameter"].append(float(line[2]))
                self._mirrors["flen"].append(float(line[3]))
                self._mirrors["shape"].append(float(line[4]))
                mirror_counter += 1
        self.number_of_mirrors = mirror_counter
        if self.number_of_mirrors == 0:
            msg = "Problem reading mirror list file"
            self._logger.error(msg)
            raise InvalidMirrorListFile()

    def get_single_mirror_parameters(self, number):
        """
        Get parameters for a single mirror given by number.

        Parameters
        ----------
        number: int
            Mirror number of desired parameters.

        Returns
        -------
        (pos_x, pos_y, diameter, flen, shape): tuple of float
            X, Y positions, diameter, focal length and shape.
        """

        if number > self.number_of_mirrors - 1:
            self._logger.error("Mirror number is out range")
            return None
        return (
            self._mirrors["pos_x"][number],
            self._mirrors["pos_y"][number],
            self._mirrors["diameter"][number],
            self._mirrors["flen"][number],
            self._mirrors["shape"][number],
        )

    def plot_mirror_layout(self):
        """
        Plot the mirror layout.

        TODO
        """
