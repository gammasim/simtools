import logging

import astropy.units as u
import numpy as np
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

        self.mirror_table = Table()
        self.mirror_diameter = None
        self.shape_type = None
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
        # Getting mirror parameters from mirror list.

        self.mirror_table = Table.read(self._mirror_list_file, format="ascii.ecsv")
        self._logger.debug(f"Reading mirror properties from {self._mirror_list_file}")
        try:
            self.shape_type = u.Quantity(self.mirror_table["shape_type"])[0]
            self.mirror_diameter = u.Quantity(self.mirror_table["mirror_diameter"])[0]
            self.number_of_mirrors = len(self.mirror_table["focal_length"])

            self._logger.debug(f"Mirror shape_type = {self.shape_type}")
            self._logger.debug(f"Mirror diameter = {self.mirror_diameter}")
            self._logger.debug(f"Number of Mirrors = {self.number_of_mirrors}")
        except KeyError:
            self._logger.debug("Mirror shape_type not in mirror file")
            self.number_of_mirrors = len(self.mirror_table["mirror_id"])

        if "focal_length" not in self.mirror_table.colnames:
            self.mirror_table["focal_length"] = (
                self.mirror_table["mirror_diameter"].to("cm").value / 4.0
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

        self.mirror_table = Table.read(
            self._mirror_list_file,
            format="ascii.no_header",
            names=[
                "mirror_x",
                "mirror_y",
                "mirror_diameter",
                "focal_length",
                "shape_type",
                "mirror_z",
                "sep",
                "mirror_id",
            ],
        )

        self.shape_type = self.mirror_table["shape_type"][0]
        self.mirror_diameter = self.mirror_table["mirror_diameter"][0]
        self.number_of_mirrors = len(self.mirror_table["focal_length"])

        self._logger.debug(f"Mirror shape_type = {self.shape_type}")
        self._logger.debug(f"Mirror diameter = {self.mirror_diameter}")
        self._logger.debug(f"Number of Mirrors = {self.number_of_mirrors}")

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
        (pos_x, pos_y, mirror_diameter, focal_length, shape_type): tuple of float
            X, Y positions, mirror_diameter, focal length and shape_type.
        """

        def get_mirror_table_mask():
            if isinstance(self.mirror_table["mirror_id"][0], np.str_):
                mask = self.mirror_table["mirror_id"] == f"id={number}"
            if isinstance(self.mirror_table["mirror_id"][0], np.int32):
                mask = self.mirror_table["mirror_id"] == number
            return mask

        mask = get_mirror_table_mask()
        if not np.any(mask):
            self._logger.error(f"Mirror id{number} not in table, using first mirror instead")
            mask[0] = True

        try:
            return (
                self.mirror_table[mask]["mirror_x"].value[0],
                self.mirror_table[mask]["mirror_y"].value[0],
                self.mirror_table[mask]["mirror_diameter"].value[0],
                self.mirror_table[mask]["focal_length"].value[0],
                self.mirror_table[mask]["shape_type"].value[0],
            )
        except KeyError:
            self._logger.debug("Mirror list missing required column")
            return None

    def plot_mirror_layout(self):
        """
        Plot the mirror layout.

        TODO
        """
