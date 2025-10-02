"""Definition and modeling of mirror panels."""

import logging
from pathlib import Path

import astropy.io.ascii
import astropy.units as u
import numpy as np
from astropy.table import Table


class InvalidMirrorListFileError(Exception):
    """Exception for invalid mirror list file."""


class Mirrors:
    """
    Mirrors class, created from a mirror list file.

    Parameters
    ----------
    mirror_list_file: Union[str, Path]
        Mirror list in sim_telarray or ecsv format (with panel focal length only).
    parameters: dict, optional
        Dictionary of parameters from the database.
    """

    def __init__(self, mirror_list_file: str | Path, parameters: dict | None = None):
        """Initialize Mirrors."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Mirrors Init")

        self.mirror_table = Table()
        self.mirror_diameter = None
        self.shape_type = None
        self.number_of_mirrors = 0
        self.parameters = parameters

        self._mirror_list_file = mirror_list_file
        self._read_mirror_list()

    def _read_mirror_list(self):
        """
        Read the mirror lists from disk and store the data.

        Allow reading of mirror lists in sim_telarray and ecsv format.
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
        InvalidMirrorListFileError
            If number of mirrors is 0.
        """
        self._logger.debug(f"Reading mirror properties from {self._mirror_list_file}")
        self.mirror_table = Table.read(self._mirror_list_file, format="ascii.ecsv")

        self.number_of_mirrors = np.shape(self.mirror_table)[0]
        self._logger.debug(f"Number of Mirrors = {self.number_of_mirrors}")

        if self.number_of_mirrors == 0:
            msg = "Problem reading mirror list file"
            self._logger.error(msg)
            raise InvalidMirrorListFileError

        try:
            self.mirror_diameter = u.Quantity(self.mirror_table["mirror_diameter"])[0]
            self._logger.debug(f"Mirror diameter = {self.mirror_diameter}")
        except KeyError:
            self._logger.debug("Mirror mirror_panel_diameter not in mirror file")
            try:
                self.mirror_diameter = u.Quantity(
                    self.parameters["mirror_panel_diameter"]["value"],
                    self.parameters["mirror_panel_diameter"]["unit"],
                )
                self._logger.debug("Take mirror_panel_diameter from parameters")
            except TypeError as error:
                msg = "Mirror mirror_panel_diameter not contained in DB"
                self._logger.error(msg)
                raise TypeError(msg) from error
        if "focal_length" not in self.mirror_table.colnames:
            try:
                self.mirror_table["focal_length"] = (
                    self.mirror_table["mirror_curvature_radius"].to("cm") / 2
                )
            except KeyError:
                self._logger.debug("mirror_curvature_radius not contained in mirror list")
                try:
                    self.mirror_table["focal_length"] = self.number_of_mirrors * [
                        u.Quantity(
                            self.parameters["mirror_focal_length"]["value"],
                            self.parameters["mirror_focal_length"]["unit"],
                        )
                    ]
                    self._logger.debug("Take mirror_focal_length from parameters")
                except TypeError as error:
                    msg = "mirror_focal_length not contained in DB"
                    self._logger.error(msg)
                    raise TypeError(msg) from error

        try:
            self.shape_type = u.Quantity(self.mirror_table["shape_type"])[0]
            self._logger.debug(f"Mirror shape_type = {self.shape_type}")
        except KeyError:
            self._logger.debug("Mirror shape_type not in mirror file")
            try:
                self.shape_type = self.parameters["mirror_panel_shape"]["value"]
                self._logger.debug("Take shape_type from parameters")
            except TypeError as error:
                msg = "Mirror shape_type not contained in DB"
                self._logger.error(msg)
                raise TypeError(msg) from error

    def _read_mirror_list_from_sim_telarray(self):
        """
        Read the mirror list in sim_telarray format and store the data.

        Allow to read mirror lists with different number of columns.

        Raises
        ------
        InvalidMirrorListFileError
            If number of mirrors is 0.
        """
        self._logger.debug(f"Reading mirror properties from {self._mirror_list_file}")

        try:
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
                    "mirror_panel_id",
                ],
                units=["cm", "cm", "cm", "cm", None, "cm", None, None],
            )
            self.mirror_table["mirror_panel_id"] = np.array(
                [
                    int("".join(filter(str.isdigit, string)))
                    for string in self.mirror_table["mirror_panel_id"]
                ]
            )
        except astropy.io.ascii.core.InconsistentTableError:
            self._logger.debug("Try and read mirror list with low number of columns")
            self.mirror_table = Table.read(
                self._mirror_list_file,
                format="ascii.no_header",
                names=[
                    "mirror_x",
                    "mirror_y",
                    "mirror_diameter",
                    "focal_length",
                    "shape_type",
                ],
                units=["cm", "cm", "cm", "cm", None],
            )
            self.mirror_table["mirror_panel_id"] = np.arange(len(self.mirror_table["mirror_x"]))

        self.shape_type = self.mirror_table["shape_type"][0]
        self.mirror_diameter = u.Quantity(
            self.mirror_table["mirror_diameter"][0], self.mirror_table["mirror_diameter"].unit
        )
        self.number_of_mirrors = len(self.mirror_table["focal_length"])

        self._logger.debug(f"Mirror shape_type = {self.shape_type}")
        self._logger.debug(f"Mirror diameter = {self.mirror_diameter}")
        self._logger.debug(f"Number of Mirrors = {self.number_of_mirrors}")

    def get_single_mirror_parameters(self, number: int) -> tuple:
        """
        Get parameters for a single mirror given by number.

        Parameters
        ----------
        number: int
            Mirror number of desired parameters.

        Returns
        -------
        tuple
            (pos_x, pos_y, mirror_diameter, focal_length, shape_type): tuple of float
            X, Y positions, mirror_diameter, focal length and shape_type.
        """
        mask = self.mirror_table["mirror_panel_id"] == number
        if not np.any(mask):
            self._logger.debug(f"Mirror id{number} not in table, using first mirror instead")
            mask[0] = True
        try:
            return_values = (
                u.Quantity(
                    self.mirror_table[mask]["mirror_x"].value[0],
                    self.mirror_table[mask]["mirror_x"].unit,
                ),
                u.Quantity(
                    self.mirror_table[mask]["mirror_y"].value[0],
                    self.mirror_table[mask]["mirror_y"].unit,
                ),
                u.Quantity(
                    self.mirror_table[mask]["mirror_diameter"].value[0],
                    self.mirror_table[mask]["mirror_diameter"].unit,
                ),
                u.Quantity(
                    self.mirror_table[mask]["focal_length"].value[0],
                    self.mirror_table[mask]["focal_length"].unit,
                ),
                u.Quantity(
                    self.mirror_table[mask]["shape_type"].value[0],
                    self.mirror_table[mask]["shape_type"].unit,
                ),
            )
        except KeyError:
            self._logger.debug("Mirror list missing required column")
            return_values = (
                0,
                0,
                self.mirror_diameter,
                u.Quantity(
                    self.mirror_table[mask]["focal_length"].value[0],
                    self.mirror_table[mask]["focal_length"].unit,
                ),
                self.shape_type,
            )
        return return_values

    def plot_mirror_layout(self):
        """Plot the mirror layout (not implemented yet)."""
