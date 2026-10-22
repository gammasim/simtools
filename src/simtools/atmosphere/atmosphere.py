"""Profiles of atmospheric parameters as a function of altitude."""

import logging
from pathlib import Path

import astropy.units as u
import numpy as np


class AtmosphereProfile:
    """
    Profiles of atmospheric parameters as a function of altitude.

    Parameters
    ----------
    filename: str or Path
        Path to the atmosphere profile file (CORSIKA table format)
    """

    def __init__(self, filename):
        """AtmosphereProfile initialization."""
        self._logger = logging.getLogger(__name__)
        self.columns = {}
        self.read_atmosphere_profile(filename)

    def read_atmosphere_profile(self, filename):
        """
        Read atmosphere profile from file.

        Parameters
        ----------
        filename: str or Path
            Path to the atmosphere profile file (CORSIKA table format)

        """
        filename = Path(filename)
        data = []
        self._logger.debug(f"Reading atmosphere profile from {filename}")
        with filename.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                data.append([float(x) for x in line.split()])

        self.data = np.array(data)
        self.columns = {
            "alt": 0,
            "rho": 1,
            "thick": 2,
            "n_minus_1": 3,
            "T": 4,
            "p": 5,
            "pw_over_p": 6,
        }

    def interpolate(self, altitude, column="thick"):
        """
        Interpolate the atmosphere profile at a given altitude.

        Parameters
        ----------
        altitude: astropy.units.Quantity
            Altitude
        column: str
            Column name to interpolate.

        Returns
        -------
        float
            Interpolated value.
        """
        if column not in self.columns:
            raise KeyError(f"Unknown column: {column}")

        altitude = altitude.to(u.km).value

        x = self.data[:, 0]
        y = self.data[:, self.columns[column]]

        if altitude < x.min() or altitude > x.max():
            raise ValueError("Altitude out of bounds")

        return np.interp(altitude, x, y)
