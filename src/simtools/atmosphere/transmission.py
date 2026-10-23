"""Atmospheric transmission along a slant path."""

import astropy.units as u
import numpy as np


class AtmosphericTransmission:
    """
    Transmission of light between two altitudes along a slant path.

    Uses an atmospheric transmission table of the simulation models: per
    wavelength, the vertical optical depth (``extinction``) from each altitude
    down to the observatory level (``observatory_level`` in the table metadata,
    depth zero). Depths are interpolated linearly in log(altitude) over the
    altitudes tabulated for a wavelength and clamped beyond them; a wavelength
    uses the tabulated wavelength at or below it.

    Parameters
    ----------
    table: astropy.table.Table
        Atmospheric transmission table (``wavelength``, ``altitude``, ``extinction``).
    """

    def __init__(self, table):
        self.observation_level_m = u.Quantity(table.meta["observatory_level"]).to_value(u.m)
        wavelength = u.Quantity(table["wavelength"], u.nm).to_value(u.nm)
        altitude = u.Quantity(table["altitude"], u.km).to_value(u.m)
        depth = np.asarray(table["extinction"], dtype=float)
        self._wavelength = np.unique(wavelength)
        # Per wavelength the altitudes tabulated for it (some list fewer).
        self._profiles = []
        for value in self._wavelength:
            selected = wavelength == value
            order = np.argsort(altitude[selected])
            self._profiles.append(
                (
                    np.log(np.concatenate([[self.observation_level_m], altitude[selected][order]])),
                    np.concatenate([[0.0], depth[selected][order]]),
                )
            )

    def optical_depth(self, wavelength_nm, altitude_m):
        """Vertical optical depth between the observation level and an altitude a.s.l. [m]."""
        row = np.searchsorted(self._wavelength, wavelength_nm, side="right") - 1
        log_altitude, depth = self._profiles[int(np.clip(row, 0, len(self._wavelength) - 1))]
        return np.interp(np.log(np.asarray(altitude_m, dtype=float)), log_altitude, depth)

    def transmission(self, wavelength_nm, emission_altitude_m, telescope_altitude_m, airmass):
        """
        Transmission from the emission point to the telescope along a slant path.

        The vertical optical depth between the two altitudes times the airmass,
        without instrument terms. Emission below the observation level is not
        transmitted.
        """
        depth = np.abs(
            self.optical_depth(wavelength_nm, emission_altitude_m)
            - self.optical_depth(wavelength_nm, telescope_altitude_m)
        )
        transmitted = np.exp(-depth * np.asarray(airmass))
        return np.where(
            np.asarray(emission_altitude_m) < self.observation_level_m, 0.0, transmitted
        )
