"""Atmospheric transmission along a slant path, as in sim_telarray."""

from pathlib import Path

import numpy as np


class AtmosphericTransmission:
    """
    Vertical optical depths of a sim_telarray atmospheric transmission table.

    The table lists, per integer wavelength, the optical depth from each
    altitude down to the observation level (the first level, depth zero).
    Depths are interpolated linearly in log10(altitude) and clamped at the
    table edges, as in sim_telarray (sim_absorb.c).

    Parameters
    ----------
    path: str or Path
        sim_telarray atmospheric transmission table.
    """

    def __init__(self, path):
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("# H2="))
        observation_km, _, levels = lines[first][len("# H2=") :].partition("H1=")
        observation_km = float(observation_km.strip().rstrip(","))
        self.observation_level_m = 1000.0 * observation_km
        levels_km = [observation_km] + [float(v) for v in levels.split("#")[0].split()]
        self._log_altitude = np.log10(np.array(levels_km) * 1e5)  # [cm]
        rows = [
            [float(v) for v in line.split("#")[0].split()]
            for line in lines[first + 1 :]
            if line.split("#")[0].split() and int(float(line.split()[0])) != 0
        ]
        rows = [row for row in rows if len(row) == len(levels_km)]
        self._wavelength = np.array([int(row[0]) for row in rows])
        self._depth = np.column_stack([np.zeros(len(rows)), np.array([row[1:] for row in rows])])

    def optical_depth(self, wavelength_nm, altitude_m):
        """Vertical optical depth between the observation level and an altitude a.s.l."""
        row = np.clip(int(wavelength_nm), self._wavelength[0], self._wavelength[-1])
        depth = self._depth[row - self._wavelength[0]]
        return np.interp(np.log10(100.0 * np.asarray(altitude_m)), self._log_altitude, depth)

    def transmission(self, wavelength_nm, emission_altitude_m, telescope_altitude_m, airmass):
        """
        Transmission from the emission point to the telescope along a slant path.

        sim_telarray's formula without the instrument term. Emission below the
        observation level is not transmitted.
        """
        depth = np.abs(
            self.optical_depth(wavelength_nm, emission_altitude_m)
            - self.optical_depth(wavelength_nm, telescope_altitude_m)
        )
        transmitted = np.exp(-depth * np.asarray(airmass))
        return np.where(
            np.asarray(emission_altitude_m) < self.observation_level_m, 0.0, transmitted
        )
