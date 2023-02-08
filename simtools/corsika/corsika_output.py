import logging
from pathlib import Path

import numpy as np
from astropy import units as u
from eventio import IACTFile


class CorsikaOutput:
    """CorsikaOutput reads the CORSIKA output file (IACT file) of a simulation and save the
    information about the Chernekov photons. It relies on pyeventio.

    Parameters
    ----------
    input_file: str or Path
        Input file (IACT file) provided by the CORSIKA simulation.
    """

    def __init__(self, input_file):

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaOutput")
        self.input_file = input_file
        self.tel_positions = None
        self._extract_information()

    def _extract_information(self):
        """
        Extract the information of the Cherenkov photons from a CORSIKA output IACT file and save it
         into the class instance.

        Raises
        ------
        FileNotFoundError:
            if the input file given does not exist.
        """

        if not isinstance(self.input_file, Path):
            self.input_file = Path(self.input_file)
        if not self.input_file.exists():
            raise FileNotFoundError

        x_photon_positions = []
        y_photon_positions = []
        x_cos = []
        y_cos = []
        wave_tot = []
        z_photon_emission = []
        num_photons_tot = []
        time_since_first_interatcion = []
        with IACTFile(self.input_file) as f:
            for event in f:

                photons = list(event.photon_bunches.values())
                if self.tel_positions is None:
                    self.tel_positions = np.array(f.telescope_positions)
                for onetel_position, photons_rel_position in zip(self.tel_positions, photons):
                    x_one_photon = -onetel_position["x"] + photons_rel_position["x"]
                    y_one_photon = -onetel_position["y"] + photons_rel_position["y"]
                    x_photon_positions.append(x_one_photon)
                    y_photon_positions.append(y_one_photon)
                    x_cos.append(photons_rel_position["cx"])
                    y_cos.append(photons_rel_position["cy"])
                    wave_tot.append(photons_rel_position["wavelength"])
                    z_photon_emission.append(photons_rel_position["zem"])
                    time_since_first_interatcion.append(photons_rel_position["time"])

                num_photons_tot.append(np.sum(event.photon_bunches[0]["photons"]))

        self.wave_tot = np.abs(np.concatenate(np.array(wave_tot)).flatten()) * u.nm

        self.x_photon_positions = (
            np.concatenate(np.array(x_photon_positions)).flatten() * u.cm
        ).to(u.m)
        self.y_photon_positions = (
            np.concatenate(np.array(y_photon_positions)).flatten() * u.cm
        ).to(u.m)
        self.x_cos = np.concatenate(np.array(x_cos)).flatten()
        self.y_cos = np.concatenate(np.array(y_cos)).flatten()
        self.z_photon_emission = (np.concatenate(np.array(z_photon_emission)).flatten() * u.cm).to(
            u.m
        )
        self.time_since_first_interaction = (
            np.concatenate(np.array(time_since_first_interatcion)).flatten() * u.ns
        )
        self.distance = np.sqrt(self.x_photon_positions**2 + self.y_photon_positions**2)
        self.num_photons_tot = np.array(num_photons_tot, dtype=float)

    def get_number_of_photons(self):
        """
        Gets the number of Cherenkov photons on the ground per event.

        Returns
        -------
        numpy.array
            Total number of Cherenkov photons for each gamma-ray/cosmic-ray event.
        """
        return self.num_photons_tot

    def get_wavelength(self):
        """
        Gets the wavelength distribution of the Cherenkov photons on the ground.

        Returns
        -------
        numpy.array
            Wavelength distribution of the Cherenkov photons on the ground.
        """
        return self.wave_tot

    def get_photon_positions(self):
        """
        Gets the Cherenkov photon positions on the ground.

        Returns
        -------
        2-tuple of numpy.array
            X and Y positions of the Cherenkov photons on the ground.
        """
        return self.x_photon_positions, self.y_photon_positions

    def get_incoming_direction(self):
        """
        Gets the Cherenkov photon incoming direction.

        Returns
        -------
        2-tuple of numpy.array
            Cosinus of the angles between the incoming Cherenkov photons and the X and Y axes,
            respectively.
        """

        return self.y_cos, self.x_cos

    def get_height(self):
        """
        Gets the Cherenkov photon emission height.

        Returns
        -------
        numpy.array
            Height of the Cherenkov photons in meters.
        """
        return self.z_photon_emission

    def get_time(self):
        """
        Gets the Cherenkov time of arrival of the Cherenkov photons since first interaction (in s).

        Returns
        -------
        numpy.array
            Time of arrival of the photons since first interaction in seconds.
        """
        return self.time_since_first_interaction

    def get_distance(self):
        """
        Gets the distance of the Cherenkov photons on the ground to the array center in meters.

        Returns
        -------
        numpy.array
            The distance of the Cherenkov photons to the array center.
        """
        return self.distance

    def get_telescope_positions(self):
        """
        Gets the telescope positions.

        Returns
        -------
        numpy.ndarray
            X and Y positions of the telescopes (the centers of the CORSIKA spheres).
        """
        return self.tel_positions
