import functools
import logging
import operator
import time
from pathlib import Path

import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from eventio import IACTFile

from simtools.util.general import convert_2D_to_radial_distr


class CorsikaOutput:
    """CorsikaOutput reads the CORSIKA output file (IACT file) of a simulation and save the
    information about the Chernekov photons. It relies on pyeventio.

    Parameters
    ----------
    input_file: str or Path
        Input file (IACT file) provided by the CORSIKA simulation.

    Raises
    ------
    FileNotFoundError:
        if the input file given does not exist.
    """

    def __init__(self, input_file):

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaOutput")
        self.input_file = input_file

        if not isinstance(self.input_file, Path):
            self.input_file = Path(self.input_file)
        if not self.input_file.exists():
            raise FileNotFoundError

        self.tel_positions = None

    def create_histogram(self, telescope_index=None):
        """
        Extract the information of the Cherenkov photons from a CORSIKA output IACT file and save it
         into the class instance.

        Parameters
        ----------
        telescope_index: int or list of int
            The index of the specific telescope to plot the data. If not specified, all telescopes
            are considered. More telescopes are also allowed.

        Returns
        -------
        histogram: instance of boost_histogram.Histogram if telescope_index=None or list of
        boost_histogram.Histogram if telescope_index is not None.

        Raises
        ------
        TypeError:
            if the index or indices passed through telescope_index are not of type int.
        IndexError:
            If the index or indices passed though telescope_index are out of range.
        """

        if telescope_index is not None:
            if not isinstance(telescope_index, list):
                telescope_index = [telescope_index]
            for one_telescope in telescope_index:
                if not isinstance(one_telescope, int):
                    msg = "The index or indices given are not of type int."
                    self._logger.error(msg)
                    raise TypeError

        if telescope_index is None:
            xy_maximum = 1000
            self.hist = bh.Histogram(
                bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                bh.axis.Regular(bins=100, start=200, stop=1000),
            )
        else:
            xy_maximum = 15
            self.hist = [
                bh.Histogram(
                    bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=100, start=200, stop=1000),
                )
                * len(telescope_index)
            ]

        self.num_photon_bunches_per_event = []
        start_time = time.time()
        self._logger.debug("Starting reading the file at {}.".format(start_time))
        with IACTFile(self.input_file) as f:

            if self.tel_positions is None:
                self.tel_positions = np.array(f.telescope_positions)

            for event in f:

                num_photons_per_event_per_telescope = 0
                for step, telescope_now in enumerate(self.tel_positions):
                    num_photons_per_event_per_telescope += np.sum(
                        event.photon_bunches[step]["photons"]
                    )
                self.num_photon_bunches_per_event = np.append(
                    self.num_photon_bunches_per_event, num_photons_per_event_per_telescope
                )

                photons = list(event.photon_bunches.values())
                if telescope_index is None:

                    for onetel_position, photons_rel_position in zip(self.tel_positions, photons):

                        x_one_photon = -onetel_position["x"] + photons_rel_position["x"]
                        y_one_photon = -onetel_position["y"] + photons_rel_position["y"]
                        # photons_rel_position["cx"], photons_rel_position["cy"],
                        # photons_rel_position["zem"], photons_rel_position["time"]

                        self.hist.fill(
                            (np.array(x_one_photon) * u.cm).to(u.m),
                            (np.array(y_one_photon) * u.cm).to(u.m),
                            np.abs(photons_rel_position["wavelength"]) * u.nm,
                        )
                else:
                    for step, one_telescope in enumerate(telescope_index):
                        try:
                            # will need one hist for each telescope
                            self.hist[step].fill(
                                (np.array(photons[one_telescope]["x"]) * u.cm).to(u.m),
                                (np.array(photons[one_telescope]["y"]) * u.cm).to(u.m),
                                np.abs(photons[one_telescope]["wavelength"]) * u.nm,
                            )
                        except IndexError:
                            msg = (
                                "Index {} is out of range. There are only {} telescopes in the "
                                "array.".format(one_telescope, len(self.tel_positions))
                            )
                            self._logger.error(msg)
                            raise

        self._logger.debug(
            "Finished reading the file and creating the histogram in {} seconds".format(
                time.time() - start_time
            )
        )
        return self.hist

    def get_2D_position_distr(self, density=True):
        """
        Gets a 2D histogram of position of the Cherenkov photons on the ground.

        Parameters
        ----------
        density: bool
            If True, returns the density distribution. If False, returns the distribution of counts.

        Returns
        -------
        3-tuple of numpy.array
            The edges of the histogram in X, Y and the matrix with the counts

        Raises
        ------
        TypeError:
            if density is not of type bool.
        """
        mini = self.hist[:, :, sum]
        if density is True:
            areas = functools.reduce(operator.mul, mini.axes.widths)
            return mini.axes.edges[0].flatten(), mini.axes.edges[1].flatten(), mini.view().T / areas
        elif density is False:
            return mini.axes.edges[0].flatten(), mini.axes.edges[1].flatten(), mini.view().T
        else:
            msg = "density has to be of type bool."
            self._logger.error(msg)
            raise TypeError

    def get_radial_distr(self, bin_size=40, max_dist=1000, density=True):
        """
        Gets the radial distribution of the photons on the ground in relation to the center of the
        array.
        Parameters
        ----------
        bin_size: float
            Size of the step in distance (in meters).
        max_dist: float
            Maximum distance to consider in the 1D histogram (in meters).
        density: bool
            If True, returns the density distribution. If False, returns the distribution of counts.

        Returns
        -------
        np.array
            The edges of the 1D histogram in meters with size = int(max_dist/bin_size) + 1.
        np.array
            The values of the 1D histogram with size = int(max_dist/bin_size).
        """

        x, y, hist2D = self.get_2D_position_distr(density=density)
        edges, hist1D = convert_2D_to_radial_distr(
            x, y, hist2D, bin_size=bin_size, max_dist=max_dist
        )
        return edges, hist1D

    def get_wavelength_distr(self):
        """
        Gets a histogram with the wavelength of the photon bunches.
        """
        mini = self.hist[sum, sum, :]
        return mini.axes.edges.T.flatten()[0], mini.view().T

    def get_num_photon_bunches(self):
        """
        Gets the number of photon bunches per event.
        """

    def get_telescope_positions(self):
        """
        Gets the telescope positions.

        Returns
        -------
        numpy.ndarray
            X and Y positions of the telescopes (the centers of the CORSIKA spheres).
        """
        return self.tel_positions

    def plot_2D_on_ground(self, density):
        """
        Plots the histogram of the photon positions on the ground.

        Parameters
        ----------
        density: bool
            If True, returns the density distribution. If False, returns the distribution of counts.
        """
        x, y, hist = self.get_2D_position_distr(density=density)
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(x, y, hist)
        fig.colorbar(mesh)
        fig.savefig("boost_histogram_dens.png")

    def plot_wavelength_distr(self):
        """
        Plots the 1D distribution of the photon wavelengths
        """
        wavelength, hist = self.get_wavelength_distr()
        fig, ax = plt.subplots()
        ax.bar(wavelength[:-1], hist, align="edge", width=np.diff(wavelength))
        fig.savefig("boost_histogram_1D.png")

    def plot1D_on_ground(self, radial_edges, histogram_1D):
        """
        Plots the 1D distribution, i.e. the radial distribution, of the photons on the ground.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(radial_edges[:-1], histogram_1D, xerr=int(np.diff(radial_edges)[0] / 2), ls="")
        # ax.scatter(distance_sorted,hist_sorted, alpha=0.5, c='r')
        fig.savefig("test.png")

    # Reformulate
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
