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

from simtools.util.general import convert_2D_to_radial_distr, rotate


class HistogramNotCreated(Exception):
    """Exception for histogram not created."""


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
        self.telescope_indices = None

    def _set_telescope_indices(self, telescope_indices):
        """
        Set the telescope index (or indices) as the class attribute.

        Parameters
        ----------
        telescope_index: int or list of int
            The index of the specific telescopes to be inspected. If not specified, all telescopes
            are treated together in one histogram.

        Raises
        ------
        TypeError:
            if the index or indices passed through telescope_index are not of type int.
        """
        if telescope_indices is not None:
            if not isinstance(telescope_indices, list):
                telescope_indices = [telescope_indices]
            for one_telescope in telescope_indices:
                if not isinstance(one_telescope, int):
                    msg = "The index or indices given are not of type int."
                    self._logger.error(msg)
                    raise TypeError
        self.telescope_indices = telescope_indices

    def _create_histograms(self):
        """
        Create the histogram instances based on the given telescope indices.
        """
        if self.telescope_indices is None:
            xy_maximum = 1000
            self.hist = [
                bh.Histogram(
                    bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=100, start=200, stop=1000),
                )
            ]
        else:
            xy_maximum = 15
            self.hist = [
                bh.Histogram(
                    bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=100, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=100, start=200, stop=1000),
                )
            ] * len(self.telescope_indices)

        np.array(self.hist)

    @u.quantity_input(rotation_angle=u.rad)
    def _fill_histogram(self, onetel_position, photons_rel_position, rotation_angle):
        """Fill the histogram created by self._create_histogram

        Parameters
        ----------
        onetel_position: numpy.record
            Wrapped structured 4-tuple with X, Y, Z positions and the radius of the CORSIKA
             representation of the telescope (in cm).

        photons_rel_position: numpy.ndarray
            Array with the following information of the Chernekov photons on the ground: x, y, cx,
            cy, time, zem, photons, wavelength.

        rotation_angle:

        Raises
        ------
        IndexError:
            If the index or indices passed though telescope_index are out of range.
        """

        photon_rel_position_rotated_x, photon_rel_position_rotated_y = rotate(
            photons_rel_position["x"],
            photons_rel_position["y"],
            (-4.533 * u.deg).to(u.rad),
            rotation_angle,
        )

        if self.telescope_indices is None:
            self.hist[0].fill(
                ((-onetel_position["x"] + photon_rel_position_rotated_x) * u.cm).to(u.m),
                ((-onetel_position["y"] + photon_rel_position_rotated_y) * u.cm).to(u.m),
                np.abs(photons_rel_position["wavelength"]) * u.nm,
            )
        else:

            for step, one_telescope in enumerate(self.telescope_indices):
                try:
                    self.hist[step].fill(
                        (photon_rel_position_rotated_x * u.cm).to(u.m),
                        (photon_rel_position_rotated_y * u.cm).to(u.m),
                        np.abs(photons_rel_position[one_telescope]["wavelength"]) * u.nm,
                    )
                except IndexError:
                    msg = (
                        "Index {} is out of range. There are only {} telescopes in the "
                        "array.".format(one_telescope, len(self.tel_positions))
                    )
                    self._logger.error(msg)

    def set_histograms(self, telescope_indices=None, rotation_angle=0 * u.rad):
        """
        Extract the information of the Cherenkov photons from a CORSIKA output IACT file and creates
        a histogram

        Parameters
        ----------
        telescope_indices: int or list of int
            The index of the specific telescopes to be inspected. If not specified, all telescopes
            are treated together in one histogram.
        rotation_angle: astropy.units.rad
            Angle to rotate the observation plane in radians. It allows one to compensate for the
            zenith angle of observations and get the photon distribution in the plane of the
            telescope cameras.

        Returns
        -------
        numpy.array: array of instances of boost_histogram.Histogram.

        """

        self._set_telescope_indices(telescope_indices)
        self._create_histograms()

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
                for onetel_position, photons_rel_position in zip(self.tel_positions, photons):

                    # photons_rel_position["cx"], photons_rel_position["cy"],
                    # photons_rel_position["zem"], photons_rel_position["time"]
                    self._fill_histogram(onetel_position, photons_rel_position, rotation_angle)

        self._logger.debug(
            "Finished reading the file and creating the histogram in {} seconds".format(
                time.time() - start_time
            )
        )

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
            The edges of the histograms in X, Y and the matrices with the counts

        Raises
        ------
        TypeError:
            if density is not of type bool.
        HistogramNotCreated:
            if the histogram was not previously created.
        """
        try:
            mini_hist = []
            if self.telescope_indices is None:
                include_all = [0]
            else:
                include_all = self.telescope_indices
            for step, _ in enumerate(include_all):
                mini_hist.append(self.hist[step][:, :, sum])
        except AttributeError:
            msg = (
                "The histogram(s) was(were) not created. Please, use `create_histograms` to create "
                "histograms from the CORSIKA output file."
            )
            self._logger.error(msg)
            raise HistogramNotCreated

        x_edges, y_edges, hist_values = [], [], []
        for step, _ in enumerate(mini_hist):
            x_edges.append(mini_hist[step].axes.edges[0].flatten())
            y_edges.append(mini_hist[step].axes.edges[1].flatten())
            if density is True:
                areas = functools.reduce(operator.mul, mini_hist[step].axes.widths)
                hist_values.append(mini_hist[step].view().T / areas)
            elif density is False:
                hist_values.append(mini_hist[step].view().T)
            else:
                msg = "`density` parameter has to be of type bool."
                self._logger.error(msg)
                raise TypeError
        return np.array(x_edges), np.array(y_edges), np.array(hist_values)

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
        edges_1D_list, hist1D_list = [], []
        x_edges_list, y_edges_list, hist2D_values_list = self.get_2D_position_distr(density=density)
        for step, _ in enumerate(x_edges_list):
            edges_1D, hist1D = convert_2D_to_radial_distr(
                x_edges_list[step],
                y_edges_list[step],
                hist2D_values_list[step],
                bin_size=bin_size,
                max_dist=max_dist,
            )
            edges_1D_list.append(edges_1D)
            hist1D_list.append(hist1D)
        return np.array(edges_1D_list), np.array(hist1D_list)

    def get_wavelength_distr(self):
        """
        Gets histograms with the wavelengths of the photon bunches.
        """
        x_edges_list, hist_1D_list = [], []
        for step, _ in enumerate(self.hist):
            mini_hist = self.hist[step][sum, sum, :]
            x_edges_list.append(mini_hist.axes.edges.T.flatten()[0])
            hist_1D_list.append(mini_hist.view().T)
        return np.array(x_edges_list), np.array(hist_1D_list)

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
        x_edges, y_edges, hist_values = self.get_2D_position_distr(density=density)
        for step, _ in enumerate(x_edges):
            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x_edges[step], y_edges[step], hist_values[step])
            fig.colorbar(mesh)
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_dens_all_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_dens_tel_" + str(self.telescope_indices[step]) + ".png"
                )

    def plot_wavelength_distr(self):
        """
        Plots the 1D distribution of the photon wavelengths
        """
        wavelengths, hist_values = self.get_wavelength_distr()
        for step, _ in enumerate(wavelengths):
            fig, ax = plt.subplots()
            ax.bar(
                wavelengths[step][:-1],
                hist_values[step],
                align="edge",
                width=np.diff(wavelengths[step]),
            )
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_wavelength_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_wavelength_tel_" + str(self.telescope_indices[step]) + ".png"
                )

    def plot1D_on_ground(self, radial_edges, histograms_1D):
        """
        Plots the 1D distribution, i.e. the radial distribution, of the photons on the ground.
        """
        for step, _ in enumerate(radial_edges):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(
                radial_edges[step][:-1],
                histograms_1D[step],
                xerr=int(np.diff(radial_edges[step])[0] / 2),
                ls="",
            )
            # ax.scatter(distance_sorted,hist_sorted, alpha=0.5, c='r')
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_1D_pos_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_1D_pos_tel_" + str(self.telescope_indices[step]) + ".png"
                )

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
