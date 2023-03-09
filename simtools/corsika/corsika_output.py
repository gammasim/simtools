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
    zenith_angle: astropy.units.rad
        Zenith angle (in radians) of the observations (in the CORSIKA coordinate system).
        It is used to rotate the observation plane (in zenith) of the telescope and to plot
        the sky map of the incoming direction of photons.
    azimuth_angle: astropy.units.rad
        Azimuth angle of observation (in radians).
        See above for more details.
    Raises
    ------
    FileNotFoundError:
        if the input file given does not exist.
    """

    @u.quantity_input(zenith_angle=u.rad, azimuth_angle=u.rad)
    def __init__(self, input_file, zenith_angle=0 * u.rad, azimuth_angle=0 * u.rad):

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaOutput")
        self.input_file = input_file
        self.zenith_angle = zenith_angle
        self.azimuth_angle = azimuth_angle

        if not isinstance(self.input_file, Path):
            self.input_file = Path(self.input_file)
        if not self.input_file.exists():
            raise FileNotFoundError

        self.tel_positions = None
        self.telescope_indices = None

    def _set_telescope_indices(self, telescope_indices):
        """
        Set the telescope index (or indices) as a class attribute.

        Parameters
        ----------
        telescope_indices: int or list of int
            The indices of the specific telescopes to be inspected. If not specified, all telescopes
            are treated together in one histogram.

        Raises
        ------
        TypeError:
            if the indices passed through telescope_index are not of type int.
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

    def _get_directive_cosinus(self):
        """
        Get the direction cosinus (X and Y) from the incoming particle, which helps defining the
        range of the histograms built from the photon incoming directions.
        """
        cosx_obs = np.sin(self.zenith_angle) * np.cos(self.azimuth_angle)
        cosy_obs = np.sin(self.zenith_angle) * np.sin(self.azimuth_angle)
        return cosx_obs, cosy_obs

    def _create_histograms(self, bin_size=None, xy_maximum=None):
        """
        Create the histogram instances based on the given telescope indices.
        """
        if bin_size is None:
            bin_size = 100

        cosx_obs, cosy_obs = self._get_directive_cosinus()

        if self.telescope_indices is None:
            if xy_maximum is None:
                xy_maximum = 1000
            self.hist_position = [
                bh.Histogram(
                    bh.axis.Regular(bins=bin_size, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=bin_size, start=-xy_maximum, stop=xy_maximum),
                    bh.axis.Regular(bins=bin_size, start=200, stop=100),
                )
            ]

            self.hist_direction = [
                bh.Histogram(
                    bh.axis.Regular(bins=bin_size, start=cosx_obs - 0.1, stop=cosx_obs + 0.1),
                    bh.axis.Regular(bins=bin_size, start=cosy_obs - 0.1, stop=cosy_obs + 0.1),
                )
            ]

            self.hist_time_altitude = [
                bh.Histogram(
                    bh.axis.Regular(bins=bin_size, start=0, stop=500),
                    bh.axis.Regular(bins=bin_size, start=15, stop=0),
                )
            ]
        else:
            if xy_maximum is None:
                xy_maximum = 15

            self.hist_position, self.hist_direction, self.hist_time_altitude = [], [], []
            for step, _ in enumerate(self.telescope_indices):
                self.hist_position.append(
                    bh.Histogram(
                        bh.axis.Regular(bins=bin_size, start=-xy_maximum, stop=xy_maximum),
                        bh.axis.Regular(bins=bin_size, start=-xy_maximum, stop=xy_maximum),
                        bh.axis.Regular(bins=bin_size, start=200, stop=1000),
                    )
                )

                self.hist_direction.append(
                    bh.Histogram(
                        bh.axis.Regular(bins=bin_size, start=cosx_obs - 0.1, stop=cosx_obs + 0.1),
                        bh.axis.Regular(bins=bin_size, start=cosy_obs - 0.1, stop=cosy_obs + 0.1),
                    )
                )

                self.hist_time_altitude.append(
                    bh.Histogram(
                        bh.axis.Regular(bins=bin_size, start=0, stop=500),
                        bh.axis.Regular(bins=bin_size, start=15, stop=0),
                    )
                )

    def _fill_histograms(self, photons):
        """Fill the histograms created by self._create_histogram

        Parameters
        ----------
        photons: list
            List of size M of numpy.void of size (N,8), where M is the number of telescopes in the
            array, N is the number of photons that reached each telescope. The following information
             of the Cherenkov photons on the ground are saved: x, y, cx, cy, time, zem, photons,
             wavelength.

        Raises
        ------
        IndexError:
            If the index or indices passed though telescope_index are out of range.
        """

        for one_tel_info, photons_info in zip(self.tel_positions, photons):

            if self.telescope_indices is None:
                photon_x, photon_y = rotate(
                    photons_info["x"],
                    photons_info["y"],
                    self.azimuth_angle,
                    self.zenith_angle,
                )
                self.hist_position[0].fill(
                    ((-one_tel_info["x"] + photon_x) * u.cm).to(u.m),
                    ((-one_tel_info["y"] + photon_y) * u.cm).to(u.m),
                    np.abs(photons_info["wavelength"]) * u.nm,
                )
                self.hist_direction[0].fill(photons_info["cx"], photons_info["cy"])
                self.hist_time_altitude[0].fill(
                    photons_info["time"] * u.ns, (photons_info["zem"] * u.cm).to(u.km)
                )

            else:
                photon_x, photon_y = photons_info["x"], photons_info["y"]
                for step, one_index in enumerate(self.telescope_indices):
                    try:
                        if (
                            one_tel_info["x"]
                            == self.tel_positions[self.telescope_indices[step]]["x"]
                            and one_tel_info["y"]
                            == self.tel_positions[self.telescope_indices[step]]["y"]
                        ):

                            self.hist_position[step].fill(
                                (photon_x * u.cm).to(u.m),
                                (photon_y * u.cm).to(u.m),
                                np.abs(photons_info["wavelength"]) * u.nm,
                            )

                            self.hist_direction[step].fill(photons_info["cx"], photons_info["cy"])
                            self.hist_time_altitude[step].fill(
                                photons_info["time"] * u.ns, (photons_info["zem"] * u.cm).to(u.km)
                            )

                    except IndexError:
                        msg = (
                            "Index {} is out of range. There are only {} telescopes in the "
                            "array.".format(one_index, len(self.tel_positions))
                        )
                        self._logger.error(msg)

    def set_histograms(self, telescope_indices=None):
        """
        Extract the information of the Cherenkov photons from a CORSIKA output IACT file, create
         and fill the histograms

        Parameters
        ----------
        telescope_indices: int or list of int
            The indices of the specific telescopes to be inspected. If not specified, all telescopes
            are treated together in one histogram.

        Returns
        -------
        list: list of boost_histogram.Histogram instances.

        """

        self._set_telescope_indices(telescope_indices)
        self._create_histograms()

        self.num_photons_per_event = []
        start_time = time.time()
        self._logger.debug("Starting reading the file at {}.".format(start_time))
        with IACTFile(self.input_file) as f:

            if self.tel_positions is None:
                self.tel_positions = np.array(f.telescope_positions)
            for event in f:
                """print("event.count")
                print(event.count)
                print("event.event_number")
                print(event.event_number)
                print("event.header")
                print(event.header)
                print("event.impact_x")
                print(event.impact_x)
                print("event.impact_y")
                print(event.impact_y)
                print("event.n_bunches")
                print(event.n_bunches)
                print("event.n_photons")
                print(event.n_photons)
                print("event.particles")
                print(event.particles)
                print("event.photon_bunches")
                print(event.photon_bunches)"""

                num_photons_partial_sum = 0
                for step, _ in enumerate(self.tel_positions):
                    num_photons_partial_sum += np.sum(event.n_photons[step])
                self.num_photons_per_event = np.append(
                    self.num_photons_per_event, num_photons_partial_sum
                )
                photons = list(event.photon_bunches.values())
                self._fill_histograms(photons)
                # photons_rel_position["zem"], photons_rel_position["time"]

        self._logger.debug(
            "Finished reading the file and creating the histogram in {} seconds".format(
                time.time() - start_time
            )
        )
        print(np.mean(self.num_photons_per_event), np.std(self.num_photons_per_event))

    def get_2D_position_distr(self, density=True):
        """
        Get 2D histograms of position of the Cherenkov photons on the ground.

        Parameters
        ----------
        density: bool
            If True, returns the density distribution. If False, returns the distribution of counts.

        Returns
        -------
        numpy.array
            The edges of the direction histograms in X.
        numpy.array
            The edges of the direction histograms in Y
        numpy.ndarray
            The values (counts) of the histogram.

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
                mini_hist.append(self.hist_position[step][:, :, sum])
        except AttributeError:
            msg = (
                "The histograms were not created. Please, use `create_histograms` to create "
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

    def get_radial_distr(self, bin_size=None, max_dist=None, density=True):
        """
        Get the radial distribution of the photons on the ground in relation to the center of the
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
        if self.telescope_indices is None:
            if bin_size is None:
                bin_size = 40
            if max_dist is None:
                max_dist = 100
        else:
            if bin_size is None:
                bin_size = 4
            if max_dist is None:
                max_dist = 10
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
        Get histograms with the wavelengths of the photon bunches.

        Returns
        -------
        np.array
            The edges of the wavelength histogram in nanometers.
        np.array
            The values of the wavelength histogram.
        """
        x_edges_list, hist_1D_list = [], []
        for step, _ in enumerate(self.hist_position):
            mini_hist = self.hist_position[step][sum, sum, :]
            x_edges_list.append(mini_hist.axes.edges.T.flatten()[0])
            hist_1D_list.append(mini_hist.view().T)
        return np.array(x_edges_list), np.array(hist_1D_list)

    def get_2D_direction_distr(self):
        """
        Get 2D histograms of incoming direction of the Cherenkov photons on the ground.

        Returns
        -------
        numpy.array
            The edges of the direction histograms in cos(X).
        numpy.array
            The edges of the direction histograms in cos(Y)
        numpy.ndarray
            The values (counts) of the histogram.
        """
        x_edges, y_edges, mini_hist = [], [], []
        if self.telescope_indices is None:
            size = 1
        else:
            size = len(self.telescope_indices)
        for step in range(size):
            x_edges.append(self.hist_direction[step].axes.edges[0].flatten())
            y_edges.append(self.hist_direction[step].axes.edges[1].flatten())
            mini_hist.append(self.hist_direction[step].view().T)
        return np.array(x_edges), np.array(y_edges), np.array(mini_hist)

    def get_2D_time_altitude(self):
        """
        Get 2D histograms of the time and altitude of the photon production.

        Returns
        -------
        numpy.array
            The edges of the direction histograms in ns.
        numpy.array
            The edges of the direction histograms in km.
        numpy.ndarray
            The values (counts) of the histogram.
        """
        x_edges, y_edges, mini_hist = [], [], []
        if self.telescope_indices is None:
            size = 1
        else:
            size = len(self.telescope_indices)
        for step in range(size):
            x_edges.append(self.hist_time_altitude[step].axes.edges[0].flatten())
            y_edges.append(self.hist_time_altitude[step].axes.edges[1].flatten())
            mini_hist.append(self.hist_time_altitude[step].view().T)
        return np.array(x_edges), np.array(y_edges), np.array(mini_hist)

    def get_time_distr(self):
        """
        Get the distribution of the emitted time of the Cherenkov photons. The start of the time
        is given according to the CORSIKA configuration.

        Returns
        -------
        numpy.array
            The edges of the direction histograms in ns.
        numpy.ndarray
            The values (counts) of the histogram.
        """
        x_edges_list, hist_1D_list = [], []
        for step, _ in enumerate(self.hist_time_altitude):
            mini_hist = self.hist_time_altitude[step][:, sum]
            x_edges_list.append(mini_hist.axes.edges.T.flatten()[0])
            hist_1D_list.append(mini_hist.view().T)
        return np.array(x_edges_list), np.array(hist_1D_list)

    def get_altitude_distr(self):
        """
        Get the emission altitude of the Cherenkov photons.

        Returns
        -------
        numpy.array
            The edges of the direction histograms in km.
        numpy.ndarray
            The values (counts) of the histogram.
        """
        x_edges_list, hist_1D_list = [], []
        for step, _ in enumerate(self.hist_time_altitude):
            mini_hist = self.hist_time_altitude[step][sum, :]
            x_edges_list.append(mini_hist.axes.edges.T.flatten()[0])
            print(mini_hist.axes.edges.T.flatten()[0])
            print(mini_hist.view().T)
            hist_1D_list.append(mini_hist.view().T)

        return np.array(x_edges_list), np.array(hist_1D_list)

    def get_num_photons_per_event(self):
        """
        Get the number of photon bunches per event.
        """
        return self.num_photons_per_event

    def get_total_num_photons(self):
        """
        Get the total number of photon bunches.
        """
        return np.sum(self.get_num_photons_per_event())

    def get_telescope_positions(self):
        """
        Get the telescope positions.

        Returns
        -------
        numpy.ndarray
            X, Y and Z positions of the telescopes and their radius according to the CORSIKA
            spherical representation of the telescopes.
        """
        return self.tel_positions

    def plot_2D_on_ground(self, density):
        """
        Plot the 2D histogram of the photon positions on the ground.

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

    def plot_2D_direction(self):
        """
        Plot the 2D histogram of the incoming direction of photons.
        """
        x_edges, y_edges, hist_values = self.get_2D_direction_distr()
        for step, _ in enumerate(x_edges):
            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x_edges[step], y_edges[step], hist_values[step])
            fig.colorbar(mesh)
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_direction_all_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_direction_tel_" + str(self.telescope_indices[step]) + ".png"
                )

    def plot_2D_time_altitude(self):
        """
        Plot the 2D histogram of the time and altitude where the photon was produced.
        """
        x_edges, y_edges, hist_values = self.get_2D_time_altitude()
        for step, _ in enumerate(x_edges):
            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x_edges[step], y_edges[step], hist_values[step])
            fig.colorbar(mesh)
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_time_altitude_all_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_time_altitude_tel_"
                    + str(self.telescope_indices[step])
                    + ".png"
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
                xerr=np.abs(np.diff(radial_edges[step])[0] / 2),
                ls="",
            )
            # ax.scatter(distance_sorted,hist_sorted, alpha=0.5, c='r')
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_1D_pos_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_1D_pos_tel_" + str(self.telescope_indices[step]) + ".png"
                )

    def plot_time_distr(self, time_edges, histograms_1D):
        """
        Plots the 1D distribution, i.e. the radial distribution, of the photons on the ground.
        """
        for step, _ in enumerate(time_edges):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(
                time_edges[step][:-1],
                histograms_1D[step],
                xerr=np.abs(np.diff(time_edges[step])[0] / 2),
                ls="",
            )
            # ax.scatter(distance_sorted,hist_sorted, alpha=0.5, c='r')
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_1D_time_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_1D_time_tel_" + str(self.telescope_indices[step]) + ".png"
                )

    def plot_altitude_distr(self, altitude_edges, histograms_1D):
        """
        Plots the 1D distribution, i.e. the radial distribution, of the photons on the ground.
        """
        for step, _ in enumerate(altitude_edges):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(
                altitude_edges[step][:-1],
                histograms_1D[step],
                xerr=np.abs(np.diff(altitude_edges[step])[0] / 2),
                ls="",
            )
            # ax.scatter(distance_sorted,hist_sorted, alpha=0.5, c='r')
            if self.telescope_indices is None:
                fig.savefig("boost_histogram_1D_altitude_tels.png")
            else:
                fig.savefig(
                    "boost_histogram_1D_altitude_tel_" + str(self.telescope_indices[step]) + ".png"
                )
