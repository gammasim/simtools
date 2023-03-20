import functools
import logging
import operator
import time
from pathlib import Path

import boost_histogram as bh
import numpy as np
from astropy import units as u
from eventio import IACTFile

from simtools.util.general import (
    collect_data_from_yaml_or_dict,
    convert_2D_to_radial_distr,
    rotate,
)


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
    def __init__(self, input_file, zenith_angle=0 * u.rad, azimuth_angle=0 * u.rad, label=None):

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaOutput")
        self.input_file = input_file
        self.zenith_angle = zenith_angle
        self.azimuth_angle = azimuth_angle
        self.label = label

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

        Returns
        -------
        np.array:
            Directive cosinus in the X direction.
        np.array:
            Directive cosinus in the Y direction.
        """
        cosx_obs = np.sin(self.zenith_angle) * np.cos(self.azimuth_angle)
        cosy_obs = np.sin(self.zenith_angle) * np.sin(self.azimuth_angle)
        return cosx_obs, cosy_obs

    def _set_histogram_config(self, in_yaml=None, in_dict=None):
        """
        Return the dictionary with the configuration to create the histograms. The inputs are
        allowed either through an yaml file or a dictionary. If nothing is given, the dictionary
        is created with default values.

        Parameters
        ----------
        in_yaml: str or Path
            Yaml file with the configuration parameters to create the histograms. For the correct
            format, please look at the docstring at `_create_histogram_config`.

        in_dict: dict
            Dictionary with the configuration parameterse to create the histograms.

        Raises
        ------
        TypeError:
            if type of yaml_config is not valid.
        FileNotFoundError:
            if yaml_config does not exist.

        """
        if isinstance(in_yaml, str):
            in_yaml = Path(in_yaml)
        if not isinstance(in_yaml, (Path, type(None))):
            msg = "The type of `in_yaml` is not valid. Valid types are Path and str."
            self._logger.error(msg)
            raise TypeError

        if in_yaml is not None and not in_yaml.exists():
            raise FileNotFoundError

        self.hist_config = collect_data_from_yaml_or_dict(in_yaml, in_dict, allow_empty=True)
        if self.hist_config is None:
            self.hist_config = self._create_histogram_config()

    def _create_histogram_config(self):
        """
        Create a dictionary with the configuration necessary to create the histograms. It is used
        only in case the configuration is not provided in a yaml file.

        Three histograms are created: hist_position with 3 dimensions (X, Y positions and the
        wavelength), hist_direction with 2 dimensions (directive cosinus in X and Y directions),
        hist_time_altitude with 2 dimensions (time and altitude of emission).

        Four arguments are passed to each dimension in the dictionary:

        "bins": the number of bins,
        "start": the first element of the histogram,
        "stop": the last element of the histogram, and
        "scale" to define the scale of the binswhich can be "linear" or "log". If "log",
            the start and stop values have to be valid, i.e., >0.

        Returns
        -------
        dict:
            Dictionary with the configuration parameters to create the histograms.
        """
        cosx_obs, cosy_obs = self._get_directive_cosinus()
        if self.telescope_indices is None:
            xy_maximum = 1000 * u.m
        else:
            xy_maximum = 15 * u.m

        histogram_config = {
            "hist_position": {
                "X axis": {
                    "bins": 100,
                    "start": -xy_maximum,
                    "stop": xy_maximum,
                    "scale": "linear",
                },
                "Y axis": {
                    "bins": 100,
                    "start": -xy_maximum,
                    "stop": xy_maximum,
                    "scale": "linear",
                },
                "Z axis": {
                    "bins": 100,
                    "start": 200 * u.nm,
                    "stop": 1000 * u.nm,
                    "scale": "linear",
                },
            },
            "hist_direction": {
                "X axis": {
                    "bins": 100,
                    "start": cosx_obs - 0.5,
                    "stop": cosx_obs + 0.5,
                    "scale": "linear",
                },
                "Y axis": {
                    "bins": 100,
                    "start": cosy_obs - 0.5,
                    "stop": cosy_obs + 0.5,
                    "scale": "linear",
                },
            },
            "hist_time_altitude": {
                "X axis": {
                    "bins": 100,
                    "start": -2000 * u.ns,
                    "stop": 2000 * u.ns,
                    "scale": "linear",
                },
                "Y axis": {"bins": 100, "start": 50 * u.km, "stop": 0 * u.km, "scale": "linear"},
            },
        }
        return histogram_config

    def _create_regular_axes(self, label):
        """
        Helper function to _create_histograms.

        Parameters
        ----------
        label: str
            Label to identify to which histogram the new axis belongs.

        Raises
        ------
        ValueError:
            if label is now valid.
        """
        allowed_labels = {"hist_position", "hist_direction", "hist_time_altitude"}
        transform = {"log": bh.axis.transform.log, "linear": None}

        if label not in allowed_labels:
            msg = "allowed labels must be one of the following: {}".format(allowed_labels)
            self._logger.error(msg)
            raise (ValueError)

        all_axis = ["X axis", "Y axis"]
        if label == "hist_position":
            all_axis.append("Z axis")

        boost_axes = []
        for axis in all_axis:
            boost_axes.append(
                bh.axis.Regular(
                    bins=self.hist_config[label][axis]["bins"],
                    start=self.hist_config[label][axis]["start"].value,
                    stop=self.hist_config[label][axis]["stop"].value,
                    transform=transform[self.hist_config[label][axis]["scale"]],
                )
            )
        return boost_axes

    def _create_histograms(self):
        """
        Create the histogram instances.
        """
        self._set_histogram_config()

        if self.telescope_indices is None:
            self.num_of_hist = 1
        else:
            self.num_of_hist = len(self.telescope_indices)

        self.hist_position, self.hist_direction, self.hist_time_altitude = [], [], []

        for step in range(self.num_of_hist):
            boost_axes_position = self._create_regular_axes("hist_position")
            self.hist_position.append(
                bh.Histogram(
                    boost_axes_position[0],
                    boost_axes_position[1],
                    boost_axes_position[2],
                )
            )
            boost_axes_direction = self._create_regular_axes("hist_direction")
            self.hist_direction.append(
                bh.Histogram(
                    boost_axes_direction[0],
                    boost_axes_direction[1],
                )
            )

            boost_axes_time_altitude = self._create_regular_axes("hist_time_altitude")
            self.hist_time_altitude.append(
                bh.Histogram(
                    boost_axes_time_altitude[0],
                    boost_axes_time_altitude[1],
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

        hist_num = 0
        for one_tel_info, photons_info in zip(self.tel_positions, photons):

            photon_x, photon_y = rotate(
                photons_info["x"],
                photons_info["y"],
                self.azimuth_angle,
                self.zenith_angle,
            )

            if self.telescope_indices is None:
                photon_x = -one_tel_info["x"] + photon_x
                photon_y = -one_tel_info["y"] + photon_y

            else:
                photon_x, photon_y = photons_info["x"], photons_info["y"]

            if one_tel_info in self.tel_positions[self.telescope_indices]:

                self.hist_position[hist_num].fill(
                    (photon_x * u.cm).to(u.m),
                    (photon_y * u.cm).to(u.m),
                    np.abs(photons_info["wavelength"]) * u.nm,
                )

                self.hist_direction[hist_num].fill(photons_info["cx"], photons_info["cy"])
                self.hist_time_altitude[hist_num].fill(
                    photons_info["time"] * u.ns, (photons_info["zem"] * u.cm).to(u.km)
                )
                hist_num += 1

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

        self.num_photons_per_event_per_telescope = []
        start_time = time.time()
        self._logger.debug("Starting reading the file at {}.".format(start_time))
        with IACTFile(self.input_file) as f:

            self.tel_positions = np.array(f.telescope_positions)
            self.num_telescopes = np.size(self.tel_positions, axis=0)
            # print((f.input_card))

            self.num_events = 0
            for event in f:
                print(event.header)
                for step, _ in enumerate(self.tel_positions):
                    self.num_photons_per_event_per_telescope.append(event.n_photons[step])

                photons = list(event.photon_bunches.values())
                self._fill_histograms(photons)
                self.num_events += 1
        self._logger.debug(
            "Finished reading the file and creating the histograms in {} seconds".format(
                time.time() - start_time
            )
        )

    def get_2D_position_distr(self, density=True):
        """
        Get 2D histograms of position of the Cherenkov photons on the ground. If density is True,
        it returns the photon density per square meter.

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
                max_dist = 1000
        else:
            if bin_size is None:
                bin_size = 1
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

    def _get_1D(self, label):
        """
        Helper function to get 1D distributions.

        Parameters
        ----------
        label: str
            Label to indicate which histogram.
        """
        x_edges_list, hist_1D_list = [], []
        for step, _ in enumerate(self.hist_position):
            if label == "wavelength":
                mini_hist = self.hist_position[step][sum, sum, :]
            elif label == "time":
                mini_hist = self.hist_time_altitude[step][:, sum]
            elif label == "altitude":
                mini_hist = self.hist_time_altitude[step][sum, :]

            x_edges_list.append(mini_hist.axes.edges.T.flatten()[0])
            hist_1D_list.append(mini_hist.view().T)
        return np.array(x_edges_list), np.array(hist_1D_list)

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
        return self._get_1D("wavelength")

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
        return self._get_1D("time")

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
        return self._get_1D("altitude")

    def get_num_photons_per_event_per_telescope(self):
        """
        Get the number of photons per event per telescope.
        """
        return (
            np.array(self.num_photons_per_event_per_telescope)
            .reshape(self.num_events, self.num_telescopes)
            .T
        )

    def get_2D_num_photons_distr(self):
        """
        Get the distribution of Cherenkov photons per event per telescope.

        Parameters
        ----------
        bins: float
            Number of bins for the histogram.
        range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.array
            Number of photons per event per telescope.
        numpy.ndarray
            The values (counts) of the histogram.
        """
        num_of_photons_per_event_per_telescope = np.array(
            self.get_num_photons_per_event_per_telescope()
        )
        num_events_array = np.arange(self.num_events + 1)
        telescope_indices_array = np.arange(self.num_telescopes + 1)
        return num_events_array, telescope_indices_array, num_of_photons_per_event_per_telescope

    def get_num_photons_distr(self, bins=50, range=None):
        """
        Get the distribution of the number of photons, including the telescopes indicated by
        `self.telescope_indices` or all telescopes if `self.telescope_indices` is None.

        Parameters
        ----------
        bins: float
            Number of bins for the histogram.
        range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.array
            Number of photons per event.
        numpy.ndarray
            The values (counts) of the histogram.
        """

        num_of_photons_per_event_per_telescope = self.get_num_photons_per_event_per_telescope()
        if self.telescope_indices is None:
            num_photons_per_event = np.sum(num_of_photons_per_event_per_telescope, axis=1)
        else:
            num_photons_per_event = np.sum(
                num_of_photons_per_event_per_telescope[self.telescope_indices], axis=1
            )
        hist, edges = np.histogram(num_photons_per_event, bins=bins, range=range)
        return edges, hist

    def get_num_photons_per_event(self):
        """
        Get the number of photons per event.

        Returns
        -------
        numpy.array
            Number of photons per event.
        numpy.ndarray
            The values (counts) of the histogram.
        """
        num_of_photons_per_event_per_telescope = self.get_num_photons_per_event_per_telescope()
        return np.sum(np.array(num_of_photons_per_event_per_telescope), axis=0)

    def get_total_num_photons(self):
        """
        Get the total number of photons.

        Returns
        -------
        float
            Total number photons.
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
