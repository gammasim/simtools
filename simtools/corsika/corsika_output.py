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
from simtools.util.names import corsika7_event_header


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

        self._tel_positions = None

        self._allowed_histograms = {"hist_position", "hist_direction", "hist_time_altitude"}
        self._allowed_1D_labels = {"wavelength", "time", "altitude"}
        self._allowed_2D_labels = {"counts", "density", "direction", "time_altitude"}
        self._events_information = None
        self._get_event_information()

    @property
    def telescope_indices(self):
        """
        The telescope index (or indices) as a class attribute.
        """
        return self._telescope_indices

    @telescope_indices.setter
    def telescope_indices(self, telescope_new_indices):
        """
        Set the telescope index (or indices) as a class attribute.

        Parameters
        ----------
        telescope_new_indices: int or list of int
            The indices of the specific telescopes to be inspected. If not specified, all telescopes
            are treated together in one histogram.

        Raises
        ------
        TypeError:
            if the indices passed through telescope_index are not of type int.
        """
        if telescope_new_indices is not None:
            if not isinstance(telescope_new_indices, list):
                telescope_new_indices = [telescope_new_indices]
            for one_telescope in telescope_new_indices:
                if not isinstance(one_telescope, int):
                    msg = "The index or indices given are not of type int."
                    self._logger.error(msg)
                    raise TypeError
        self._telescope_indices = telescope_new_indices

    def _get_directive_cosinus(self, zenith_angle, azimuth_angle):
        """
        Get the direction cosinus (X and Y) from the incoming particle, which helps defining the
        range of the histograms built from the photon incoming directions.

        Parameters
        ----------
        zenith_angle: astropy.units.rad
            Zenith angle (in radians) of the observations (in the CORSIKA coordinate system).
            It is used to rotate the observation plane (in zenith) of the telescope and to plot
            the sky map of the incoming direction of photons.
        azimuth_angle: astropy.units.rad
            Azimuth angle of observation (in radians).
            See above for more details.

        Returns
        -------
        np.array:
            Directive cosinus in the X direction.
        np.array:
            Directive cosinus in the Y direction.
        """
        cosx_obs = np.sin(zenith_angle) * np.cos(azimuth_angle)
        cosy_obs = np.sin(zenith_angle) * np.sin(azimuth_angle)
        return cosx_obs, cosy_obs

    @property
    def hist_config(self, in_yaml=None, in_dict=None):
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

        self._hist_config = collect_data_from_yaml_or_dict(in_yaml, in_dict, allow_empty=True)
        if self._hist_config is None:
            self._hist_config = self._create_histogram_config()
        return self._hist_config

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

        if self.telescope_indices is None:
            self._xy_maximum = 1000 * u.m
        else:
            self._xy_maximum = 15 * u.m

        histogram_config = {
            "hist_position": {
                "X axis": {
                    "bins": 100,
                    "start": -self._xy_maximum,
                    "stop": self._xy_maximum,
                    "scale": "linear",
                },
                "Y axis": {
                    "bins": 100,
                    "start": -self._xy_maximum,
                    "stop": self._xy_maximum,
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
                    "start": -1,
                    "stop": 1,
                    "scale": "linear",
                },
                "Y axis": {
                    "bins": 100,
                    "start": -1,
                    "stop": 1,
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
            if label is not valid.
        """
        transform = {"log": bh.axis.transform.log, "linear": None}

        if label not in self._allowed_histograms:
            msg = "allowed labels must be one of the following: {}".format(self._allowed_histograms)
            self._logger.error(msg)
            raise (ValueError)

        all_axes = ["X axis", "Y axis"]
        if label == "hist_position":
            all_axes.append("Z axis")

        boost_axes = []
        for axis in all_axes:
            try:
                boost_axes.append(
                    bh.axis.Regular(
                        bins=self.hist_config[label][axis]["bins"],
                        start=self.hist_config[label][axis]["start"].value,
                        stop=self.hist_config[label][axis]["stop"].value,
                        transform=transform[self.hist_config[label][axis]["scale"]],
                    )
                )
            except AttributeError:
                boost_axes.append(
                    bh.axis.Regular(
                        bins=self.hist_config[label][axis]["bins"],
                        start=self.hist_config[label][axis]["start"],
                        stop=self.hist_config[label][axis]["stop"],
                        transform=transform[self.hist_config[label][axis]["scale"]],
                    )
                )
        return boost_axes

    def _create_histograms(self):
        """
        Create the histogram instances.
        """
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

    def _fill_histograms(self, photons, azimuth_angle=None, zenith_angle=None):
        """Fill the histograms created by self._create_histogram

        Parameters
        ----------
        photons: list
            List of size M of numpy.void of size (N,8), where M is the number of telescopes in the
            array, N is the number of photons that reached each telescope. The following information
             of the Cherenkov photons on the ground are saved: x, y, cx, cy, time, zem, photons,
             wavelength.
        azimuth_angle: astropy.Quantity
            Azimuth angle to rotate the observational plane and obtain it perpendicular to the
            incoming event. It can be passed in radians or degrees.
            If not given, no rotation is performed.
        zenith_angle: astropy.Quantity
            Zenith angle to rotate the observational plane and obtain it perpendicular to the
            incoming event. It can be passed in radians or degrees.
            If not given, no rotation is performed.

        Raises
        ------
        IndexError:
            If the index or indices passed though telescope_index are out of range.
        """

        hist_num = 0
        for one_tel_info, photons_info in zip(self._tel_positions, photons):

            if azimuth_angle is None or zenith_angle is None:
                photon_x, photon_y = photons_info["x"], photons_info["y"]
            else:
                photon_x, photon_y = rotate(
                    photons_info["x"],
                    photons_info["y"],
                    azimuth_angle,
                    zenith_angle,
                )

            if self.telescope_indices is None:
                hist_num = 0
                photon_x = -one_tel_info["x"] + photon_x
                photon_y = -one_tel_info["y"] + photon_y

            if (
                one_tel_info in self._tel_positions[self.telescope_indices]
                or self.telescope_indices is None
            ):
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
        self._telescope_indices = telescope_indices
        self._create_histograms()

        self.num_photons_per_event_per_telescope = []
        start_time = time.time()
        self._logger.debug("Starting reading the file at {}.".format(start_time))
        with IACTFile(self.input_file) as f:
            event_counter = 0
            for event in f:
                for step, _ in enumerate(self._tel_positions):
                    self.num_photons_per_event_per_telescope.append(event.n_photons[step])

                photons = list(event.photon_bunches.values())
                self._fill_histograms(
                    photons,
                    self.event_azimuth_angles[event_counter],
                    self.event_zenith_angles[event_counter],
                )
                event_counter += 1
        self._logger.debug(
            "Finished reading the file and creating the histograms in {} seconds".format(
                time.time() - start_time
            )
        )

    def _raise_if_no_histogram(self):
        """
        Raise an error if the histograms were not created.

        Raises
        ------
        HistogramNotCreated:
            if the histogram was not previously created.
        """

        for histogram in self._allowed_histograms:
            if not hasattr(self, histogram):
                msg = (
                    "The histograms were not created. Please, use `create_histograms` to create "
                    "histograms from the CORSIKA output file."
                )
                self._logger.error(msg)
                raise HistogramNotCreated

    def _get_2D(self, label):
        """
        Helper function to get 2D distributions.

        Parameters
        ----------
        label: str
            Label to indicate which histogram.

        Raises
        ------
        ValueError:
            if label is not valid.
        """

        if label not in self._allowed_2D_labels:
            msg = "label is not valid. Valid entries are {}".format(self._allowed_2D_labels)
            self._logger.error(msg)
            raise ValueError
        self._raise_if_no_histogram()

        if self.telescope_indices is None:
            size = 1
        else:
            size = len(self.telescope_indices)

        x_edges, y_edges, hist_values = [], [], []
        for step in range(size):
            if label == "counts":
                mini_hist = self.hist_position[step][:, :, sum]
                hist_values.append(mini_hist.view().T)
            elif label == "density":
                mini_hist = self.hist_position[step][:, :, sum]
                areas = functools.reduce(operator.mul, mini_hist.axes.widths)
                hist_values.append(mini_hist.view().T / areas)
            elif label == "direction":
                mini_hist = self.hist_direction[step]
                hist_values.append(self.hist_direction[step].view().T)
            elif label == "time_altitude":
                mini_hist = self.hist_time_altitude[step]
                hist_values.append(self.hist_time_altitude[step].view().T)
            x_edges.append(mini_hist.axes.edges[0].flatten())
            y_edges.append(mini_hist.axes.edges[1].flatten())

        return np.array(x_edges), np.array(y_edges), np.array(hist_values)

    def get_2D_photon_position_distr(self, density=True):
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
        if density is True:
            return self._get_2D("density")
        else:
            return self._get_2D("counts")

    def get_2D_photon_direction_distr(self):
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
        return self._get_2D("direction")

    def get_2D_photon_time_altitude(self):
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
        return self._get_2D("time_altitude")

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

    def _get_1D(self, label):
        """
        Helper function to get 1D distributions.

        Parameters
        ----------
        label: str
            Label to indicate which histogram.

        Raises
        ------
        ValueError:
            if label is not valid.
        """

        if label not in self._allowed_1D_labels:
            msg = "`label` is not valid. Valid entries are {}".format(self._allowed_1D_labels)
            self._logger.error(msg)
            raise ValueError
        self._raise_if_no_histogram()

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

    def get_photon_radial_distr(self, bin_size=None, max_dist=None, density=True):
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
        x_edges_list, y_edges_list, hist2D_values_list = self.get_2D_photon_position_distr(
            density=density
        )
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

    def get_photon_wavelength_distr(self):
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

    def get_photon_time_distr(self):
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

    def get_photon_altitude_distr(self):
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

    @property
    def num_photons_per_event(self):
        """
        The number of photons per event.

        Returns
        -------
        numpy.array
            Number of photons per event.
        """
        num_of_photons_per_event_per_telescope = self.get_num_photons_per_event_per_telescope()
        self._num_photons_per_event = np.sum(
            np.array(num_of_photons_per_event_per_telescope), axis=0
        )
        return self._num_photons_per_event

    @property
    def total_num_photons(self):
        """
        The total number of photons.

        Returns
        -------
        float
            Total number photons.
        """
        self._total_num_photons = np.sum(self.num_photons_per_event)
        return self._total_num_photons

    @property
    def telescope_positions(self):
        """
        The telescope positions.

        Returns
        -------
        numpy.ndarray
            X, Y and Z positions of the telescopes and their radius according to the CORSIKA
            spherical representation of the telescopes.
        """
        return self._tel_positions

    def _get_event_information(self):
        """
        Get information from the event header and save into dictionary.
        """
        if self._events_information is None:
            with IACTFile(self.input_file) as f:
                self._file_header = f.header
                self._tel_positions = np.array(f.telescope_positions)
                self.num_telescopes = np.size(self._tel_positions, axis=0)
                self._events_information = {
                    key: {"value": [], "unit": None} for key in corsika7_event_header
                }
                self.num_events = 0
                for _, event in enumerate(f):
                    for key in corsika7_event_header:
                        self._events_information[key]["value"].append(
                            (event.header[corsika7_event_header[key]["value"]])
                        )
                        if self._events_information[key]["unit"] is None:
                            self._events_information[key]["unit"] = corsika7_event_header[key][
                                "unit"
                            ]
                    self.num_events += 1

    @property
    def events_information(self):
        """
        Get the information about the events from their headers.
        The main information can also be fetched individually through the functions below.
        For the remaining information (such as px, py, pz), use this function.

        Returns
        -------
        dict
            Dictionary with the events information.
        """
        return self._events_information

    @property
    def event_zenith_angles(self):
        """
        Get the zenith angles of the simulated events in degrees.

        Returns
        -------
        numpy.array
            The zenith angles in degrees for each event.
        """
        self._event_zenith_angles = np.around(
            (
                self.events_information["zenith_angle"]["value"]
                * self.events_information["zenith_angle"]["unit"]
            ).to(u.deg),
            4,
        )
        return self._event_zenith_angles

    @property
    def event_azimuth_angles(self):
        """
        Get the azimuth angles of the simulated events in degrees.

        Returns
        -------
        numpy.array
            The azimuth angles in degrees for each event.
        """
        self._event_azimuth_angles = np.around(
            (
                self.events_information["azimuth_angle"]["value"]
                * self.events_information["azimuth_angle"]["unit"]
            ).to(u.deg),
            4,
        )
        return self._event_azimuth_angles

    @property
    def event_energies(self):
        """
        Get the energy of the simulated events in TeV.

        Returns
        -------
        numpy.array
            The total energies in TeV for each event.
        """
        self._event_total_energies = np.around(
            (
                self.events_information["total_energy"]["value"]
                * self.events_information["total_energy"]["unit"]
            ).to(u.TeV),
            4,
        )
        return self._event_total_energies

    @property
    def event_first_interaction_heights(self):
        """
        Get the height of the first interaction in km.
        If negative, tracking starts at margin of atmosphere, see TSTART in the CORSIKA 7 user guide
        .

        Returns
        -------
        numpy.array
            The first interaction height in km for each event.
        """
        self._event_first_interaction_heights = np.around(
            (
                self.events_information["first_interaction_height"]["value"]
                * self.events_information["first_interaction_height"]["unit"]
            ).to(u.km),
            4,
        )
        return self._event_first_interaction_heights

    @property
    def corsika_version(self):
        """
        Get the CORSIKA version from the events header.

        Returns
        -------
        numpy.array
            The CORSIKA version used for each event.
        """
        self._corsika_version = self.events_information["software_version"]["value"]
        return self._corsika_version

    @property
    def magnetic_field(self):
        """
        Get the Earth magnetic field from the events header in microT.

        Returns
        -------
        numpy.array
            The Earth magnetic field in the X direction used for each event in microT.
        numpy.array
            The Earth magnetic field in the Y direction used for each event in microT.
        """
        self._magnetic_field_x = (
            self.events_information["Earth_B_field_x"]["value"]
            * self.events_information["Earth_B_field_x"]["unit"]
        )
        self._magnetic_field_x = (
            self.events_information["Earth_B_field_y"]["value"]
            * self.events_information["Earth_B_field_y"]["unit"]
        )

        return self._magnetic_field_x, self._magnetic_field_y

    def event_1D_histogram(self, key, bins=50, range=None):
        """
        Create a histogram for the all events using `key` as parameter.
        Valid keys are in ~simtools.names.corsika7_event_header.

        Parameters
        ----------
        key: str
            The information from which to build the histogram, e.g. total_energy, zenith_angle
            or first_interaction_height.
        bins: float
            Number of bins for the histogram.
        range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.array
            Edges of the histogram.
        numpy.ndarray
            The values (counts) of the histogram.

        Raises
        ------
        KeyError:
            If key is not valid.
        """
        if key not in corsika7_event_header:
            msg = "`key` is not valid. Valid entries are {}".format(corsika7_event_header)
            self._logger.error(msg)
            raise KeyError
        hist, edges = np.histogram(
            self.events_information[key]["value"] * self.events_information[key]["unit"],
            bins=bins,
            range=range,
        )
        return edges, hist

    def event_2D_histogram(self, key_1, key_2, bins=50, range=None):
        """
        Create a 2D histogram for the all events using `key_1` and `key_2` as parameters.
        Valid keys are in ~simtools.names.corsika7_event_header.

        Parameters
        ----------
        key_1: str
            The first key from which to build the histogram, e.g. total_energy, zenith_angle
            or first_interaction_height.
        key_2: str
            The second key from which to build the histogram, e.g. total_energy, zenith_angle
            or first_interaction_height.
        bins: float
            Number of bins for the histogram.
        range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.array
            X Edges of the histogram.
        numpy.array
            Y Edges of the histogram.
        numpy.ndarray
            The values (counts) of the histogram.

        Raises
        ------
        KeyError:
            If at least one of the keys is not valid.
        """
        for key in [key_1, key_2]:
            if key not in corsika7_event_header:
                msg = "At least one of the keys given is not valid. Valid entries are {}".format(
                    corsika7_event_header
                )
                self._logger.error(msg)
                raise KeyError
        hist, x_edges, y_edges = np.histogram2d(
            self.events_information[key_1]["value"] * self.events_information[key_1]["unit"],
            self.events_information[key_2]["value"] * self.events_information[key_2]["unit"],
            bins=bins,
            range=range,
        )
        return x_edges, y_edges, hist
