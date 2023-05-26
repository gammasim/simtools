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
    """CorsikaOutput extracts the Cherenkov photons information from a CORSIKA output file
    (IACT file) using pyeventio.

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

        self.input_file = Path(self.input_file)
        if not self.input_file.exists():
            msg = f"file {self.input_file} does not exist."
            self._logger.error(msg)
            raise FileNotFoundError

        self._initialize_attributes()
        self._read_event_information()

    def _initialize_attributes(self):
        """
        Initializes the class attributes.
        """
        self._tel_positions = None
        self.num_events = None
        self.num_of_hist = None
        self.num_telescopes = None
        self._num_photons_per_event_per_telescope = None
        self._num_photons_per_event = None
        self._num_photons_per_telescope = None
        self._event_azimuth_angles = None
        self._event_zenith_angles = None
        self._events_information = None
        self._hist_config = None
        self._total_num_photons = None
        self._magnetic_field_x = None
        self._magnetic_field_y = None
        self._event_total_energies = None
        self._event_first_interaction_heights = None
        self._corsika_version = None
        self._allowed_histograms = {"hist_position", "hist_direction", "hist_time_altitude"}
        self._allowed_1D_labels = {"wavelength", "time", "altitude"}
        self._allowed_2D_labels = {"counts", "density", "direction", "time_altitude"}

    @property
    def telescope_indices(self):
        """
        The telescope index (or indices).
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
            are treated together in one histogram and the value of self._telescope_indices is None.

        Raises
        ------
        TypeError:
            if the indices passed through telescope_index are not of type int.
        """

        if telescope_new_indices is not None:
            if not isinstance(telescope_new_indices, list):
                telescope_new_indices = [telescope_new_indices]
            for i_telescope in telescope_new_indices:
                if not isinstance(i_telescope, int):
                    msg = "The index or indices given are not of type int."
                    self._logger.error(msg)
                    raise TypeError
        # if self.individual_telescopes is True, the indices of the telescopes passed are analyzed
        # individually (different histograms for each telescope) even if all telescopes are listed.
        self._telescope_indices = telescope_new_indices

    @property
    def hist_config(self):
        """
        The configuration of the histograms.
        """
        if self._hist_config is None:
            self._hist_config = self._create_histogram_default_config()
        return self._hist_config

    @hist_config.setter
    def hist_config(self, in_yaml, in_dict):
        """
        Set the configuration for the histograms (e.g., bin size, min and max values, etc).
        The inputs are allowed either through a yaml file or a dictionary. If nothing is given,
        the dictionary is created with default values.

        Parameters
        ----------
        in_yaml: str or Path
            yaml file with the configuration parameters to create the histograms. For the correct
            format, please look at the docstring at `_create_histogram_default_config`.

        in_dict: dict
            Dictionary with the configuration parameters to create the histograms.
        """
        self._hist_config = collect_data_from_yaml_or_dict(in_yaml, in_dict, allow_empty=True)

    def _create_histogram_default_config(self):
        """
        Create a dictionary with the configuration necessary to create the histograms. It is used
        only in case the configuration is not provided in a yaml file or dict.

        Three histograms are created: hist_position with 3 dimensions (x, y positions and the
        wavelength), hist_direction with 2 dimensions (directive cosinus in x and y directions),
        hist_time_altitude with 2 dimensions (time and altitude of emission).

        Four arguments are passed to each dimension in the dictionary:

        "bins": the number of bins,
        "start": the first element of the histogram,
        "stop": the last element of the histogram, and
        "scale" to define the scale of the bins which can be "linear" or "log". If "log",
            the start and stop values have to be valid, i.e., >0.

        Returns
        -------
        dict:
            Dictionary with the configuration parameters to create the histograms.
        """

        if self.individual_telescopes is False:
            xy_maximum = 1000 * u.m
            xy_bin = 100 * u.m

        else:
            xy_maximum = 16 * u.m
            xy_bin = 64 * u.m

        histogram_config = {
            "hist_position": {
                "x axis": {
                    "bins": xy_bin,
                    "start": -xy_maximum,
                    "stop": xy_maximum,
                    "scale": "linear",
                },
                "y axis": {
                    "bins": xy_bin,
                    "start": -xy_maximum,
                    "stop": xy_maximum,
                    "scale": "linear",
                },
                "z axis": {
                    "bins": 80,
                    "start": 200 * u.nm,
                    "stop": 1000 * u.nm,
                    "scale": "linear",
                },
            },
            "hist_direction": {
                "x axis": {
                    "bins": 100,
                    "start": -1,
                    "stop": 1,
                    "scale": "linear",
                },
                "y axis": {
                    "bins": 100,
                    "start": -1,
                    "stop": 1,
                    "scale": "linear",
                },
            },
            "hist_time_altitude": {
                "x axis": {
                    "bins": 100,
                    "start": -2000 * u.ns,
                    "stop": 2000 * u.ns,
                    "scale": "linear",
                },
                "y axis": {"bins": 100, "start": 50 * u.km, "stop": 0 * u.km, "scale": "linear"},
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
            msg = f"allowed labels must be one of the following: {self._allowed_histograms}"
            self._logger.error(msg)
            raise ValueError

        all_axes = ["x axis", "y axis"]
        if label == "hist_position":
            all_axes.append("z axis")

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

        self.num_of_hist = len(self.telescope_indices) if self.individual_telescopes is True else 1

        self.hist_position, self.hist_direction, self.hist_time_altitude = [], [], []

        for _ in range(self.num_of_hist):
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
        """Fill all the histograms created by self._create_histogram with the information of the
         photons on the ground.

        Parameters
        ----------
        photons: list
            List of size M of numpy.array of size (N,8), where M is the number of telescopes in the
            array, N is the number of photons that reached each telescope. The following information
             of the Cherenkov photons on the ground are saved:
             x: x position on the ground (CORSIKA coordinate system),
             y: y position on the ground (CORSIKA coordinate system),
             cx: direction cosinus in the x direction, i.e., the cosinus of the angle between the
             incoming direction and the x axis,
             cy: direction cosinus in the y direction, i.e., the cosinus of the angle between the
             incoming direction and the y axis,
             time: time of arrival of the photon in ns. The clock starts when the particle crosses
             the top of the atmosphere (CORSIKA-defined) if `self.event_first_interaction_heights`
             is positive or at first interaction if otherwise.
             zem: altitude where the photon was generated in cm,
             photons: number of photons associated to this bunch,
             wavelength: the wavelength of the photons in nm.
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
        telescope_mask = np.isin(self.all_telescope_indices, self.telescope_indices)
        for i_tel_info, photons_info in np.array(list(zip(self.telescope_positions, photons)))[
            telescope_mask
        ]:

            if azimuth_angle is None or zenith_angle is None:
                photon_x, photon_y = photons_info["x"], photons_info["y"]
            else:
                photon_x, photon_y = rotate(
                    photons_info["x"],
                    photons_info["y"],
                    azimuth_angle,
                    zenith_angle,
                )

            if self.individual_telescopes is False:
                hist_num = 0
                # Adding the position of the telescopes to the relative position of the photons
                # such that we have a common coordinate system.
                photon_x = -i_tel_info["x"] + photon_x
                photon_y = -i_tel_info["y"] + photon_y

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

    def set_histograms(self, telescope_indices=None, individual_telescopes=False):
        """
        Extract the information of the Cherenkov photons from a CORSIKA output IACT file, create
         and fill the histograms

        Parameters
        ----------
        telescope_indices: int or list of int
            The indices of the specific telescopes to be inspected. If not specified, all telescopes
            are treated together in one histogram.
        individual_telescopes: bool
            if False, the histograms are supposed to be filled for all telescopes.
            if True, one histogram is set for each telescope sepparately.

        Returns
        -------
        list: list of boost_histogram.Histogram instances.

        """
        self.all_telescope_indices = np.arange(self.num_telescopes)
        if telescope_indices is None:
            telescope_indices = self.all_telescope_indices.tolist()
        self.telescope_indices = telescope_indices
        self.individual_telescopes = individual_telescopes
        self._create_histograms()

        num_photons_per_event_per_telescope_to_set = []
        start_time = time.time()
        self._logger.debug("Starting reading the file at {}.".format(start_time))
        with IACTFile(self.input_file) as f:
            event_counter = 0
            for event in f:
                for i_telescope in self.telescope_indices:
                    # Count photons only from the telescopes given by self.telescope_indices.
                    num_photons_per_event_per_telescope_to_set.append(event.n_photons[i_telescope])

                photons = list(event.photon_bunches.values())
                self._fill_histograms(
                    photons,
                    self.event_azimuth_angles[event_counter],
                    self.event_zenith_angles[event_counter],
                )
                event_counter += 1
        self.num_photons_per_event_per_telescope = num_photons_per_event_per_telescope_to_set
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

    def _get_hist_2D_projection(self, label):
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

        num_telescopes_to_fill = (
            len(self.telescope_indices) if self.individual_telescopes is True else 1
        )

        x_edges, y_edges, hist_values = [], [], []
        for i_telescope in range(num_telescopes_to_fill):
            if label == "counts":
                mini_hist = self.hist_position[i_telescope][:, :, sum]
                hist_values.append(mini_hist.view().T)
            elif label == "density":
                mini_hist = self.hist_position[i_telescope][:, :, sum]
                areas = functools.reduce(operator.mul, mini_hist.axes.widths)
                hist_values.append(mini_hist.view().T / areas)
            elif label == "direction":
                mini_hist = self.hist_direction[i_telescope]
                hist_values.append(self.hist_direction[i_telescope].view().T)
            elif label == "time_altitude":
                mini_hist = self.hist_time_altitude[i_telescope]
                hist_values.append(self.hist_time_altitude[i_telescope].view().T)
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
            The x edges of the density/count histograms in x, usually in meters.
        numpy.array
            The y edges of the density/count histograms in y, usually in meters.
        numpy.ndarray
            The counts of the histogram.
        """
        if density is True:
            return self._get_hist_2D_projection("density")
        else:
            return self._get_hist_2D_projection("counts")

    def get_2D_photon_direction_distr(self):
        """
        Get 2D histograms of incoming direction of the Cherenkov photons on the ground.

        Returns
        -------
        numpy.array
            The x edges of the direction histograms in cos(x).
        numpy.array
            The y edges of the direction histograms in cos(y)
        numpy.ndarray
            The counts of the histogram.
        """
        return self._get_hist_2D_projection("direction")

    def get_2D_photon_time_altitude(self):
        """
        Get 2D histograms of the time and altitude of the photon production.

        Returns
        -------
        numpy.array
            The x edges of the time_altitude histograms, usually in ns.
        numpy.array
            The y edges of the time_altitude histograms, usually in km.
        numpy.ndarray
            The counts of the histogram.
        """
        return self._get_hist_2D_projection("time_altitude")

    def get_2D_num_photons_distr(self):
        """
        Get the distribution of Cherenkov photons per event per telescope. It returns the 2D array
        accounting for the events from the telescopes given by `self.telescope_indices`.

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
        numpy.array
            Indices of the telescopes.
        numpy.ndarray
            The counts of the histogram.
        """
        num_events_array = np.arange(self.num_events + 1)
        telescope_indices_array = np.arange(len(self.telescope_indices) + 1)
        return (
            num_events_array,
            telescope_indices_array,
            np.array(self.num_photons_per_event_per_telescope),
        )

    def _get_hist_1D_projection(self, label):
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
        for i_hist, _ in enumerate(self.hist_position):
            if label == "wavelength":
                mini_hist = self.hist_position[i_hist][sum, sum, :]
            elif label == "time":
                mini_hist = self.hist_time_altitude[i_hist][:, sum]
            elif label == "altitude":
                mini_hist = self.hist_time_altitude[i_hist][sum, :]

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
            Bin size of the radial distribution (in meters).
        max_dist: float
            Maximum distance to consider in the 1D histogram (in meters).
        density: bool
            If True, returns the density distribution. If False, returns the distribution of counts.

        Returns
        -------
        np.array
            The edges of the 1D histogram in meters with size = int(max_dist/bin_size) + 1.
        np.array
            The counts of the 1D histogram with size = int(max_dist/bin_size).
        """
        if self.individual_telescopes is False:
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
        x_position_list, y_position_list, hist2D_values_list = self.get_2D_photon_position_distr(
            density=density
        )
        for i_hist, _ in enumerate(x_position_list):
            edges_1D, hist1D = convert_2D_to_radial_distr(
                x_position_list[i_hist],
                y_position_list[i_hist],
                hist2D_values_list[i_hist],
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
            The counts of the wavelength histogram.
        """
        return self._get_hist_1D_projection("wavelength")

    def get_photon_time_distr(self):
        """
        Get the distribution of the emitted time of the Cherenkov photons. The clock starts when the
         particle crosses the top of the atmosphere (CORSIKA-defined) if
         `self.event_first_interaction_heights` is positive or at first interaction if otherwise.

        Returns
        -------
        numpy.array
            The edges of the direction histograms in ns.
        numpy.ndarray
            The counts of the histogram.
        """
        return self._get_hist_1D_projection("time")

    def get_photon_altitude_distr(self):
        """
        Get the emission altitude of the Cherenkov photons.

        Returns
        -------
        numpy.array
            The edges of the direction histograms in km.
        numpy.ndarray
            The counts of the histogram.
        """
        return self._get_hist_1D_projection("altitude")

    @property
    def num_photons_per_event_per_telescope(self):
        """
        The number of photons per event per telescope.
        """
        return self._num_photons_per_event_per_telescope

    @num_photons_per_event_per_telescope.setter
    def num_photons_per_event_per_telescope(self, num_photons_per_event_per_telescope_to_set):
        """
        Set the number of photons per event per telescope.
        """
        if self._num_photons_per_event_per_telescope is None:
            self._num_photons_per_event_per_telescope = (
                np.array(num_photons_per_event_per_telescope_to_set)
                .reshape(self.num_events, len(self.telescope_indices))
                .T
            )

    @property
    def num_photons_per_event(self):
        """
        Get the distribution of the number of photons amongst the events,
         including the telescopes indicated by `self.telescope_indices`.

        Returns
        -------
        numpy.array
            Number of photons per event.
        """
        if self._num_photons_per_event is None:
            self._num_photons_per_event = np.sum(self.num_photons_per_event_per_telescope, axis=1)
        return self._num_photons_per_event

    def get_num_photons_distr(self, bins=50, range=None):
        """
        Get the distribution of

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
            The counts of the histogram.
        """
        hist, edges = np.histogram(self.num_photons_per_event, bins=bins, range=range)
        return edges, hist

    @property
    def num_photons_per_telescope(self):
        """
        The number of photons per event, considering the telescopes given by
        self.telescope_indices.

        Returns
        -------
        numpy.array
            Number of photons per telescope.
        """
        if self._num_photons_per_telescope is None:
            self._num_photons_per_telescope = np.sum(
                np.array(self.num_photons_per_event_per_telescope), axis=0
            )
        return self._num_photons_per_telescope

    @property
    def total_num_photons(self):
        """
        The total number of photons.

        Returns
        -------
        float
            Total number photons.
        """
        if self._total_num_photons is None:
            self._total_num_photons = np.sum(self.num_photons_per_event)
        return self._total_num_photons

    @property
    def telescope_positions(self):
        """
        The telescope positions.

        Returns
        -------
        numpy.ndarray
            x, y and z positions of the telescopes and their radius according to the CORSIKA
            spherical representation of the telescopes.
        """
        return self._telescope_positions

    @telescope_positions.setter
    def telescope_positions(self, new_positions):
        """
        Set the telescope positions.

        Parameters
        ----------
        numpy.ndarray
            x, y and z positions of the telescopes and their radius according to the CORSIKA
            spherical representation of the telescopes.
        """
        self._telescope_positions = new_positions

    def _read_event_information(self):
        """
        Get information from the event header and save into dictionary.
        """
        if self._events_information is None:
            with IACTFile(self.input_file) as f:
                self._file_header = f.header
                self.telescope_positions = np.array(f.telescope_positions)
                self.num_telescopes = np.size(self.telescope_positions, axis=0)
                self._events_information = {
                    key: {"value": [], "unit": None} for key in corsika7_event_header
                }
                self.num_events = 0
                for event in f:
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
        astropy.Quantity
            The zenith angles for each event, usually in degrees.
        """
        if self._event_zenith_angles is None:
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
        astropy.Quantity
            The azimuth angles for each event, usually in degrees.
        """
        if self._event_azimuth_angles is None:
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
        astropy.Quantity
            The total energies of the incoming particles for each event, usually in TeV.
        """
        if self._event_total_energies is None:
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
        astropy.Quantity
            The first interaction height for each event, usually in km.
        """
        if self._event_first_interaction_heights is None:
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
        if self._corsika_version is None:
            self._corsika_version = self.events_information["software_version"]["value"]
        return self._corsika_version

    @property
    def magnetic_field(self):
        """
        Get the Earth magnetic field from the events header in microT.

        Returns
        -------
        astropy.Quantity
            The Earth magnetic field in the x direction used for each event.
        astropy.Quantity
            The Earth magnetic field in the y direction used for each event.
        """
        if self._magnetic_field_x is None:
            self._magnetic_field_x = (
                self.events_information["Earth_B_field_x"]["value"]
                * self.events_information["Earth_B_field_x"]["unit"]
            )
        if self._magnetic_field_y is None:
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
            The counts of the histogram.

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
            x Edges of the histogram.
        numpy.array
            y Edges of the histogram.
        numpy.ndarray
            The counts of the histogram.

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
