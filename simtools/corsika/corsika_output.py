import functools
import logging
import operator
import time
from pathlib import Path

import boost_histogram as bh
import numpy as np
from astropy import units as u
from astropy.io.misc import yaml
from astropy.units import cds
from corsikaio.subblocks import event_header, get_units_from_fields, run_header
from eventio import IACTFile

from simtools import io_handler
from simtools.util.general import (
    collect_data_from_yaml_or_dict,
    convert_2D_to_radial_distr,
    rotate,
)


class HistogramNotCreated(Exception):
    """Exception for histogram not created."""


class CorsikaOutput:
    """CorsikaOutput extracts the Cherenkov photons information from a CORSIKA output file
    (IACT file) using pyeventio.

    Parameters
    ----------
    input_file: str or Path
        Input file (IACT file) provided by the CORSIKA simulation.
    label: str
        Instance label.

    Raises
    ------
    FileNotFoundError:
        if the input file given does not exist.
    """

    def __init__(self, input_file, label=None):
        self.label = label
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaOutput")
        self.input_file = input_file

        self.input_file = Path(self.input_file)
        if not self.input_file.exists():
            msg = f"file {self.input_file} does not exist."
            self._logger.error(msg)
            raise FileNotFoundError

        self.io_handler = io_handler.IOHandler()

        self._initialize_attributes()
        self.read_event_information()

    def _initialize_attributes(self):
        """
        Initializes the class attributes.
        """
        self._telescope_indices = None
        self._telescope_positions = None
        self.num_events = None
        self.num_of_hist = None
        self.num_telescopes = None
        self._num_photons_per_event_per_telescope = None
        self._num_photons_per_event = None
        self._num_photons_per_telescope = None
        self._event_azimuth_angles = None
        self._event_zenith_angles = None
        self._hist_config = None
        self._total_num_photons = None
        self._magnetic_field_x = None
        self._magnetic_field_y = None
        self._event_total_energies = None
        self._event_first_interaction_heights = None
        self._version = None
        self._header = None
        self.event_information = None
        self._individual_telescopes = None
        self._allowed_histograms = {"hist_position", "hist_direction", "hist_time_altitude"}
        self._allowed_1D_labels = {"wavelength", "time", "altitude"}
        self._allowed_2D_labels = {"counts", "density", "direction", "time_altitude"}

    @property
    def version(self):
        """
        Get the version of the Corsika output file.

        Returns
        -------
        float:
            The version of CORSIKA used to produce the output given by `self.input_file`.
        """

        if self._version is None:
            all_corsika_versions = list(run_header.run_header_types.keys())
            header = np.array(list(self.iact_file.header))

            for i_version in reversed(all_corsika_versions):
                # Get the event header for this software version being tested.
                single_run_header = run_header.run_header_types[i_version]
                # Get the position in the dictionary, where the version is.
                version_index_position = np.argwhere(
                    np.array(list(single_run_header.names)) == "version"
                )[0]

                # Check if version tested is the same as the version written in the file header.
                if i_version == np.trunc(float(header[version_index_position]) * 10) / 10:
                    # If the version found is the same as the initial guess, leave the loop,
                    # otherwise, iterate until we find the correct version.
                    self._version = np.around(float(header[version_index_position]), 3)
                    break
        return self._version

    @property
    def header(self):
        """
        Get the run header.

        Returns
        -------
        dict:
            The run header.
        """

        # Get keys in the header
        if self._header is None:
            self.all_run_keys = list(run_header.run_header_types[np.around(self.version, 1)].names)
            self._header = {}

            # Get units of the header
            all_run_units = get_units_from_fields(
                run_header.run_header_fields[np.trunc(self.version * 10) / 10]
            )
            all_header_astropy_units = self._get_header_astropy_units(
                self.all_run_keys, all_run_units
            )

            # Fill the header dictionary
            for i_key, key in enumerate(self.all_run_keys[1:]):  # starting at the second
                # element to avoid the non-numeric key.
                self._header[key] = self.iact_file.header[i_key + 1] * all_header_astropy_units[key]
        return self._header

    def read_event_information(self):
        """
        Read the information about the events from their headers and save as a class instance.
        The main information can also be fetched individually through the functions below.
        For the remaining information (such as px, py, pz), use this function.

        """

        if self.event_information is None:

            with IACTFile(self.input_file) as self.iact_file:
                self.telescope_positions = np.array(self.iact_file.telescope_positions)
                self.num_telescopes = np.size(self.telescope_positions, axis=0)
                self.all_event_keys = list(
                    event_header.event_header_types[np.trunc(self.version * 10) / 10].names
                )
                all_event_units = get_units_from_fields(
                    event_header.event_header_fields[np.trunc(self.version * 10) / 10]
                )

                self.event_information = {key: [] for key in self.all_event_keys}

                self.num_events = 0
                # Build a dictionary with the parameters for the events.
                for event in self.iact_file:
                    for i_key, key in enumerate(self.all_event_keys[1:]):
                        self.event_information[key].append(event.header[i_key + 1])

                    self.num_events += 1

                all_event_astropy_units = self._get_header_astropy_units(
                    self.all_event_keys, all_event_units
                )

                # Add the unity to dictionary with the parameters and turn it into
                # astropy.Quantities.
                for i_key, key in enumerate(self.all_event_keys[1:]):  # starting at the second
                    # element to avoid the non-numeric (e.g. 'EVTH') key.
                    self.event_information[key] = (
                        np.array(self.event_information[key]) * all_event_astropy_units[key]
                    )

    def _get_header_astropy_units(self, parameters, non_astropy_units):
        """
        Return the dictionary with astropy units from the given list of parameters.

        Parameters
        ----------
        parameters: list
            The list of parameters to extract the astropy units.
        non_astropy_units: dict
            A dictionary with the parameter units (in strings).

        Returns
        -------
        dict:
            A dictionary with the astropy units.
        """

        # Build a dictionary with astropy units for the unit of the event's (header's) parameters.
        all_event_astropy_units = {}
        for i_key, key in enumerate(parameters[1:]):  # starting at the second
            # element to avoid the non-numeric (e.g. 'EVTH') key.

            # We extract the astropy unit (dimensionless in case no unit is provided).
            if key in non_astropy_units:
                with cds.enable():
                    unit = u.Unit(non_astropy_units[key])
            else:
                unit = u.dimensionless_unscaled
            all_event_astropy_units[key] = unit
        return all_event_astropy_units

    @property
    def telescope_indices(self):
        """
        The telescope index (or indices), which are considered for the production of the histograms.

        Returns
        -------
        list:
            The indices of the telescopes of interest.
        """
        return self._telescope_indices

    @telescope_indices.setter
    def telescope_indices(self, telescope_new_indices):
        """
        Set the telescope index (or indices).
        If self.individual_telescopes is True, the indices of the telescopes passed are analyzed
        individually (different histograms for each telescope) even if all telescopes are listed.

        Parameters
        ----------
        telescope_new_indices: int or list of int or np.array of int
            The indices of the specific telescopes to be inspected. If not specified, all telescopes
            are treated together in one histogram and the value of self._telescope_indices is a list
            of all telescope indices.

        Raises
        ------
        TypeError:
            if the indices passed through telescope_index are not of type int.
        """

        if telescope_new_indices is None:
            self._telescope_indices = np.arange(self.num_telescopes)
        else:
            if not isinstance(telescope_new_indices, (list, np.ndarray)):
                telescope_new_indices = np.array([telescope_new_indices])
            for i_telescope in telescope_new_indices:
                if not isinstance(i_telescope, (int, np.int32, np.int64)):
                    msg = "The index or indices given are not of type int."
                    self._logger.error(msg)
                    raise TypeError
            self._telescope_indices = np.sort(telescope_new_indices)

    @property
    def hist_config(self):
        """
        The configuration of the histograms.

        Returns
        -------
        dict:
            the dictionary with the histogram configuration.
        """
        if self._hist_config is None:
            msg = "No configuration was defined before. The default config is being created now."
            self._logger.warning(msg)
            self._hist_config = self._create_histogram_default_config()
        return self._hist_config

    @hist_config.setter
    def hist_config(self, input_config):
        """
        Set the configuration for the histograms (e.g., bin size, min and max values, etc).
        The input is allowed either through a yaml file or a dictionary. If nothing is given,
        the dictionary is created with default values.

        Parameters
        ----------
        input_config: str, Path, dict or NoneType
            yaml file with the configuration parameters to create the histograms. For the correct
            format, please look at the docstring at `_create_histogram_default_config`.
            Alternatively, it can be a dictionary with the configuration parameters to create
            the histograms.
        """
        input_dict, input_yaml = None, None
        if isinstance(input_config, dict):
            input_dict = input_config
        elif isinstance(input_config, (type(Path("")), str)):
            input_yaml = input_config
        self._hist_config = collect_data_from_yaml_or_dict(input_yaml, input_dict, allow_empty=True)

    def hist_config_to_yaml(self, file_name=None):
        """
        Save the histogram configuration dictionary to a yaml file.

        Parameters
        ----------
        file_name: str
            Name of the output file, in which to save the histogram configuration.

        """
        if file_name is None:
            file_name = "hist_config"
        if file_name[-4:] != ".yml":
            file_name = f"{file_name}.yml"
        output_config_path = self.io_handler.get_output_directory(self.label, "corsika")
        output_config_file = output_config_path.joinpath(file_name)
        self._logger.info(f"Exporting histogram configuration to {output_config_file}")
        with open(output_config_file, "w") as file:
            yaml.dump(self.hist_config, file)

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
            xy_bin = 100

        else:
            xy_maximum = 16 * u.m
            xy_bin = 64

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

            if isinstance(self.hist_config[label][axis]["start"], u.quantity.Quantity):
                start = self.hist_config[label][axis]["start"].value
                stop = self.hist_config[label][axis]["stop"].value
            else:
                start = self.hist_config[label][axis]["start"]
                stop = self.hist_config[label][axis]["stop"]
            boost_axes.append(
                bh.axis.Regular(
                    bins=self.hist_config[label][axis]["bins"],
                    start=start,
                    stop=stop,
                    transform=transform[self.hist_config[label][axis]["scale"]],
                )
            )
        return boost_axes

    def _create_histograms(self, individual_telescopes=False):
        """
        Create the histogram instances.

        Parameters
        ----------
        individual_telescopes: bool
            if False, the histograms are supposed to be filled for all telescopes.
            if True, one histogram is set for each telescope sepparately.
        """
        self.individual_telescopes = individual_telescopes
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
        for i_tel_info, photons_info in np.array(list(zip(self.telescope_positions, photons)))[
            self.telescope_indices
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
            if self.individual_telescopes is True:
                hist_num += 1

    def set_histograms(self, telescope_indices=None, individual_telescopes=None, hist_config=None):
        """
        Extract the information of the Cherenkov photons from a CORSIKA output IACT file, create
         and fill the histograms

        Parameters
        ----------
        telescope_indices: int or list of int
            The indices of the specific telescopes to be inspected.
        individual_telescopes: bool
            if False, the histograms are supposed to be filled for all telescopes. Default is False.
            if True, one histogram is set for each telescope sepparately.
        hist_config:
            yaml file with the configuration parameters to create the histograms. For the correct
            format, please look at the docstring at `_create_histogram_default_config`.
            Alternatively, it can be a dictionary with the configuration parameters to create
            the histograms.

        Returns
        -------
        list: list of boost_histogram.Histogram instances.

        Raises
        ------
        AttributeError:
            if event has not photon saved.
        """
        self.telescope_indices = telescope_indices
        self.individual_telescopes = individual_telescopes
        self.hist_config = hist_config
        self._create_histograms(individual_telescopes=self.individual_telescopes)

        num_photons_per_event_per_telescope_to_set = []
        start_time = time.time()
        self._logger.debug("Starting reading the file at {}.".format(start_time))
        with IACTFile(self.input_file) as f:
            event_counter = 0
            for event in f:
                for i_telescope in self.telescope_indices:

                    if hasattr(event, "photon_bunches"):
                        photons = list(event.photon_bunches.values())
                    else:
                        msg = "The event has no associated photon bunches saved. "
                        self._logger.error(msg)
                        raise AttributeError

                    # Count photons only from the telescopes given by self.telescope_indices.
                    num_photons_per_event_per_telescope_to_set.append(event.n_photons[i_telescope])
                self._fill_histograms(
                    photons,
                    self.event_azimuth_angles[event_counter],
                    self.event_zenith_angles[event_counter],
                )
                event_counter += 1
        self.num_photons_per_event_per_telescope = num_photons_per_event_per_telescope_to_set
        self._logger.debug(
            f"Finished reading the file and creating the histograms in {time.time() - start_time} "
            f"seconds"
        )

    @property
    def individual_telescopes(self):
        """
        Return the individual telescopes as property.
        """
        return self._individual_telescopes

    @individual_telescopes.setter
    def individual_telescopes(self, new_individual_telescopes):
        """
        The following lines allow `individual_telescopes` to be defined before using this function
        but if any parameter is passed in this function, it overwrites the class attribute.

        Parameters
        ----------
        new_individual_telescopes: bool
            if False, the histograms are supposed to be filled for all telescopes.
            if True, one histogram is set for each telescope sepparately.

        Raises
        ------
        TypeError:
            if new_individual_telescopes passed are not of type bool.
        """

        if new_individual_telescopes is None:
            if self._individual_telescopes is None:
                self._individual_telescopes = False
        else:
            if isinstance(new_individual_telescopes, bool):
                self._individual_telescopes = new_individual_telescopes
            else:
                msg = (
                    f"`individual_telescopes` passed {new_individual_telescopes} is not of type "
                    f"bool."
                )
                self._logger.error(msg)
                raise TypeError

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
            msg = f"label is not valid. Valid entries are {self._allowed_2D_labels}"
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
            Number of photons per event per telescope in self.telescope_indices.
        numpy.array
            An array that counts the telescopes in self.telescope_indices
        numpy.ndarray
            The counts of the histogram.
        """
        num_events_array = np.arange(self.num_events + 1)
        # It counts only the telescope indices given by self.telescope_indices.
        # The + 1 closes the last edge.
        telescope_counter = np.arange(len(self.telescope_indices) + 1)
        return (
            num_events_array,
            telescope_counter,
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
            msg = f"{label} is not valid. Valid entries are {self._allowed_1D_labels}"
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
                bin_size = 50
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
            The edges of the time histograms in ns.
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
            The edges of the photon altitude histograms in km.
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

    def get_num_photons_distr(self, bins=50, range=None, event_or_telescope="event"):
        """
        Get the distribution of photons per event.

        Parameters
        ----------
        bins: float
            Number of bins for the histogram.
        range: 2-tuple
            Tuple to define the range of the histogram.
        event_or_telescope: str
            Indicates if the distribution of photons is given for the events, or for the telescopes.
            Allowed values are: "event" or "telescope".


        Returns
        -------
        numpy.array
            Number of photons per event.
        numpy.ndarray
            The counts of the histogram.

        Raises
        ------
        ValueError:
            if event_or_telescope not valid.
        """
        if event_or_telescope == "event":
            hist, edges = np.histogram(self.num_photons_per_event, bins=bins, range=range)
        elif event_or_telescope == "telescope":
            hist, edges = np.histogram(self.num_photons_per_telescope, bins=bins, range=range)
        else:
            msg = "`event_or_telescope` has to be either 'event' or 'telescope'."
            self._logger.error(msg)
            raise ValueError
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

    # In the next five functions, we provide dedicated functions to retrieve specific information
    # about the runs, i.e. zenith, azimuth, total energy, interaction height and Earth magnetic
    # field defined for the run. For other information, please use the `get_event_parameter_info`
    # function.
    @property
    def event_zenith_angles(self):
        """
        Get the zenith angles of the simulated events in astropy units of degrees.

        Returns
        -------
        astropy.Quantity
            The zenith angles for each event, usually in degrees.
        """
        if self._event_zenith_angles is None:

            self._event_zenith_angles = np.around(
                (self.event_information["zenith"]).to(u.deg),
                4,
            )
        return self._event_zenith_angles

    @property
    def event_azimuth_angles(self):
        """
        Get the azimuth angles of the simulated events in astropy units of degrees.

        Returns
        -------
        astropy.Quantity
            The azimuth angles for each event, usually in degrees.
        """
        if self._event_azimuth_angles is None:
            self._event_azimuth_angles = np.around(
                (self.event_information["azimuth"]).to(u.deg),
                4,
            )
        return self._event_azimuth_angles

    @property
    def event_energies(self):
        """
        Get the energy of the simulated events in astropy units of TeV.

        Returns
        -------
        astropy.Quantity
            The total energies of the incoming particles for each event, usually in TeV.
        """
        if self._event_total_energies is None:
            self._event_total_energies = np.around(
                (self.event_information["total_energy"]).to(u.TeV),
                4,
            )
        return self._event_total_energies

    @property
    def event_first_interaction_heights(self):
        """
        Get the height of the first interaction in astropy units of km.
        If negative, tracking starts at margin of atmosphere, see TSTART in the CORSIKA 7 user guide
        .

        Returns
        -------
        astropy.Quantity
            The first interaction height for each event, usually in km.
        """
        if self._event_first_interaction_heights is None:
            self._event_first_interaction_heights = np.around(
                (self.event_information["first_interaction_height"]).to(u.km),
                4,
            )
        return self._event_first_interaction_heights

    @property
    def magnetic_field(self):
        """
        Get the Earth magnetic field from the events header in astropy units of microT.
        A tuple with Earth's magnetic field in the x and z directions are returned.

        Returns
        -------
        astropy.Quantity
            The Earth magnetic field in the x direction used for each event.
        astropy.Quantity
            The Earth magnetic field in the y direction used for each event.
        """
        if self._magnetic_field_x is None:
            self._magnetic_field_x = (self.event_information["earth_magnetic_field_x"]).to(u.uT)
        if self._magnetic_field_y is None:
            self._magnetic_field_x = (self.event_information["earth_magnetic_field_z"]).to(u.uT)
        return self._magnetic_field_x, self._magnetic_field_y

    def get_event_parameter_info(self, parameter):
        """
        Get specific information (i.e. any parameter) of the events. The parameter is passed through
        the key word `parameter`. Available options are to be found under `self.all_event_keys`.
        The unit of the parameter, if any, is given according to the CORSIKA version
        (please see user guide in this case).

        Parameters
        ----------
        parameter: str
            The parameter of interest. Available options are to be found under
            `self.all_event_keys`.

        Returns
        -------
        astropy.Quantity
            The array with the event information as required by the parameter.

        Raises
        ------
        KeyError:
            If parameter is not valid.
        """

        if parameter not in self.all_event_keys:
            msg = f"`key` is not valid. Valid entries are {self.all_event_keys}"
            self._logger.error(msg)
            raise KeyError
        return self.event_information[parameter]

    def get_run_info(self, parameter):
        """
        Get specific information (i.e. any parameter) of the run. The parameter is passed through
        the key word `parameter`. Available options are to be found under `self.all_run_keys`.
        The unit of the parameter, if any, is given according to the CORSIKA version
        (please see user guide in this case).

        Parameters
        ----------
        parameter: str
            The parameter of interest. Available options are to be found under
            `self.all_run_keys`.

        Raises
        ------
        KeyError:
            If parameter is not valid.
        """

        if parameter not in self.all_run_keys:
            msg = f"`key` is not valid. Valid entries are {self.all_run_keys}"
            self._logger.error(msg)
            raise KeyError
        return self.header[parameter]

    def event_1D_histogram(self, key, bins=50, range=None):
        """
        Create a histogram for the all events using `key` as parameter.
        Valid keys are stored in `self.all_event_keys` (CORSIKA defined).

        Parameters
        ----------
        key: str
            The information from which to build the histogram, e.g. total_energy, zenith or
            first_interaction_height.
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
        if key not in self.all_event_keys:
            msg = f"`key` is not valid. Valid entries are {self.all_event_keys}"
            self._logger.error(msg)
            raise KeyError
        hist, edges = np.histogram(
            self.event_information[key].value,
            bins=bins,
            range=range,
        )
        return edges, hist

    def event_2D_histogram(self, key_1, key_2, bins=50, range=None):
        """
        Create a 2D histogram for the all events using `key_1` and `key_2` as parameters.
        Valid keys are stored in `self.all_event_keys` (CORSIKA defined).

        Parameters
        ----------
        key_1: str
            The information from which to build the histogram, e.g. total_energy, zenith or
            first_interaction_height.
        key_2: str
            The information from which to build the histogram, e.g. total_energy, zenith or
            first_interaction_height.
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
            if key not in self.all_event_keys:
                msg = (
                    f"At least one of the keys given is not valid. Valid entries are "
                    f"{self.all_event_keys}"
                )
                self._logger.error(msg)
                raise KeyError
        hist, x_edges, y_edges = np.histogram2d(
            self.event_information[key_1].value,
            self.event_information[key_2].value,
            bins=bins,
            range=range,
        )
        return x_edges, y_edges, hist
