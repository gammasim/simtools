"""Extract Cherenkov photons information from a CORSIKA IACT file."""

import functools
import logging
import operator
import re
import time
from pathlib import Path

import boost_histogram as bh
import numpy as np
from astropy import units as u
from astropy.io.misc import yaml
from astropy.units import cds
from corsikaio.subblocks import event_header, get_units_from_fields, run_header
from ctapipe.io import write_table
from eventio import IACTFile

from simtools import version
from simtools.io import io_handler
from simtools.io.ascii_handler import collect_data_from_file
from simtools.io.hdf5_handler import fill_hdf5_table
from simtools.utils.geometry import convert_2d_to_radial_distr, rotate
from simtools.utils.names import sanitize_name

X_AXIS_STRING = "x axis"
Y_AXIS_STRING = "y axis"
Z_AXIS_STRING = "z axis"


class HistogramNotCreatedError(Exception):
    """Exception for histogram not created."""


class CorsikaHistograms:
    """
    Extracts the Cherenkov photons information from a CORSIKA IACT file.

    Parameters
    ----------
    input_file: str or Path
        CORSIKA IACT file provided by the CORSIKA simulation.
    label: str
        Instance label.
    output_path: str
        Path where to save the output of the class methods.
    hdf5_file_name: str
        HDF5 file name for histogram storage.

    Raises
    ------
    FileNotFoundError:
        if the input file given does not exist.
    """

    def __init__(self, input_file, label=None, output_path=None, hdf5_file_name=None):
        self.label = label
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaHistograms")
        self.input_file = input_file

        self.input_file = Path(self.input_file)
        if not self.input_file.exists():
            msg = f"file {self.input_file} does not exist."
            self._logger.error(msg)
            raise FileNotFoundError

        self.io_handler = io_handler.IOHandler()
        _default_output_path = self.io_handler.get_output_directory(self.label, "corsika")

        if output_path is None:
            self.output_path = _default_output_path
        else:
            self.output_path = Path(output_path)

        if hdf5_file_name is None:
            self.hdf5_file_name = re.split(r"\.", self.input_file.name)[0] + ".hdf5"
        else:
            self.hdf5_file_name = hdf5_file_name

        self._telescope_indices = None
        self._telescope_positions = None
        self.num_events = None
        self.num_of_hist = None
        self.num_telescopes = None
        self._num_photons_per_event_per_telescope = None
        self._num_photons_per_event = None
        self._num_photons_per_telescope = None
        self.__meta_dict = None
        self._dict_2d_distributions = None
        self._dict_1d_distributions = None
        self._event_azimuth_angles = None
        self._event_zenith_angles = None
        self._hist_config = None
        self._total_num_photons = None
        self._magnetic_field_x = None
        self._magnetic_field_z = None
        self._event_total_energies = None
        self._event_first_interaction_heights = None
        self._corsika_version = None
        self.event_information = None
        self._individual_telescopes = None
        self._allowed_histograms = {"hist_position", "hist_direction", "hist_time_altitude"}
        self._allowed_1d_labels = {"wavelength", "time", "altitude"}
        self._allowed_2d_labels = {"counts", "density", "direction", "time_altitude"}
        self._header = None
        self.hist_position = None
        self.hist_direction = None
        self.hist_time_altitude = None

        self.read_event_information()
        self._initialize_header()

    @property
    def hdf5_file_name(self):
        """
        Property for the hdf5 file name.

        The idea of this property is to allow setting (or changing) the name of the hdf5 file
        even after creating the class instance.
        """
        return self._hdf5_file_name

    @hdf5_file_name.setter
    def hdf5_file_name(self, hdf5_file_name):
        """
        Set the hdf5_file_name to the argument passed.

        Parameters
        ----------
        hdf5_file_name: str
            The name of hdf5 file to be set.
        """
        self._hdf5_file_name = Path(self.output_path).joinpath(hdf5_file_name).absolute().as_posix()

    @property
    def corsika_version(self):
        """
        Get the version of the CORSIKA IACT file.

        Returns
        -------
        float:
            The version of CORSIKA used to produce the CORSIKA IACT file given by self.input_file.
        """
        if self._corsika_version is None:
            all_corsika_versions = list(run_header.run_header_types.keys())
            header = list(self.iact_file.header)

            for i_version in reversed(all_corsika_versions):
                # Get the event header for this software version being tested.
                single_run_header = run_header.run_header_types[i_version]
                # Get the position in the dictionary, where the version is.
                version_index_position = np.argwhere(
                    np.array(list(single_run_header.names)) == "version"
                )[0]

                # Check if version tested is the same as the version written in the file header.
                if i_version == np.trunc(float(header[version_index_position[0]]) * 10) / 10:
                    # If the version found is the same as the initial guess, leave the loop,
                    # otherwise, iterate until we find the correct version.
                    self._corsika_version = np.around(float(header[version_index_position[0]]), 3)
                    break
        return self._corsika_version

    def _initialize_header(self):
        """Initialize the header."""
        self.all_run_keys = list(
            run_header.run_header_types[np.around(self.corsika_version, 1)].names
        )
        self._header = {}

        # Get units of the header
        all_run_units = get_units_from_fields(
            run_header.run_header_fields[np.trunc(self.corsika_version * 10) / 10]
        )
        all_header_astropy_units = self._get_header_astropy_units(self.all_run_keys, all_run_units)

        # Fill the header dictionary
        for i_key, key in enumerate(self.all_run_keys[1:]):  # starting at the second
            # element to avoid the non-numeric key.
            self._header[key] = self.iact_file.header[i_key + 1] * all_header_astropy_units[key]

    @property
    def header(self):
        """
        Get the run header.

        Returns
        -------
        dict:
            The run header.
        """
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
                    event_header.event_header_types[np.trunc(self.corsika_version * 10) / 10].names
                )
                all_event_units = get_units_from_fields(
                    event_header.event_header_fields[np.trunc(self.corsika_version * 10) / 10]
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

                # Add the units to dictionary with the parameters and turn it into
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
        for key in parameters[1:]:  # starting at the second
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
            if not isinstance(telescope_new_indices, list | np.ndarray):
                telescope_new_indices = np.array([telescope_new_indices])
            for i_telescope in telescope_new_indices:
                if not isinstance(i_telescope, int | np.integer):
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
            msg = (
                "No histogram configuration was defined before. The default config is being "
                "created now."
            )
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
            format, please look at the docstring at _create_histogram_default_config.
            Alternatively, it can be a dictionary with the configuration parameters to create
            the histograms.
        """
        if isinstance(input_config, dict):
            self._hist_config = input_config
        else:
            self._hist_config = collect_data_from_file(input_config) if input_config else None

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
        file_name = Path(file_name).with_suffix(".yml")
        output_config_file_name = Path(self.output_path).joinpath(file_name)
        with open(output_config_file_name, "w", encoding="utf-8") as file:
            yaml.dump(self.hist_config, file)

    def _create_histogram_default_config(self):
        """
        Create a dictionary with the configuration necessary to create the histograms.

        It is used only in case the configuration is not provided in a yaml file or dict.

        Three histograms are created: hist_position with 3 dimensions (x, y positions and the
        wavelength), hist_direction with 2 dimensions (direction cosines in x and y directions),
        hist_time_altitude with 2 dimensions (time and altitude of emission).

        Four arguments are passed to each dimension in the dictionary:

        "bins": the number of bins,
        "start": the first element of the histogram,
        "stop": the last element of the histogram, and
        "scale" to define the scale of the bins which can be "linear" or "log". If "log", the
         common logarithm (log10) is applied to the axes. The start and stop values have to be
         valid, i.e., >0.

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
        return {
            "hist_position": {
                X_AXIS_STRING: {
                    "bins": xy_bin,
                    "start": -xy_maximum,
                    "stop": xy_maximum,
                    "scale": "linear",
                },
                Y_AXIS_STRING: {
                    "bins": xy_bin,
                    "start": -xy_maximum,
                    "stop": xy_maximum,
                    "scale": "linear",
                },
                Z_AXIS_STRING: {
                    "bins": 80,
                    "start": 200 * u.nm,
                    "stop": 1000 * u.nm,
                    "scale": "linear",
                },
            },
            "hist_direction": {
                X_AXIS_STRING: {
                    "bins": 100,
                    "start": -1,
                    "stop": 1,
                    "scale": "linear",
                },
                Y_AXIS_STRING: {
                    "bins": 100,
                    "start": -1,
                    "stop": 1,
                    "scale": "linear",
                },
            },
            "hist_time_altitude": {
                X_AXIS_STRING: {
                    "bins": 100,
                    "start": -2000 * u.ns,
                    "stop": 2000 * u.ns,
                    "scale": "linear",
                },
                Y_AXIS_STRING: {
                    "bins": 100,
                    "start": 120 * u.km,
                    "stop": 0 * u.km,
                    "scale": "linear",
                },
            },
        }

    def _create_regular_axes(self, label):
        """
        Create regular axis for histograms.

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
            raise ValueError(msg)

        all_axes = [X_AXIS_STRING, Y_AXIS_STRING]
        if label == "hist_position":
            all_axes.append(Z_AXIS_STRING)

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
            if False, the histograms are filled for all given telescopes together.
            if True, one histogram is set for each telescope separately.
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

    def _fill_histograms(self, photons, rotation_around_z_axis=None, rotation_around_y_axis=None):
        """
        Fill histograms with the information of the photons on the ground.

        if the azimuth and zenith angles are provided, the Cherenkov photon's coordinates are
        filled in the plane perpendicular to the incoming direction of the particle.

        Parameters
        ----------
        photons: list
            List of size M of numpy.array of size (N,8), where M is the number of telescopes in the
            array, N is the number of photons that reached each telescope. The following information
             of the Cherenkov photons on the ground are saved:
             x: x position on the ground (CORSIKA coordinate system),
             y: y position on the ground (CORSIKA coordinate system),
             cx: direction cosine in the x direction, i.e., the cosine of the angle between the
             incoming direction and the x axis,
             cy: direction cosine in the y direction, i.e., the cosine of the angle between the
             incoming direction and the y axis,
             time: time of arrival of the photon in ns. The clock starts when the particle crosses
             the top of the atmosphere (CORSIKA-defined) if self.event_first_interaction_heights
             is positive or at first interaction if otherwise.
             zem: altitude where the photon was generated in cm,
             photons: number of photons associated to this bunch,
             wavelength: the wavelength of the photons in nm.
        rotation_around_z_axis: astropy.Quantity
            Angle to rotate the observational plane around the Z axis.
            It can be passed in radians or degrees.
            If not given, no rotation is performed.
        rotation_around_y_axis: astropy.Quantity
            Angle to rotate the observational plane around the Y axis.
            It can be passed in radians or degrees.
            If not given, no rotation is performed.
            rotation_around_z_axis and rotation_around_y_axis are used to align the observational
            plane normal to the incoming direction of the shower particles for the individual
             telescopes (useful for fixed targets).

        Raises
        ------
        IndexError:
            If the index or indices passed though telescope_index are out of range.
        """
        hist_num = 0
        for i_tel_info, photons_info in np.array(
            list(zip(self.telescope_positions, photons)), dtype=object
        )[self.telescope_indices]:
            if rotation_around_z_axis is None or rotation_around_y_axis is None:
                photon_x, photon_y = photons_info["x"], photons_info["y"]
            else:
                photon_x, photon_y = rotate(
                    photons_info["x"],
                    photons_info["y"],
                    rotation_around_z_axis,
                    rotation_around_y_axis,
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
        Create and fill Cherenkov photons histograms using information from the CORSIKA IACT file.

        Parameters
        ----------
        telescope_indices: int or list of int
            The indices of the specific telescopes to be inspected.
        individual_telescopes: bool
            if False, the histograms are supposed to be filled for all telescopes. Default is False.
            if True, one histogram is set for each telescope separately.
        hist_config:
            yaml file with the configuration parameters to create the histograms. For the correct
            format, please look at the docstring of _create_histogram_default_config.
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
        self._logger.debug(f"Starting reading the file at {start_time}.")
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
        """Return the individual telescopes as property."""
        return self._individual_telescopes

    @individual_telescopes.setter
    def individual_telescopes(self, new_individual_telescopes: bool):
        """
        Define individual telescopes.

        Parameters
        ----------
        new_individual_telescopes: bool
            if False, the histograms are supposed to be filled for all telescopes.
            if True, one histogram is set for each telescope separately.
        """
        if new_individual_telescopes is None:
            self._individual_telescopes = False
        else:
            self._individual_telescopes = new_individual_telescopes

    def _raise_if_no_histogram(self):
        """
        Raise an error if the histograms were not created.

        Raises
        ------
        HistogramNotCreatedError:
            if the histogram was not previously created.
        """
        for histogram in self._allowed_histograms:
            if not hasattr(self, histogram) or getattr(self, histogram) is None:
                msg = (
                    "The histograms were not created. Please, use create_histograms to create "
                    "histograms from the CORSIKA output file."
                )
                self._logger.error(msg)
                raise HistogramNotCreatedError

    def _get_hist_2d_projection(self, label):
        """
        Get 2D distributions.

        Parameters
        ----------
        label: str
            Label to indicate which histogram.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The x bin edges of the histograms.
        numpy.array
            The y bin edges of the histograms.

        Raises
        ------
        ValueError:
            if label is not valid.
        """
        if label not in self._allowed_2d_labels:
            msg = f"label is not valid. Valid entries are {self._allowed_2d_labels}"
            self._logger.error(msg)
            raise ValueError(msg)
        self._raise_if_no_histogram()

        num_telescopes_to_fill = (
            len(self.telescope_indices) if self.individual_telescopes is True else 1
        )

        x_bin_edges, y_bin_edges, hist_values = [], [], []
        for i_telescope in range(num_telescopes_to_fill):
            mini_hist = None
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
            x_bin_edges.append(mini_hist.axes.edges[0].flatten())
            y_bin_edges.append(mini_hist.axes.edges[1].flatten())

        return np.array(hist_values), np.array(x_bin_edges), np.array(y_bin_edges)

    def get_2d_photon_position_distr(self):
        """
        Get 2D histograms of position of the Cherenkov photons on the ground.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The x bin edges of the count histograms in x, usually in meters.
        numpy.array
            The y bin edges of the count histograms in y, usually in meters.
        """
        return self._get_hist_2d_projection("counts")

    def get_2d_photon_density_distr(self):
        """
        Get 2D histograms of position of the Cherenkov photons on the ground.

        It returns the photon density per square meter.

        Returns
        -------
        numpy.ndarray
            The values of the histogram, usually in $m^{-2}$
        numpy.array
            The x bin edges of the density/count histograms in x, usually in meters.
        numpy.array
            The y bin edges of the density/count histograms in y, usually in meters.
        """
        return self._get_hist_2d_projection("density")

    def get_2d_photon_direction_distr(self):
        """
        Get 2D histograms of incoming direction of the Cherenkov photons on the ground.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The x bin edges of the direction histograms in cos(x).
        numpy.array
            The y bin edges of the direction histograms in cos(y)
        """
        return self._get_hist_2d_projection("direction")

    def get_2d_photon_time_altitude_distr(self):
        """
        Get 2D histograms of the time and altitude of the photon production.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The x bin edges of the time_altitude histograms, usually in ns.
        numpy.array
            The y bin edges of the time_altitude histograms, usually in km.
        """
        return self._get_hist_2d_projection("time_altitude")

    def get_2d_num_photons_distr(self):
        """
        Get the distribution of Cherenkov photons per event per telescope.

        It returns the 2D array accounting for the events from the telescopes given
        by self.telescope_indices.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            An array that counts the telescopes in self.telescope_indices
        numpy.array
            Number of photons per event per telescope in self.telescope_indices.
        """
        num_events_array = np.arange(self.num_events + 1).reshape(1, self.num_events + 1)
        # It counts only the telescope indices given by self.telescope_indices.
        # The + 1 closes the last edge.
        telescope_counter = np.arange(len(self.telescope_indices) + 1).reshape(
            1, len(self.telescope_indices) + 1
        )
        hist_2d = np.array(self.num_photons_per_event_per_telescope)
        hist_2d = hist_2d.reshape((1, len(self.telescope_indices), self.num_events))
        return (hist_2d, num_events_array, telescope_counter)

    def _get_hist_1d_projection(self, label):
        """
        Get 1D distributions.

        Parameters
        ----------
        label: str
            Label to indicate which histogram.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The bin edges of the histogram.

        Raises
        ------
        ValueError:
            if label is not valid.
        """
        if label not in self._allowed_1d_labels:
            msg = f"{label} is not valid. Valid entries are {self._allowed_1d_labels}"
            self._logger.error(msg)
            raise ValueError(msg)
        self._raise_if_no_histogram()

        x_bin_edges_list, hist_1d_list = [], []
        for i_hist, _ in enumerate(self.hist_position):
            mini_hist = None
            if label == "wavelength":
                mini_hist = self.hist_position[i_hist][sum, sum, :]
            elif label == "time":
                mini_hist = self.hist_time_altitude[i_hist][:, sum]
            elif label == "altitude":
                mini_hist = self.hist_time_altitude[i_hist][sum, :]

            x_bin_edges_list.append(mini_hist.axes.edges.T.flatten()[0])
            hist_1d_list.append(mini_hist.view().T)
        return np.array(hist_1d_list), np.array(x_bin_edges_list)

    def _get_bins_max_dist(self, bins=None, max_dist=None):
        """
        Get the number of bins and the max distance to generate the radial and density histograms.

        Parameters
        ----------
        bins: float
            Number of bins of the radial distribution.
        max_dist: float
            Maximum distance to consider in the 1D histogram (in meters).
        """
        hist_position = "hist_position"
        if max_dist is None:
            max_dist = np.amax(
                [
                    self.hist_config[hist_position][X_AXIS_STRING]["start"].to(u.m).value,
                    self.hist_config[hist_position][X_AXIS_STRING]["stop"].to(u.m).value,
                    self.hist_config[hist_position][Y_AXIS_STRING]["start"].to(u.m).value,
                    self.hist_config[hist_position][Y_AXIS_STRING]["stop"].to(u.m).value,
                ]
            )
        if bins is None:
            bins = (
                np.amax(
                    [
                        self.hist_config[hist_position][X_AXIS_STRING]["bins"],
                        self.hist_config[hist_position][Y_AXIS_STRING]["bins"],
                    ]
                )
                // 2
            )  # //2 because of the 2D array going into the negative and
            # positive axis
        return bins, max_dist

    def get_photon_radial_distr(self, bins=None, max_dist=None):
        """
        Get the phton radial distribution on the ground in relation to the center of the array.

        Parameters
        ----------
        bins: float
            Number of bins of the radial distribution.
        max_dist: float
            Maximum distance to consider in the 1D histogram (in meters).

        Returns
        -------
        np.array
            The counts of the 1D histogram with size = int(max_dist/bin_size).
        np.array
            The bin edges of the 1D histogram in meters with size = int(max_dist/bin_size) + 1,
            usually in meter.
        """
        bins, max_dist = self._get_bins_max_dist(bins=bins, max_dist=max_dist)
        bin_edges_1d_list, hist_1d_list = [], []

        hist_2d_values_list, x_position_list, y_position_list = self.get_2d_photon_position_distr()

        for i_hist, x_pos in enumerate(x_position_list):
            hist_1d, bin_edges_1d = convert_2d_to_radial_distr(
                hist_2d_values_list[i_hist],
                x_pos,
                y_position_list[i_hist],
                bins=bins,
                max_dist=max_dist,
            )
            bin_edges_1d_list.append(bin_edges_1d)
            hist_1d_list.append(hist_1d)
        return np.array(hist_1d_list), np.array(bin_edges_1d_list)

    def get_photon_density_distr(self, bins=None, max_dist=None):
        """
        Get the photon density distribution on the ground in relation to the center of the array.

        Parameters
        ----------
        bins: float
            Number of bins of the radial distribution.
        max_dist: float
            Maximum distance to consider in the 1D histogram (in meters).

        Returns
        -------
        np.array
            The density distribution of the 1D histogram with size = int(max_dist/bin_size),
            usually in $m^{-2}$.
        np.array
            The bin edges of the 1D histogram in meters with size = int(max_dist/bin_size) + 1,
            usually in meter.
        """
        bins, max_dist = self._get_bins_max_dist(bins=bins, max_dist=max_dist)
        bin_edges_1d_list, hist_1d_list = [], []

        hist_2d_values_list, x_position_list, y_position_list = self.get_2d_photon_density_distr()

        for i_hist, _ in enumerate(x_position_list):
            hist_1d, bin_edges_1d = convert_2d_to_radial_distr(
                hist_2d_values_list[i_hist],
                x_position_list[i_hist],  # pylint: disable=unnecessary-list-index-lookup
                y_position_list[i_hist],
                bins=bins,
                max_dist=max_dist,
            )
            bin_edges_1d_list.append(bin_edges_1d)
            hist_1d_list.append(hist_1d)
        return np.array(hist_1d_list), np.array(bin_edges_1d_list)

    def get_photon_wavelength_distr(self):
        """
        Get histograms with the wavelengths of the photon bunches.

        Returns
        -------
        np.array
            The counts of the wavelength histogram.
        np.array
            The bin edges of the wavelength histogram in nanometers.

        """
        return self._get_hist_1d_projection("wavelength")

    def get_photon_time_of_emission_distr(self):
        """
        Get the distribution of the emitted time of the Cherenkov photons.

        The clock starts when the particle crosses the top of the atmosphere (CORSIKA-defined) if
        self.event_first_interaction_heights is positive or at first interaction if otherwise.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The bin edges of the time histograms in ns.

        """
        return self._get_hist_1d_projection("time")

    def get_photon_altitude_distr(self):
        """
        Get the emission altitude of the Cherenkov photons.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The bin edges of the photon altitude histograms in km.

        """
        return self._get_hist_1d_projection("altitude")

    @property
    def num_photons_per_event_per_telescope(self):
        """The number of photons per event per telescope."""
        return self._num_photons_per_event_per_telescope

    @num_photons_per_event_per_telescope.setter
    def num_photons_per_event_per_telescope(self, num_photons_per_event_per_telescope_to_set):
        """Set the number of photons per event per telescope."""
        self._num_photons_per_event_per_telescope = (
            np.array(num_photons_per_event_per_telescope_to_set)
            .reshape(self.num_events, len(self.telescope_indices))
            .T
        )

    @property
    def num_photons_per_event(self):
        """
        Get the the number of photons per events.

        Includes the telescopes indicated by self.telescope_indices.

        Returns
        -------
        numpy.array
            Number of photons per event.
        """
        self._num_photons_per_event = np.sum(self.num_photons_per_event_per_telescope, axis=0)
        return self._num_photons_per_event

    def get_num_photons_per_event_distr(self, bins=50, hist_range=None):
        """
        Get the distribution of photons per event.

        Parameters
        ----------
        bins: float
            Number of bins for the histogram.
        hist_range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            Number of photons per event.
        """
        hist, bin_edges = np.histogram(self.num_photons_per_event, bins=bins, range=hist_range)
        return hist.reshape(1, bins), bin_edges.reshape(1, bins + 1)

    def get_num_photons_per_telescope_distr(self, bins=50, hist_range=None):
        """
        Get the distribution of photons per telescope.

        Parameters
        ----------
        bins: float
            Number of bins for the histogram.
        hist_range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            Number of photons per telescope.
        """
        hist, bin_edges = np.histogram(self.num_photons_per_telescope, bins=bins, range=hist_range)
        return hist.reshape(1, bins), bin_edges.reshape(1, bins + 1)

    def export_histograms(self, overwrite=False):
        """
        Export the histograms to hdf5 files.

        Parameters
        ----------
        overwrite: bool
            If True overwrites the histograms already saved in the hdf5 file.
        """
        self._export_1d_histograms(overwrite=overwrite)
        self._export_2d_histograms(overwrite=False)

    @property
    def _meta_dict(self):
        """
        Define the meta dictionary for exporting the histograms.

        Returns
        -------
        dict
            Meta dictionary for the hdf5 files with the histograms.
        """
        if self.__meta_dict is None:
            self.__meta_dict = {
                "corsika_version": self.corsika_version,
                "simtools_version": version.__version__,
                "iact_file": self.input_file.name,
                "telescope_indices": list(self.telescope_indices),
                "individual_telescopes": self.individual_telescopes,
                "note": "Only lower bin edges are given.",
            }
        return self.__meta_dict

    @property
    def dict_1d_distributions(self):
        """
        Dictionary to label the 1D distributions according to the class methods.

        Returns
        -------
        dict:
            The dictionary with information about the 1D distributions.
        """
        fn_key = "function"
        file_name = "file name"
        title = "title"
        bin_edges = "bin edges"
        axis_unit = "axis unit"
        self._dict_1d_distributions = {
            "wavelength": {
                fn_key: "get_photon_wavelength_distr",
                file_name: "hist_1d_photon_wavelength_distr",
                title: "Photon wavelength distribution",
                bin_edges: "wavelength",
                axis_unit: self.hist_config["hist_position"][Z_AXIS_STRING]["start"].unit,
            },
            "counts": {
                fn_key: "get_photon_radial_distr",
                file_name: "hist_1d_photon_radial_distr",
                title: "Radial photon distribution on the ground",
                bin_edges: "Distance to center",
                axis_unit: self.hist_config["hist_position"][X_AXIS_STRING]["start"].unit,
            },
            "density": {
                fn_key: "get_photon_density_distr",
                file_name: "hist_1d_photon_density_distr",
                title: "Photon density distribution on the ground",
                bin_edges: "Distance to center",
                axis_unit: self.hist_config["hist_position"][X_AXIS_STRING]["start"].unit,
            },
            "time": {
                fn_key: "get_photon_time_of_emission_distr",
                file_name: "hist_1d_photon_time_distr",
                title: "Photon time of arrival distribution",
                bin_edges: "Time of arrival",
                axis_unit: self.hist_config["hist_time_altitude"][X_AXIS_STRING]["start"].unit,
            },
            "altitude": {
                fn_key: "get_photon_altitude_distr",
                file_name: "hist_1d_photon_altitude_distr",
                title: "Photon altitude of emission distribution",
                bin_edges: "Altitude of emission",
                axis_unit: self.hist_config["hist_time_altitude"][Y_AXIS_STRING]["start"].unit,
            },
            "num_photons_per_event": {
                fn_key: "get_num_photons_per_event_distr",
                file_name: "hist_1d_photon_per_event_distr",
                title: "Photons per event distribution",
                bin_edges: "Event counter",
                axis_unit: u.dimensionless_unscaled,
            },
            "num_photons_per_telescope": {
                fn_key: "get_num_photons_per_telescope_distr",
                file_name: "hist_1d_photon_per_telescope_distr",
                title: "Photons per telescope distribution",
                bin_edges: "Telescope counter",
                axis_unit: u.dimensionless_unscaled,
            },
        }
        return self._dict_1d_distributions

    def _export_1d_histograms(self, overwrite=False):
        """
        Auxiliary function to export only the 1D histograms.

        Parameters
        ----------
        overwrite: bool
            If True overwrites the histograms already saved in the hdf5 file.
        """
        axis_unit = "axis unit"
        for _, function_dict in self.dict_1d_distributions.items():
            self._meta_dict["Title"] = sanitize_name(function_dict["title"])
            histogram_function = getattr(self, function_dict["function"])
            hist_1d_list, x_bin_edges_list = histogram_function()
            x_bin_edges_list = x_bin_edges_list * function_dict[axis_unit]
            if function_dict["function"] == "get_photon_density_distr":
                histogram_value_unit = 1 / (function_dict[axis_unit] ** 2)
            else:
                histogram_value_unit = u.dimensionless_unscaled
            hist_1d_list = hist_1d_list * histogram_value_unit
            for i_histogram, _ in enumerate(x_bin_edges_list):
                if self.individual_telescopes:
                    hdf5_table_name = (
                        f"/{function_dict['file name']}_"
                        f"tel_index_{self.telescope_indices[i_histogram]}"
                    )
                else:
                    hdf5_table_name = f"/{function_dict['file name']}_all_tels"

                table = fill_hdf5_table(
                    hist=hist_1d_list[i_histogram],
                    x_bin_edges=x_bin_edges_list[i_histogram],
                    y_bin_edges=None,
                    x_label=function_dict["bin edges"],
                    y_label=None,
                    meta_data=self._meta_dict,
                )
                self._logger.info(
                    f"Writing 1D histogram with name {hdf5_table_name} to {self.hdf5_file_name}."
                )
                # overwrite takes precedence over append
                if overwrite is True:
                    append = False
                else:
                    append = True
                write_table(
                    table, self.hdf5_file_name, hdf5_table_name, append=append, overwrite=overwrite
                )

    @property
    def dict_2d_distributions(self):
        """
        Dictionary to label the 2D distributions according to the class methods.

        Returns
        -------
        dict:
            The dictionary with information about the 2D distributions.
        """
        fn_key = "function"
        file_name = "file name"
        title = "title"
        x_bin_edges = "x bin edges"
        x_axis_unit = "x axis unit"
        y_bin_edges = "y bin edges"
        y_axis_unit = "y axis unit"
        if self._dict_2d_distributions is None:
            self._dict_2d_distributions = {
                "counts": {
                    fn_key: "get_2d_photon_position_distr",
                    file_name: "hist_2d_photon_count_distr",
                    title: "Photon count distribution on the ground",
                    x_bin_edges: "x position on the ground",
                    x_axis_unit: self.hist_config["hist_position"][X_AXIS_STRING]["start"].unit,
                    y_bin_edges: "y position on the ground",
                    y_axis_unit: self.hist_config["hist_position"][Y_AXIS_STRING]["start"].unit,
                },
                "density": {
                    fn_key: "get_2d_photon_density_distr",
                    file_name: "hist_2d_photon_density_distr",
                    title: "Photon density distribution on the ground",
                    x_bin_edges: "x position on the ground",
                    x_axis_unit: self.hist_config["hist_position"][X_AXIS_STRING]["start"].unit,
                    y_bin_edges: "y position on the ground",
                    y_axis_unit: self.hist_config["hist_position"][Y_AXIS_STRING]["start"].unit,
                },
                "direction": {
                    fn_key: "get_2d_photon_direction_distr",
                    file_name: "hist_2d_photon_direction_distr",
                    title: "Photon arrival direction",
                    x_bin_edges: "x direction cosine",
                    x_axis_unit: u.dimensionless_unscaled,
                    y_bin_edges: "y direction cosine",
                    y_axis_unit: u.dimensionless_unscaled,
                },
                "time_altitude": {
                    fn_key: "get_2d_photon_time_altitude_distr",
                    file_name: "hist_2d_photon_time_altitude_distr",
                    title: "Time of arrival vs altitude of emission",
                    x_bin_edges: "Time of arrival",
                    x_axis_unit: self.hist_config["hist_time_altitude"][X_AXIS_STRING][
                        "start"
                    ].unit,
                    y_bin_edges: "Altitude of emission",
                    y_axis_unit: self.hist_config["hist_time_altitude"][Y_AXIS_STRING][
                        "start"
                    ].unit,
                },
                "num_photons_per_telescope": {
                    fn_key: "get_2d_num_photons_distr",
                    file_name: "hist_2d_photon_telescope_event_distr",
                    title: "Number of photons per telescope and per event",
                    x_bin_edges: "Telescope counter",
                    x_axis_unit: u.dimensionless_unscaled,
                    y_bin_edges: "Event counter",
                    y_axis_unit: u.dimensionless_unscaled,
                },
            }
        return self._dict_2d_distributions

    def _export_2d_histograms(self, overwrite):
        """
        Auxiliary function to export only the 2D histograms.

        Parameters
        ----------
        overwrite: bool
            If True overwrites the histograms already saved in the hdf5 file.
        """
        x_axis_unit = "x axis unit"
        y_axis_unit = "y axis unit"

        for property_name, function_dict in self.dict_2d_distributions.items():
            self._meta_dict["Title"] = sanitize_name(function_dict["title"])
            histogram_function = getattr(self, function_dict["function"])

            hist_2d_list, x_bin_edges_list, y_bin_edges_list = histogram_function()
            if function_dict["function"] == "get_2d_photon_density_distr":
                histogram_value_unit = 1 / (
                    self.dict_2d_distributions[property_name][x_axis_unit]
                    * self.dict_2d_distributions[property_name][y_axis_unit]
                )
            else:
                histogram_value_unit = u.dimensionless_unscaled

            hist_2d_list, x_bin_edges_list, y_bin_edges_list = (
                hist_2d_list * histogram_value_unit,
                x_bin_edges_list * self.dict_2d_distributions[property_name][x_axis_unit],
                y_bin_edges_list * self.dict_2d_distributions[property_name][y_axis_unit],
            )

            for i_histogram, _ in enumerate(x_bin_edges_list):
                if self.individual_telescopes:
                    hdf5_table_name = (
                        f"/{self.dict_2d_distributions[property_name]['file name']}"
                        f"_tel_index_{self.telescope_indices[i_histogram]}"
                    )
                else:
                    hdf5_table_name = (
                        f"/{self.dict_2d_distributions[property_name]['file name']}_all_tels"
                    )
                table = fill_hdf5_table(
                    hist=hist_2d_list[i_histogram],
                    x_bin_edges=x_bin_edges_list[i_histogram],
                    y_bin_edges=y_bin_edges_list[i_histogram],
                    x_label=function_dict["x bin edges"],
                    y_label=function_dict["y bin edges"],
                    meta_data=self._meta_dict,
                )

                self._logger.info(
                    f"Writing 2D histogram with name {hdf5_table_name} to {self.hdf5_file_name}."
                )
                # Always appending to table due to the file previously created
                # by self._export_1d_histograms.
                write_table(
                    table, self.hdf5_file_name, hdf5_table_name, append=True, overwrite=overwrite
                )

    def export_event_header_1d_histogram(
        self, event_header_element, bins=50, hist_range=None, overwrite=False
    ):
        """
        Export 'event_header_element' from CORSIKA to hd5 for a 1D histogram.

        Parameters
        ----------
        event_header_element: str
            The key to the CORSIKA event header element.
            Possible choices are stored in 'self.all_event_keys'.
        bins: float
            Number of bins for the histogram.
        hist_range: 2-tuple
            Tuple to define the range of the histogram.
        overwrite: bool
            If True overwrites the histograms already saved in the hdf5 file.
        """
        hist, bin_edges = self.event_1d_histogram(
            event_header_element, bins=bins, hist_range=hist_range
        )
        bin_edges *= self.event_information[event_header_element].unit
        table = fill_hdf5_table(
            hist=hist,
            x_bin_edges=bin_edges,
            y_bin_edges=None,
            x_label=event_header_element,
            y_label=None,
            meta_data=self._meta_dict,
        )
        hdf5_table_name = f"/event_2d_histograms_{event_header_element}"

        self._logger.info(
            f"Exporting histogram with name {hdf5_table_name} to {self.hdf5_file_name}."
        )
        # overwrite takes precedence over append
        if overwrite is True:
            append = False
        else:
            append = True
        write_table(table, self.hdf5_file_name, hdf5_table_name, append=append, overwrite=overwrite)

    def export_event_header_2d_histogram(
        self,
        event_header_element_1,
        event_header_element_2,
        bins=50,
        hist_range=None,
        overwrite=False,
    ):
        """
        Export event_header of a 2D histogram to a hdf5 file.

        Searches the 2D histogram for the key 'event_header_element_1' and
        'event_header_element_2'from the CORSIKA event header.

        Parameters
        ----------
        event_header_element_1: str
            The key to the CORSIKA event header element.
        event_header_element_2: str
            The key to the CORSIKA event header element.
            Possible choices for 'event_header_element_1' and 'event_header_element_2' are stored
            in 'self.all_event_keys'.
        bins: float
            Number of bins for the histogram.
        hist_range: 2-tuple
            Tuple to define the range of the histogram.
        overwrite: bool
            If True overwrites the histograms already saved in the hdf5 file.

        """
        hist, x_bin_edges, y_bin_edges = self.event_2d_histogram(
            event_header_element_1, event_header_element_2, bins=bins, hist_range=hist_range
        )
        x_bin_edges *= self.event_information[event_header_element_1].unit
        y_bin_edges *= self.event_information[event_header_element_2].unit

        table = fill_hdf5_table(
            hist=hist,
            x_bin_edges=x_bin_edges,
            y_bin_edges=y_bin_edges,
            x_label=event_header_element_1,
            y_label=event_header_element_2,
            meta_data=self._meta_dict,
        )

        hdf5_table_name = f"/event_2d_histograms_{event_header_element_1}_{event_header_element_2}"

        self._logger.info(
            f"Exporting histogram with name {hdf5_table_name} to {self.hdf5_file_name}."
        )
        # overwrite takes precedence over append
        if overwrite is True:
            append = False
        else:
            append = True
        write_table(table, self.hdf5_file_name, hdf5_table_name, append=append, overwrite=overwrite)

    @property
    def num_photons_per_telescope(self):
        """
        The number of photons per event, considering the telescopes given by self.telescope_indices.

        Returns
        -------
        numpy.array
            Number of photons per telescope.
        """
        self._num_photons_per_telescope = np.sum(
            np.array(self.num_photons_per_event_per_telescope), axis=1
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
        self._total_num_photons = np.sum(self.num_photons_per_event)
        return self._total_num_photons

    @property
    def telescope_positions(self):
        """
        The telescope positions found in the CORSIKA output file.

        It does not depend on the telescope_indices attribute.

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
    # field defined for the run. For other information, please use the get_event_parameter_info
    # function.
    @property
    def event_zenith_angles(self):
        """
        Get the zenith angles of the simulated events in astropy units of degrees.

        Returns
        -------
        astropy.Quantity
            The zenith angles for each event.
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

        If negative, tracking starts at margin of atmosphere,
        see TSTART in the CORSIKA 7 user guide.

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
            The Earth magnetic field in the z direction used for each event.
        """
        if self._magnetic_field_x is None:
            self._magnetic_field_x = (self.event_information["earth_magnetic_field_x"]).to(u.uT)
        if self._magnetic_field_z is None:
            self._magnetic_field_z = (self.event_information["earth_magnetic_field_z"]).to(u.uT)
        return self._magnetic_field_x, self._magnetic_field_z

    def get_event_parameter_info(self, parameter):
        """
        Get specific information (i.e. any parameter) of the events.

        The parameter is passed through the key word parameter.
        Available options are to be found under self.all_event_keys.
        The unit of the parameter, if any, is given according to the CORSIKA version
        (please see user guide in this case).

        Parameters
        ----------
        parameter: str
            The parameter of interest. Available options are to be found under
            self.all_event_keys.

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
            msg = f"key is not valid. Valid entries are {self.all_event_keys}"
            self._logger.error(msg)
            raise KeyError
        return self.event_information[parameter]

    def get_run_info(self, parameter):
        """
        Get specific information (i.e. any parameter) of the run.

        The parameter is passed through the key word parameter.
        Available options are to be found under self.all_run_keys.
        The unit of the parameter, if any, is given according to the CORSIKA version
        (please see user guide in this case).

        Parameters
        ----------
        parameter: str
            The parameter of interest. Available options are to be found under
            self.all_run_keys.

        Raises
        ------
        KeyError:
            If parameter is not valid.
        """
        if parameter not in self.all_run_keys:
            msg = f"key is not valid. Valid entries are {self.all_run_keys}"
            self._logger.error(msg)
            raise KeyError
        return self.header[parameter]

    def event_1d_histogram(self, key, bins=50, hist_range=None):
        """
        Create a histogram for the all events using key as parameter.

        Valid keys are stored in self.all_event_keys (CORSIKA defined).

        Parameters
        ----------
        key: str
            The information from which to build the histogram, e.g. total_energy, zenith or
            first_interaction_height.
        bins: float
            Number of bins for the histogram.
        hist_range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            Edges of the histogram.


        Raises
        ------
        KeyError:
            If key is not valid.
        """
        if key not in self.all_event_keys:
            msg = f"key is not valid. Valid entries are {self.all_event_keys}"
            self._logger.error(msg)
            raise KeyError
        hist, bin_edges = np.histogram(
            self.event_information[key].value,
            bins=bins,
            range=hist_range,
        )
        return hist, bin_edges

    def event_2d_histogram(self, key_1, key_2, bins=50, hist_range=None):
        """
        Create a 2D histogram for the all events using key_1 and key_2 as parameters.

        Valid keys are stored in self.all_event_keys (CORSIKA defined).

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
        hist_range: 2-tuple
            Tuple to define the range of the histogram.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            x Edges of the histogram.
        numpy.array
            y Edges of the histogram.


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
        hist, x_bin_edges, y_bin_edges = np.histogram2d(
            self.event_information[key_1].value,
            self.event_information[key_2].value,
            bins=bins,
            range=hist_range,
        )
        return hist, x_bin_edges, y_bin_edges
