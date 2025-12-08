"""Extract Cherenkov photons information from a CORSIKA IACT file."""

import functools
import logging
import operator
from pathlib import Path

import boost_histogram as bh
import numpy as np
from astropy import units as u
from eventio import IACTFile

from simtools.io import io_handler
from simtools.utils.geometry import convert_2d_to_radial_distr, rotate
from simtools.visualization import plot_corsika_histograms as visualize

X_AXIS_STRING = "x axis"
Y_AXIS_STRING = "y axis"
Z_AXIS_STRING = "z axis"


class CorsikaHistograms:
    """
    Extracts the Cherenkov photons information from a CORSIKA IACT file.

    Parameters
    ----------
    input_file: str or Path
        CORSIKA IACT file.
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
        self._logger.debug("Init CorsikaHistograms")
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"file {self.input_file} does not exist.")

        self.io_handler = io_handler.IOHandler()
        self.output_path = self.io_handler.get_output_directory("corsika")

        self.events = None

        self.hist = {}
        self.hist_config = self._create_histogram_default_config()
        self._dict_2d_distributions = None
        self._dict_1d_distributions = None

        self._allowed_1d_labels = {"wavelength", "time", "altitude"}
        self._allowed_2d_labels = {"counts", "density", "direction", "time_altitude"}

    def fill(self):
        """
        Create and fill Cherenkov photons histograms.

        Returns
        -------
        list: list of boost_histogram.Histogram instances.

        Raises
        ------
        AttributeError:
            if event has not photon saved.
        """
        self._read_event_headers()
        self._create_histograms()

        with IACTFile(self.input_file) as f:
            event_counter = 0
            for event in f:
                if hasattr(event, "photon_bunches"):
                    photons = list(event.photon_bunches.values())
                else:
                    raise AttributeError("The event has no associated photon bunches saved.")

                self._fill_histograms(photons, event_counter, True)
                event_counter += 1

    def plot(self, pdf_file=None):
        """Generate plots."""
        if pdf_file is None:
            pdf_file = Path(self.input_file.name).stem + ".pdf"

        pdf_file = Path(self.output_path).joinpath(pdf_file)
        self._logger.info(f"Saving histograms to {pdf_file}")

        visualize.export_all_photon_figures_pdf(self, pdf_file)

    def _read_event_headers(self):
        """Read event information from headers."""
        event_dtype = np.dtype(
            [
                ("particle_id", "i4"),
                ("total_energy", "f8"),
                ("azimuth_deg", "f8"),
                ("zenith_deg", "f8"),
                ("num_photons", "f8"),
            ]
        )

        with IACTFile(self.input_file) as iact_file:
            records = []
            for event in iact_file:
                records.append(
                    (
                        event.header["particle_id"],
                        event.header["total_energy"],
                        np.rad2deg(event.header["azimuth"]),
                        np.rad2deg(event.header["zenith"]),
                        0.0,  # filled later when reading photon bunches
                    )
                )

        self.events = np.array(records, dtype=event_dtype)

    def _create_histogram_default_config(self):
        """
        Create a dictionary with the configuration necessary to create the histograms.

        Three histograms are created: hist_position with 3 dimensions (x, y positions),
        hist_direction with 2 dimensions (direction cosines in x and y directions),
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
        xy_maximum = 1000 * u.m
        xy_bin = 100

        return {
            "position": {
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
            "direction": {
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
            "time_altitude": {
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

        all_axes = [X_AXIS_STRING, Y_AXIS_STRING]
        if label == "position":
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

    def _create_histograms(self):
        """Create the histograms."""
        boost_axes_position = self._create_regular_axes("position")
        self.hist["position"] = bh.Histogram(
            boost_axes_position[0],
            boost_axes_position[1],
            boost_axes_position[2],
        )

        for axis in "direction", "time_altitude":
            boost_axes = self._create_regular_axes(axis)
            self.hist[axis] = bh.Histogram(boost_axes[0], boost_axes[1])

    def _fill_histograms(self, photons, event_counter, rotate_photons=True):
        """
        Fill Cherenkov photon histograms.

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
             time: time of arrival of the photon in ns.
             zem: altitude where the photon was generated in cm,
             photons: number of photons associated to this bunch,
        event_counter: int
            Event counter.

        Raises
        ------
        IndexError:
            If the index or indices passed though telescope_index are out of range.
        """
        for photon in photons:
            if not rotate_photons:
                photon_x, photon_y = photon["x"], photon["y"]
            else:
                photon_x, photon_y = rotate(
                    photon["x"],
                    photon["y"],
                    self.events["azimuth_deg"][event_counter],
                    self.events["zenith_deg"][event_counter],
                )

            self.hist["position"].fill(
                (photon_x * u.cm).to(u.m),
                (photon_y * u.cm).to(u.m),
                np.abs(photon["wavelength"]) * u.nm,
            )

            self.hist["direction"].fill(photon["cx"], photon["cy"])
            self.hist["time_altitude"].fill(photon["time"] * u.ns, (photon["zem"] * u.cm).to(u.km))
            self.events["num_photons"][event_counter] += photon["photons"]

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

        x_bin_edges, y_bin_edges, hist_values = [], [], []
        mini_hist = None
        if label == "counts":
            mini_hist = self.hist["position"][:, :, sum]
            hist_values.append(mini_hist.view().T)
        elif label == "density":
            mini_hist = self.hist["position"][:, :, sum]
            areas = functools.reduce(operator.mul, mini_hist.axes.widths)
            hist_values.append(mini_hist.view().T / areas)
        elif label == "direction":
            mini_hist = self.hist["direction"]
            hist_values.append(self.hist["direction"].view().T)
        elif label == "time_altitude":
            mini_hist = self.hist["time_altitude"]
            hist_values.append(self.hist["time_altitude"].view().T)
        if mini_hist is not None:
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

        x_bin_edges_list, hist_1d_list = [], []
        mini_hist = None
        if label == "wavelength":
            mini_hist = self.hist["position"][sum, sum, :]
        elif label == "time":
            mini_hist = self.hist["time_altitude"][:, sum]
        elif label == "altitude":
            mini_hist = self.hist["time_altitude"][sum, :]

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
        hist_position = "position"
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
        first interaction heights is positive or at first interaction if otherwise.

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
        hist, bin_edges = np.histogram(self.events["num_photons"], bins=bins, range=hist_range)
        return hist.reshape(1, bins), bin_edges.reshape(1, bins + 1)

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
                axis_unit: self.hist_config["position"][Z_AXIS_STRING]["start"].unit,
            },
            "counts": {
                fn_key: "get_photon_radial_distr",
                file_name: "hist_1d_photon_radial_distr",
                title: "Radial photon distribution on the ground",
                bin_edges: "Distance to center",
                axis_unit: self.hist_config["position"][X_AXIS_STRING]["start"].unit,
            },
            "density": {
                fn_key: "get_photon_density_distr",
                file_name: "hist_1d_photon_density_distr",
                title: "Photon density distribution on the ground",
                bin_edges: "Distance to center",
                axis_unit: self.hist_config["position"][X_AXIS_STRING]["start"].unit,
            },
            "time": {
                fn_key: "get_photon_time_of_emission_distr",
                file_name: "hist_1d_photon_time_distr",
                title: "Photon time of arrival distribution",
                bin_edges: "Time of arrival",
                axis_unit: self.hist_config["time_altitude"][X_AXIS_STRING]["start"].unit,
            },
            "altitude": {
                fn_key: "get_photon_altitude_distr",
                file_name: "hist_1d_photon_altitude_distr",
                title: "Photon altitude of emission distribution",
                bin_edges: "Altitude of emission",
                axis_unit: self.hist_config["time_altitude"][Y_AXIS_STRING]["start"].unit,
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
                    x_axis_unit: self.hist_config["position"][X_AXIS_STRING]["start"].unit,
                    y_bin_edges: "y position on the ground",
                    y_axis_unit: self.hist_config["position"][Y_AXIS_STRING]["start"].unit,
                },
                "density": {
                    fn_key: "get_2d_photon_density_distr",
                    file_name: "hist_2d_photon_density_distr",
                    title: "Photon density distribution on the ground",
                    x_bin_edges: "x position on the ground",
                    x_axis_unit: self.hist_config["position"][X_AXIS_STRING]["start"].unit,
                    y_bin_edges: "y position on the ground",
                    y_axis_unit: self.hist_config["position"][Y_AXIS_STRING]["start"].unit,
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
                    x_axis_unit: self.hist_config["time_altitude"][X_AXIS_STRING]["start"].unit,
                    y_bin_edges: "Altitude of emission",
                    y_axis_unit: self.hist_config["time_altitude"][Y_AXIS_STRING]["start"].unit,
                },
            }
        return self._dict_2d_distributions
