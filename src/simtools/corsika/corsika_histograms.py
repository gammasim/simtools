"""Extract Cherenkov photons from a CORSIKA IACT file and fills histograms."""

import functools
import logging
import operator
from pathlib import Path

import boost_histogram as bh
import numpy as np
from astropy import units as u
from eventio import IACTFile

from simtools.io import io_handler
from simtools.utils.geometry import rotate

X_AXIS_STRING = "x axis"
Y_AXIS_STRING = "y axis"


class CorsikaHistograms:
    """
    Extracts Cherenkov photons from CORSIKA IACT file and fills histograms.

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
            raise FileNotFoundError(f"File {self.input_file} does not exist.")

        self.io_handler = io_handler.IOHandler()
        self.output_path = self.io_handler.get_output_directory("corsika")

        self.events = None
        self.hist = self._set_2d_distributions()
        self.hist.update(self._set_1d_distributions())

    def fill(self):
        """
        Fill Cherenkov photons histograms.

        Returns
        -------
        list: list of boost_histogram.Histogram instances.

        Raises
        ------
        AttributeError:
            if event has not photon saved.
        """
        self._read_event_headers()

        with IACTFile(self.input_file) as f:
            telescope_positions = np.array(f.telescope_positions)
            event_counter = 0
            for event in f:
                if hasattr(event, "photon_bunches"):
                    photons = list(event.photon_bunches.values())
                    self._fill_histograms(photons, event_counter, telescope_positions, True)
                event_counter += 1

        self._update_distributions()

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

    def _create_regular_axes(self, hist, axes):
        """Create regular axis for a single histogram."""
        transform = {"log": bh.axis.transform.log, "linear": None}

        boost_axes = []
        for axis in axes:
            bins, start, stop = hist[axis][:3]
            if isinstance(start, u.quantity.Quantity):
                start, stop = start.value, stop.value
            boost_axes.append(
                bh.axis.Regular(
                    bins=bins,
                    start=start,
                    stop=stop,
                    transform=transform[hist["scale"]],
                )
            )
        return boost_axes

    def _fill_histograms(self, photons, event_counter, telescope_positions, rotate_photons=True):
        """
        Fill Cherenkov photon histograms.

        For rotate_photons, the Cherenkov photon's coordinates are filled in the shower plane.

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
        telescope_positions: numpy.array
            Array with the telescope positions with shape (M, 2), where M is the number of
            telescopes in the array. The two columns are the x and y positions of the telescopes
            in the CORSIKA coordinate system.
        rotate_photons: bool
            If True, the photon's coordinates are rotated to the plane perpendicular to the
            incoming direction of the primary particle.

        Raises
        ------
        IndexError:
            If the index or indices passed though telescope_index are out of range.
        """
        hist_str = "histogram"
        for photon, telescope in zip(photons, telescope_positions):
            if not rotate_photons:
                photon_x, photon_y = photon["x"], photon["y"]
            else:
                photon_x, photon_y = rotate(
                    photon["x"],
                    photon["y"],
                    self.events["azimuth_deg"][event_counter],
                    self.events["zenith_deg"][event_counter],
                )

            photon_x -= -telescope["x"]
            photon_y -= -telescope["y"]

            self.hist["counts_xy"][hist_str].fill(
                (photon_x * u.cm).to(u.m), (photon_y * u.cm).to(u.m)
            )
            self.hist["density_xy"][hist_str].fill(
                (photon_x * u.cm).to(u.m), (photon_y * u.cm).to(u.m)
            )
            self.hist["direction_xy"][hist_str].fill(photon["cx"], photon["cy"])
            self.hist["time_altitude"][hist_str].fill(
                photon["time"] * u.ns, (photon["zem"] * u.cm).to(u.km)
            )
            self.hist["wavelength_altitude"][hist_str].fill(
                np.abs(photon["wavelength"]) * u.nm, (photon["zem"] * u.cm).to(u.km)
            )

            photon_r = np.sqrt(photon_x**2 + photon_y**2)
            self.hist["counts_r"][hist_str].fill(photon_r * u.cm.to(u.m))
            self.hist["density_r"][hist_str].fill(photon_r * u.cm.to(u.m))

            self.events["num_photons"][event_counter] += np.sum(photon["photons"])

    def get_hist_2d_projection(self, label, hist):
        """
        Get 2D distributions.

        Parameters
        ----------
        label: str
            Histogram label.
        hist: dict
            Histogram dictionary.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The x bin edges of the histograms.
        numpy.array
            The y bin edges of the histograms.
        """
        h = hist["histogram"]

        v = h.view().T
        if label == "density_xy":  # TODO density
            v = v / functools.reduce(operator.mul, h.axes.widths)

        xb = h.axes.edges[0].flatten()
        yb = h.axes.edges[1].flatten()

        return np.asarray([v]), np.asarray([xb]), np.asarray([yb])

    def get_hist_1d_projection(self, label, hist, bins=50, hist_range=None):
        """
        Get 1D distributions.

        Parameters
        ----------
        label: str
            Histogram label.
        hist: dict
            Histogram dictionary.

        Returns
        -------
        numpy.ndarray
            The counts of the histogram.
        numpy.array
            The bin edges of the histogram.
        """
        # plain numpy histogram
        if hist.get("projection") is None and label in self.events.dtype.names:
            histo_1d, bin_edges = np.histogram(self.events[label], bins=bins, range=hist_range)
            return histo_1d.reshape(1, bins), bin_edges.reshape(1, bins + 1)

        # boost 1D histogram
        if hist.get("projection") is None:
            histo_1d = self.hist[label]["histogram"]
            edges = histo_1d.axes.edges.T.flatten()[0]
            return np.asarray([histo_1d.view().T]), np.asarray([edges])

        # boost 2D histogram projection
        histo_2d = self.hist[hist["projection"][0]]["histogram"]
        if hist["projection"][1] == "x":
            h = histo_2d[:, sum]
        else:
            h = histo_2d[sum, :]
        edges = h.axes.edges.T.flatten()[0]
        return np.asarray([h.view().T]), np.asarray([edges])

    def _set_1d_distributions(self, r_max=1000 * u.m, bins=100):
        """
        Define 1D histograms.

        Returns
        -------
        dict:
            Dictionary with 1D histogram information.
        """
        file_name = "file name"
        title = "title"
        projection = "projection"
        x_bins = "x_bins"
        scale = "scale"
        x_axis_unit = "x_axis_unit"
        x_axis_title = "x_axis_title"
        hist_1d = {
            "wavelength": {
                file_name: "hist_1d_photon_wavelength_distr",
                title: "Photon wavelength distribution",
                projection: ["wavelength_altitude", "x"],
            },
            "counts_r": {
                file_name: "hist_1d_photon_radial_distr",
                title: "Radial photon distribution on the ground",
                x_bins: [bins, 0 * u.m, r_max],
                scale: "linear",
                x_axis_title: "Distance to center",
                x_axis_unit: u.m,
            },
            "density_r": {
                file_name: "hist_1d_photon_density_distr",
                title: "Photon density distribution on the ground",
                x_bins: [bins, 0 * u.m, r_max],
                scale: "linear",
                x_axis_title: "Distance to center",
                x_axis_unit: u.m,
            },
            "time": {
                file_name: "hist_1d_photon_time_distr",
                title: "Photon time of arrival distribution",
                projection: ["time_altitude", "x"],
            },
            "altitude": {
                file_name: "hist_1d_photon_altitude_distr",
                title: "Photon altitude of emission distribution",
                projection: ["time_altitude", "y"],
            },
            "num_photons": {
                file_name: "hist_1d_photon_per_event_distr",
                title: "Photons per event distribution",
                x_axis_title: "Event counter",
                x_axis_unit: u.dimensionless_unscaled,
            },
        }

        for value in hist_1d.values():
            value["is_1d"] = True
            value["log_y"] = True
            if value.get("projection") is not None:
                hist_2d_name = value["projection"][0]
                if value["projection"][1] == "x":
                    value[x_bins] = self.hist[hist_2d_name]["x_bins"]
                    value[x_axis_title] = self.hist[hist_2d_name]["x_axis_title"]
                    value[x_axis_unit] = self.hist[hist_2d_name]["x_axis_unit"]
                else:
                    value[x_bins] = self.hist[hist_2d_name]["y_bins"]
                    value[x_axis_title] = self.hist[hist_2d_name]["y_axis_title"]
                    value[x_axis_unit] = self.hist[hist_2d_name]["y_axis_unit"]
            elif value.get(x_bins) is not None:
                boost_axes = self._create_regular_axes(value, ["x_bins"])
                value["histogram"] = bh.Histogram(boost_axes[0])
        return hist_1d

    def _set_2d_distributions(self, xy_maximum=1000 * u.m, xy_bin=100):
        """
        Define 2D histograms.

        Returns
        -------
        dict:
            Dictionary with 2D histogram information.
        """
        file_name = "file name"
        title = "title"
        x_bins, y_bins = "x_bins", "y_bins"
        scale = "scale"
        x_axis_title = "x_axis_title"
        x_axis_unit = "x_axis_unit"
        y_axis_title = "y_axis_title"
        y_axis_unit = "y_axis_unit"

        hist_2d = {
            "counts_xy": {
                file_name: "hist_2d_photon_count_distr",
                title: "Photon count distribution on the ground",
                x_bins: [xy_bin, -xy_maximum, xy_maximum],
                y_bins: [xy_bin, -xy_maximum, xy_maximum],
                scale: "linear",
                x_axis_title: "x position on the ground",
                x_axis_unit: xy_maximum.unit,
                y_axis_title: "y position on the ground",
                y_axis_unit: xy_maximum.unit,
            },
            "density_xy": {
                file_name: "hist_2d_photon_density_distr",
                title: "Photon density distribution on the ground",
                x_bins: [xy_bin, -xy_maximum, xy_maximum],
                y_bins: [xy_bin, -xy_maximum, xy_maximum],
                scale: "linear",
                x_axis_title: "x position on the ground",
                x_axis_unit: xy_maximum.unit,
                y_axis_title: "y position on the ground",
                y_axis_unit: xy_maximum.unit,
            },
            "direction_xy": {
                file_name: "hist_2d_photon_direction_distr",
                title: "Photon arrival direction",
                x_bins: [100, -1, 1],
                y_bins: [100, -1, 1],
                scale: "linear",
                x_axis_title: "x direction cosine",
                x_axis_unit: u.dimensionless_unscaled,
                y_axis_title: "y direction cosine",
                y_axis_unit: u.dimensionless_unscaled,
            },
            "time_altitude": {
                file_name: "hist_2d_photon_time_altitude_distr",
                title: "Time of arrival vs altitude of emission",
                x_bins: [100, -2000 * u.ns, 2000 * u.ns],
                y_bins: [100, 120 * u.km, 0 * u.km],
                scale: "linear",
                x_axis_title: "Time of arrival",
                x_axis_unit: u.ns,
                y_axis_title: "Altitude of emission",
                y_axis_unit: u.km,
            },
            "wavelength_altitude": {
                file_name: "hist_2d_photon_wavelength_altitude_distr",
                title: "Wavelength vs altitude of emission",
                x_bins: [100, -2000 * u.ns, 2000 * u.ns],
                y_bins: [100, 120 * u.km, 0 * u.km],
                scale: "linear",
                x_axis_title: "Wavelength",
                x_axis_unit: u.nm,
                y_axis_title: "Altitude of emission",
                y_axis_unit: u.km,
            },
        }

        for value in hist_2d.values():
            value["is_1d"] = False
            value["log_z"] = True
            boost_axes = self._create_regular_axes(value, ["x_bins", "y_bins"])
            value["histogram"] = bh.Histogram(boost_axes[0], boost_axes[1])

        return hist_2d

    def _update_distributions(self):
        """Update the distributions dictionary with the histogram values and bin edges."""
        for key, value in self.hist.items():
            if value["is_1d"]:
                value["hist_values"], value["x_bin_edges"] = self.get_hist_1d_projection(key, value)
            else:
                value["hist_values"], value["x_bin_edges"], value["y_bin_edges"] = (
                    self.get_hist_2d_projection(key, value)
                )
