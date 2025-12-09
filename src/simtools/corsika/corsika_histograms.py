"""Extract Cherenkov photons from a CORSIKA IACT file and fills histograms."""

import functools
import logging
import operator
from pathlib import Path

import boost_histogram as bh
import numpy as np
from astropy import units as u
from eventio import IACTFile

from simtools.utils.geometry import rotate


class CorsikaHistograms:
    """
    Extracts Cherenkov photons from CORSIKA IACT file and fills histograms.

    Parameters
    ----------
    input_file: str or Path
        CORSIKA IACT file.

    Raises
    ------
    FileNotFoundError:
        if the input file given does not exist.
    """

    def __init__(self, input_file):
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaHistograms")
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"File {self.input_file} does not exist.")

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
        """
        Create regular axis for a single histogram.

        Parameters
        ----------
        hist: dict
            Histogram dictionary.
        axes: list
            List of axis names (e.g. ["x_bins", "y_bins"]).

        Returns
        -------
        list:
            List of boost_histogram axis instances.
        """
        transform = {"log": bh.axis.transform.log, "linear": None}

        boost_axes = []
        for axis in axes:
            bins, start, stop = hist[axis][:3]
            scale = hist[axis][3] if len(hist[axis]) > 3 else "linear"
            if isinstance(start, u.quantity.Quantity):
                start, stop = start.value, stop.value
            boost_axes.append(
                bh.axis.Regular(
                    bins=bins,
                    start=start,
                    stop=stop,
                    transform=transform[scale],
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
        """
        hist_str = "histogram"
        for photon, telescope in zip(photons, telescope_positions):
            if rotate_photons:
                px, py = rotate(
                    photon["x"],
                    photon["y"],
                    self.events["azimuth_deg"][event_counter],
                    self.events["zenith_deg"][event_counter],
                )
            else:
                px, py = photon["x"], photon["y"]

            px -= -telescope["x"]
            py -= -telescope["y"]
            w = photon["photons"]

            pxm = px * u.cm.to(u.m)
            pym = py * u.cm.to(u.m)
            zem = (photon["zem"] * u.cm).to(u.km)

            self.hist["counts_xy"][hist_str].fill(pxm, pym, weight=w)
            self.hist["density_xy"][hist_str].fill(pxm, pym, weight=w)
            self.hist["direction_xy"][hist_str].fill(photon["cx"], photon["cy"], weight=w)
            self.hist["time_altitude"][hist_str].fill(photon["time"] * u.ns, zem, weight=w)
            self.hist["wavelength_altitude"][hist_str].fill(
                np.abs(photon["wavelength"]) * u.nm, zem, weight=w
            )

            r = np.hypot(px, py) * u.cm.to(u.m)
            self.hist["counts_r"][hist_str].fill(r, weight=w)
            self.hist["density_r"][hist_str].fill(r, weight=w)

            self.events["num_photons"][event_counter] += np.sum(photon["photons"] * w)

    def get_hist_2d_projection(self, hist):
        """
        Get 2D distributions.

        Parameters
        ----------
        hist: boost_histogram.Histogram
            Histogram.

        Returns
        -------
        numpy.ndarray
            Histogram counts.
        numpy.array
            Histogram x bin edges.
        numpy.array
            Histogram y bin edges
        """
        return (
            np.asarray([hist.view().T]),
            np.asarray([hist.axes.edges[0].flatten()]),
            np.asarray([hist.axes.edges[1].flatten()]),
        )

    def _get_hist_1d_from_numpy(self, label, hist):
        """Get 1D histogram from numpy histogram."""
        bins = hist["x_bins"][0]
        start = hist["x_bins"][1] if hist["x_bins"][1] else np.min(self.events[label])
        stop = hist["x_bins"][2] if hist["x_bins"][2] is not None else np.max(self.events[label])
        scale = hist["x_bins"][3] if len(hist["x_bins"]) > 3 else "linear"
        if scale == "log":
            bin_edges = np.logspace(np.log10(start), np.log10(stop), bins + 1)
        else:
            bin_edges = np.linspace(start, stop, bins + 1)
        histo_1d, _ = np.histogram(self.events[label], bins=bin_edges)
        return histo_1d.reshape(1, bins), bin_edges.reshape(1, bins + 1)

    def get_hist_1d_projection(self, label, hist):
        """
        Get 1D distributions from numpy or boost histograms (1D and 2D).

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
            return self._get_hist_1d_from_numpy(label, hist)
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

    def _set_1d_distributions(self, r_max=2000 * u.m, bins=100):
        """
        Define 1D histograms.

        Returns
        -------
        dict:
            Dictionary with 1D histogram information.
        """
        file_name = "file_name"
        title = "title"
        projection = "projection"
        x_bins = "x_bins"
        x_axis_unit = "x_axis_unit"
        x_axis_title = "x_axis_title"
        y_axis_unit = "y_axis_unit"
        y_axis_title = "y_axis_title"
        hist_1d = {
            "wavelength": {
                file_name: "hist_1d_photon_wavelength_distr",
                title: "Photon wavelength distribution",
                projection: ["wavelength_altitude", "x"],
            },
            "counts_r": {
                file_name: "hist_1d_photon_radial_distr",
                title: "Photon lateral distribution (ground level)",
                x_bins: [bins, 0 * u.m, r_max, "linear"],
                x_axis_title: "Distance to center",
                x_axis_unit: u.m,
            },
            "density_r": {
                file_name: "hist_1d_photon_density_distr",
                title: "Photon lateral density distribution (ground level)",
                x_bins: [bins, 0 * u.m, r_max, "linear"],
                x_axis_title: "Distance to center",
                x_axis_unit: u.m,
                y_axis_title: "Photon density",
                y_axis_unit: u.m**-2,
            },
            "time": {
                file_name: "hist_1d_photon_time_distr",
                title: "Photon arrival time distribution",
                projection: ["time_altitude", "x"],
            },
            "altitude": {
                file_name: "hist_1d_photon_altitude_distr",
                title: "Photon emission altitude distribution",
                projection: ["time_altitude", "y"],
            },
            "num_photons": {
                file_name: "hist_1d_photon_per_event_distr",
                title: "Photons per event distribution",
                "event_type": True,
                x_bins: [100, 0, None, "log"],
                x_axis_title: "Cherenkov photons per event",
                x_axis_unit: u.dimensionless_unscaled,
            },
        }

        for value in hist_1d.values():
            value["is_1d"] = True
            value["log_y"] = True
            value[y_axis_title] = (
                "Counts" if value.get(y_axis_title) is None else value[y_axis_title]
            )
            value[y_axis_unit] = (
                u.dimensionless_unscaled if value.get(y_axis_unit) is None else value[y_axis_unit]
            )
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
            elif value.get("event_type", False) is False:
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
        file_name = "file_name"
        title = "title"
        x_bins, y_bins = "x_bins", "y_bins"
        x_axis_title, x_axis_unit = "x_axis_title", "x_axis_unit"
        y_axis_title, y_axis_unit = "y_axis_title", "y_axis_unit"
        z_axis_title, z_axis_unit = "z_axis_title", "z_axis_unit"

        hist_2d = {
            "counts_xy": {
                file_name: "hist_2d_photon_count_distr",
                title: "Photon count distribution (ground level)",
                x_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                y_bins: [xy_bin, -xy_maximum, xy_maximum],
                x_axis_title: "x position on the ground",
                x_axis_unit: xy_maximum.unit,
                y_axis_title: "y position on the ground",
                y_axis_unit: xy_maximum.unit,
            },
            "density_xy": {
                file_name: "hist_2d_photon_density_distr",
                title: "Photon lateral density distribution (ground level)",
                x_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                y_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                x_axis_title: "x position on the ground",
                x_axis_unit: xy_maximum.unit,
                y_axis_title: "y position on the ground",
                y_axis_unit: xy_maximum.unit,
                z_axis_title: "Photon density",
                z_axis_unit: u.m**-2,
            },
            "direction_xy": {
                file_name: "hist_2d_photon_direction_distr",
                title: "Photon arrival direction",
                x_bins: [100, -1, 1, "linear"],
                y_bins: [100, -1, 1, "linear"],
                x_axis_title: "x direction cosine",
                x_axis_unit: u.dimensionless_unscaled,
                y_axis_title: "y direction cosine",
                y_axis_unit: u.dimensionless_unscaled,
            },
            "time_altitude": {
                file_name: "hist_2d_photon_time_altitude_distr",
                title: "Arrival time vs emission altitude",
                x_bins: [100, -2000 * u.ns, 2000 * u.ns, "linear"],
                y_bins: [100, 120 * u.km, 0 * u.km, "linear"],
                x_axis_title: "Arrival time",
                x_axis_unit: u.ns,
                y_axis_title: "Emission altitude",
                y_axis_unit: u.km,
            },
            "wavelength_altitude": {
                file_name: "hist_2d_photon_wavelength_altitude_distr",
                title: "Wavelength vs emission altitude ",
                x_bins: [100, -2000 * u.ns, 2000 * u.ns, "linear"],
                y_bins: [100, 120 * u.km, 0 * u.km, "linear"],
                x_axis_title: "Wavelength",
                x_axis_unit: u.nm,
                y_axis_title: "Emission altitude",
                y_axis_unit: u.km,
            },
        }

        for value in hist_2d.values():
            value["is_1d"] = False
            value["log_z"] = True
            value[z_axis_title] = (
                "Counts" if value.get(z_axis_title) is None else value[z_axis_title]
            )
            value[z_axis_unit] = (
                u.dimensionless_unscaled if value.get(z_axis_unit) is None else value[z_axis_unit]
            )
            boost_axes = self._create_regular_axes(value, ["x_bins", "y_bins"])
            value["histogram"] = bh.Histogram(boost_axes[0], boost_axes[1])

        return hist_2d

    def _update_distributions(self):
        """Update the distributions dictionary with the histogram values and bin edges."""
        self._normalize_density_histograms()

        for key, value in self.hist.items():
            value["input_file_name"] = str(self.input_file)
            if value["is_1d"]:
                value["hist_values"], value["x_bin_edges"] = self.get_hist_1d_projection(key, value)
            else:
                value["hist_values"], value["x_bin_edges"], value["y_bin_edges"] = (
                    self.get_hist_2d_projection(value["histogram"])
                )

    def _normalize_density_histograms(self):
        """Normalize the density histograms by the area of each bin."""
        for key in ["density_xy", "density_r"]:
            hist = self.hist[key]["histogram"]
            if key == "density_xy":
                bin_areas = functools.reduce(operator.mul, hist.axes.widths)
                hist /= bin_areas
            elif key == "density_r":
                bin_edges = hist.axes.edges[0]
                bin_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
                for i, area in enumerate(bin_areas):
                    hist.view()[i] /= area
