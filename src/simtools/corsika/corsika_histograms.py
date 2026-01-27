"""Extract Cherenkov photons from a CORSIKA IACT file and fill histograms."""

import logging
from pathlib import Path

import boost_histogram as bh
import numpy as np
from astropy import units as u
from eventio import IACTFile

from simtools.utils.geometry import rotate


class CorsikaHistograms:
    """
    Extract Cherenkov photons from a CORSIKA IACT file and fill histograms.

    Parameters
    ----------
    input_file: str or Path
        CORSIKA IACT file.
    axis_distance: astropy.units.Quantity or float
        Distance from the axis to consider when calculating the lateral density profiles
        along x and y axes. If a float is given, it is assumed to be in meters.

    Raises
    ------
    FileNotFoundError:
        if the input file given does not exist.
    """

    def __init__(self, input_file, normalization_method="per-telescope", axis_distance=1000 * u.m):
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CorsikaHistograms")
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(f"File {self.input_file} does not exist.")

        self.axis_distance = (
            axis_distance.to(u.m).value if isinstance(axis_distance, u.Quantity) else axis_distance
        )
        self.events = None
        self.hist = self._set_2d_distributions()
        self.hist.update(self._set_1d_distributions())
        self._density_samples = []
        self.normalization_method = normalization_method

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
            for event_counter, event in enumerate(f):
                if hasattr(event, "photon_bunches"):
                    photons = list(event.photon_bunches.values())
                    self._fill_histograms(photons, event_counter, telescope_positions, False)

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
            records = [
                (
                    event.header["particle_id"],
                    event.header["total_energy"],
                    np.rad2deg(event.header["azimuth"]),
                    np.rad2deg(event.header["zenith"]),
                    0.0,  # filled later when reading photon bunches
                )
                for event in iact_file
            ]

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
        photons_per_telescope = np.zeros(len(telescope_positions))
        zenith_rad = np.deg2rad(self.events["zenith_deg"][event_counter])

        for tel_idx, (photon, telescope) in enumerate(zip(photons, telescope_positions)):
            if rotate_photons:
                px, py = rotate(
                    photon["x"],
                    photon["y"],
                    self.events["azimuth_deg"][event_counter],
                    self.events["zenith_deg"][event_counter],
                )
            else:
                px, py = photon["x"], photon["y"]

            px = px - telescope["x"]
            py = py - telescope["y"]
            w = photon["photons"]

            pxm = px * u.cm.to(u.m)
            pym = py * u.cm.to(u.m)
            zem = (photon["zem"] * u.cm).to(u.km)
            photons_per_telescope[tel_idx] += np.sum(w)

            self.hist["counts_xy"][hist_str].fill(pxm, pym, weight=w)
            self.hist["direction_xy"][hist_str].fill(photon["cx"], photon["cy"], weight=w)
            self.hist["time_altitude"][hist_str].fill(photon["time"] * u.ns, zem, weight=w)
            self.hist["wavelength_altitude"][hist_str].fill(
                np.abs(photon["wavelength"]) * u.nm, zem, weight=w
            )

            r = np.hypot(px, py) * u.cm.to(u.m)
            self.hist["counts_r"][hist_str].fill(r, weight=w)

            self.events["num_photons"][event_counter] += np.sum(w)

        for tel_idx, telescope in enumerate(telescope_positions):
            tel_r = np.hypot(telescope["x"], telescope["y"]) * u.cm.to(u.m)
            area = np.pi * (tel_r**2) / np.cos(zenith_rad) / 1.0e4  # in m^2
            n_photons = photons_per_telescope[tel_idx]
            density = n_photons / area if area > 0 else 0.0
            density_error = np.sqrt(n_photons) / area if area > 0 else 0.0
            self._density_samples.append(
                {
                    "x": telescope["x"] * u.cm.to(u.m),
                    "y": telescope["y"] * u.cm.to(u.m),
                    "r": tel_r,
                    "density": density,
                    "density_error": density_error,
                }
            )

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
            Histogram y bin edges.
        numpy.ndarray or None
            Histogram uncertainties (sqrt of variance) if available.
        """
        view = hist.view()
        if self._check_for_all_attributes(view):
            counts = np.asarray([view["value"].T])
            uncertainties = np.asarray([np.sqrt(view["variance"].T)])
        else:
            counts = np.asarray([view.T])
            uncertainties = None

        x_edges = np.asarray([hist.axes.edges[0].flatten()])
        y_edges = np.asarray([hist.axes.edges[1].flatten()])

        return counts, x_edges, y_edges, uncertainties

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
        uncertainties = np.sqrt(histo_1d)
        return (
            histo_1d.reshape(1, bins),
            bin_edges.reshape(1, bins + 1),
            uncertainties.reshape(1, bins),
        )

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
            Histogram counts.
        numpy.array
            Histogram x bin edges.
        numpy.ndarray or None
            Histogram uncertainties (if available).
        """
        # plain numpy histogram
        if (
            hist.get("projection") is None
            and hasattr(self, "events")
            and label in self.events.dtype.names
        ):
            return self._get_hist_1d_from_numpy(label, hist)

        # boost 1D histogram
        if hist.get("projection") is None:
            # Use histogram from hist dict if available, otherwise from self.hist
            if "histogram" in hist:
                histo_1d = hist["histogram"]
            else:
                # No histogram available, return None values
                return None, None, None
            edges = histo_1d.axes.edges.T.flatten()[0]
            view = histo_1d.view()
            if self._check_for_all_attributes(view):
                counts = np.asarray([view["value"].T])
                uncertainties = np.asarray([np.sqrt(view["variance"].T)])
            else:
                counts = np.asarray([view.T])
                uncertainties = None
            return counts, np.asarray([edges]), uncertainties

        # boost 2D histogram projection
        histo_2d = self.hist[hist["projection"][0]]["histogram"]
        if hist["projection"][1] == "x":
            h = histo_2d[:, sum]
        else:
            h = histo_2d[sum, :]
        edges = h.axes.edges.T.flatten()[0]
        view = h.view()
        if self._check_for_all_attributes(view):
            counts = np.asarray([view["value"].T])
            uncertainties = np.asarray([np.sqrt(view["variance"].T)])
        else:
            counts = np.asarray([view.T])
            uncertainties = None
        return counts, np.asarray([edges]), uncertainties

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
        log_y = "log_y"
        photon_density = "Photon density"
        distance_to_center = "Distance to center"
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
                x_axis_title: distance_to_center,
                x_axis_unit: u.m,
            },
            "density_r": {
                file_name: "hist_1d_photon_density_distr",
                title: "Photon lateral density distribution (ground level)",
                x_bins: [bins, 0 * u.m, r_max, "linear"],
                x_axis_title: distance_to_center,
                x_axis_unit: u.m,
                y_axis_title: photon_density,
                y_axis_unit: u.m**-2,
            },
            "density_r_from_counts": {
                file_name: "hist_1d_photon_density_from_counts_distr",
                title: "Photon lateral density from counts distribution (ground level)",
                x_bins: [bins, 0 * u.m, r_max, "linear"],
                x_axis_title: distance_to_center,
                x_axis_unit: u.m,
                y_axis_title: photon_density,
                y_axis_unit: u.m**-2,
            },
            "density_x": {
                file_name: "hist_1d_photon_density_x_distr",
                title: "Photon lateral density x distribution (ground level)",
                projection: ["counts_xy", "x"],  # projection requires counts_xy histogram
                x_axis_title: distance_to_center,
                x_axis_unit: u.m,
                y_axis_title: photon_density,
                y_axis_unit: u.m**-2,
            },
            "density_y": {
                file_name: "hist_1d_photon_density_y_distr",
                title: "Photon lateral density y distribution (ground level)",
                projection: ["counts_xy", "y"],  # projection requires counts_xy histogram
                y_axis_title: photon_density,
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
            "direction_cosine_x": {
                file_name: "hist_1d_photon_direction_cosine_x_distr",
                title: "Photon direction cosine x distribution",
                projection: ["direction_xy", "x"],
            },
            "direction_cosine_y": {
                file_name: "hist_1d_photon_direction_cosine_y_distr",
                title: "Photon direction cosine y distribution",
                projection: ["direction_xy", "y"],
            },
            "num_photons": {
                file_name: "hist_1d_photon_per_event_distr",
                title: "Photons per event distribution",
                "event_type": True,
                x_bins: [100, 0, None, "log"],
                x_axis_title: "Cherenkov photons per event",
                x_axis_unit: u.dimensionless_unscaled,
                log_y: False,
            },
        }

        for value in hist_1d.values():
            value["is_1d"] = True
            value["log_y"] = value.get("log_y", True)
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
                value["histogram"] = bh.Histogram(boost_axes[0], storage=bh.storage.Weight())
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
        photon_density = "Photon density"
        x_pos = "x position on the ground"
        y_pos = "y position on the ground"

        hist_2d = {
            "counts_xy": {
                file_name: "hist_2d_photon_count_distr",
                title: "Photon count distribution (ground level)",
                x_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                y_bins: [xy_bin, -xy_maximum, xy_maximum],
                x_axis_title: x_pos,
                x_axis_unit: xy_maximum.unit,
                y_axis_title: y_pos,
                y_axis_unit: xy_maximum.unit,
            },
            "density_xy": {
                file_name: "hist_2d_photon_density_distr",
                title: "Photon density distribution (ground level)",
                x_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                y_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                x_axis_title: x_pos,
                x_axis_unit: xy_maximum.unit,
                y_axis_title: y_pos,
                y_axis_unit: xy_maximum.unit,
                z_axis_title: photon_density,
                z_axis_unit: u.m**-2,
            },
            "density_xy_from_counts": {
                file_name: "hist_2d_photon_density_from_counts_distr",
                title: "Photon density from counts distribution (ground level)",
                x_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                y_bins: [xy_bin, -xy_maximum, xy_maximum, "linear"],
                x_axis_title: x_pos,
                x_axis_unit: xy_maximum.unit,
                y_axis_title: y_pos,
                y_axis_unit: xy_maximum.unit,
                z_axis_title: photon_density,
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
                x_bins: [100, 100 * u.nm, 1000 * u.nm, "linear"],
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
            value["histogram"] = bh.Histogram(
                boost_axes[0], boost_axes[1], storage=bh.storage.Weight()
            )

        return hist_2d

    def _update_distributions(self):
        """Update the distributions dictionary with the histogram values and bin edges."""
        self._populate_density_from_probes()
        self._populate_density_from_counts()
        self._filter_density_histograms()

        for key, value in self.hist.items():
            value["input_file_name"] = str(self.input_file)
            if "hist_values" not in value:
                if value["is_1d"]:
                    (
                        value["hist_values"],
                        value["x_bin_edges"],
                        value["uncertainties"],
                    ) = self.get_hist_1d_projection(key, value)
                else:
                    (
                        value["hist_values"],
                        value["x_bin_edges"],
                        value["y_bin_edges"],
                        value["uncertainties"],
                    ) = self.get_hist_2d_projection(value["histogram"])

    def _filter_density_histograms(self):
        """Filter density histograms based on the normalization method."""
        if self.normalization_method == "per-telescope":
            keys_to_remove = ["density_xy_from_counts", "density_r_from_counts"]
        elif self.normalization_method == "per-bin":
            keys_to_remove = ["density_xy", "density_x", "density_y", "density_r"]
        else:
            raise ValueError(
                f"Unknown normalization_method: {self.normalization_method}. "
                "Must be 'per-telescope' or 'per-bin'."
            )

        for key in keys_to_remove:
            if key in self.hist:
                del self.hist[key]

    def _fill_projected_density_values(self, value):
        """Extract 1D density by using projection Counts and normalizing by area."""
        histo_2d = value["projection"][0]
        source_h = self.hist[histo_2d]["histogram"]
        project_axis = value["projection"][1]

        if project_axis == "x":
            h_1d = source_h[:, sum]
            total_ortho_width = source_h.axes[1].edges[-1] - source_h.axes[1].edges[0]
        else:
            h_1d = source_h[sum, :]
            total_ortho_width = source_h.axes[0].edges[-1] - source_h.axes[0].edges[0]

        areas_1d = h_1d.axes[0].widths * total_ortho_width

        view = h_1d.view()
        if self._check_for_all_attributes(view):
            vals = view["value"] / areas_1d
            uncs = np.sqrt(view["variance"]) / areas_1d
        else:
            vals = view / areas_1d
            uncs = np.sqrt(vals)  # Fallback if no weights

        value["hist_values"] = np.asarray([vals.T])
        value["x_bin_edges"] = np.asarray([h_1d.axes.edges[0]])
        value["uncertainties"] = np.asarray([uncs.T])

    def _populate_density_from_probes(self):
        """Build density distributions from per-telescope sampling."""
        if not self._density_samples:
            return

        s = self._density_samples
        xs, ys, rs = (np.array([p[k] for p in s]) for k in ("x", "y", "r"))
        dens = np.array([p["density"] for p in s])
        errs = np.array([p["density_error"] for p in s])

        hxy = self.hist["counts_xy"]["histogram"]
        x_edges, y_edges = hxy.axes[0].edges, hxy.axes[1].edges
        r_edges = self.hist["density_r"]["histogram"].axes[0].edges

        def avg_unc_nd(coords, edges, values, errors):
            num = np.histogramdd(coords, bins=edges, weights=values)[0]
            den = np.histogramdd(coords, bins=edges)[0]
            var = np.histogramdd(coords, bins=edges, weights=errors**2)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                avg = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                unc = np.sqrt(np.divide(var, den**2, out=np.zeros_like(var), where=den > 0))
            return avg, unc

        # 2D
        avg_xy, unc_xy = avg_unc_nd(np.column_stack((xs, ys)), (x_edges, y_edges), dens, errs)

        self.hist["density_xy"].update(
            {
                "hist_values": np.asarray([avg_xy.T]),
                "x_bin_edges": np.asarray([x_edges]),
                "y_bin_edges": np.asarray([y_edges]),
                "uncertainties": np.asarray([unc_xy.T]),
            }
        )

        # 1D helpers
        def avg_unc_1d(x, e, v, err):
            return avg_unc_nd(x[:, None], (e,), v, err)

        ax = self.axis_distance

        avg_x, unc_x = (
            avg_unc_1d(xs[np.abs(ys) < ax], x_edges, dens[np.abs(ys) < ax], errs[np.abs(ys) < ax])
            if np.any(np.abs(ys) < ax)
            else (np.zeros(len(x_edges) - 1),) * 2
        )

        avg_y, unc_y = (
            avg_unc_1d(ys[np.abs(xs) < ax], y_edges, dens[np.abs(xs) < ax], errs[np.abs(xs) < ax])
            if np.any(np.abs(xs) < ax)
            else (np.zeros(len(y_edges) - 1),) * 2
        )

        avg_r, unc_r = avg_unc_1d(rs, r_edges, dens, errs)

        for k, avg, unc, edges in (
            ("density_x", avg_x, unc_x, x_edges),
            ("density_y", avg_y, unc_y, y_edges),
            ("density_r", avg_r, unc_r, r_edges),
        ):
            self.hist[k].update(
                {
                    "hist_values": np.asarray([avg]),
                    "x_bin_edges": np.asarray([edges]),
                    "uncertainties": np.asarray([unc]),
                }
            )

    def _populate_density_from_counts(self):
        """Build density distributions by dividing counts histograms by bin area."""
        # --- 2D ---
        hxy = self.hist["counts_xy"]["histogram"]
        xw = np.diff(hxy.axes[0].edges)
        yw = np.diff(hxy.axes[1].edges)
        areas2d = np.outer(xw, yw)

        dens_xy, unc_xy = self._density_and_unc(hxy.view(), areas2d)

        self.hist["density_xy_from_counts"].update(
            {
                "hist_values": np.asarray([dens_xy.T]),
                "x_bin_edges": np.asarray([hxy.axes[0].edges]),
                "y_bin_edges": np.asarray([hxy.axes[1].edges]),
                "uncertainties": np.asarray([unc_xy.T]),
            }
        )

        # --- 1D ---
        hr = self.hist["counts_r"]["histogram"]
        r = hr.axes[0].edges
        areas1d = np.pi * (r[1:] ** 2 - r[:-1] ** 2)

        dens_r, unc_r = self._density_and_unc(hr.view(), areas1d)

        self.hist["density_r_from_counts"].update(
            {
                "hist_values": np.asarray([dens_r]),
                "x_bin_edges": np.asarray([r]),
                "uncertainties": np.asarray([unc_r]),
            }
        )

    def _density_and_unc(self, view, areas):
        """Calculate density and uncertainty by dividing histogram values by areas."""
        if self._check_for_all_attributes(view):
            values = view["value"]
            unc = np.sqrt(view["variance"])
        else:
            values = view
            unc = np.sqrt(view)
        return values / areas, unc / areas

    def _check_for_all_attributes(self, view):
        """Check if view has dtype fields ('value', 'variance')."""
        if hasattr(view, "dtype") and view.dtype.names == ("value", "variance"):
            return True
        return False
