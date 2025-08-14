"""Histograms for shower and triggered events."""

import logging

import astropy.units as u
import numpy as np

from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader


class SimtelIOEventHistograms:
    """
    Generate and fill histograms for shower and triggered events.

    Event data is read from the reduced MC event data file.

    Parameters
    ----------
    event_data_file : str
        Path to the event-data file.
    array_name : str, optional
        Name of the telescope array configuration (default is None).
    telescope_list : list, optional
        List of telescope IDs to filter the events (default is None).
    """

    def __init__(self, event_data_file, array_name=None, telescope_list=None):
        """Initialize."""
        self._logger = logging.getLogger(__name__)
        self.event_data_file = event_data_file
        self.array_name = array_name
        self.telescope_list = telescope_list

        self.histograms = {}
        self.file_info = {}

        self.reader = SimtelIOEventDataReader(event_data_file, telescope_list=telescope_list)

    def get(self, key, default=None):
        """
        Provide direct access to histogram dictionary.

        Parameters
        ----------
        key : str
            Key for the histogram.
        default : any
            Default value to return if key is not found.

        Returns
        -------
        any
            Histogram data or default value.
        """
        return self.histograms.get(key, default)

    def fill(self):
        """
        Fill histograms with event data.

        Involves looping over all event data, and therefore is the slowest part of the
        histogram module. Adds the histograms to the histogram dictionary.

        Assume that all event data files are generated with similar configurations
        (self.file_info contains the file info of the last file).
        """
        for data_set in self.reader.data_sets:
            self._logger.info(f"Reading event data from {self.event_data_file} for {data_set}")
            _file_info_table, shower_data, event_data, triggered_data = self.reader.read_event_data(
                self.event_data_file, table_name_map=data_set
            )
            _file_info_table = self.reader.get_reduced_simulation_file_info(_file_info_table)
            self.file_info = {
                "energy_min": _file_info_table["energy_min"].to("TeV"),
                "core_scatter_max": _file_info_table["core_scatter_max"].to("m"),
                "viewcone_max": _file_info_table["viewcone_max"].to("deg"),
            }

            hist_defs = self._histogram_definitions(event_data, triggered_data, shower_data)

            for name, data, bins, hist1d in hist_defs:
                self._fill_histogram_and_bin_edges(name, data, bins, hist1d=hist1d)

            # TODO temporary break to run one file only
            break

    def _histogram_definitions(self, event_data, triggered_data, shower_data):
        """
        Generate list with definitions and data for filling of histograms.

        All histograms are defined for simulated and triggered event (note
        the subtlety of triggered events being read from event_data and triggered_data).

        Parameters
        ----------
        event_data : EventData
            The event data to use for filling the histograms.
        triggered_data : TriggeredData
            The triggered data to use for filling the histograms.
        shower_data : ShowerData
            The shower data to use for filling the histograms.

        Returns
        -------
         list
            List with histogram definitions and data.
        """
        xy_bins = np.linspace(
            -1.0 * self.core_distance_bins.max(),
            self.core_distance_bins.max(),
            len(self.core_distance_bins),
        )
        hist_specs = [
            (
                "energy",
                [(event_data, "simulated_energy")],
                self.energy_bins,
                True,
                [(shower_data, "simulated_energy")],
            ),
            (
                "core_distance",
                [(event_data, "core_distance_shower")],
                self.core_distance_bins,
                True,
                [(shower_data, "core_distance_shower")],
            ),
            (
                "angular_distance",
                [(triggered_data, "angular_distance")],
                self.view_cone_bins,
                True,
                [(shower_data, "angular_distance")],
            ),
            (
                "x_core_shower_vs_y_core_shower",
                [(event_data, "x_core_shower"), (event_data, "y_core_shower")],
                [xy_bins, xy_bins],
                False,
                [(shower_data, "x_core_shower"), (shower_data, "y_core_shower")],
            ),
            (
                "core_vs_energy",
                [(event_data, "core_distance_shower"), (event_data, "simulated_energy")],
                [self.core_distance_bins, self.energy_bins],
                False,
                [(shower_data, "core_distance_shower"), (shower_data, "simulated_energy")],
            ),
            (
                "angular_distance_vs_energy",
                [(triggered_data, "angular_distance"), (event_data, "simulated_energy")],
                [self.view_cone_bins, self.energy_bins],
                False,
                [(shower_data, "angular_distance"), (shower_data, "simulated_energy")],
            ),
        ]

        hists = []
        for name, fields_ev, bins, one_d, fields_mc in hist_specs:
            hists.append(
                (name, tuple(getattr(obj, f) for obj, f in fields_ev), bins, one_d)
                if len(fields_ev) > 1
                else (name, getattr(fields_ev[0][0], fields_ev[0][1]), bins, one_d)
            )

            hists.append(
                (f"{name}_mc", tuple(getattr(obj, f) for obj, f in fields_mc), bins, one_d)
                if len(fields_mc) > 1
                else (f"{name}_mc", getattr(fields_mc[0][0], fields_mc[0][1]), bins, one_d)
            )
        return hists

    def _fill_histogram_and_bin_edges(self, name, data, bins, hist1d=True):
        """
        Fill histogram and bin edges and it both to histogram dictionary.

        Adds histogram to existing histogram if it exists, otherwise initializes it.

        """
        if name in self.histograms:
            if hist1d:
                bins = self.histograms[f"{name}_bin_edges"]
                hist, _ = np.histogram(data, bins=bins)
                self.histograms[name] += hist
            else:
                x_bins = self.histograms[f"{name}_bin_x_edges"]
                y_bins = self.histograms[f"{name}_bin_y_edges"]
                hist, _, _ = np.histogram2d(data[0], data[1], bins=[x_bins, y_bins])
                self.histograms[name] += hist
        else:
            if hist1d:
                hist, bin_edges = np.histogram(data, bins=bins)
                self.histograms[name] = hist
                self.histograms[f"{name}_bin_edges"] = bin_edges
            else:
                hist, x_edges, y_edges = np.histogram2d(data[0], data[1], bins=bins)
                self.histograms[name] = hist
                self.histograms[f"{name}_bin_x_edges"] = x_edges
                self.histograms[f"{name}_bin_y_edges"] = y_edges

    @property
    def energy_bins(self):
        """Return bins for the energy histogram."""
        if "energy_bin_edges" in self.histograms:
            return self.histograms["energy_bin_edges"]
        return np.logspace(
            np.log10(self.file_info.get("energy_min", 1.0e-3 * u.TeV).to("TeV").value),
            np.log10(self.file_info.get("energy_max", 1.0e3 * u.TeV).to("TeV").value),
            100,
        )

    @property
    def core_distance_bins(self):
        """Return bins for the core distance histogram."""
        if "core_distance_bin_edges" in self.histograms:
            return self.histograms["core_distance_bin_edges"]
        return np.linspace(
            self.file_info.get("core_scatter_min", 0.0 * u.m).to("m").value,
            self.file_info.get("core_scatter_max", 1.0e5 * u.m).to("m").value,
            100,
        )

    @property
    def view_cone_bins(self):
        """Return bins for the viewcone histogram."""
        if "viewcone_bin_edges" in self.histograms:
            return self.histograms["viewcone_bin_edges"]
        return np.linspace(
            self.file_info.get("viewcone_min", 0.0 * u.deg).to("deg").value,
            self.file_info.get("viewcone_max", 20.0 * u.deg).to("deg").value,
            100,
        )

    def calculate_cumulative_data(self):
        """
        Calculate cumulative distributions for triggered event histograms.

        Takes into account the different histogram types and their axes.

        Returns
        -------
        dict
            Dictionary containing the cumulative histograms.

        """
        cumulative_data = {}

        # Calculate normalized cumulative for 2D vs_energy histograms
        for hist_key in self.histograms:
            if hist_key.endswith("_vs_energy") and not hist_key.endswith("_mc"):
                output_key = f"normalized_cumulative_{hist_key}"
                hist = self.histograms.get(hist_key)
                cumulative_data[output_key] = self._calculate_cumulative_histogram(
                    hist, axis=0, normalize=True
                )

        # Calculate cumulative for 1D histograms
        for hist_key in ["energy", "core_distance", "angular_distance"]:
            if hist_key in self.histograms:
                output_key = f"cumulative_{hist_key}"
                hist = self.histograms.get(hist_key)
                reverse = hist_key == "energy"  # Only energy uses reverse cumulative
                cumulative_data[output_key] = self._calculate_cumulative_histogram(
                    hist, reverse=reverse
                )

        return cumulative_data

    def _calculate_cumulative_histogram(self, hist, reverse=False, axis=None, normalize=False):
        """
        Calculate cumulative distribution of a histogram.

        Works with both 1D and 2D histograms.

        Parameters
        ----------
        hist : np.ndarray
            Histogram (1D or 2D)
        reverse : bool, optional
            If True, sum from high to low values
        axis : int, optional
            For 2D histograms, axis along which to compute cumulative sum
            None means default behavior: for 1D just cumsum, for 2D along rows
        normalize : bool, optional
            If True, normalize by the total sum for each slice along the specified axis
            For 1D histograms, normalizes by the total sum

        Returns
        -------
        np.ndarray
            Histogram with cumulative counts, optionally normalized
        """
        if hist is None:
            return None

        if hist.ndim == 1:
            result = self._calculate_cumulative_1d(hist, reverse)
            if normalize and np.sum(hist) > 0:
                result = result / np.sum(hist)
            return result

        if axis is None:
            axis = 1

        result = self._apply_cumsum_along_axis(hist.copy(), axis, reverse)

        if normalize:
            # Ensure floating dtype to allow in-place normalization without casting errors
            if not np.issubdtype(result.dtype, np.floating):
                result = result.astype(float)
            self._normalize_along_axis(result, hist, axis)

        return result

    def _normalize_along_axis(self, result, hist, axis):
        """
        Normalize cumulative histogram along the specified axis.

        Parameters
        ----------
        result : np.ndarray
            Cumulative histogram to normalize (modified in-place)
        hist : np.ndarray
            Original histogram (for calculating totals)
        axis : int
            Axis along which normalization should be applied
        """
        normalized = np.zeros_like(result, dtype=float)

        if axis == 0:
            for i in range(result.shape[1]):
                col_total = np.sum(hist[:, i])
                if col_total > 0:
                    normalized[:, i] = result[:, i] / col_total
        else:  # axis == 1
            for i in range(result.shape[0]):
                row_total = np.sum(hist[i, :])
                if row_total > 0:
                    normalized[i, :] = result[i, :] / row_total

        np.copyto(result, normalized)

    def _calculate_cumulative_1d(self, hist, reverse):
        """Calculate cumulative distribution for 1D histogram."""
        if reverse:
            return np.cumsum(hist[::-1])[::-1]
        return np.cumsum(hist)

    def _calculate_cumulative_2d(self, hist, reverse, axis=None):
        """Calculate cumulative distribution for 2D histogram."""
        if axis is None:
            axis = 1

        return self._apply_cumsum_along_axis(hist, axis, reverse)

    def _apply_cumsum_along_axis(self, hist, axis, reverse):
        """Apply cumulative sum along the specified axis of a 2D histogram."""

        def cumsum_func(arr):
            return np.cumsum(arr[::-1])[::-1] if reverse else np.cumsum(arr)

        return np.apply_along_axis(cumsum_func, axis, hist)

    @staticmethod
    def rebin_2d_histogram(hist, x_bins, y_bins, rebin_factor=2):
        """
        Rebin a 2D histogram by merging neighboring bins along the energy dimension (y-axis) only.

        Parameters
        ----------
        hist : np.ndarray
            Original 2D histogram data
        x_bins : np.ndarray
            Original x-axis bin edges (preserved)
        y_bins : np.ndarray
            Original y-axis (energy) bin edges
        rebin_factor : int, optional
            Factor by which to reduce the number of bins in the energy dimension
            Default is 2 (merge every 2 bins)

        Returns
        -------
        tuple
            (re-binned_hist, x_bins, re-binned_y_bins)
        """
        if rebin_factor <= 1:
            return hist, x_bins, y_bins

        x_size = hist.shape[0]
        new_y_size = hist.shape[1] // rebin_factor

        new_hist = np.zeros((x_size, new_y_size), dtype=float)

        for i in range(x_size):
            for j in range(new_y_size):
                y_start = j * rebin_factor
                y_end = (j + 1) * rebin_factor
                new_hist[i, j] = np.sum(hist[i, y_start:y_end])

        new_y_bins = y_bins[::rebin_factor]

        return new_hist, x_bins, new_y_bins
