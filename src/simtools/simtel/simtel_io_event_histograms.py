"""Histograms for shower and (if available) triggered events."""

import copy
import logging

import astropy.units as u
import numpy as np

from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader


class SimtelIOEventHistograms:
    """
    Generate and fill histograms for shower and (if available) triggered events.

    Event data is read from the reduced MC event data file.
    Calculate cumulative and relative (efficiency) distributions.

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

        self.histograms = {}
        self.file_info = {}

        self.reader = SimtelIOEventDataReader(event_data_file, telescope_list=telescope_list)

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
                "solid_angle": _file_info_table["solid_angle"].to("sr"),
                "scatter_area": _file_info_table["scatter_area"].to("cm2"),
            }

            self.histograms = self._define_histograms(event_data, triggered_data, shower_data)

            for data in self.histograms.values():
                self._fill_histogram_and_bin_edges(data)

        self.print_summary()
        self.calculate_efficiency_data()
        self.calculate_cumulative_data()

    def _define_histograms(self, event_data, triggered_data, shower_data):
        """
        Define histograms including event data, binning, naming, and labels.

        All histograms are defined for simulated and triggered events (note
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
        dict
            Dictionary with histogram definitions.
        """
        xy_bins = np.linspace(
            -1.0 * self.core_distance_bins.max(),
            self.core_distance_bins.max(),
            len(self.core_distance_bins),
        )
        hists = {}

        energy_axis_title = "Energy (TeV)"
        event_count_axis_title = "Event Count"

        definitions = {
            "energy": {
                "event_data_column": "simulated_energy",
                "event_data": event_data,
                "bin_edges": self.energy_bins,
                "axis_titles": [energy_axis_title, event_count_axis_title],
                "plot_scales": {"x": "log", "y": "log"},
            },
            "core_distance": {
                "event_data_column": "core_distance_shower",
                "event_data": event_data,
                "bin_edges": self.core_distance_bins,
                "axis_titles": ["Core Distance (m)", event_count_axis_title],
            },
            "angular_distance": {
                "event_data_column": "angular_distance",
                "event_data": triggered_data,
                "bin_edges": self.view_cone_bins,
                "axis_titles": ["Angular Distance (deg)", event_count_axis_title],
            },
            "x_core_shower_vs_y_core_shower": {
                "event_data_column": ("x_core_shower", "y_core_shower"),
                "event_data": (event_data, event_data),
                "bin_edges": (xy_bins, xy_bins),
                "is_1d": False,
                "axis_titles": ["Core X (m)", "Core Y (m)", event_count_axis_title],
            },
            "core_vs_energy": {
                "event_data_column": ("core_distance_shower", "simulated_energy"),
                "event_data": (event_data, event_data),
                "bin_edges": (self.core_distance_bins, self.energy_bins),
                "is_1d": False,
                "axis_titles": ["Core Distance (m)", energy_axis_title, event_count_axis_title],
                "plot_scales": {"y": "log"},
            },
            "angular_distance_vs_energy": {
                "event_data_column": ("angular_distance", "simulated_energy"),
                "event_data": (triggered_data, event_data),
                "bin_edges": (self.view_cone_bins, self.energy_bins),
                "is_1d": False,
                "axis_titles": [
                    "Angular Distance (deg)",
                    energy_axis_title,
                    event_count_axis_title,
                ],
                "plot_scales": {"y": "log"},
            },
        }

        hists = {
            name: self.get_histogram_definition(**cfg) | {"suffix": "", "title": "Triggered Events"}
            for name, cfg in definitions.items()
        }

        hists_mc = {}
        for key, hist in hists.items():
            key_mc = f"{key}_mc"
            hists_mc[key_mc] = copy.copy(hist)
            hists_mc[key_mc]["suffix"] = "_mc"
            hists_mc[key_mc]["title"] = "Simulated Events"
            hists_mc[key_mc]["event_data"] = (
                shower_data if hist["1d"] else (shower_data, shower_data)
            )

        hists.update(hists_mc)
        return hists

    def get_histogram_definition(
        self,
        event_data_column=None,
        event_data=None,
        histogram=None,
        bin_edges=None,
        title=None,
        axis_titles=None,
        suffix=None,
        is_1d=True,
        plot_scales=None,
    ):
        """Return a single histogram definition."""
        return {
            "histogram": histogram,
            "event_data_column": event_data_column,
            "event_data": event_data,
            "1d": is_1d,
            "bin_edges": bin_edges,
            "title": title,
            "axis_titles": axis_titles,
            "suffix": suffix,
            "plot_scales": plot_scales,
        }

    def _fill_histogram_and_bin_edges(self, data):
        """
        Fill histogram and bin edges into the histogram dictionary.

        Adds to existing histogram if present, otherwise initializes it.
        """
        if data["1d"]:
            if data["event_data"] is None:
                self._logger.debug(f"DEBUG: event_data is None for {data.get('event_data_column')}")
                return
            hist, _ = np.histogram(
                getattr(data["event_data"], data["event_data_column"]), bins=data["bin_edges"]
            )
        else:
            if data["event_data"][0] is None or data["event_data"][1] is None:
                return
            hist, _, _ = np.histogram2d(
                getattr(data["event_data"][0], data["event_data_column"][0]),
                getattr(data["event_data"][1], data["event_data_column"][1]),
                bins=[data["bin_edges"][0], data["bin_edges"][1]],
            )

        data["histogram"] = hist if data["histogram"] is None else data["histogram"] + hist

    def calculate_efficiency_data(self):
        """
        Calculate efficiency histograms (triggered divided by simulated).

        Assumes that for each histogram with simulated events, there is a
        corresponding histogram with triggered events.

        Returns
        -------
        dict
            Dictionary containing the efficiency histograms.
        """
        if "TRIGGERS" not in self.reader.data_sets:
            return None

        def calculate_efficiency(trig_hist, mc_hist):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.divide(
                    trig_hist,
                    mc_hist,
                    out=np.zeros_like(trig_hist, dtype=float),
                    where=mc_hist > 0,
                )

        eff_histograms = {}
        for name, mc_hist in self.histograms.items():
            if not name.endswith("_mc"):
                continue

            base_name = name[:-3]
            trig_hist = self.histograms.get(base_name)
            if trig_hist is None:
                continue

            if mc_hist["histogram"].shape != trig_hist["histogram"].shape:
                self._logger.warning(
                    f"Shape mismatch for {base_name} and {name}, skipping efficiency calculation."
                )
                continue

            eff = copy.copy(mc_hist)
            eff.update(
                {
                    "histogram": calculate_efficiency(trig_hist["histogram"], mc_hist["histogram"]),
                    "suffix": "_eff",
                    "title": "Efficiency",
                }
            )
            eff["axis_titles"] = copy.copy(mc_hist["axis_titles"])
            eff["axis_titles"][-1] = "Efficiency"
            eff_histograms[f"{base_name}_eff"] = eff

        self.histograms.update(eff_histograms)
        return eff_histograms

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

        viewcone_min = self.file_info.get("viewcone_min", 0.0 * u.deg).to("deg").value
        viewcone_max = self.file_info.get("viewcone_max", 20.0 * u.deg).to("deg").value

        # avoid zero-width bins
        if viewcone_min == viewcone_max:
            viewcone_max = 0.5

        return np.linspace(viewcone_min, viewcone_max, 100)

    def calculate_cumulative_data(self):
        """
        Calculate cumulative distributions for triggered histograms.

        Returns
        -------
        dict
            Dictionary containing the cumulative histograms.
        """
        if "TRIGGERS" not in self.reader.data_sets:
            return None
        cumulative_data = {}
        suffix = "_cumulative"

        def add_cumulative(name, hist, **kwargs):
            new = copy.copy(hist)
            new["histogram"] = self._calculate_cumulative_histogram(hist["histogram"], **kwargs)
            new["axis_titles"] = copy.copy(hist["axis_titles"])
            new.update(
                {
                    "suffix": suffix,
                    "title": "Cumulative triggered events",
                }
            )
            new["axis_titles"][-1] = "Fraction of Events"
            cumulative_data[f"{name}{suffix}"] = new

        # 2D histograms vs energy
        for name, hist in self.histograms.items():
            if name.endswith("_vs_energy") and not name.endswith("_mc"):
                add_cumulative(name, hist, axis=0, normalize=True)

        # 1D histograms
        for name in ["energy", "core_distance", "angular_distance"]:
            if (hist := self.histograms.get(name)) is not None:
                add_cumulative(name, hist, reverse=name == "energy")

        self.histograms.update(cumulative_data)
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

        axis = axis if axis is not None else 1
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
        axis = axis if axis is not None else 1
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

    def print_summary(self):
        """
        Print a summary of the histogram statistics.

        Total number of events is retrieved from the 'energy' histograms.
        """
        total_simulated = np.sum(self.histograms.get("energy_mc", {}).get("histogram", []))
        total_triggered = np.sum(self.histograms.get("energy", {}).get("histogram", []))

        self._logger.info(f"Total simulated events: {total_simulated}")
        self._logger.info(f"Total triggered events: {total_triggered}")
