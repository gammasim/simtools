"""Histograms for shower and (if available) triggered events."""

import copy
import logging

import astropy.units as u
import numpy as np

from simtools.sim_events.reader import EventDataReader
from simtools.utils.general import resolve_file_patterns
from simtools.utils.value_conversion import get_value_as_quantity

_ANGULAR_DISTANCE = "Angular Distance (deg)"
_CORE_DISTANCE = "Core Distance (m)"
_REUSE_STAT_LABELS = {
    "mean": "Mean Reuse",
    "max": "Max Reuse",
    "std": "Reuse Std Dev",
}


class EventDataHistograms:
    """
    Generate and fill histograms for shower and (if available) triggered events.

    Event data is read from the reduced MC event data file.
    Calculate cumulative and relative (efficiency) distributions.

    Parameters
    ----------
    event_data_file : str or list[pathlib.Path | str]
        Path to the event-data file or a list of event-data files.
    array_name : str, optional
        Name of the telescope array configuration (default is None).
    telescope_list : list, optional
        List of telescope IDs to filter the events (default is None).
    energy_bins_per_decade : int, optional
        Number of energy bins per decade for logarithmic energy histograms.
    skip_invalid_event_data_files : bool, optional
        Skip invalid event-data files while reading resolved input files.
    require_triggered_data : bool, optional
        Require triggered-event tables in each input file.
    """

    def __init__(
        self,
        event_data_file,
        array_name=None,
        telescope_list=None,
        energy_bins_per_decade=10,
        angular_distance_bin_count=100,
        angular_distance_bin_width=None,
        core_distance_bin_count=100,
        skip_invalid_event_data_files=False,
        require_triggered_data=False,
    ):
        """Initialize."""
        self._logger = logging.getLogger(__name__)
        self.event_data_file = event_data_file
        self.event_data_files = self._normalize_event_data_files(event_data_file)
        self.array_name = array_name
        self.energy_bins_per_decade = max(int(energy_bins_per_decade), 1)
        self.angular_distance_bin_count = max(int(angular_distance_bin_count), 2)
        self.angular_distance_bin_width = self._validate_angular_distance_bin_width(
            angular_distance_bin_width
        )
        self.core_distance_bin_count = max(int(core_distance_bin_count), 2)
        self.skip_invalid_event_data_files = skip_invalid_event_data_files
        self.require_triggered_data = require_triggered_data
        self.telescope_list = telescope_list

        self.histograms = {}
        self.file_info = {}
        self.data_ranges = {}
        self.loaded_bin_edges = {}
        self._contains_triggered_data = False
        self._filled_data_sets = 0
        self._release_event_data_after_fill = False
        self._reuse_stat_accumulators = {}

        self.reader = None
        if not self.skip_invalid_event_data_files:
            self.reader = EventDataReader(
                self.event_data_files[0], telescope_list=self.telescope_list
            )
            self._contains_triggered_data = self._reader_has_triggered_data(self.reader)

    @classmethod
    def create_accumulator(
        cls,
        array_name=None,
        telescope_list=None,
        energy_bins_per_decade=10,
        angular_distance_bin_count=100,
        angular_distance_bin_width=None,
        core_distance_bin_count=100,
    ):
        """Create an empty histogram accumulator for externally supplied event data.

        This avoids opening the same input files for every telescope configuration when
        one caller reads the event data once and distributes it to several accumulators.
        """
        instance = cls.__new__(cls)
        instance._logger = logging.getLogger(__name__)
        instance.event_data_file = None
        instance.event_data_files = []
        instance.array_name = array_name
        instance.energy_bins_per_decade = max(int(energy_bins_per_decade), 1)
        instance.angular_distance_bin_count = max(int(angular_distance_bin_count), 2)
        instance.angular_distance_bin_width = cls._validate_angular_distance_bin_width(
            angular_distance_bin_width
        )
        instance.core_distance_bin_count = max(int(core_distance_bin_count), 2)
        instance.skip_invalid_event_data_files = False
        instance.require_triggered_data = True
        instance.telescope_list = telescope_list
        instance.histograms = {}
        instance.file_info = {}
        instance.data_ranges = {}
        instance.loaded_bin_edges = {}
        instance._contains_triggered_data = True
        instance._filled_data_sets = 0
        instance._release_event_data_after_fill = True
        instance._reuse_stat_accumulators = {}
        instance.reader = None
        return instance

    def get_empty_histogram_definitions(self):
        """Return empty histogram definitions without attached event-data arrays."""
        histograms = self._define_histograms(None, None, None)
        for histogram in histograms.values():
            histogram["event_data"] = (
                None if histogram["1d"] else tuple(None for _ in histogram["event_data_column"])
            )
        return histograms

    def set_loaded_histograms(
        self,
        histograms,
        file_info=None,
        data_ranges=None,
        loaded_bin_edges=None,
        contains_triggered_data=True,
    ):
        """Install histogram data loaded from a serialized histogram product."""
        self.histograms = histograms
        if file_info is not None:
            self.file_info = file_info
        if data_ranges is not None:
            self.data_ranges = data_ranges
        if loaded_bin_edges is not None:
            self.loaded_bin_edges = loaded_bin_edges
        self._filled_data_sets = 1
        self._contains_triggered_data = contains_triggered_data
        self._reuse_stat_accumulators = {}

    @staticmethod
    def _validate_angular_distance_bin_width(angular_distance_bin_width):
        """Return angular-distance bin width in deg, or None for count-based binning."""
        if angular_distance_bin_width is None:
            return None
        angular_distance_bin_width = get_value_as_quantity(angular_distance_bin_width, "deg")
        if angular_distance_bin_width.value <= 0.0:
            raise ValueError("angular_distance_bin_width must be positive.")
        return angular_distance_bin_width

    def _normalize_event_data_files(self, event_data_file):
        """Return event-data files as a list of resolved file names."""
        return [str(file_name) for file_name in resolve_file_patterns(event_data_file)]

    def _reader_has_triggered_data(self, reader):
        """Check if a reader exposes triggered event tables."""
        return any(isinstance(ds, dict) and "TRIGGERS" in ds for ds in reader.data_sets)

    def _log_skipped_event_data_file(self, event_data_file, exception):
        """Log skipped invalid event-data file."""
        self._logger.warning(f"Skipping invalid event data file '{event_data_file}': {exception}")

    def _iter_readers(self):
        """Yield one reader per input file to keep memory usage bounded."""
        for index, event_data_file in enumerate(self.event_data_files):
            try:
                if index == 0 and self.reader is not None:
                    reader = self.reader
                else:
                    reader = EventDataReader(event_data_file, telescope_list=self.telescope_list)
                if self.require_triggered_data and not self._reader_has_triggered_data(reader):
                    raise ValueError("Missing triggered event table(s).")
            except (OSError, KeyError, ValueError) as exc:
                if self.skip_invalid_event_data_files:
                    self._log_skipped_event_data_file(event_data_file, exc)
                    continue
                raise

            self.reader = reader
            self._contains_triggered_data = (
                self._contains_triggered_data or self._reader_has_triggered_data(self.reader)
            )
            yield event_data_file, self.reader

    def _read_data_set(self, reader, event_data_file, data_set, file_index=None, total_files=None):
        """Read one dataset and return reduced file information with event tables."""
        progress = f" ({file_index}/{total_files})" if file_index and total_files else ""
        self._logger.info(f"Reading event data from {event_data_file} for {data_set}{progress}")
        file_info_table, shower_data, event_data, triggered_data = reader.read_event_data(
            event_data_file, table_name_map=data_set
        )
        file_info_table = reader.get_reduced_simulation_file_info(file_info_table)
        return file_info_table, shower_data, event_data, triggered_data

    def _get_file_info_value(self, file_info_table, key, unit=None):
        """Return file-info value, converting to the requested unit when provided."""
        value = file_info_table.get(key)
        if value is None or unit is None:
            return value
        return get_value_as_quantity(value, unit)

    def _update_file_info(self, file_info_table):
        """Store normalized metadata from the reduced file-info table."""
        spectral_index = self._get_file_info_value(file_info_table, "spectral_index")
        if spectral_index is None:
            spectral_index = np.nan
        self.file_info = {
            "primary_particle": self._get_file_info_value(file_info_table, "primary_particle"),
            "zenith": self._get_file_info_value(file_info_table, "zenith", "deg"),
            "azimuth": self._get_file_info_value(file_info_table, "azimuth", "deg"),
            "nsb_level": self._get_file_info_value(file_info_table, "nsb_level"),
            "spectral_index": spectral_index,
            "energy_min": self._get_file_info_value(file_info_table, "energy_min", "TeV"),
            "energy_max": self._get_file_info_value(file_info_table, "energy_max", "TeV"),
            "core_scatter_max": self._get_file_info_value(file_info_table, "core_scatter_max", "m"),
            "viewcone_min": self._get_file_info_value(file_info_table, "viewcone_min", "deg"),
            "viewcone_max": self._get_file_info_value(file_info_table, "viewcone_max", "deg"),
            "solid_angle": self._get_file_info_value(file_info_table, "solid_angle", "sr"),
            "scatter_area": self._get_file_info_value(file_info_table, "scatter_area", "cm2"),
        }

    def _merge_histograms(self, current_histograms):
        """Carry over accumulated histogram counts before filling new data."""
        for name, hist in current_histograms.items():
            previous = self.histograms.get(name)
            if previous is not None:
                hist["histogram"] = previous["histogram"]
        self.histograms = current_histograms

    def _fill_current_histograms(self):
        """Fill all currently defined histograms with their event data."""
        for data in self.histograms.values():
            self._fill_histogram_and_bin_edges(data)

    def _update_data_range(self, name, values):
        """Accumulate the finite minimum and maximum of an event-data field."""
        try:
            values = np.asarray(values, dtype=float)
        except TypeError, ValueError:
            return
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        current_min, current_max = self.data_ranges.get(name, (np.inf, -np.inf))
        self.data_ranges[name] = (
            min(current_min, float(np.min(values))),
            max(current_max, float(np.max(values))),
        )

    def _release_event_data(self):
        """Release raw event references after their histogram counts have been accumulated."""
        for data in self.histograms.values():
            if data["1d"] or data["event_data"] is None:
                data["event_data"] = None
            else:
                data["event_data"] = tuple(None for _ in data["event_data"])

    def accumulate(self, file_info_table, shower_data, event_data, triggered_data):
        """Accumulate one already-read event dataset into all configured histograms."""
        self._update_file_info(file_info_table)
        self._update_data_range("angular_distance", triggered_data.angular_distance)
        current_histograms = self._define_histograms(event_data, triggered_data, shower_data)
        self._merge_histograms(current_histograms)
        self._fill_current_histograms()
        self._accumulate_reuse_histograms(event_data, triggered_data)
        if self._release_event_data_after_fill:
            self._release_event_data()
        self._filled_data_sets += 1

    def finalize(self, fill_efficiency_histogram=True):
        """Finalize accumulated histograms after all event datasets have been added."""
        if self._filled_data_sets == 0:
            raise ValueError("No readable event data files or datasets found.")
        self.print_summary()
        if fill_efficiency_histogram:
            self.calculate_efficiency_data()
        self.calculate_cumulative_data()

    def iter_event_data(self):
        """Yield reduced event datasets sequentially, keeping input memory bounded."""
        total_files = len(self.event_data_files)
        for file_index, (event_data_file, reader) in enumerate(self._iter_readers(), start=1):
            for data_set in reader.data_sets:
                try:
                    values = self._read_data_set(
                        reader,
                        event_data_file,
                        data_set,
                        file_index=file_index,
                        total_files=total_files,
                    )
                except (OSError, KeyError, ValueError) as exc:
                    if self.skip_invalid_event_data_files:
                        self._log_skipped_event_data_file(event_data_file, exc)
                        break
                    raise
                yield reader, values

    def fill(self, fill_efficiency_histogram=True):
        """
        Fill histograms with event data.

        Involves looping over all event data, and therefore is the slowest part of the
        histogram module. Adds the histograms to the histogram dictionary.

        Assume that all event data files are generated with similar configurations
        (self.file_info contains the file info of the last file).

        Parameters
        ----------
        fill_efficiency_histogram : bool, optional
            Whether to calculate and fill the efficiency histograms.
        """
        for _, values in self.iter_event_data():
            self.accumulate(*values)
        self.finalize(fill_efficiency_histogram=fill_efficiency_histogram)

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
                "axis_titles": [_CORE_DISTANCE, event_count_axis_title],
                "plot_scales": {"y": "log"},
            },
            "angular_distance": {
                "event_data_column": "angular_distance",
                "event_data": triggered_data,
                "bin_edges": self.view_cone_bins,
                "axis_titles": [_ANGULAR_DISTANCE, event_count_axis_title],
                "plot_scales": {"y": "log"},
            },
            "x_core_shower_vs_y_core_shower": {
                "event_data_column": ("x_core_shower", "y_core_shower"),
                "event_data": (event_data, event_data),
                "bin_edges": (xy_bins, xy_bins),
                "is_1d": False,
                "axis_titles": ["Core X (m)", "Core Y (m)", event_count_axis_title],
            },
            "core_distance_vs_energy": {
                "event_data_column": ("core_distance_shower", "simulated_energy"),
                "event_data": (event_data, event_data),
                "bin_edges": (self.core_distance_bins, self.energy_bins),
                "is_1d": False,
                "axis_titles": [_CORE_DISTANCE, energy_axis_title, event_count_axis_title],
                "plot_scales": {"y": "log"},
            },
            "angular_distance_vs_energy": {
                "event_data_column": ("angular_distance", "simulated_energy"),
                "event_data": (triggered_data, event_data),
                "bin_edges": (self.view_cone_bins, self.energy_bins),
                "is_1d": False,
                "axis_titles": [
                    _ANGULAR_DISTANCE,
                    energy_axis_title,
                    event_count_axis_title,
                ],
                "plot_scales": {"y": "log"},
            },
            "angular_distance_vs_energy_vs_core_distance": {
                "event_data_column": (
                    "angular_distance",
                    "simulated_energy",
                    "core_distance_shower",
                ),
                "event_data": (triggered_data, event_data, event_data),
                "bin_edges": (self.view_cone_bins, self.energy_bins, self.core_distance_bins),
                "is_1d": False,
                "axis_titles": [
                    _ANGULAR_DISTANCE,
                    energy_axis_title,
                    _CORE_DISTANCE,
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
                shower_data if hist["1d"] else tuple(shower_data for _ in hist["event_data_column"])
            )

        hists.update(hists_mc)
        hists.update(self._define_reuse_histograms())
        return hists

    def _define_reuse_histograms(self):
        """Return definitions for reuse summary histograms."""
        energy_axis_title = "Energy (TeV)"
        reuse_histograms = {}
        definitions = (
            (
                "energy",
                self.energy_bins,
                [energy_axis_title],
                {"x": "log", "y": "linear"},
            ),
            (
                "core_distance",
                self.core_distance_bins,
                [_CORE_DISTANCE],
                {"x": "linear", "y": "linear"},
            ),
            (
                "angular_distance",
                self.view_cone_bins,
                [_ANGULAR_DISTANCE],
                {"x": "linear", "y": "linear"},
            ),
            (
                "core_distance_vs_energy",
                (self.core_distance_bins, self.energy_bins),
                [_CORE_DISTANCE, energy_axis_title],
                {"x": "linear", "y": "log"},
            ),
            (
                "angular_distance_vs_energy",
                (self.view_cone_bins, self.energy_bins),
                [_ANGULAR_DISTANCE, energy_axis_title],
                {"x": "linear", "y": "log"},
            ),
        )

        for statistic, label in _REUSE_STAT_LABELS.items():
            for base_name, bin_edges, axis_titles, plot_scales in definitions:
                reuse_histograms[f"reuse_{statistic}_vs_{base_name}"] = (
                    self.get_histogram_definition(
                        event_data_column=() if isinstance(bin_edges, tuple) else None,
                        histogram=None,
                        bin_edges=bin_edges,
                        title=f"Triggered reuse {statistic}",
                        axis_titles=[*axis_titles, label],
                        suffix="",
                        is_1d=not isinstance(bin_edges, tuple),
                        plot_scales=plot_scales,
                    )
                )
        return reuse_histograms

    def _accumulate_reuse_histograms(self, event_data, triggered_data):
        """Accumulate reuse summary statistics into their histogram definitions."""
        if not self._can_accumulate_reuse_histograms(event_data, triggered_data):
            return

        try:
            reuse_counts = self._triggered_reuse_counts(triggered_data)
            coordinates = {
                "energy": np.asarray(event_data.simulated_energy, dtype=float),
                "core_distance": np.asarray(event_data.core_distance_shower, dtype=float),
                "angular_distance": np.asarray(triggered_data.angular_distance, dtype=float),
            }
        except AttributeError, TypeError, ValueError:
            return
        histogram_specs = {
            "energy": (coordinates["energy"], self.energy_bins),
            "core_distance": (coordinates["core_distance"], self.core_distance_bins),
            "angular_distance": (coordinates["angular_distance"], self.view_cone_bins),
            "core_distance_vs_energy": (
                (coordinates["core_distance"], coordinates["energy"]),
                (self.core_distance_bins, self.energy_bins),
            ),
            "angular_distance_vs_energy": (
                (coordinates["angular_distance"], coordinates["energy"]),
                (self.view_cone_bins, self.energy_bins),
            ),
        }

        for statistic in _REUSE_STAT_LABELS:
            for name, (values, bin_edges) in histogram_specs.items():
                histogram_name = f"reuse_{statistic}_vs_{name}"
                histogram = self.histograms.get(histogram_name)
                if histogram is None:
                    continue
                accumulator = self._reuse_stat_accumulators.setdefault(
                    histogram_name,
                    self._create_reuse_stat_accumulator(bin_edges),
                )
                self._update_reuse_stat_accumulator(accumulator, values, reuse_counts, bin_edges)
                histogram["histogram"] = self._finalize_reuse_statistic(accumulator, statistic)

    @staticmethod
    def _can_accumulate_reuse_histograms(event_data, triggered_data):
        """Return whether the required inputs for reuse summaries are available."""
        required_event_attrs = (
            "file_id",
            "shower_id",
            "simulated_energy",
            "core_distance_shower",
        )
        required_triggered_attrs = ("file_id", "shower_id", "angular_distance")
        return (
            event_data is not None
            and triggered_data is not None
            and all(hasattr(event_data, attr) for attr in required_event_attrs)
            and all(hasattr(triggered_data, attr) for attr in required_triggered_attrs)
        )

    @staticmethod
    def _triggered_reuse_counts(triggered_data):
        """Return triggered reuse count per triggered event row."""
        triggered_keys = np.column_stack(
            (
                np.asarray(triggered_data.file_id, dtype=np.int64),
                np.asarray(triggered_data.shower_id, dtype=np.int64),
            )
        )
        _, inverse, counts = np.unique(
            triggered_keys, axis=0, return_inverse=True, return_counts=True
        )
        return counts[inverse].astype(float)

    def _create_reuse_stat_accumulator(self, bin_edges):
        """Return empty accumulation arrays for one reuse-statistic histogram."""
        shape = (
            tuple(len(edges) - 1 for edges in bin_edges)
            if isinstance(bin_edges, tuple)
            else (len(bin_edges) - 1,)
        )
        return {
            "count": np.zeros(shape, dtype=int),
            "sum": np.zeros(shape, dtype=float),
            "sum_sq": np.zeros(shape, dtype=float),
            "max": np.full(shape, np.nan, dtype=float),
        }

    def _update_reuse_stat_accumulator(self, accumulator, values, reuse_counts, bin_edges):
        """Update one reuse-statistic accumulator with one dataset."""
        if isinstance(values, tuple):
            x_values, y_values = values
            shape = accumulator["count"].shape
            x_indices = np.digitize(x_values, bin_edges[0]) - 1
            y_indices = np.digitize(y_values, bin_edges[1]) - 1
            valid = (
                np.isfinite(reuse_counts)
                & np.isfinite(x_values)
                & np.isfinite(y_values)
                & (x_indices >= 0)
                & (x_indices < shape[0])
                & (y_indices >= 0)
                & (y_indices < shape[1])
            )
            self._accumulate_reuse_bin_values(
                accumulator,
                np.column_stack((x_indices[valid], y_indices[valid])),
                reuse_counts[valid],
            )
            return

        indices = np.digitize(values, bin_edges) - 1
        valid = (
            np.isfinite(reuse_counts)
            & np.isfinite(values)
            & (indices >= 0)
            & (indices < accumulator["count"].shape[0])
        )
        self._accumulate_reuse_bin_values(accumulator, indices[valid], reuse_counts[valid])

    @staticmethod
    def _accumulate_reuse_bin_values(accumulator, indices, reuse_counts):
        """Update count, sum, sum-of-squares, and max for addressed bins."""
        reuse_counts = np.asarray(reuse_counts, dtype=float)
        positive = reuse_counts > 0.0
        if not np.any(positive):
            return

        indices = np.asarray(indices, dtype=np.int64)
        shape = accumulator["count"].shape
        if indices.ndim == 1:
            flat_indices = indices[positive]
        else:
            flat_indices = np.ravel_multi_index(indices[positive].T, shape)

        values = reuse_counts[positive]
        flat_size = int(np.prod(shape, dtype=np.int64))

        count = np.bincount(flat_indices, minlength=flat_size).reshape(shape)
        sum_ = np.bincount(flat_indices, weights=values, minlength=flat_size).reshape(shape)
        sum_sq = np.bincount(flat_indices, weights=values**2, minlength=flat_size).reshape(shape)

        accumulator["count"] += count
        accumulator["sum"] += sum_
        accumulator["sum_sq"] += sum_sq
        np.fmax.at(accumulator["max"].reshape(-1), flat_indices, values)

    @staticmethod
    def _finalize_reuse_statistic(accumulator, statistic):
        """Return finalized reuse statistic values from accumulation arrays."""
        count = accumulator["count"]
        result = np.full(count.shape, np.nan, dtype=float)
        valid = count > 0

        if statistic == "mean":
            result[valid] = accumulator["sum"][valid] / count[valid]
            return result
        if statistic == "max":
            result[valid] = accumulator["max"][valid]
            return result
        if statistic == "std":
            mean = np.zeros(count.shape, dtype=float)
            mean[valid] = accumulator["sum"][valid] / count[valid]
            variance = np.zeros(count.shape, dtype=float)
            variance[valid] = accumulator["sum_sq"][valid] / count[valid] - mean[valid] ** 2
            result[valid] = np.sqrt(np.maximum(variance[valid], 0.0))
            return result

        raise ValueError(f"Unsupported reuse statistic: {statistic}")

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
            "title_fontsize": "xx-small",
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
                return
            hist, _ = np.histogram(
                getattr(data["event_data"], data["event_data_column"]), bins=data["bin_edges"]
            )
        else:
            if data["event_data"] is None:
                return
            if any(event_data is None for event_data in data["event_data"]):
                return
            values = [
                getattr(event_data, column_name)
                for event_data, column_name in zip(data["event_data"], data["event_data_column"])
            ]
            if len(values) == 2:
                hist, _, _ = np.histogram2d(
                    values[0],
                    values[1],
                    bins=[data["bin_edges"][0], data["bin_edges"][1]],
                )
            else:
                hist, _ = np.histogramdd(
                    np.column_stack(values),
                    bins=data["bin_edges"],
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
        if not self._contains_triggered_data and not self._reader_has_triggered_data(self.reader):
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
        """
        Return bins for the energy histogram.

        Align bins to full decades of energy, using the configured bins per decade,
        and ensure that the range covers the energy range of the events.

        Returns
        -------
        np.ndarray            Array of energy bin edges in TeV.
        """
        if "energy" in self.loaded_bin_edges:
            return self.loaded_bin_edges["energy"]
        if "energy_bin_edges" in self.histograms:
            return self.histograms["energy_bin_edges"]

        energy_min = self.file_info.get("energy_min", 1.0e-3 * u.TeV).to("TeV").value
        energy_max = self.file_info.get("energy_max", 1.0e3 * u.TeV).to("TeV").value
        energy_min = max(energy_min, 1e-3)
        energy_max = max(energy_max, 10 * energy_min)

        lower_decade = np.floor(np.log10(energy_min))
        upper_decade = np.ceil(np.log10(energy_max))
        if upper_decade <= lower_decade:
            upper_decade = lower_decade + 1

        n_bins = int((upper_decade - lower_decade) * self.energy_bins_per_decade)
        return np.logspace(lower_decade, upper_decade, n_bins + 1)

    @property
    def core_distance_bins(self):
        """
        Return bins for the core distance histogram.

        CORSIKA CSCAT ('core_scatter_max') is defined in the shower plane.
        """
        if "core_distance" in self.loaded_bin_edges:
            return self.loaded_bin_edges["core_distance"]
        if "core_distance_bin_edges" in self.histograms:
            return self.histograms["core_distance_bin_edges"]

        return np.linspace(
            self.file_info.get("core_scatter_min", 0.0 * u.m).to("m").value,
            self.file_info.get("core_scatter_max", 1.0e5 * u.m).to("m").value,
            self.core_distance_bin_count,
        )

    @property
    def view_cone_bins(self):
        """Return bins for the viewcone histogram."""
        if "viewcone" in self.loaded_bin_edges:
            return self.loaded_bin_edges["viewcone"]
        if "viewcone_bin_edges" in self.histograms:
            return self.histograms["viewcone_bin_edges"]

        viewcone_min = self.file_info.get("viewcone_min", 0.0 * u.deg).to("deg").value
        viewcone_max = self.file_info.get("viewcone_max", 20.0 * u.deg).to("deg").value

        # avoid zero-width bins
        if viewcone_min == viewcone_max:
            viewcone_max = viewcone_min + 0.5

        if self.angular_distance_bin_width is not None:
            return self._fixed_width_bins(
                viewcone_min,
                viewcone_max,
                self.angular_distance_bin_width.to_value(u.deg),
            )

        return np.linspace(viewcone_min, viewcone_max, self.angular_distance_bin_count)

    @staticmethod
    def _fixed_width_bins(lower_edge, upper_edge, bin_width):
        """Return bin edges from lower to upper edge using fixed-width bins."""
        if upper_edge <= lower_edge:
            upper_edge = lower_edge + bin_width
        bin_count = int(np.floor((upper_edge - lower_edge) / bin_width))
        edges = lower_edge + np.arange(bin_count + 1) * bin_width
        if np.isclose(edges[-1], upper_edge):
            edges[-1] = upper_edge
        elif edges[-1] < upper_edge:
            edges = np.append(edges, upper_edge)
        return edges

    def calculate_cumulative_data(self):
        """
        Calculate cumulative distributions for triggered histograms.

        Returns
        -------
        dict
            Dictionary containing the cumulative histograms.
        """
        if not self._contains_triggered_data and not self._reader_has_triggered_data(self.reader):
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
            if (
                name.endswith("_vs_energy")
                and not name.endswith("_mc")
                and not name.startswith("reuse_")
            ):
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
