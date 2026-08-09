"""Utilities for comparison across multiple simulation productions."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from simtools.camera import trace_analysis as trace
from simtools.io import table_handler
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
    _load_dense_histogram_payloads,
)
from simtools.simtel.simtel_event_reader import read_events
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)
from simtools.utils.general import ensure_string_lists, resolve_file_patterns


@dataclass
class ProductionDescriptor:
    """Descriptor for one production input provided."""

    label: str
    input_files: list[str]


@dataclass
class ProductionEventMetrics:
    """Aggregated event-level metrics for one production."""

    label: str
    simulated_energies: np.ndarray
    triggered_energies: np.ndarray
    simulated_core_distances: np.ndarray
    triggered_core_distances: np.ndarray
    trigger_multiplicity: np.ndarray
    trigger_combinations: Counter
    telescope_participation: Counter
    simulated_event_count: int
    triggered_event_count: int
    simulated_angular_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    triggered_angular_distances: np.ndarray = field(default_factory=lambda: np.array([]))
    per_type: dict = field(default_factory=dict)
    quantity_histograms: dict = field(default_factory=dict)
    trigger_multiplicity_histogram: tuple = field(default_factory=tuple)

    @property
    def trigger_fraction(self):
        """Return triggered/simulated fraction."""
        if self.simulated_event_count <= 0:
            return 0.0
        return self.triggered_event_count / self.simulated_event_count


@dataclass
class ProductionSignalMetrics:
    """Aggregated signal metrics for one production and telescope."""

    label: str
    pedestals: np.ndarray
    signals: np.ndarray
    peak_timing: np.ndarray
    triggered_pixels: np.ndarray


def parse_production_arguments(production_arguments):
    """Parse repeated production arguments into validated descriptors.

    Parameters
    ----------
    production_arguments : list[list[str]]
        Repeated ``--production`` arguments in the shape ``[label, patterns]``.

    Returns
    -------
    list[ProductionDescriptor]
        Validated and normalized production descriptors.

    Raises
    ------
    ValueError
        If configuration is malformed or does not contain any production.
    """
    parsed_productions = _normalize_production_arguments(production_arguments)
    if not parsed_productions:
        raise ValueError("At least one production is required.")

    labels = [label for label, _ in parsed_productions]
    if len(set(labels)) != len(labels):
        raise ValueError("Production labels must be unique.")

    descriptors = []
    for label, pattern_list in parsed_productions:
        patterns = [pattern.strip() for pattern in pattern_list.split(",") if pattern.strip()]
        if len(patterns) == 0:
            raise ValueError(f"Production '{label}' has no input_file pattern.")

        resolved_files = [str(path) for path in resolve_file_patterns(patterns)]
        if len(resolved_files) == 0:
            raise ValueError(f"Production '{label}' does not resolve to any files.")
        descriptors.append(ProductionDescriptor(label=label, input_files=resolved_files))

    return descriptors


def _ensure_single_array_layout_name(array_layout_name):
    """Return one array layout name for telescope-level comparisons.

    Parameters
    ----------
    array_layout_name : str or list[str]
        Array layout selection.

    Returns
    -------
    str
        The selected array layout name.

    Raises
    ------
    ValueError
        If zero or multiple layouts are provided.
    """
    if isinstance(array_layout_name, str):
        return array_layout_name
    if not array_layout_name or len(array_layout_name) != 1:
        raise ValueError("Signal comparison requires exactly one array_layout_name.")
    return array_layout_name[0]


def _normalize_production_arguments(production_arguments):
    """Normalize raw production arguments into ``[(label, files), ...]``."""
    if not production_arguments:
        return []

    normalized = []
    if all(isinstance(item, str) for item in production_arguments):
        return _pairwise_label_file_arguments(production_arguments)

    for item in production_arguments:
        normalized.extend(_normalize_single_production_argument(item))

    return normalized


def _pairwise_label_file_arguments(flat_arguments):
    """Convert a flat list of strings into ``[(label, files), ...]`` pairs."""
    if len(flat_arguments) % 2 != 0:
        _raise_invalid_production_arguments()
    return [
        (flat_arguments[index], flat_arguments[index + 1])
        for index in range(0, len(flat_arguments), 2)
    ]


def _normalize_single_production_argument(argument):
    """Normalize one nested production argument into label/file pairs."""
    if not isinstance(argument, list | tuple):
        _raise_invalid_production_arguments()
    if not all(isinstance(value, str) for value in argument):
        _raise_invalid_production_arguments()
    if len(argument) == 2:
        return [(argument[0], argument[1])]
    return _pairwise_label_file_arguments(list(argument))


def _raise_invalid_production_arguments():
    """Raise a standardized parser error for malformed production arguments."""
    raise ValueError("Production arguments must be provided as label/file pairs.")


def collect_production_metrics(production_descriptors, array_names=None):
    """Collect comparison metrics from trigger histogram files for each production.

    Parameters
    ----------
    production_descriptors : list[ProductionDescriptor]
        Input descriptor for each production.
    array_names : list[str] or str, optional
        Restrict loaded histogram references to these array layout names.

    Returns
    -------
    list[ProductionEventMetrics]
        Aggregated metrics per production.
    """
    return [
        _collect_single_production_histogram_metrics(descriptor, array_names=array_names)
        for descriptor in production_descriptors
    ]


def _collect_single_production_histogram_metrics(production_descriptor, array_names=None):
    """Collect comparison metrics for one trigger-histogram production descriptor."""
    selected_array_names = ensure_string_lists(array_names)
    accumulators = _initialize_histogram_metric_accumulators()
    matched_references = 0
    for trigger_histogram_file in production_descriptor.input_files:
        matched_references += _collect_metrics_from_trigger_histogram_file(
            trigger_histogram_file,
            accumulators,
            array_names=selected_array_names,
        )
    if selected_array_names and matched_references == 0:
        raise ValueError(
            "Array layout selection did not match any trigger-histogram references for "
            f"production '{production_descriptor.label}'."
        )

    simulated_histograms = accumulators["quantity_histograms"]["simulated"]
    triggered_histograms = accumulators["quantity_histograms"]["triggered"]
    per_type = _build_per_type_histogram_metrics(
        production_descriptor.label,
        simulated_histograms,
        accumulators,
    )

    return ProductionEventMetrics(
        label=production_descriptor.label,
        simulated_energies=np.array([]),
        triggered_energies=np.array([]),
        simulated_core_distances=np.array([]),
        triggered_core_distances=np.array([]),
        simulated_angular_distances=np.array([]),
        triggered_angular_distances=np.array([]),
        trigger_multiplicity=np.array([], dtype=int),
        trigger_combinations=accumulators["trigger_combinations"],
        telescope_participation=accumulators["telescope_participation"],
        simulated_event_count=accumulators["simulated_event_count"],
        triggered_event_count=accumulators["triggered_event_count"],
        per_type=per_type,
        quantity_histograms={
            quantity: {
                "simulated": simulated_histogram,
                "triggered": triggered_histograms[quantity],
            }
            for quantity, simulated_histogram in simulated_histograms.items()
        },
        trigger_multiplicity_histogram=_counter_to_histogram(accumulators["trigger_multiplicity"]),
    )


def _initialize_histogram_metric_accumulators():
    """Initialize accumulators for metrics loaded from trigger histogram files."""
    return {
        "quantity_histograms": {
            "simulated": {},
            "triggered": {},
            "subset_triggered": defaultdict(dict),
        },
        "trigger_multiplicity": Counter(),
        "trigger_combinations": Counter(),
        "telescope_participation": Counter(),
        "subset_multiplicity": defaultdict(Counter),
        "simulated_event_count": 0,
        "triggered_event_count": 0,
    }


def _collect_metrics_from_trigger_histogram_file(
    trigger_histogram_file, accumulators, array_names=None
):
    """Collect metrics from one trigger-histogram HDF5 file."""
    dense_payloads = _load_dense_histogram_payloads(trigger_histogram_file)
    tables = table_handler.read_tables(
        trigger_histogram_file,
        [
            TRIGGER_HISTOGRAM_METADATA_TABLE,
            TRIGGER_TOPOLOGY_COUNTS_TABLE,
            TRIGGER_SUBSET_HISTOGRAMS_TABLE,
        ],
        file_type="HDF5",
    )
    metadata = tables[TRIGGER_HISTOGRAM_METADATA_TABLE]
    topology_counts = tables[TRIGGER_TOPOLOGY_COUNTS_TABLE]
    subset_histograms = tables[TRIGGER_SUBSET_HISTOGRAMS_TABLE]
    topology_rows_by_reference = table_handler.group_table_rows(topology_counts, "reference_id")
    subset_rows_by_reference = table_handler.group_table_rows(subset_histograms, "reference_id")

    matched_references = 0
    for row in metadata:
        if array_names and str(row["array_name"]) not in array_names:
            continue
        matched_references += 1
        reference_id = str(row["reference_id"])
        values_by_name, edges_by_name = dense_payloads.get(reference_id, ({}, {}))
        _accumulate_quantity_histograms_from_dense_payload(
            values_by_name, edges_by_name, accumulators
        )
        _accumulate_topology_counts_for_reference(
            topology_rows_by_reference.get(row["reference_id"], topology_counts[:0]),
            accumulators,
        )
        _accumulate_subset_histograms_for_reference(
            subset_rows_by_reference.get(row["reference_id"], subset_histograms[:0]),
            accumulators,
        )
        accumulators["simulated_event_count"] += int(row["total_simulated_events"])
        accumulators["triggered_event_count"] += int(row["total_triggered_events"])
    return matched_references


def _accumulate_quantity_histograms_from_dense_payload(values_by_name, edges_by_name, accumulators):
    """Accumulate base simulated and triggered histograms from dense payloads."""
    histogram_map = {
        "energy_mc": ("energy", "simulated"),
        "energy": ("energy", "triggered"),
        "core_distance_mc": ("core_distance", "simulated"),
        "core_distance": ("core_distance", "triggered"),
        "angular_distance_mc": ("angular_distance", "simulated"),
        "angular_distance": ("angular_distance", "triggered"),
    }
    for histogram_name, (quantity, event_kind) in histogram_map.items():
        counts = values_by_name.get(histogram_name)
        axis_edges = edges_by_name.get(histogram_name, {})
        if counts is None or counts.ndim != 1 or 0 not in axis_edges:
            continue
        _add_histogram(
            accumulators["quantity_histograms"][event_kind],
            quantity,
            np.asarray(counts, dtype=float),
            np.asarray(axis_edges[0], dtype=float),
        )


def _add_histogram(target, quantity, counts, bin_edges):
    """Add histogram counts into a target mapping, requiring consistent bin edges."""
    if quantity not in target:
        target[quantity] = (np.asarray(counts, dtype=float), np.asarray(bin_edges, dtype=float))
        return
    existing_counts, existing_edges = target[quantity]
    bin_edges = np.asarray(bin_edges, dtype=float)
    if np.array_equal(existing_edges, bin_edges):
        target[quantity] = (existing_counts + counts, existing_edges)
        return

    merged_edges = np.union1d(existing_edges, bin_edges)
    if merged_edges.size < 2:
        raise ValueError(f"Inconsistent bin edges for quantity '{quantity}'.")
    target[quantity] = (
        _rebin_histogram(existing_counts, existing_edges, merged_edges)
        + _rebin_histogram(np.asarray(counts, dtype=float), bin_edges, merged_edges),
        merged_edges,
    )


def _rebin_histogram(counts, source_edges, target_edges):
    """Approximate histogram rebinning by projecting source-bin centers onto target edges."""
    centers = 0.5 * (source_edges[:-1] + source_edges[1:])
    rebinned, _ = np.histogram(centers, bins=target_edges, weights=counts)
    return rebinned


def _accumulate_topology_counts_for_reference(topology_rows, accumulators):
    """Accumulate trigger topology count rows."""
    for row in topology_rows:
        count_type = str(row["count_type"])
        key = str(row["key"])
        count = int(row["count"])
        if count_type == "trigger_multiplicity":
            accumulators["trigger_multiplicity"][int(key)] += count
        elif count_type == "trigger_combinations":
            accumulators["trigger_combinations"][key] += count
        elif count_type == "telescope_participation":
            accumulators["telescope_participation"][key] += count
        elif count_type == "subset_multiplicity":
            accumulators["subset_multiplicity"][str(row["subset"])][int(key)] += count


def _accumulate_subset_histograms_for_reference(subset_rows, accumulators):
    """Accumulate per-subset triggered quantity histograms."""
    for subset_name, subset_selected in table_handler.group_table_rows(
        subset_rows, "subset"
    ).items():
        for quantity, rows in table_handler.group_table_rows(subset_selected, "quantity").items():
            rows.sort("bin_index")
            counts = np.asarray(rows["count"], dtype=float)
            bin_edges = np.concatenate(
                [
                    np.asarray(rows["bin_low"][:1], dtype=float),
                    np.asarray(rows["bin_high"], dtype=float),
                ]
            )
            _add_histogram(
                accumulators["quantity_histograms"]["subset_triggered"][str(subset_name)],
                str(quantity),
                counts,
                bin_edges,
            )


def _counter_to_histogram(counter):
    """Convert integer-key count data to histogram counts and bin edges."""
    if not counter:
        return np.array([], dtype=float), np.array([], dtype=float)
    max_key = max(int(key) for key in counter)
    bin_edges = np.arange(1, max_key + 2)
    counts = np.array([counter.get(index, 0) for index in range(1, max_key + 1)], dtype=float)
    return counts, bin_edges


def _build_per_type_histogram_metrics(label, simulated_histograms, accumulators):
    """Build per-subset metrics from histogram-backed accumulators."""
    per_type = {}
    for subset_name, triggered_histograms in accumulators["quantity_histograms"][
        "subset_triggered"
    ].items():
        quantity_histograms = {
            quantity: {
                "simulated": simulated_histograms[quantity],
                "triggered": triggered_histograms[quantity],
            }
            for quantity in triggered_histograms
            if quantity in simulated_histograms
        }
        per_type[subset_name] = ProductionEventMetrics(
            label=label,
            simulated_energies=np.array([]),
            triggered_energies=np.array([]),
            simulated_core_distances=np.array([]),
            triggered_core_distances=np.array([]),
            simulated_angular_distances=np.array([]),
            triggered_angular_distances=np.array([]),
            trigger_multiplicity=np.array([], dtype=int),
            trigger_combinations=Counter(),
            telescope_participation=Counter(),
            simulated_event_count=accumulators["simulated_event_count"],
            triggered_event_count=int(
                sum(accumulators["subset_multiplicity"].get(subset_name, {}).values())
            ),
            quantity_histograms=quantity_histograms,
            trigger_multiplicity_histogram=_counter_to_histogram(
                accumulators["subset_multiplicity"].get(subset_name, Counter())
            ),
        )
    return per_type


def collect_signal_metrics(production_descriptors, array_layout_name):
    """Collect telescope-level signal metrics for each production.

    Parameters
    ----------
    production_descriptors : list[ProductionDescriptor]
        Production descriptors containing sim_telarray input files.
    array_layout_name : str or list[str]
        Single array layout to compare.

    Returns
    -------
    dict[str, list[ProductionSignalMetrics]]
        Metrics grouped by telescope name, with one entry per production.

    Raises
    ------
    ValueError
        If the layout selection is invalid, a required telescope is absent, or
        no event data is available for a telescope.
    """
    layout_name = _ensure_single_array_layout_name(array_layout_name)
    telescope_names = _discover_telescope_names(production_descriptors, layout_name)
    if not telescope_names:
        raise ValueError(f"Array layout '{layout_name}' contains no telescopes.")

    metrics_by_telescope = {name: [] for name in telescope_names}
    for production in production_descriptors:
        metrics = _collect_production_signal_metrics(production, telescope_names)
        for telescope_name, telescope_metrics in metrics.items():
            metrics_by_telescope[telescope_name].append(telescope_metrics)
    return metrics_by_telescope


def _discover_telescope_names(production_descriptors, layout_name):
    """Discover and validate the telescope set represented by all input files."""
    input_files = [
        input_file for production in production_descriptors for input_file in production.input_files
    ]
    if not input_files:
        raise ValueError(f"Array layout '{layout_name}' has no sim_telarray input files.")

    expected = None
    for input_file in input_files:
        mapping = get_sim_telarray_telescope_id_to_telescope_name_mapping(input_file)
        available = {str(name) for name in mapping.values()}
        if expected is None:
            expected = available
        elif available != expected:
            raise ValueError(
                f"Input '{input_file}' has telescope set {sorted(available)}; expected "
                f"the layout telescope set {sorted(expected)}."
            )
    return sorted(expected)


def _collect_production_signal_metrics(production, telescope_names):
    """Collect all telescope metrics for one production."""
    values = {
        name: {"pedestals": [], "signals": [], "peak_timing": [], "triggered_pixels": []}
        for name in telescope_names
    }
    for input_file in production.input_files:
        for telescope_name in telescope_names:
            _collect_file_metrics(input_file, telescope_name, values[telescope_name])

    result = {}
    for telescope_name, telescope_values in values.items():
        if not any(telescope_values.values()):
            raise ValueError(
                f"Production '{production.label}' has no event data for telescope "
                f"'{telescope_name}'."
            )
        result[telescope_name] = ProductionSignalMetrics(
            label=production.label,
            **{key: _concatenate_values(value) for key, value in telescope_values.items()},
        )
    return result


def _concatenate_values(values):
    """Concatenate per-event values, preserving an empty observable."""
    return np.concatenate(values) if values else np.array([])


def _collect_file_metrics(input_file, telescope_name, values):
    """Collect metrics from one file and telescope."""
    _event_ids, _telescope_description, events = read_events(
        input_file,
        telescope_name,
        event_ids=None,
        max_events=None,
    )
    if events is None:
        raise ValueError(f"Telescope '{telescope_name}' was not found in input '{input_file}'.")

    for event in events:
        try:
            samples, pedestals, signals = trace.get_trace_data(event["adc_samples"])
            peak_samples, _pixel_ids, _found_count = trace.trace_maxima(
                samples, sum_threshold=trace.DEFAULT_SUM_THRESHOLD
            )
            trigger_pixels = _get_triggered_pixel_count(event)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Input '{input_file}' has incomplete signal data for telescope '{telescope_name}'."
            ) from exc

        values["pedestals"].append(np.asarray(pedestals))
        values["signals"].append(np.asarray(signals))
        if peak_samples is not None:
            values["peak_timing"].append(np.asarray(peak_samples))
        values["triggered_pixels"].append(np.asarray([trigger_pixels]))


def _get_triggered_pixel_count(event):
    """Return the triggered pixel count, falling back to the selected pixel list."""
    pixel_lists = event["pixel_lists"]
    pixel_list = pixel_lists.get(0)
    if pixel_list is None:
        pixel_list = pixel_lists.get(1)
    if pixel_list is None:
        raise ValueError("Event contains no triggered or selected pixel list.")
    return pixel_list["pixels"]
