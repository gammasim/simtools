"""Utilities for event-level comparison across multiple simulation productions."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from simtools.io import table_handler
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_EDGES_TABLE,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_HISTOGRAM_VALUES_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
)
from simtools.utils.general import resolve_file_patterns


@dataclass
class ProductionDescriptor:
    """Descriptor for one production input provided."""

    label: str
    trigger_histogram_files: list[str]


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
            raise ValueError(f"Production '{label}' has no trigger_histogram_file pattern.")

        resolved_files = [str(path) for path in resolve_file_patterns(patterns)]
        if len(resolved_files) == 0:
            raise ValueError(f"Production '{label}' does not resolve to any files.")
        descriptors.append(
            ProductionDescriptor(label=label, trigger_histogram_files=resolved_files)
        )

    return descriptors


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


def collect_production_metrics(production_descriptors):
    """Collect comparison metrics from trigger histogram files for each production.

    Parameters
    ----------
    production_descriptors : list[ProductionDescriptor]
        Input descriptor for each production.

    Returns
    -------
    list[ProductionEventMetrics]
        Aggregated metrics per production.
    """
    return [
        _collect_single_production_histogram_metrics(descriptor)
        for descriptor in production_descriptors
    ]


def _collect_single_production_histogram_metrics(production_descriptor):
    """Collect comparison metrics for one trigger-histogram production descriptor."""
    accumulators = _initialize_histogram_metric_accumulators()
    for trigger_histogram_file in production_descriptor.trigger_histogram_files:
        _collect_metrics_from_trigger_histogram_file(trigger_histogram_file, accumulators)

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


def _collect_metrics_from_trigger_histogram_file(trigger_histogram_file, accumulators):
    """Collect metrics from one trigger-histogram HDF5 file."""
    tables = table_handler.read_tables(
        trigger_histogram_file,
        [
            TRIGGER_HISTOGRAM_METADATA_TABLE,
            TRIGGER_HISTOGRAM_VALUES_TABLE,
            TRIGGER_HISTOGRAM_EDGES_TABLE,
            TRIGGER_TOPOLOGY_COUNTS_TABLE,
            TRIGGER_SUBSET_HISTOGRAMS_TABLE,
        ],
        file_type="HDF5",
    )
    metadata = tables[TRIGGER_HISTOGRAM_METADATA_TABLE]
    values = tables[TRIGGER_HISTOGRAM_VALUES_TABLE]
    edges = tables[TRIGGER_HISTOGRAM_EDGES_TABLE]
    topology_counts = tables[TRIGGER_TOPOLOGY_COUNTS_TABLE]
    subset_histograms = tables[TRIGGER_SUBSET_HISTOGRAMS_TABLE]
    value_rows_by_reference = table_handler.group_table_rows(values, "reference_id")
    edge_rows_by_reference = table_handler.group_table_rows(edges, "reference_id")
    topology_rows_by_reference = table_handler.group_table_rows(topology_counts, "reference_id")
    subset_rows_by_reference = table_handler.group_table_rows(subset_histograms, "reference_id")

    for row in metadata:
        reference_id = row["reference_id"]
        _accumulate_quantity_histograms_for_reference(
            value_rows_by_reference.get(reference_id, values[:0]),
            edge_rows_by_reference.get(reference_id, edges[:0]),
            accumulators,
        )
        _accumulate_topology_counts_for_reference(
            topology_rows_by_reference.get(reference_id, topology_counts[:0]),
            accumulators,
        )
        _accumulate_subset_histograms_for_reference(
            subset_rows_by_reference.get(reference_id, subset_histograms[:0]),
            accumulators,
        )
        accumulators["simulated_event_count"] += int(row["total_simulated_events"])
        accumulators["triggered_event_count"] += int(row["total_triggered_events"])


def _accumulate_quantity_histograms_for_reference(value_rows, edge_rows, accumulators):
    """Accumulate base simulated and triggered histograms for comparable quantities."""
    histogram_map = {
        "energy_mc": ("energy", "simulated"),
        "energy": ("energy", "triggered"),
        "core_distance_mc": ("core_distance", "simulated"),
        "core_distance": ("core_distance", "triggered"),
        "angular_distance_mc": ("angular_distance", "simulated"),
        "angular_distance": ("angular_distance", "triggered"),
    }
    for histogram_name, (quantity, event_kind) in histogram_map.items():
        counts, bin_edges = _histogram_counts_and_edges(value_rows, edge_rows, histogram_name)
        if counts is None:
            continue
        _add_histogram(
            accumulators["quantity_histograms"][event_kind],
            quantity,
            counts,
            bin_edges,
        )


def _histogram_counts_and_edges(value_rows, edge_rows, histogram_name):
    """Return 1D histogram counts and bin edges for one persisted histogram."""
    selected_values = value_rows[value_rows["histogram_name"] == histogram_name]
    if len(selected_values) == 0:
        return None, None
    selected_values.sort("index_0")
    selected_edges = edge_rows[edge_rows["histogram_name"] == histogram_name]
    selected_edges.sort("bin_index")
    return (
        np.asarray(selected_values["value"], dtype=float),
        np.asarray(selected_edges["edge"], dtype=float),
    )


def _add_histogram(target, quantity, counts, bin_edges):
    """Add histogram counts into a target mapping, requiring consistent bin edges."""
    if quantity not in target:
        target[quantity] = (np.asarray(counts, dtype=float), np.asarray(bin_edges, dtype=float))
        return
    existing_counts, existing_edges = target[quantity]
    if not np.array_equal(existing_edges, bin_edges):
        raise ValueError(f"Inconsistent bin edges for quantity '{quantity}'.")
    target[quantity] = (existing_counts + counts, existing_edges)


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
        return ()
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
