"""Utilities for event-level comparison across multiple simulation productions."""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from simtools.sim_events.reader import EventDataReader
from simtools.utils import names
from simtools.utils.general import resolve_file_patterns


@dataclass
class ProductionDescriptor:
    """Descriptor for one production input provided by the user."""

    label: str
    event_data_files: list[str]


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
    if len(parsed_productions) == 0:
        raise ValueError("At least one production is required.")

    labels = [item[0] for item in parsed_productions]
    if len(set(labels)) != len(labels):
        raise ValueError("Production labels must be unique.")

    descriptors = []
    for label, pattern_list in parsed_productions:
        patterns = [pattern.strip() for pattern in pattern_list.split(",") if pattern.strip()]
        if len(patterns) == 0:
            raise ValueError(f"Production '{label}' has no event_data_file pattern.")

        resolved_files = [str(path) for path in resolve_file_patterns(patterns)]
        if len(resolved_files) == 0:
            raise ValueError(f"Production '{label}' does not resolve to any files.")

        descriptors.append(ProductionDescriptor(label=label, event_data_files=resolved_files))

    return descriptors


def _normalize_production_arguments(production_arguments):
    """Normalize raw production arguments into ``[(label, files), ...]``."""
    if not production_arguments:
        return []

    if all(isinstance(item, str) for item in production_arguments):
        if len(production_arguments) % 2 != 0:
            raise ValueError("Production arguments must be provided as label/file pairs.")
        return [
            (production_arguments[index], production_arguments[index + 1])
            for index in range(0, len(production_arguments), 2)
        ]

    normalized = []
    for item in production_arguments:
        if isinstance(item, list | tuple):
            if len(item) == 2 and all(isinstance(value, str) for value in item):
                normalized.append((item[0], item[1]))
                continue

            if all(isinstance(value, str) for value in item):
                normalized.extend(_normalize_production_arguments(list(item)))
                continue

        raise ValueError("Production arguments must be provided as label/file pairs.")

    return normalized


def collect_production_metrics(production_descriptors, telescope_list=None):
    """Collect event-level comparison metrics for each production.

    Parameters
    ----------
    production_descriptors : list[ProductionDescriptor]
        Input descriptor for each production.
    telescope_list : list[str], optional
        Telescope IDs to filter triggered events.

    Returns
    -------
    list[ProductionEventMetrics]
        Aggregated metrics per production.
    """
    return [
        _collect_single_production_metrics(descriptor, telescope_list=telescope_list)
        for descriptor in production_descriptors
    ]


def _collect_single_production_metrics(production_descriptor, telescope_list=None):
    """Collect event-level metrics for one production descriptor."""
    logger = logging.getLogger(__name__)

    metric_accumulators = _initialize_metric_accumulators()
    for event_data_file in production_descriptor.event_data_files:
        _collect_metrics_from_event_data_file(event_data_file, telescope_list, metric_accumulators)

    logger.info(
        f"Collected production {production_descriptor.label}: "
        f"simulated={metric_accumulators['simulated_event_count']}, "
        f"triggered={metric_accumulators['triggered_event_count']}"
    )

    all_simulated_energies = _safe_concat(metric_accumulators["simulated_energies"])
    all_simulated_core_distances = _safe_concat(metric_accumulators["simulated_core_distances"])
    all_simulated_angular_distances = _safe_concat(
        metric_accumulators["simulated_angular_distances"]
    )
    per_type = _build_per_type_metrics(
        production_descriptor.label,
        all_simulated_energies,
        all_simulated_core_distances,
        all_simulated_angular_distances,
        metric_accumulators,
    )

    return ProductionEventMetrics(
        label=production_descriptor.label,
        simulated_energies=all_simulated_energies,
        triggered_energies=_safe_concat(metric_accumulators["triggered_energies"]),
        simulated_core_distances=all_simulated_core_distances,
        triggered_core_distances=_safe_concat(metric_accumulators["triggered_core_distances"]),
        simulated_angular_distances=all_simulated_angular_distances,
        triggered_angular_distances=_safe_concat(
            metric_accumulators["triggered_angular_distances"]
        ),
        trigger_multiplicity=_safe_concat(metric_accumulators["trigger_multiplicity"], dtype=int),
        trigger_combinations=metric_accumulators["trigger_combinations"],
        telescope_participation=metric_accumulators["telescope_participation"],
        simulated_event_count=metric_accumulators["simulated_event_count"],
        triggered_event_count=metric_accumulators["triggered_event_count"],
        per_type=per_type,
    )


def _initialize_metric_accumulators():
    """Initialize mutable accumulators used while scanning event files."""
    return {
        "simulated_energies": [],
        "triggered_energies": [],
        "simulated_core_distances": [],
        "triggered_core_distances": [],
        "simulated_angular_distances": [],
        "triggered_angular_distances": [],
        "trigger_multiplicity": [],
        "trigger_combinations": Counter(),
        "telescope_participation": Counter(),
        "simulated_event_count": 0,
        "triggered_event_count": 0,
        "subset_accumulators": {
            "energies": defaultdict(list),
            "core_distances": defaultdict(list),
            "angular_distances": defaultdict(list),
            "multiplicities": defaultdict(list),
            "counts": defaultdict(int),
        },
    }


def _collect_metrics_from_event_data_file(event_data_file, telescope_list, metric_accumulators):
    """Collect metrics from all data sets in one event data file."""
    reader = EventDataReader(event_data_file, telescope_list=telescope_list)
    for data_set in reader.data_sets:
        _, shower_data, triggered_shower_data, triggered_data = reader.read_event_data(
            event_data_file,
            table_name_map=data_set,
        )
        if shower_data is None:
            continue

        _accumulate_simulated_events(shower_data, metric_accumulators)
        if triggered_data is None or triggered_shower_data is None:
            continue
        _accumulate_triggered_events(triggered_shower_data, triggered_data, metric_accumulators)


def _accumulate_simulated_events(shower_data, metric_accumulators):
    """Accumulate simulated event quantities."""
    metric_accumulators["simulated_energies"].append(np.asarray(shower_data.simulated_energy))
    metric_accumulators["simulated_core_distances"].append(
        np.asarray(shower_data.core_distance_shower)
    )
    metric_accumulators["simulated_angular_distances"].append(
        np.asarray(shower_data.angular_distance)
    )
    metric_accumulators["simulated_event_count"] += len(shower_data.simulated_energy)


def _accumulate_triggered_events(triggered_shower_data, triggered_data, metric_accumulators):
    """Accumulate triggered event quantities and subset counters."""
    trig_energies_arr = np.asarray(triggered_shower_data.simulated_energy)
    trig_core_dist_arr = np.asarray(triggered_shower_data.core_distance_shower)
    trig_angular_dist_arr = np.asarray(triggered_shower_data.angular_distance)
    metric_accumulators["triggered_energies"].append(trig_energies_arr)
    metric_accumulators["triggered_core_distances"].append(trig_core_dist_arr)
    metric_accumulators["triggered_angular_distances"].append(trig_angular_dist_arr)

    multiplicity = np.array([len(tel_list) for tel_list in triggered_data.telescope_list])
    metric_accumulators["trigger_multiplicity"].append(multiplicity)
    metric_accumulators["triggered_event_count"] += len(multiplicity)

    for event_idx, tel_list in enumerate(triggered_data.telescope_list):
        telescopes = tuple(sorted(str(telescope) for telescope in tel_list))
        combination_key = ",".join(telescopes)
        metric_accumulators["trigger_combinations"][combination_key] += 1
        for telescope in set(telescopes):
            metric_accumulators["telescope_participation"][telescope] += 1
        _accumulate_per_subset(
            telescopes,
            float(trig_energies_arr[event_idx]),
            float(trig_core_dist_arr[event_idx]),
            float(trig_angular_dist_arr[event_idx]),
            metric_accumulators["subset_accumulators"],
        )


def _build_per_type_metrics(
    label,
    all_simulated_energies,
    all_simulated_core_distances,
    all_simulated_angular_distances,
    metric_accumulators,
):
    """Build per-subset metrics from collected accumulators."""
    subset_accumulators = metric_accumulators["subset_accumulators"]
    return {
        key: ProductionEventMetrics(
            label=label,
            simulated_energies=all_simulated_energies,
            triggered_energies=np.array(subset_accumulators["energies"][key]),
            simulated_core_distances=all_simulated_core_distances,
            triggered_core_distances=np.array(subset_accumulators["core_distances"][key]),
            simulated_angular_distances=all_simulated_angular_distances,
            triggered_angular_distances=np.array(subset_accumulators["angular_distances"][key]),
            trigger_multiplicity=np.array(subset_accumulators["multiplicities"][key], dtype=int),
            trigger_combinations=Counter(),
            telescope_participation=Counter(),
            simulated_event_count=metric_accumulators["simulated_event_count"],
            triggered_event_count=subset_accumulators["counts"][key],
        )
        for key in subset_accumulators["energies"]
    }


def _accumulate_per_subset(telescopes, energy, core_dist, angular_dist, accumulators):
    """Accumulate event data per telescope-type subset and trigger topology.

    Parameters
    ----------
    telescopes : tuple[str]
        Sorted telescope identifiers that triggered for this event.
    energy : float
        Primary energy of the simulated shower.
    core_dist : float
        Core distance of the simulated shower.
    angular_dist : float
        Angular distance of the shower from the pointing direction.
    accumulators : dict
        Mapping of quantity lists keyed by subset name.
    """
    type_counts = Counter()
    valid_types = set()
    for telescope in telescopes:
        try:
            tel_type = names.get_array_element_type_from_name(telescope)
            type_counts[tel_type] += 1
            valid_types.add(tel_type)
        except ValueError:
            pass
    for tel_type, count in type_counts.items():
        _accumulate_event_for_key(tel_type, count, energy, core_dist, angular_dist, accumulators)
    if len(telescopes) == 1:
        _accumulate_event_for_key(
            "single_telescope", 1, energy, core_dist, angular_dist, accumulators
        )
    elif len(valid_types) > 1:
        _accumulate_event_for_key(
            "mixed_type", len(telescopes), energy, core_dist, angular_dist, accumulators
        )


def _accumulate_event_for_key(key, count, energy, core_dist, angular_dist, accumulators):
    """Accumulate one event's data into a named subset key.

    Parameters
    ----------
    key : str
        Subset key (telescope type, ``"single_telescope"``, or ``"mixed_type"``).
    count : int
        Number of telescopes (of this type, or total) that triggered.
    energy : float
        Primary energy of the shower.
    core_dist : float
        Core distance of the shower.
    angular_dist : float
        Angular distance of the shower.
    accumulators : dict
        Mapping of quantity lists to append to.
    """
    accumulators["energies"][key].append(energy)
    accumulators["core_distances"][key].append(core_dist)
    accumulators["angular_distances"][key].append(angular_dist)
    accumulators["multiplicities"][key].append(count)
    accumulators["counts"][key] += 1


def _safe_concat(arrays, dtype=float):
    """Concatenate list of arrays, returning an empty array for empty input."""
    if len(arrays) == 0:
        return np.array([], dtype=dtype)
    return np.concatenate(arrays)
