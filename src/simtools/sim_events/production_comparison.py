"""Utilities for event-level comparison across multiple simulation productions."""

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from simtools.sim_events.reader import EventDataReader
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
        If configuration is malformed or does not contain at least two productions.
    """
    parsed_productions = _normalize_production_arguments(production_arguments)
    if len(parsed_productions) < 2:
        raise ValueError("At least two productions are required.")

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

    simulated_energies = []
    triggered_energies = []
    simulated_core_distances = []
    triggered_core_distances = []
    trigger_multiplicity = []
    trigger_combinations = Counter()
    telescope_participation = Counter()
    simulated_event_count = 0
    triggered_event_count = 0

    for event_data_file in production_descriptor.event_data_files:
        reader = EventDataReader(event_data_file, telescope_list=telescope_list)
        for data_set in reader.data_sets:
            _, shower_data, triggered_shower_data, triggered_data = reader.read_event_data(
                event_data_file,
                table_name_map=data_set,
            )

            if shower_data is None:
                continue

            simulated_energies.append(np.asarray(shower_data.simulated_energy))
            simulated_core_distances.append(np.asarray(shower_data.core_distance_shower))
            simulated_event_count += len(shower_data.simulated_energy)

            if triggered_data is None or triggered_shower_data is None:
                continue

            triggered_energies.append(np.asarray(triggered_shower_data.simulated_energy))
            triggered_core_distances.append(np.asarray(triggered_shower_data.core_distance_shower))

            multiplicity = np.array([len(tel_list) for tel_list in triggered_data.telescope_list])
            trigger_multiplicity.append(multiplicity)
            triggered_event_count += len(multiplicity)

            for tel_list in triggered_data.telescope_list:
                telescopes = tuple(sorted(str(telescope) for telescope in tel_list))
                combination_key = ",".join(telescopes)
                trigger_combinations[combination_key] += 1
                for telescope in set(telescopes):
                    telescope_participation[telescope] += 1

    logger.info(
        f"Collected production {production_descriptor.label}: "
        f"simulated={simulated_event_count}, triggered={triggered_event_count}"
    )

    return ProductionEventMetrics(
        label=production_descriptor.label,
        simulated_energies=_safe_concat(simulated_energies),
        triggered_energies=_safe_concat(triggered_energies),
        simulated_core_distances=_safe_concat(simulated_core_distances),
        triggered_core_distances=_safe_concat(triggered_core_distances),
        trigger_multiplicity=_safe_concat(trigger_multiplicity, dtype=int),
        trigger_combinations=trigger_combinations,
        telescope_participation=telescope_participation,
        simulated_event_count=simulated_event_count,
        triggered_event_count=triggered_event_count,
    )


def _safe_concat(arrays, dtype=float):
    """Concatenate list of arrays, returning an empty array for empty input."""
    if len(arrays) == 0:
        return np.array([], dtype=dtype)
    return np.concatenate(arrays)
