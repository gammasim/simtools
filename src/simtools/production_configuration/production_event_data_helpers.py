"""Shared helpers for production workflows based on reduced event-data files."""

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from simtools.layout.array_layout_utils import (
    get_array_elements_from_db_for_layouts,
    resolve_array_layout_name,
)
from simtools.sim_events.histograms import EventDataHistograms
from simtools.utils import names
from simtools.utils.general import get_uuid
from simtools.utils.names import normalize_array_element_identifier_container


def normalize_event_data_file(event_data_file):
    """
    Normalize event_data_file to an ordered list of production patterns.

    Parameters
    ----------
    event_data_file : str or list
        A single pattern string or list of pattern strings.

    Returns
    -------
    list
        Ordered list of event data-file patterns.
    """
    if isinstance(event_data_file, str):
        return [event_data_file]
    if isinstance(event_data_file, list):
        return list(event_data_file)
    raise TypeError(f"event_data_file must be str or list, got {type(event_data_file)}")


def get_production_directory_name(production_pattern, existing_names=None):
    """
    Generate a readable, filesystem-safe production directory name.

    Parameters
    ----------
    production_pattern : str
        Glob pattern identifying one production input.
    existing_names : set[str] or None
        Existing directory names already assigned in this run.

    Returns
    -------
    str
        Safe directory name.
    """

    def _sanitize(name):
        name = re.sub(r"[^A-Za-z0-9]+", "_", name)
        return re.sub(r"_+", "_", name).strip("_")

    pattern_path = Path(production_pattern)
    parent_name = _sanitize(pattern_path.parent.name) if pattern_path.parent.name != "." else ""
    readable_name = parent_name or _sanitize(pattern_path.stem) or "production"
    base_name = f"production_{readable_name}"

    if existing_names is None or base_name not in existing_names:
        return base_name
    return f"{base_name}_{get_uuid()}"


def build_production_subdirectories(production_patterns, output_dir):
    """
    Build and create per-production output subdirectories when needed.

    Parameters
    ----------
    production_patterns : list[str]
        Ordered production patterns processed by the workflow.
    output_dir : pathlib.Path
        Root output directory.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of production pattern to output subdirectory.
    """
    production_subdirs = {}
    used_subdir_names = set()
    for production_pattern in production_patterns:
        subdir_name = get_production_directory_name(production_pattern, used_subdir_names)
        used_subdir_names.add(subdir_name)
        output_subdir = output_dir / subdir_name
        Path(output_subdir).mkdir(parents=True, exist_ok=True)
        production_subdirs[production_pattern] = output_subdir

    return production_subdirs


def resolve_telescope_configs(args_dict):
    """
    Resolve telescope configurations from supported input options.

    Parameters
    ----------
    args_dict : dict
        Application arguments containing layout or array-element selection.

    Returns
    -------
    dict or list[dict]
        Telescope configuration payload returned by the chosen selector.

    Raises
    ------
    ValueError
        If no supported telescope selector is provided.
    """
    if args_dict.get("array_layout_name"):
        layouts = resolve_array_layout_name(
            args_dict["array_layout_name"],
            args_dict.get("model_version"),
        )
        if not isinstance(layouts, list):
            layouts = [layouts]
        return get_array_elements_from_db_for_layouts(
            layouts,
            args_dict.get("site"),
            args_dict.get("model_version"),
        )
    if args_dict.get("array_element_list"):
        return {"array_element_list": args_dict["array_element_list"]}

    raise ValueError(
        "No telescope configuration provided. Use one of --array_layout_name "
        "or --array_element_list."
    )


def normalize_telescope_configs(telescope_configs):
    """
    Normalize telescope configurations to a list of array-name descriptors.

    Parameters
    ----------
    telescope_configs : dict or list[dict]
        Raw telescope configuration payload.

    Returns
    -------
    list[dict]
        List of dictionaries with ``array_name`` and normalized ``telescope_ids``.
    """
    if isinstance(telescope_configs, dict):
        return [
            {
                "array_name": array_name,
                "telescope_ids": normalize_array_element_identifier_container(telescope_ids_raw),
            }
            for array_name, telescope_ids_raw in telescope_configs.items()
        ]
    return [
        {
            "array_name": config["array_name"],
            "telescope_ids": normalize_array_element_identifier_container(config["telescope_ids"]),
        }
        for config in telescope_configs
    ]


def accumulate_histograms_by_telescope_config(
    file_path,
    telescope_configs,
    *,
    energy_bins_per_decade,
    angular_distance_bin_count=100,
    angular_distance_bin_width=None,
    skip_invalid_event_data_files=False,
    fill_efficiency_histogram=False,
    collect_trigger_topology=False,
):
    """
    Read one production once and accumulate histograms for all telescope configurations.

    Parameters
    ----------
    file_path : str
        Reduced event-data input path or glob pattern for one production.
    telescope_configs : list[dict]
        Telescope configurations with ``array_name`` and normalized ``telescope_ids``.
    energy_bins_per_decade : int
        Number of logarithmic energy bins per decade.
    angular_distance_bin_count : int, optional
        Number of angular-distance bins used by the accumulators.
    angular_distance_bin_width : astropy.units.Quantity, optional
        Angular-distance bin width used to derive bins from broad-range viewcone limits.
    skip_invalid_event_data_files : bool, optional
        Skip malformed event-data files instead of aborting the run.
    fill_efficiency_histogram : bool, optional
        Whether to finalize with efficiency histograms enabled.

    Returns
    -------
    list[tuple[dict, EventDataHistograms]]
        One finalized histogram accumulator per telescope configuration.
    """
    event_source = EventDataHistograms(
        file_path,
        energy_bins_per_decade=energy_bins_per_decade,
        angular_distance_bin_count=angular_distance_bin_count,
        angular_distance_bin_width=angular_distance_bin_width,
        skip_invalid_event_data_files=skip_invalid_event_data_files,
        require_triggered_data=True,
    )
    accumulators = [
        EventDataHistograms.create_accumulator(
            array_name=config["array_name"],
            telescope_list=config["telescope_ids"],
            energy_bins_per_decade=energy_bins_per_decade,
            angular_distance_bin_count=angular_distance_bin_count,
            angular_distance_bin_width=angular_distance_bin_width,
        )
        for config in telescope_configs
    ]
    topology_accumulators = (
        [_initialize_trigger_topology_accumulator() for _ in telescope_configs]
        if collect_trigger_topology
        else None
    )

    for reader, values in event_source.iter_event_data():
        file_info_table, shower_data, triggered_shower, triggered_data = values
        for config_index, (config, histograms) in enumerate(zip(telescope_configs, accumulators)):
            if config["telescope_ids"]:
                filtered_triggered_data, filtered_triggered_shower = reader.filter_by_telescopes(
                    triggered_data,
                    triggered_shower,
                    telescope_list=config["telescope_ids"],
                )
            else:
                filtered_triggered_data = triggered_data
                filtered_triggered_shower = triggered_shower
            histograms.accumulate(
                file_info_table,
                shower_data,
                filtered_triggered_shower,
                filtered_triggered_data,
            )
            if collect_trigger_topology:
                _accumulate_trigger_topology(
                    filtered_triggered_shower,
                    filtered_triggered_data,
                    topology_accumulators[config_index],
                    allowed_telescopes=config["telescope_ids"],
                )

    for histograms in accumulators:
        histograms.finalize(fill_efficiency_histogram=fill_efficiency_histogram)
    if collect_trigger_topology:
        return list(zip(telescope_configs, accumulators, topology_accumulators))
    return list(zip(telescope_configs, accumulators))


def _initialize_trigger_topology_accumulator():
    """Initialize counters and quantity buffers for trigger topology summaries."""
    return {
        "trigger_multiplicity": Counter(),
        "trigger_combinations": Counter(),
        "telescope_participation": Counter(),
        "subset_multiplicity": defaultdict(Counter),
        "subset_values": {
            "energy": defaultdict(list),
            "core_distance": defaultdict(list),
            "angular_distance": defaultdict(list),
        },
    }


def _accumulate_trigger_topology(
    triggered_shower_data, triggered_data, accumulator, allowed_telescopes=None
):
    """Accumulate trigger topology counters and per-subset triggered quantities."""
    if triggered_shower_data is None or triggered_data is None:
        return

    allowed_telescope_set = set(allowed_telescopes or [])
    energies = np.asarray(triggered_shower_data.simulated_energy)
    core_distances = np.asarray(triggered_shower_data.core_distance_shower)
    angular_distances = np.asarray(triggered_shower_data.angular_distance)
    for event_index, telescope_list in enumerate(triggered_data.telescope_list):
        telescopes = _normalize_trigger_telescopes(telescope_list, allowed_telescope_set)
        if not telescopes:
            continue
        multiplicity = len(telescopes)
        accumulator["trigger_multiplicity"][multiplicity] += 1
        accumulator["trigger_combinations"][",".join(telescopes)] += 1
        for telescope in set(telescopes):
            accumulator["telescope_participation"][telescope] += 1

        for subset_name, subset_multiplicity in _subset_counts_for_trigger(telescopes).items():
            accumulator["subset_multiplicity"][subset_name][subset_multiplicity] += 1
            accumulator["subset_values"]["energy"][subset_name].append(float(energies[event_index]))
            accumulator["subset_values"]["core_distance"][subset_name].append(
                float(core_distances[event_index])
            )
            accumulator["subset_values"]["angular_distance"][subset_name].append(
                float(angular_distances[event_index])
            )


def _subset_counts_for_trigger(telescopes):
    """Return per-type and special subset multiplicities for one trigger."""
    type_counts = Counter()
    for telescope in telescopes:
        tel_type = names.get_array_element_type_from_name(telescope)
        type_counts[tel_type] += 1

    subset_counts = {tel_type: count for tel_type, count in type_counts.items() if count >= 2}
    if len(telescopes) == 1:
        subset_counts["single_telescope"] = 1
    elif len(type_counts) > 1:
        subset_counts["mixed_type"] = len(telescopes)
    return subset_counts


def _normalize_trigger_telescopes(telescope_list, allowed_telescope_set=None):
    """Return sorted valid telescope names for topology summaries."""
    allowed_telescope_set = allowed_telescope_set or set()
    normalized = []
    for telescope in telescope_list:
        telescope_name = str(telescope)
        if allowed_telescope_set and telescope_name not in allowed_telescope_set:
            continue
        try:
            names.get_array_element_type_from_name(telescope_name)
        except ValueError:
            continue
        normalized.append(telescope_name)
    return tuple(sorted(normalized))
