"""Shared helpers for production workflows based on reduced event-data files."""

from collections import Counter, defaultdict

import numpy as np

from simtools.layout.array_layout_utils import (
    get_array_elements_from_model_repository,
    resolve_array_layout_name,
)
from simtools.sim_events.histograms import EventDataHistograms
from simtools.utils import names
from simtools.utils.names import normalize_array_element_identifier_container


def resolve_telescope_configs(args_dict, context=None):
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
        model_version = args_dict.get("model_version")
        if isinstance(model_version, list):
            if len(model_version) != 1:
                raise ValueError(
                    "Telescope layout resolution requires exactly one model version, "
                    f"got {model_version}."
                )
            model_version = model_version[0]
        layouts = resolve_array_layout_name(args_dict["array_layout_name"], model_version)
        return get_array_elements_from_model_repository(
            layouts if isinstance(layouts, list) else [layouts],
            args_dict.get("site"),
            model_version,
            context=context,
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
    pairs = (
        telescope_configs.items()
        if isinstance(telescope_configs, dict)
        else [(config["array_name"], config["telescope_ids"]) for config in telescope_configs]
    )
    return [
        {
            "array_name": array_name,
            "telescope_ids": normalize_array_element_identifier_container(telescope_ids),
        }
        for array_name, telescope_ids in pairs
    ]


def accumulate_histograms_by_telescope_config(
    file_path,
    telescope_configs,
    *,
    energy_bins_per_decade,
    angular_distance_bin_count=100,
    angular_distance_bin_width=None,
    core_distance_bin_width=None,
    skip_invalid_event_data_files=False,
    fill_efficiency_histogram=False,
    collect_trigger_topology=False,
    minimum_triggered_telescopes=2,
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
    core_distance_bin_width : astropy.units.Quantity, optional
        Core-distance bin width used to derive a common physical grid across ranges.
    skip_invalid_event_data_files : bool, optional
        Skip malformed event-data files instead of aborting the run.
    fill_efficiency_histogram : bool, optional
        Whether to finalize with efficiency histograms enabled.
    collect_trigger_topology : bool, optional
        Whether to also collect trigger topology summaries.
    minimum_triggered_telescopes : int, optional
        Minimum number of triggered telescopes within each telescope configuration.

    Returns
    -------
    list[tuple]
        One finalized ``(config, histograms)`` pair per telescope configuration, with
        a trigger-topology accumulator appended when ``collect_trigger_topology`` is set.
    """
    histogram_options = {
        "energy_bins_per_decade": energy_bins_per_decade,
        "angular_distance_bin_count": angular_distance_bin_count,
        "angular_distance_bin_width": angular_distance_bin_width,
        "core_distance_bin_width": core_distance_bin_width,
    }
    event_source = EventDataHistograms(
        file_path,
        skip_invalid_event_data_files=skip_invalid_event_data_files,
        require_triggered_data=True,
        **histogram_options,
    )
    accumulators = [
        EventDataHistograms.create_accumulator(
            array_name=config["array_name"],
            telescope_list=config["telescope_ids"],
            **histogram_options,
        )
        for config in telescope_configs
    ]
    topology = (
        [_initialize_trigger_topology_accumulator() for _ in telescope_configs]
        if collect_trigger_topology
        else None
    )

    for reader, values in event_source.iter_event_data():
        file_info_table, shower_data, triggered_shower, triggered_data = values
        for index, (config, histograms) in enumerate(zip(telescope_configs, accumulators)):
            if config["telescope_ids"]:
                filtered_data, filtered_shower = reader.filter_by_telescopes(
                    triggered_data,
                    triggered_shower,
                    telescope_list=config["telescope_ids"],
                    minimum_triggered_telescopes=minimum_triggered_telescopes,
                )
            else:
                filtered_data, filtered_shower = triggered_data, triggered_shower
            histograms.accumulate(file_info_table, shower_data, filtered_shower, filtered_data)
            if collect_trigger_topology:
                _accumulate_trigger_topology(
                    filtered_shower,
                    filtered_data,
                    topology[index],
                    allowed_telescopes=config["telescope_ids"],
                )

    for histograms in accumulators:
        histograms.finalize(fill_efficiency_histogram=fill_efficiency_histogram)
    if collect_trigger_topology:
        return list(zip(telescope_configs, accumulators, topology))
    return list(zip(telescope_configs, accumulators))


def _initialize_trigger_topology_accumulator():
    """Return empty counters and quantity buffers for trigger topology summaries."""
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
    allowed = set(allowed_telescopes or [])
    energies = np.asarray(triggered_shower_data.simulated_energy)
    core_distances = np.asarray(triggered_shower_data.core_distance_shower)
    angular_distances = np.asarray(triggered_shower_data.angular_distance)
    for event_index, telescope_list in enumerate(triggered_data.telescope_list):
        telescopes = _normalize_trigger_telescopes(telescope_list, allowed)
        if not telescopes:
            continue
        accumulator["trigger_multiplicity"][len(telescopes)] += 1
        accumulator["trigger_combinations"][",".join(telescopes)] += 1
        for telescope in set(telescopes):
            accumulator["telescope_participation"][telescope] += 1
        for subset_name, multiplicity in _subset_counts_for_trigger(telescopes).items():
            accumulator["subset_multiplicity"][subset_name][multiplicity] += 1
            values = accumulator["subset_values"]
            values["energy"][subset_name].append(float(energies[event_index]))
            values["core_distance"][subset_name].append(float(core_distances[event_index]))
            values["angular_distance"][subset_name].append(float(angular_distances[event_index]))


def _subset_counts_for_trigger(telescopes):
    """Return per-type and special subset multiplicities for one trigger."""
    type_counts = Counter(names.get_array_element_type_from_name(tel) for tel in telescopes)
    subset_counts = {tel_type: count for tel_type, count in type_counts.items() if count >= 2}
    if len(telescopes) == 1:
        subset_counts["single_telescope"] = 1
    elif len(type_counts) > 1:
        subset_counts["mixed_type"] = len(telescopes)
    return subset_counts


def _normalize_trigger_telescopes(telescope_list, allowed_telescopes=None):
    """Return sorted valid telescope names, restricted to ``allowed_telescopes`` if given."""
    normalized = []
    for telescope in telescope_list:
        name = str(telescope)
        if allowed_telescopes and name not in allowed_telescopes:
            continue
        try:
            names.get_array_element_type_from_name(name)
        except ValueError:
            continue
        normalized.append(name)
    return tuple(sorted(normalized))
