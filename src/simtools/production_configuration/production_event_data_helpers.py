"""Shared helpers for production workflows based on reduced event-data files."""

import re
from pathlib import Path

from simtools.layout.array_layout_utils import (
    get_array_elements_from_db_for_layouts,
    resolve_array_layout_name,
)
from simtools.sim_events.histograms import EventDataHistograms
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


def build_production_subdirectories(production_patterns, output_dir, is_multi_production):
    """
    Build and create per-production output subdirectories when needed.

    Parameters
    ----------
    production_patterns : list[str]
        Ordered production patterns processed by the workflow.
    output_dir : pathlib.Path
        Root output directory.
    is_multi_production : bool
        Whether the current run handles multiple independent productions.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of production pattern to output subdirectory.
    """
    if not is_multi_production:
        return {}

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
    skip_invalid_event_data_files=False,
    fill_efficiency_histogram=False,
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
        skip_invalid_event_data_files=skip_invalid_event_data_files,
        require_triggered_data=True,
    )
    accumulators = [
        EventDataHistograms.create_accumulator(
            array_name=config["array_name"],
            telescope_list=config["telescope_ids"],
            energy_bins_per_decade=energy_bins_per_decade,
            angular_distance_bin_count=angular_distance_bin_count,
        )
        for config in telescope_configs
    ]

    for reader, values in event_source.iter_event_data():
        file_info_table, shower_data, triggered_shower, triggered_data = values
        for config, histograms in zip(telescope_configs, accumulators):
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

    for histograms in accumulators:
        histograms.finalize(fill_efficiency_histogram=fill_efficiency_histogram)
    return list(zip(telescope_configs, accumulators))
