"""Build metadata for completed simulation-production jobs."""

import gzip
import re
from copy import deepcopy
from pathlib import Path

import h5py
from astropy import units as u

from simtools.production_configuration.production_file_selection import (
    _resolve_relative_manifest_path,
    get_manifest_schema_metadata,
    inventory_production_files,
)
from simtools.utils import names
from simtools.utils.geometry import geographic_to_corsika_azimuth

CATALOG_SITE_NAMES = {"North": "LaPalma", "South": "Paranal"}
REQUIRED_SIMULATION_JOB_METADATA_ARGUMENTS = (
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "energy_range",
    "core_scatter",
    "view_cone",
    "showers_per_run",
    "model_version",
    "array_layout_name",
    "site",
    "simulation_software",
)

# Possessive quantifiers keep malformed log lines from triggering backtracking.
_SIMTEL_EVENT_COUNT_PATTERN = re.compile(
    r"\d++/\d++/\d++/\d++\s++tel\.,\s*+"
    r"(?P<simulated>\d++)/(?P<triggered>\d++)\s++events\s*+$"
)


def build_simulation_job_metadata(args_dict, simulator, include_sct=True):
    """Build DIRAC catalog metadata from resolved simulation configuration.

    Parameters
    ----------
    args_dict : dict
        Resolved ``simulate_prod`` application arguments.
    simulator : simtools.simulator.Simulator
        Simulator for the completed run.
    include_sct : bool, optional
        Include the SCT-presence catalog field when the resolved array elements are available.

    Returns
    -------
    dict
        Job-level metadata using DIRAC file-catalog field names.
    """
    azimuth_angle = args_dict["azimuth_angle"].to_value(u.deg)
    view_cone_min, view_cone_max = args_dict["view_cone"]
    metadata = {
        "array_layout": args_dict["array_layout_name"],
        "site": CATALOG_SITE_NAMES[args_dict["site"]],
        "particle": _primary_name(args_dict, simulator),
        "phiP": round(geographic_to_corsika_azimuth(azimuth_angle), 2),
        "thetaP": round(float(args_dict["zenith_angle"].to_value(u.deg)), 2),
        "view_cone_min": round(float(view_cone_min.to_value(u.deg)), 2),
        "view_cone_max": round(float(view_cone_max.to_value(u.deg)), 2),
        "runNumber": int(simulator.run_number),
        "model_version": _scalar_or_list(args_dict["model_version"]),
    }
    if include_sct:
        metadata["sct"] = str(_has_sct(simulator.array_models))
    _add_optional_coordinate(metadata, "dec", args_dict.get("dec"))
    _add_optional_coordinate(metadata, "ha", args_dict.get("ha"))
    return metadata


def build_production_job_manifest(
    args_dict,
    simulator,
    output_directory,
    file_inventory=None,
    catalog_metadata=None,
    atmosphere_configuration=None,
):
    """Build a versioned production-job manifest from resolved simulation output.

    Parameters
    ----------
    args_dict : dict
        Resolved ``simulate_prod`` application arguments.
    simulator : simtools.simulator.Simulator
        Simulator for the completed and validated run.
    output_directory : str or pathlib.Path
        Directory containing the packaged output files.
    file_inventory : dict, optional
        Precomputed manifest file inventory. Used when backfilling existing jobs.
    catalog_metadata : dict, optional
        Catalog metadata to store instead of deriving it from the simulator.
    atmosphere_configuration : dict, optional
        Resolved atmosphere configuration to store instead of deriving it from the simulator.

    Returns
    -------
    dict
        Versioned production-job manifest used for downstream file selection.
    """
    if catalog_metadata is None:
        catalog_metadata = build_simulation_job_metadata(args_dict, simulator)
    manifest_schema = get_manifest_schema_metadata("simulate_prod_job")
    manifest = {
        **manifest_schema,
        "product_type": "simulate_prod_job",
        "production_id": args_dict.get("production_id") or args_dict.get("label"),
        "job_id": Path(output_directory).name,
        "status": "complete",
        "catalog_metadata": catalog_metadata,
        "configuration": _build_selection_configuration(
            args_dict,
            simulator,
            atmosphere_configuration=atmosphere_configuration,
        ),
        "files": (
            file_inventory
            if file_inventory is not None
            else inventory_production_files(output_directory)
        ),
    }
    event_counts = get_sim_telarray_event_counts(
        output_directory,
        manifest["files"],
    )
    if event_counts is not None:
        manifest["statistics"] = event_counts
    return manifest


def get_sim_telarray_event_counts(output_directory, file_inventory):
    """Return sim_telarray event counts without reading simtel output files.

    Parameters
    ----------
    output_directory : str or pathlib.Path
        Directory containing the production outputs.
    file_inventory : dict
        Manifest file inventory with relative paths grouped by file type.

    Returns
    -------
    dict or None
        Mapping with ``simulated_events`` and ``triggered_events`` when counts
        can be recovered from reduced-event HDF5 files or sim_telarray logs.
    """
    output_directory = Path(output_directory)
    reduced_event_counts = _reduced_event_counts(output_directory, file_inventory)
    if reduced_event_counts is not None:
        return reduced_event_counts
    return _sim_telarray_log_event_counts(output_directory, file_inventory)


def _reduced_event_counts(output_directory, file_inventory):
    """Read event counts from reduced-event HDF5 tables when available."""
    paths = file_inventory.get("reduced_event_data", [])
    if not paths:
        return None
    simulated_events = 0
    triggered_events = 0
    try:
        for relative_path in paths:
            file_path = _resolve_relative_manifest_path(output_directory, relative_path)
            with h5py.File(file_path, "r") as data_file:
                simulated_events += len(data_file["SHOWERS"])
                trigger_table = data_file.get("TRIGGERS")
                if trigger_table is not None:
                    triggered_events += len(trigger_table)
    except KeyError, OSError, TypeError, ValueError:
        return None
    return {
        "simulated_events": simulated_events,
        "triggered_events": triggered_events,
    }


def _sim_telarray_log_event_counts(output_directory, file_inventory):
    """Extract event counts from sim_telarray log footers as a fallback."""
    paths = file_inventory.get("sim_telarray_log", [])
    counts = []
    for relative_path in paths:
        try:
            file_path = _resolve_relative_manifest_path(output_directory, relative_path)
            count = _read_log_event_count(file_path)
        except EOFError, OSError, UnicodeError, ValueError:
            continue
        if count is not None:
            counts.append(count)
    if not counts:
        return None
    return {
        "simulated_events": sum(simulated for simulated, _ in counts),
        "triggered_events": sum(triggered for _, triggered in counts),
    }


def _read_log_event_count(path):
    """Return the last event-count footer in a plain or gzip-compressed log."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as log_file:
            return _find_event_count(log_file)
    with path.open(encoding="utf-8", errors="replace") as log_file:
        return _find_event_count(log_file)


def _find_event_count(lines):
    """Return the last sim_telarray event count found in an iterable of log lines."""
    count = None
    for line in lines:
        match = _SIMTEL_EVENT_COUNT_PATTERN.search(line.rstrip())
        if match:
            count = (int(match["simulated"]), int(match["triggered"]))
    return count


def _build_selection_configuration(args_dict, simulator, atmosphere_configuration=None):
    """Return stable simulation configuration fields used for selection and grouping."""
    energy_min, energy_max = args_dict["energy_range"]
    view_cone_min, view_cone_max = args_dict["view_cone"]
    cores_per_shower, core_scatter_max = args_dict["core_scatter"]
    configuration = {
        "run_number": int(simulator.run_number),
        "primary": _primary_name(args_dict, simulator),
        "site": args_dict["site"],
        "array_layout_name": args_dict["array_layout_name"],
        "model_version": _scalar_or_list(args_dict["model_version"]),
        "simulation_software": args_dict["simulation_software"],
        "azimuth_angle": args_dict["azimuth_angle"],
        "zenith_angle": args_dict["zenith_angle"],
        "energy_min": energy_min,
        "energy_max": energy_max,
        "view_cone_min": view_cone_min,
        "view_cone_max": view_cone_max,
        "cores_per_shower": int(cores_per_shower),
        "core_scatter_max": core_scatter_max,
        "showers_per_run": _showers_per_run(args_dict, simulator),
        "eslope": args_dict.get("eslope"),
        "corsika_he_interaction": args_dict.get("corsika_he_interaction"),
        "corsika_le_interaction": args_dict.get("corsika_le_interaction"),
        "corsika_hadronic_transition_energy": args_dict.get("corsika_hadronic_transition_energy"),
        "model_parameter_overrides": _resolved_model_parameter_overrides(simulator),
        "atmosphere": atmosphere_configuration
        if atmosphere_configuration is not None
        else _resolved_atmosphere_configuration(args_dict, simulator),
    }
    _add_optional_configuration_value(configuration, "dec", args_dict.get("dec"))
    _add_optional_configuration_value(configuration, "ha", args_dict.get("ha"))
    for key in (
        "corsika_seeds",
        "event_number_first_shower",
        "correct_for_b_field_alignment",
        "sim_telarray_instrument_seed",
        "sim_telarray_random_instrument_instances",
        "sim_telarray_seed",
        "sim_telarray_seed_file",
    ):
        _add_optional_configuration_value(configuration, key, args_dict.get(key))
    return {key: value for key, value in configuration.items() if value is not None}


def _resolved_model_parameter_overrides(simulator):
    """Return resolved model-parameter overrides without source file paths."""
    overrides_by_version = {
        str(model.model_version): deepcopy(model.overwrite_model_parameter_dict)
        for model in simulator.array_models
        if getattr(model, "overwrite_model_parameter_dict", None)
    }
    if not overrides_by_version:
        return {}
    if len(overrides_by_version) == 1:
        return next(iter(overrides_by_version.values()))
    return overrides_by_version


def _resolved_atmosphere_configuration(args_dict, simulator):
    """Return resolved atmosphere settings used by the simulation."""
    atmosphere = {}
    threshold = args_dict.get("curved_atmosphere_min_zenith_angle")
    if threshold is not None:
        atmosphere["curved_atmosphere_min_zenith_angle"] = threshold

    corsika_configurations = getattr(simulator, "corsika_configurations", [])
    if not isinstance(corsika_configurations, list):
        corsika_configurations = [corsika_configurations]
    curved_values = {
        bool(configuration.use_curved_atmosphere)
        for configuration in corsika_configurations
        if configuration is not None
    }
    if len(curved_values) == 1:
        atmosphere["use_curved_atmosphere"] = curved_values.pop()

    site_parameters = {}
    for model in simulator.array_models:
        site_model = getattr(model, "site_model", None)
        parameters = getattr(site_model, "parameters", {})
        for name in (
            "atmospheric_profile",
            "atmospheric_transmission",
            "reference_point_altitude",
        ):
            if name in parameters:
                site_parameters[name] = deepcopy(parameters[name].get("value"))
    if site_parameters:
        atmosphere["site_parameters"] = site_parameters
    return atmosphere


def _add_optional_configuration_value(configuration, key, value):
    """Add an optional resolved configuration value."""
    if value is not None:
        configuration[key] = value


def _primary_name(args_dict, simulator):
    """Return the resolved primary name, including sim_telarray-only inputs."""
    if args_dict.get("primary") is not None:
        return str(args_dict["primary"]).lower()

    corsika_configurations = getattr(simulator, "corsika_configurations", [])
    if not isinstance(corsika_configurations, list):
        corsika_configurations = [corsika_configurations]
    for configuration in corsika_configurations:
        primary_particle = getattr(configuration, "primary_particle", None)
        primary_name = getattr(primary_particle, "name", None)
        if primary_name:
            return str(primary_name).lower()

    raise ValueError("Unable to determine the primary particle for production metadata.")


def _scalar_or_list(value):
    """Return model-version values without converting lists to pseudo-scalars."""
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return str(value)


def _showers_per_run(args_dict, simulator):
    """Return the configured shower count, including for sim_telarray-only inputs."""
    if args_dict.get("showers_per_run") is not None:
        return args_dict["showers_per_run"]

    corsika_configurations = getattr(simulator, "corsika_configurations", [])
    if not isinstance(corsika_configurations, list):
        corsika_configurations = [corsika_configurations]
    shower_counts = {
        configuration.shower_events
        for configuration in corsika_configurations
        if configuration is not None and getattr(configuration, "shower_events", None) is not None
    }
    if len(shower_counts) == 1:
        return shower_counts.pop()

    raise ValueError("Unable to determine the number of showers for production metadata.")


def _has_sct(array_models):
    """Return whether any resolved array model contains an SCT."""
    return any(
        names.get_array_element_type_from_name(element_name) == "SCTS"
        for array_model in array_models
        for element_name in array_model.array_elements
    )


def _add_optional_coordinate(metadata, key, value):
    """Add one optional angular coordinate in degrees to metadata."""
    if value is not None:
        metadata[key] = round(float(value.to_value(u.deg)), 2)
