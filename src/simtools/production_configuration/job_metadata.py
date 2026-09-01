"""Build metadata for completed simulation-production jobs."""

from copy import deepcopy
from pathlib import Path

from astropy import units as u

from simtools.utils import names

CATALOG_SITE_NAMES = {"North": "LaPalma", "South": "Paranal"}
PRODUCTION_JOB_MANIFEST_VERSION = "1.0.0"

_FILE_TYPE_ALIASES = {
    "sim_telarray_output": "sim_telarray",
    "sim_telarray_event_data": "reduced_event_data",
    "sim_telarray_log": "sim_telarray_log",
    "sim_telarray_histogram": "sim_telarray_histogram",
    "corsika_output": "corsika",
    "corsika_log": "corsika_log",
}


def build_simulation_job_metadata(args_dict, simulator):
    """Build DIRAC catalog metadata from resolved simulation configuration.

    Parameters
    ----------
    args_dict : dict
        Resolved ``simulate_prod`` application arguments.
    simulator : simtools.simulator.Simulator
        Simulator for the completed run.

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
        "particle": args_dict["primary"].lower(),
        "phiP": round((azimuth_angle + 180.0) % 360.0, 2),
        "thetaP": round(float(args_dict["zenith_angle"].to_value(u.deg)), 2),
        "sct": str(_has_sct(simulator.array_models)),
        "view_cone": _format_view_cone(view_cone_min, view_cone_max),
        "runNumber": int(simulator.run_number),
        "model_version": str(args_dict["model_version"]),
    }
    _add_optional_coordinate(metadata, "dec", args_dict.get("dec"))
    _add_optional_coordinate(metadata, "ha", args_dict.get("ha"))
    return metadata


def build_production_job_manifest(args_dict, simulator, output_directory, file_inventory=None):
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

    Returns
    -------
    dict
        Versioned production-job manifest used for downstream file selection.
    """
    catalog_metadata = build_simulation_job_metadata(args_dict, simulator)
    return {
        "schema_name": "simulate_prod_job_metadata",
        "schema_version": PRODUCTION_JOB_MANIFEST_VERSION,
        "product_type": "simulate_prod_job",
        "production_id": args_dict.get("production_id") or args_dict.get("label"),
        "job_id": Path(output_directory).name,
        "status": "complete",
        "catalog_metadata": catalog_metadata,
        "configuration": _build_selection_configuration(args_dict, simulator),
        "files": file_inventory or _build_output_file_inventory(simulator, output_directory),
    }


def _build_selection_configuration(args_dict, simulator):
    """Return stable simulation configuration fields used for selection and grouping."""
    energy_min, energy_max = args_dict["energy_range"]
    view_cone_min, view_cone_max = args_dict["view_cone"]
    cores_per_shower, core_scatter_max = args_dict["core_scatter"]
    configuration = {
        "run_number": int(simulator.run_number),
        "primary": str(args_dict["primary"]).lower(),
        "site": args_dict["site"],
        "array_layout_name": args_dict["array_layout_name"],
        "model_version": str(args_dict["model_version"]),
        "simulation_software": args_dict["simulation_software"],
        "azimuth_angle": args_dict["azimuth_angle"],
        "zenith_angle": args_dict["zenith_angle"],
        "energy_min": energy_min,
        "energy_max": energy_max,
        "view_cone_min": view_cone_min,
        "view_cone_max": view_cone_max,
        "cores_per_shower": int(cores_per_shower),
        "core_scatter_max": core_scatter_max,
        "showers_per_run": args_dict.get("showers_per_run"),
        "eslope": args_dict.get("eslope"),
        "corsika_he_interaction": args_dict.get("corsika_he_interaction"),
        "corsika_le_interaction": args_dict.get("corsika_le_interaction"),
        "corsika_hadronic_transition_energy": args_dict.get("corsika_hadronic_transition_energy"),
        "model_parameter_overrides": _resolved_model_parameter_overrides(simulator),
        "atmosphere": _resolved_atmosphere_configuration(args_dict, simulator),
    }
    _add_optional_configuration_value(configuration, "dec", args_dict.get("dec"))
    _add_optional_configuration_value(configuration, "ha", args_dict.get("ha"))
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


def _build_output_file_inventory(simulator, output_directory):
    """Return output files grouped by manifest file type."""
    output_directory = Path(output_directory)
    inventory = {}
    for simulator_file_type, manifest_file_type in _FILE_TYPE_ALIASES.items():
        files = []
        for file_path in _ensure_list(simulator.get_files(file_type=simulator_file_type)):
            packaged_file = output_directory / Path(file_path).name
            if packaged_file.exists():
                files.append(packaged_file.relative_to(output_directory).as_posix())
        if files:
            inventory[manifest_file_type] = sorted(files)
    return inventory


def _ensure_list(value):
    """Return value as list, preserving empty values."""
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    return [value]


def _has_sct(array_models):
    """Return whether any resolved array model contains an SCT."""
    return any(
        names.get_array_element_type_from_name(element_name) == "SCTS"
        for array_model in array_models
        for element_name in array_model.array_elements
    )


def _format_view_cone(view_cone_min, view_cone_max):
    """Format view-cone bounds in the catalog convention."""
    return (
        f"{round(view_cone_min.to_value(u.deg), 2)}_deg_"
        f"{round(view_cone_max.to_value(u.deg), 2)}_deg"
    ).replace(" ", "_")


def _add_optional_coordinate(metadata, key, value):
    """Add one optional angular coordinate in degrees to metadata."""
    if value is not None:
        metadata[key] = round(float(value.to_value(u.deg)), 2)
