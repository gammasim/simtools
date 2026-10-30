"""Write and validate metadata manifests for completed simulation production jobs."""

import gzip
import re
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
from astropy import units as u

from simtools.io.ascii_handler import write_data_to_file
from simtools.model.model_utils import read_overwrite_model_parameter_dict
from simtools.production_configuration.jobs_io import (
    job_grid_row_to_simulate_prod_args,
    read_job_grid,
)
from simtools.production_configuration.production_file_selection import (
    SIMULATE_PROD_JOB_METADATA,
    ProductionManifest,
    check_manifest,
    get_manifest_schema_metadata,
    inventory_production_files,
    resolve_relative_manifest_path,
    validate_required_production_outputs,
)
from simtools.utils import names
from simtools.utils.geometry import geographic_to_corsika_azimuth

CATALOG_SITE_NAMES = {"North": "LaPalma", "South": "Paranal"}
USE_CURVED_ATMOSPHERE = "use_curved_atmosphere"
_AZIMUTH_ANGLE = "azimuth_angle"
_ZENITH_ANGLE = "zenith_angle"
REQUIRED_SIMULATION_JOB_METADATA_ARGUMENTS = tuple(
    "primary azimuth_angle zenith_angle energy_range core_scatter view_cone showers_per_run "
    "model_version array_layout_name site simulation_software".split()
)
# Possessive quantifiers keep malformed log lines from triggering backtracking.
_SIMTEL_EVENT_COUNT_PATTERN = re.compile(
    r"\d++/\d++/\d++/\d++\s++tel\.,\s*+"
    r"(?P<simulated>\d++)/(?P<triggered>\d++)\s++events\s*+$"
)


def write_production_metadata(args_dict, context=None):
    """Write or validate production job metadata manifests.

    Parameters
    ----------
    args_dict : dict
        Application arguments containing ``production_path`` and, in write mode, ``job_grid_file``.
    context : RunContext, optional
        Run context for model access.
    """
    production_path = Path(args_dict["production_path"])
    if args_dict.get("check"):
        _check_existing_manifests(production_path)
    else:
        _write_manifests(production_path, args_dict["job_grid_file"], args_dict, context)


def _check_existing_manifests(production_path):
    """Check all existing simulate_prod job manifests below a production path."""
    job_directories = _job_directories(production_path)
    if not job_directories:
        raise FileNotFoundError(f"No production job directories found in {production_path}.")
    manifests = [directory / SIMULATE_PROD_JOB_METADATA for directory in job_directories]
    missing = [manifest for manifest in manifests if not manifest.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing production metadata manifest(s): " + ", ".join(map(str, missing))
        )
    for manifest in manifests:
        check_manifest(manifest)


def _write_manifests(production_path, job_grid_file, args_dict, context):
    """Write manifests for job directories described by a job grid."""
    rows, metadata = read_job_grid(job_grid_file)
    job_directories = _job_directories_for_rows(production_path, len(rows))
    for row, job_directory in zip(rows, job_directories):
        manifest_path = job_directory / SIMULATE_PROD_JOB_METADATA
        if manifest_path.exists() and not args_dict.get("overwrite"):
            raise FileExistsError(
                f"Metadata manifest already exists: {manifest_path}. Use --overwrite to replace it."
            )
        args = job_grid_row_to_simulate_prod_args(row, metadata)
        args.setdefault("simulation_software", "corsika_sim_telarray")
        args.setdefault("eslope", -2.0)
        missing = [
            key
            for key in ("run_number", *REQUIRED_SIMULATION_JOB_METADATA_ARGUMENTS)
            if args.get(key) is None
        ]
        if missing:
            raise ValueError(
                f"Job grid {job_grid_file} does not provide required resolved configuration "
                "field(s): " + ", ".join(missing)
            )
        overwrites = args.get("overwrite_model_parameters")
        simulator = SimpleNamespace(
            run_number=args["run_number"],
            array_models=[
                SimpleNamespace(
                    model_version=args["model_version"],
                    overwrite_model_parameter_dict=(
                        read_overwrite_model_parameter_dict(overwrites, context=context)
                        if overwrites
                        else {}
                    ),
                    array_elements={},
                    site_model=SimpleNamespace(parameters={}),
                )
            ],
            corsika_configurations=[],
        )
        file_inventory = inventory_production_files(job_directory)
        validate_required_production_outputs(
            file_inventory, args["simulation_software"], job_directory
        )
        threshold = args.get("curved_atmosphere_min_zenith_angle")
        manifest = build_production_job_manifest(
            args,
            simulator,
            job_directory,
            file_inventory=file_inventory,
            catalog_metadata=build_simulation_job_metadata(args, simulator, include_sct=False),
            atmosphere_configuration=(
                {"curved_atmosphere_min_zenith_angle": threshold} if threshold is not None else {}
            ),
        )
        check_manifest(ProductionManifest(path=manifest_path, data=manifest))
        write_data_to_file(manifest, manifest_path)


def _job_directories(production_path):
    """Return sorted direct production job directories."""
    if not Path(production_path).is_dir():
        raise FileNotFoundError(f"Production path not found: {production_path}")
    return sorted(path for path in Path(production_path).glob("job-*") if path.is_dir())


def _job_directories_for_rows(production_path, row_count):
    """Resolve a complete zero- or one-based job-directory mapping."""
    job_directories = _job_directories(production_path)
    if not job_directories:
        raise FileNotFoundError(
            f"Production job directory not found: {production_path / 'job-000001'}"
        )
    numbers = {
        int(directory.name.removeprefix("job-"))
        for directory in job_directories
        if directory.name.removeprefix("job-").isdigit()
    }
    offset = 0 if 0 in numbers else 1
    expected = list(range(offset, offset + row_count))
    if numbers != set(expected):
        raise FileNotFoundError(
            f"Production job directories do not match the {row_count}-row job grid. "
            f"Expected: {', '.join(f'job-{number:06d}' for number in expected)}; "
            f"found: {', '.join(directory.name for directory in job_directories)}."
        )
    return [production_path / f"job-{number:06d}" for number in expected]


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
    view_cone_min, view_cone_max = args_dict["view_cone"]
    metadata = {
        "array_layout": args_dict["array_layout_name"],
        "site": CATALOG_SITE_NAMES[args_dict["site"]],
        "particle": _primary_name(args_dict, simulator),
        "phiP": round(geographic_to_corsika_azimuth(args_dict[_AZIMUTH_ANGLE].to_value(u.deg)), 2),
        "thetaP": round(float(args_dict[_ZENITH_ANGLE].to_value(u.deg)), 2),
        "view_cone_min": round(float(view_cone_min.to_value(u.deg)), 2),
        "view_cone_max": round(float(view_cone_max.to_value(u.deg)), 2),
        "runNumber": int(simulator.run_number),
        "model_version": _scalar_or_list(args_dict["model_version"]),
    }
    if include_sct:
        metadata["sct"] = str(_has_sct(simulator.array_models))
    metadata.update(
        {
            key: round(float(args_dict[key].to_value(u.deg)), 2)
            for key in ("dec", "ha")
            if args_dict.get(key) is not None
        }
    )
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
    manifest = {
        **get_manifest_schema_metadata("simulate_prod_job"),
        "product_type": "simulate_prod_job",
        "production_id": args_dict.get("production_id") or args_dict.get("label"),
        "job_id": Path(output_directory).name,
        "status": "complete",
        "catalog_metadata": catalog_metadata,
        "configuration": _build_selection_configuration(
            args_dict, simulator, atmosphere_configuration=atmosphere_configuration
        ),
        "files": (
            file_inventory
            if file_inventory is not None
            else inventory_production_files(output_directory)
        ),
    }
    event_counts = get_sim_telarray_event_counts(output_directory, manifest["files"])
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
    return _reduced_event_counts(output_directory, file_inventory) or (
        _sim_telarray_log_event_counts(output_directory, file_inventory)
    )


def _reduced_event_counts(output_directory, file_inventory):
    """Read event counts from reduced-event HDF5 tables when available."""
    paths = file_inventory.get("reduced_event_data", [])
    if not paths:
        return None
    simulated_events = triggered_events = 0
    try:
        for relative_path in paths:
            file_path = resolve_relative_manifest_path(output_directory, relative_path)
            with h5py.File(file_path, "r") as data_file:
                simulated_events += len(data_file["SHOWERS"])
                if (trigger_table := data_file.get("TRIGGERS")) is not None:
                    triggered_events += len(trigger_table)
    except KeyError, OSError, TypeError, ValueError:
        return None
    return {"simulated_events": simulated_events, "triggered_events": triggered_events}


def _sim_telarray_log_event_counts(output_directory, file_inventory):
    """Extract event counts from sim_telarray log footers as a fallback."""
    counts = []
    for relative_path in file_inventory.get("sim_telarray_log", []):
        try:
            file_path = resolve_relative_manifest_path(output_directory, relative_path)
            count = _read_log_event_count(file_path)
        except EOFError, OSError, ValueError:
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
    """Return the last sim_telarray event-count footer in a plain or gzip-compressed log."""
    opener = gzip.open if path.name.endswith(".gz") else open
    count = None
    with opener(path, "rt", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = _SIMTEL_EVENT_COUNT_PATTERN.search(line.rstrip())
            if match:
                count = (int(match["simulated"]), int(match["triggered"]))
    return count


def _as_list(value):
    """Return value as a list, wrapping a single simulator object."""
    return value if isinstance(value, list) else [value]


def _corsika_configurations(simulator):
    """Return the simulator's CORSIKA configurations as a list."""
    return _as_list(getattr(simulator, "corsika_configurations", []))


def _simulation_parameters(simulator):
    """Return the simulator's simulation parameters as a list."""
    return _as_list(getattr(simulator, "simulation_parameters", []))


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
        _AZIMUTH_ANGLE: args_dict[_AZIMUTH_ANGLE],
        _ZENITH_ANGLE: args_dict[_ZENITH_ANGLE],
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
    simulation_parameters = _simulation_parameters(simulator)
    if args_dict.get("corsika_file") and simulation_parameters:
        for angle in (_AZIMUTH_ANGLE, _ZENITH_ANGLE):
            value = getattr(simulation_parameters[0], angle, None)
            if value is not None:
                configuration[angle] = float(value) * u.deg
    configuration.update(
        {
            key: args_dict.get(key)
            for key in (
                "dec",
                "ha",
                "corsika_seeds",
                "event_number_first_shower",
                "correct_for_b_field_alignment",
                "instrument_seed",
                "random_instrument_instances",
                "simulation_seed",
                "sim_telarray_seed_file",
            )
            if args_dict.get(key) is not None
        }
    )
    return {key: value for key, value in configuration.items() if value is not None}


def _resolved_model_parameter_overrides(simulator):
    """Return resolved model-parameter overrides without source file paths."""
    overrides_by_version = {
        str(model.model_version): deepcopy(model.overwrite_model_parameter_dict)
        for model in simulator.array_models
        if getattr(model, "overwrite_model_parameter_dict", None)
    }
    if len(overrides_by_version) == 1:
        return next(iter(overrides_by_version.values()))
    return overrides_by_version


def _resolved_atmosphere_configuration(args_dict, simulator):
    """Return resolved atmosphere settings used by the simulation."""
    atmosphere = {}
    if args_dict.get("curved_atmosphere_min_zenith_angle") is not None:
        atmosphere["curved_atmosphere_min_zenith_angle"] = args_dict[
            "curved_atmosphere_min_zenith_angle"
        ]
    curved_values = {
        bool(configuration.use_curved_atmosphere)
        for configuration in _corsika_configurations(simulator)
        if configuration is not None
    } or {
        bool(getattr(parameters, USE_CURVED_ATMOSPHERE))
        for parameters in _simulation_parameters(simulator)
        if parameters is not None and hasattr(parameters, USE_CURVED_ATMOSPHERE)
    }
    if len(curved_values) == 1:
        atmosphere[USE_CURVED_ATMOSPHERE] = curved_values.pop()

    site_parameters = {}
    for model in simulator.array_models:
        parameters = getattr(getattr(model, "site_model", None), "parameters", {})
        for name in ("atmospheric_profile", "atmospheric_transmission", "reference_point_altitude"):
            if name in parameters:
                site_parameters[name] = deepcopy(parameters[name].get("value"))
    if site_parameters:
        atmosphere["site_parameters"] = site_parameters
    return atmosphere


def _primary_name(args_dict, simulator):
    """Return the resolved primary name, including sim_telarray-only inputs."""
    if args_dict.get("primary") is not None:
        return str(args_dict["primary"]).lower()
    for configuration in (*_corsika_configurations(simulator), *_simulation_parameters(simulator)):
        primary_name = getattr(getattr(configuration, "primary_particle", None), "name", None)
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
    for objects in (_corsika_configurations(simulator), _simulation_parameters(simulator)):
        shower_counts = {
            obj.shower_events for obj in objects if getattr(obj, "shower_events", None) is not None
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
