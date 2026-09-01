#!/usr/bin/python3

"""Write or check production-job metadata manifests."""

from pathlib import Path
from types import SimpleNamespace

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io.ascii_handler import write_data_to_file
from simtools.model.model_utils import read_overwrite_model_parameter_dict
from simtools.production_configuration.job_grid_io import (
    job_grid_row_to_simulate_prod_args,
    read_job_grid,
)
from simtools.production_configuration.job_metadata import (
    build_production_job_manifest,
    build_simulation_job_metadata,
)
from simtools.production_configuration.production_file_selection import (
    SIMULATE_PROD_JOB_METADATA,
    ProductionManifest,
    check_manifest,
    inventory_production_files,
    validate_required_production_outputs,
)

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "production_path",
        help="Directory containing production job output directories.",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition(
        "job_grid_file",
        help="Authoritative job grid used to reconstruct resolved production configuration.",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "check",
        help="Validate existing manifests without writing files.",
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "overwrite",
        help="Overwrite existing metadata manifests in write mode.",
        action="store_true",
        default=False,
    ),
)


def _post_parse(args_dict, _config_sources, parser):
    """Validate write/check mode arguments."""
    if not args_dict.get("check") and not args_dict.get("job_grid_file"):
        parser.error(
            "'--job_grid_file' is required when writing production metadata; "
            "filenames alone are not an authoritative configuration source."
        )


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=_ARGUMENTS,
    initialize_output=False,
    post_parse=_post_parse,
)


def main():
    """Run the production metadata writer/checker."""
    app_context = APPLICATION.start()
    production_path = Path(app_context.args["production_path"])
    if app_context.args.get("check"):
        _check_existing_manifests(production_path)
    else:
        _write_manifests(production_path, app_context.args["job_grid_file"], app_context.args)


def _check_existing_manifests(production_path):
    """Check all existing simulate_prod job manifests below a production path."""
    job_directories = _job_directories(production_path)
    if not job_directories:
        raise FileNotFoundError(f"No production job directories found in {production_path}.")
    missing = [
        directory / SIMULATE_PROD_JOB_METADATA
        for directory in job_directories
        if not (directory / SIMULATE_PROD_JOB_METADATA).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing production metadata manifest(s): " + ", ".join(map(str, missing))
        )
    for job_directory in job_directories:
        check_manifest(job_directory / SIMULATE_PROD_JOB_METADATA)


def _write_manifests(production_path, job_grid_file, args_dict):
    """Write manifests for job directories described by a job grid."""
    rows, metadata = read_job_grid(job_grid_file)
    for index, row in enumerate(rows):
        job_directory = production_path / f"job-{index + 1:06d}"
        if not job_directory.is_dir():
            raise FileNotFoundError(f"Production job directory not found: {job_directory}")
        manifest_path = job_directory / SIMULATE_PROD_JOB_METADATA
        if manifest_path.exists() and not args_dict.get("overwrite"):
            raise FileExistsError(
                f"Metadata manifest already exists: {manifest_path}. Use --overwrite to replace it."
            )
        resolved_args = job_grid_row_to_simulate_prod_args(row, metadata)
        resolved_args.setdefault("simulation_software", "corsika_sim_telarray")
        resolved_args.setdefault("eslope", -2.0)
        _validate_resolved_configuration(resolved_args, job_grid_file)
        overwrite_parameter_file = resolved_args.get("overwrite_model_parameters")
        overwrite_parameters = (
            read_overwrite_model_parameter_dict(overwrite_parameter_file)
            if overwrite_parameter_file
            else {}
        )
        array_model = SimpleNamespace(
            model_version=resolved_args["model_version"],
            overwrite_model_parameter_dict=overwrite_parameters,
            array_elements={},
            site_model=SimpleNamespace(parameters={}),
        )
        simulator = SimpleNamespace(
            run_number=resolved_args["run_number"],
            array_models=[array_model],
            corsika_configurations=[],
        )
        file_inventory = inventory_production_files(job_directory)
        validate_required_production_outputs(
            file_inventory,
            resolved_args["simulation_software"],
            job_directory,
        )
        manifest = build_production_job_manifest(
            resolved_args,
            simulator,
            job_directory,
            file_inventory=file_inventory,
            catalog_metadata=build_simulation_job_metadata(
                resolved_args, simulator, include_sct=False
            ),
            atmosphere_configuration=_backfilled_atmosphere_configuration(resolved_args),
        )
        check_manifest(ProductionManifest(path=manifest_path, data=manifest))
        write_data_to_file(manifest, manifest_path)


def _backfilled_atmosphere_configuration(args_dict):
    """Return only atmosphere values known from the authoritative job grid."""
    threshold = args_dict.get("curved_atmosphere_min_zenith_angle")
    return {"curved_atmosphere_min_zenith_angle": threshold} if threshold is not None else {}


def _validate_resolved_configuration(args_dict, job_grid_file):
    """Require fields needed to write truthful production metadata."""
    missing = [
        key
        for key in (
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
        if args_dict.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"Job grid {job_grid_file} does not provide required resolved configuration "
            "field(s): " + ", ".join(missing)
        )


def _job_directories(production_path):
    """Return sorted direct production job directories."""
    production_path = Path(production_path)
    if not production_path.is_dir():
        raise FileNotFoundError(f"Production path not found: {production_path}")
    return sorted(path for path in production_path.glob("job-*") if path.is_dir())


if __name__ == "__main__":
    main()
