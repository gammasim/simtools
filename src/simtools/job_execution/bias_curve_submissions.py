r"""Generate HTCondor submit files for NSB (gamma) and proton bias curves.

This module generates two independent HTCondor submission workflows:

- NSB curve:
  Uses gamma primary, fixed low-energy range, and NSB-specific model overwrites.
- Proton curve:
  Uses proton primary, fixed proton energy range, and proton trigger overwrites.

For each curve, this module writes a scan configuration and a simtools workflow
configuration, then executes the workflow through ``simtools_runner.run_applications``.

The NSB overwrite always sets ``min_photons`` and ``min_photoelectrons`` to zero
and resets ``nsb_scaling_factor`` to 2. The proton overwrite does not touch
these parameters.
"""

import logging
from pathlib import Path

import yaml

from simtools.model.site_model import SiteModel
from simtools.runners import simtools_runner

_logger = logging.getLogger(__name__)

_PARAMETER_VERSION = "2.0.0"

_NSB_ENERGY_RANGE = "20 MeV 25 MeV"
_PROTON_ENERGY_RANGE = "2 GeV 2000 GeV"
_NSB_SCALING_FACTOR = 2

_ASUM_THRESHOLDS = [220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360]
_DSUM_THRESHOLDS = [22, 23, 24, 25, 26, 27, 28, 29, 30]

_TELESCOPE_THRESHOLD_PARAM = {"L": "asum_threshold", "M": "dsum_threshold"}

_CURVE_DEFINITIONS = {
    "nsb": {
        "primary": "gamma",
        "energy_range": _NSB_ENERGY_RANGE,
    },
    "proton": {
        "primary": "proton",
        "energy_range": _PROTON_ENERGY_RANGE,
    },
}

_PRODUCTION_GRID_ARGS = [
    "site",
    "model_version",
    "telescope",
    "array_layout_name",
    "simulation_software",
    "azimuth_angle",
    "zenith_angle",
    "showers_per_run",
    "core_scatter",
    "view_cone",
    "number_of_runs",
    "corsika_le_interaction",
    "corsika_he_interaction",
]


def _resolve_telescopes_from_layout(args):
    """Resolve telescope names from array layout and enforce single-telescope layouts."""
    site_model = SiteModel(site=args["site"], model_version=args["model_version"])
    layout_elements = list(site_model.get_array_elements_for_layout(args["array_layout_name"]))

    _logger.info(
        "Resolved array layout '%s' to elements: %s",
        args["array_layout_name"],
        layout_elements,
    )

    if len(layout_elements) != 1:
        raise ValueError(
            f"Bias-curve submissions currently support only single-telescope layouts; "
            f"got {len(layout_elements)} elements in '{args['array_layout_name']}': "
            f"{layout_elements}"
        )

    telescope = layout_elements[0]

    if "invalid" in telescope.lower():
        raise ValueError(
            f"Array layout '{args['array_layout_name']}' resolved to invalid telescope "
            f"'{telescope}'."
        )

    return [telescope]


def _threshold_param_name(telescope):
    """Return the trigger-threshold parameter name for a telescope."""
    if not telescope:
        raise ValueError("Cannot determine threshold parameter for empty telescope name.")

    telescope_type = telescope[0].upper()

    try:
        return _TELESCOPE_THRESHOLD_PARAM[telescope_type]
    except KeyError as exc:
        raise ValueError(
            f"Cannot determine threshold parameter for telescope '{telescope}'. "
            "Expected telescope name to start with L, M, or S."
        ) from exc


def _threshold_values_for_telescope(telescope):
    """Return the built-in threshold scan values for a telescope."""
    threshold_param = _threshold_param_name(telescope)

    if threshold_param == "asum_threshold":
        return _ASUM_THRESHOLDS

    if threshold_param == "dsum_threshold":
        return _DSUM_THRESHOLDS

    raise ValueError(
        f"Unsupported threshold parameter '{threshold_param}' for telescope '{telescope}'."
    )


def _threshold_label_prefix(threshold_param):
    """Return a compact prefix for threshold scan labels."""
    return threshold_param.removesuffix("_threshold")


def _parameter_scan_entry(telescope):
    """Build the parameter-scan entry for the telescope trigger threshold."""
    threshold_param = _threshold_param_name(telescope)
    return {
        "name": threshold_param,
        "path": f"changes.{telescope}.{threshold_param}",
        "version": _PARAMETER_VERSION,
        "values": _threshold_values_for_telescope(telescope),
        "label": _threshold_label_prefix(threshold_param),
        "label_separator": "",
    }


def _base_proton_overwrite(telescope, model_version):
    """Build the proton base overwrite block before threshold insertion."""
    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for proton telescope trigger scan",
        "changes": {telescope: {}},
    }


def _base_nsb_overwrite(telescope, site, model_version):
    """Build the NSB base overwrite block before threshold insertion."""
    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for NSB telescope trigger scan",
        "changes": {
            telescope: {
                "min_photons": {
                    "version": _PARAMETER_VERSION,
                    "value": 0,
                },
                "min_photoelectrons": {
                    "version": _PARAMETER_VERSION,
                    "value": 0,
                },
            },
            f"OBS-{site}": {
                "nsb_scaling_factor": {
                    "version": _PARAMETER_VERSION,
                    "value": _NSB_SCALING_FACTOR,
                },
            },
        },
    }


def _base_overwrite(curve_name, telescope, args):
    """Build the curve-specific base overwrite block before threshold insertion."""
    if curve_name == "nsb":
        return _base_nsb_overwrite(
            telescope=telescope,
            site=args["site"],
            model_version=args["model_version"],
        )

    if curve_name == "proton":
        return _base_proton_overwrite(
            telescope=telescope,
            model_version=args["model_version"],
        )

    raise ValueError(f"Unsupported curve name '{curve_name}'.")


def _write_yaml(file_path, content):
    """Write YAML content to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(content, file_handle, sort_keys=False)
    _logger.info("Wrote %s", file_path)


def _scan_config(curve_name, telescope, args):
    """Build the parameter-scan configuration for one curve."""
    return {
        "label": curve_name,
        "parameter_scan": {
            "overwrite": _base_overwrite(curve_name, telescope, args),
            "parameters": [_parameter_scan_entry(telescope)],
        },
    }


def _production_grid_configuration(args, curve_definition, base_grid_file, curve_label):
    """Build configuration for simtools-production-generate-grid."""
    configuration = {
        key: args[key]
        for key in _PRODUCTION_GRID_ARGS
        if key in args and args[key] not in (None, "")
    }
    configuration.update(
        {
            "primary": curve_definition["primary"],
            "energy_range": curve_definition["energy_range"],
            "label": curve_label,
            "output_file": str(base_grid_file),
        }
    )
    return configuration


def _scan_grid_configuration(base_grid_file, scan_config_file, scan_grid_file):
    """Build configuration for simtools-generate-parameter-scan-grid."""
    return {
        "job_grid_file": str(base_grid_file),
        "scan_config": str(scan_config_file),
        "output_file": str(scan_grid_file),
    }


def _htcondor_configuration(curve_label, curve_directory, scan_grid_file, args, output_root):
    """Build configuration for simtools-simulate-prod-htcondor-generator."""
    apptainer_image = args.get("apptainer_image")
    if not apptainer_image:
        raise ValueError("Missing required argument: --apptainer_image")

    configuration = {
        "apptainer_image": apptainer_image,
        "priority": args.get("priority", 1),
        "job_grid_file": str(scan_grid_file),
        "output_path": str(curve_directory / args.get("htcondor_output_path", "htcondor_submit")),
        "simulation_output": str(output_root),
        "label": curve_label,
        "log_level": args.get("log_level", "INFO"),
    }
    if args.get("telescope") not in (None, ""):
        configuration["telescope"] = args["telescope"]
    return configuration


def _workflow_config(
    curve_name,
    curve_definition,
    args,
    base_grid_file,
    scan_config_file,
    scan_grid_file,
    curve_directory,
    output_root,
):
    """Build the runner workflow configuration for one curve."""
    curve_label = curve_name
    return {
        "applications": [
            {
                "application": "simtools-production-generate-grid",
                "configuration": _production_grid_configuration(
                    args=args,
                    curve_definition=curve_definition,
                    base_grid_file=base_grid_file,
                    curve_label=curve_label,
                ),
            },
            {
                "application": "simtools-generate-parameter-scan-grid",
                "configuration": _scan_grid_configuration(
                    base_grid_file=base_grid_file,
                    scan_config_file=scan_config_file,
                    scan_grid_file=scan_grid_file,
                ),
            },
            {
                "application": "simtools-simulate-prod-htcondor-generator",
                "configuration": _htcondor_configuration(
                    curve_label=curve_label,
                    curve_directory=curve_directory,
                    scan_grid_file=scan_grid_file,
                    args=args,
                    output_root=output_root,
                ),
            },
        ]
    }


def _run_workflow(workflow_file, args):
    """Run a generated simtools workflow configuration through simtools runners."""
    simtools_runner.run_applications(
        {
            "config_file": str(workflow_file),
            "steps": None,
            "activity_id": args.get("activity_id"),
            "ignore_runtime_environment": True,
        }
    )


def _generate_curve_submissions(curve_name, curve_definition, args, output_root):
    """Generate grid, scan config, scan grid, and HTCondor scripts for one curve."""
    curve_directory = output_root / curve_name
    curve_directory.mkdir(parents=True, exist_ok=True)

    telescope = args["telescopes"][0]
    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_config_file = curve_directory / "scan_config.yaml"
    scan_grid_file = curve_directory / "scan_grid.ecsv"
    workflow_file = curve_directory / "workflow.yaml"

    _write_yaml(scan_config_file, _scan_config(curve_name, telescope, args))
    _write_yaml(
        workflow_file,
        _workflow_config(
            curve_name=curve_name,
            curve_definition=curve_definition,
            args=args,
            base_grid_file=base_grid_file,
            scan_config_file=scan_config_file,
            scan_grid_file=scan_grid_file,
            curve_directory=curve_directory,
            output_root=output_root,
        ),
    )
    _run_workflow(workflow_file, args)


def _validate_required_args(args):
    """Validate arguments required by this module."""
    required_args = [
        "site",
        "model_version",
        "array_layout_name",
        "azimuth_angle",
        "zenith_angle",
        "showers_per_run",
        "core_scatter",
        "view_cone",
        "number_of_runs",
        "apptainer_image",
    ]

    for key in required_args:
        if args.get(key) in (None, ""):
            raise ValueError(f"Missing required argument: --{key}")


def generate_bias_curve_submissions(args):
    """Generate NSB and proton bias-curve HTCondor submissions from CLI args."""
    _validate_required_args(args)

    args["telescopes"] = _resolve_telescopes_from_layout(args)
    if len(args["telescopes"]) == 1:
        args["telescope"] = str(args["telescopes"][0])

    output_root = Path(args.get("output_path") or ".").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for curve_name, curve_definition in _CURVE_DEFINITIONS.items():
        _generate_curve_submissions(
            curve_name=curve_name,
            curve_definition=curve_definition,
            args=args,
            output_root=output_root,
        )
