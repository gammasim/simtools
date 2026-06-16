r"""Generate HTCondor submit files for NSB (gamma) and proton bias curves.

This module generates two independent HTCondor submission sets:

- NSB curve:
  Uses gamma primary, fixed low-energy range, and NSB-specific model overwrites.
- Proton curve:
  Uses proton primary, fixed proton energy range, and proton trigger overwrites.

The trigger-threshold scan is built into this module. The user does not provide
threshold values through the CLI. The telescope is resolved from the requested
single-telescope array layout, and the correct threshold parameter is chosen from
the telescope type:

- LST: ``asum_threshold``
- MST/SST: ``dsum_threshold``

The resolved single telescope is passed to production generation as an
``array_element_list`` entry. This keeps the HTCondor generator independent of
model/layout resolution and avoids requiring a separate telescope argument there.

For each curve, a parameter-scan configuration is generated dynamically. The
scan-grid application then creates one overwrite file per threshold value and
writes the expanded scan grid. Only one scan parameter is used, so the generic
scan generator does not create an unwanted cartesian product.

The NSB overwrite always sets ``min_photons`` and ``min_photoelectrons`` to zero
and resets ``nsb_scaling_factor`` to 2. The proton overwrite does not touch
these parameters.

No external overwrite templates are used; overwrite YAML content is defined
directly in the generated scan configuration.

"""

import logging
from pathlib import Path

import yaml

from simtools.model.site_model import SiteModel
from simtools.runners import simtools_runner

_logger = logging.getLogger(__name__)

_PARAMETER_VERSION = "2.0.0"

_NSB_ENERGY_RANGE = "20 MeV 25 MeV"
_PROTON_ENERGY_RANGE = "800 GeV 2000 GeV"
_NSB_SCALING_FACTOR = 2

_ASUM_THRESHOLDS = [220, 230, 240, 250, 260, 270, 280, 290, 300]
_DSUM_THRESHOLDS = [22, 23, 24, 25, 26, 27, 28, 29, 30]

_TELESCOPE_THRESHOLD_PARAM = {
    "L": "asum_threshold",
    "M": "dsum_threshold",
    "S": "dsum_threshold",
}

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

_SIMULATION_CLI_ARGS = [
    "site",
    "model_version",
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
    layout_elements = [
        str(element)
        for element in site_model.get_array_elements_for_layout(args["array_layout_name"])
    ]

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


def _format_threshold_value(threshold):
    """Return integer threshold values without a trailing .0."""
    return int(threshold) if float(threshold).is_integer() else threshold


def _build_proton_overwrite_base(telescopes, model_version):
    """Build base overwrite YAML content for proton threshold scans."""
    changes = {}

    for telescope in telescopes:
        changes[telescope] = {}

    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for proton telescope trigger scan",
        "changes": changes,
    }


def _build_nsb_overwrite_base(telescopes, site, model_version):
    """Build base overwrite YAML content for NSB threshold scans."""
    changes = {}

    for telescope in telescopes:
        changes[telescope] = {
            "min_photons": {
                "version": _PARAMETER_VERSION,
                "value": 0,
            },
            "min_photoelectrons": {
                "version": _PARAMETER_VERSION,
                "value": 0,
            },
        }

    changes[f"OBS-{site}"] = {
        "nsb_scaling_factor": {
            "version": _PARAMETER_VERSION,
            "value": _NSB_SCALING_FACTOR,
        },
    }

    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for NSB telescope trigger scan",
        "changes": changes,
    }


def _build_overwrite_base(curve_name, telescopes, args):
    """Build base overwrite YAML content for one curve."""
    if curve_name == "nsb":
        return _build_nsb_overwrite_base(
            telescopes=telescopes,
            site=args["site"],
            model_version=args["model_version"],
        )

    if curve_name == "proton":
        return _build_proton_overwrite_base(
            telescopes=telescopes,
            model_version=args["model_version"],
        )

    raise ValueError(f"Unsupported curve name '{curve_name}'.")


def _grid_generation_configuration(args, primary, energy_range, output_file, label):
    """Build configuration for simtools-production-generate-grid."""
    configuration = {}

    for key in _SIMULATION_CLI_ARGS:
        value = args.get(key)
        if value is not None:
            configuration[key] = value

    if len(args.get("telescopes", [])) != 1:
        raise ValueError(
            "Bias-curve grid generation requires exactly one resolved telescope; "
            f"got {len(args.get('telescopes', []))}: {args.get('telescopes', [])}"
        )

    configuration.update(
        {
            "array_element_list": args["telescopes"][0],
            "primary": primary,
            "energy_range": energy_range,
            "label": label,
            "output_file": str(output_file),
            "output_path": str(output_file.parent),
        }
    )

    return configuration


def _write_scan_config(curve_name, scan_config_file, label, telescopes, args):
    """Write scan configuration for simtools-generate-parameter-scan-grid."""
    if len(telescopes) != 1:
        raise ValueError(
            f"Bias-curve scan configuration expects exactly one telescope; got {len(telescopes)}."
        )

    telescope = telescopes[0]
    threshold_param = _threshold_param_name(telescope)
    threshold_values = _threshold_values_for_telescope(telescope)

    scan_config = {
        "label": label,
        "parameter_scan": {
            "overwrite": _build_overwrite_base(
                curve_name=curve_name,
                telescopes=telescopes,
                args=args,
            ),
            "parameters": [
                {
                    "name": threshold_param,
                    "path": f"changes.{telescope}.{threshold_param}",
                    "version": _PARAMETER_VERSION,
                    "values": [_format_threshold_value(value) for value in threshold_values],
                }
            ],
        },
    }

    with open(scan_config_file, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(scan_config, file_handle, sort_keys=False)

    _logger.info(f"Parameter-scan configuration for {curve_name} written to '{scan_config_file}'.")
    return scan_config_file


def _htcondor_generator_configuration(label, curve_directory, scan_grid_file, args):
    """Build configuration for simtools-simulate-prod-htcondor-generator."""
    apptainer_image = args.get("apptainer_image")

    if not apptainer_image:
        raise ValueError("Missing required argument: --apptainer_image")

    return {
        "apptainer_image": apptainer_image,
        "priority": args.get("priority", 1),
        "job_grid_file": str(scan_grid_file),
        "output_path": str(curve_directory / args.get("htcondor_output_path", "htcondor_submit")),
        "simulation_output": str(Path(args.get("output_path") or ".").expanduser().resolve()),
        "label": label,
        "log_level": args.get("log_level", "INFO"),
    }


def _write_full_workflow_config(
    curve_name,
    args,
    primary,
    energy_range,
    base_grid_file,
    scan_config_file,
    scan_grid_file,
    label,
    curve_directory,
):
    """Write the complete workflow configuration for one bias-curve submission."""
    workflow_config = {
        "applications": [
            {
                "application": "simtools-production-generate-grid",
                "configuration": _grid_generation_configuration(
                    args=args,
                    primary=primary,
                    energy_range=energy_range,
                    output_file=base_grid_file,
                    label=label,
                ),
            },
            {
                "application": "simtools-generate-parameter-scan-grid",
                "configuration": {
                    "job_grid_file": str(base_grid_file),
                    "scan_config": str(scan_config_file),
                    "output_file": str(scan_grid_file),
                },
            },
            {
                "application": "simtools-simulate-prod-htcondor-generator",
                "configuration": _htcondor_generator_configuration(
                    label=label,
                    curve_directory=curve_directory,
                    scan_grid_file=scan_grid_file,
                    args=args,
                ),
            },
        ]
    }

    config_file = curve_directory / f"{curve_name}_workflow.yml"

    with open(config_file, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(workflow_config, file_handle, sort_keys=False)

    _logger.info(f"Full workflow configuration written to '{config_file}'.")
    return config_file


def _run_workflow(config_file, args):
    """Run the complete workflow through the simtools workflow runner."""
    runner_args = {
        "config_file": str(config_file),
        "steps": None,
        "ignore_runtime_environment": True,
    }

    if args.get("activity_id") is not None:
        runner_args["activity_id"] = args["activity_id"]

    simtools_runner.run_applications(runner_args)


def _generate_curve_submissions(curve_name, curve_definition, args, output_root):
    """Generate scan configuration and run the full workflow for one curve."""
    primary = curve_definition["primary"]
    energy_range = curve_definition["energy_range"]

    curve_directory = output_root / curve_name
    curve_directory.mkdir(parents=True, exist_ok=True)

    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_config_file = curve_directory / "scan_config.yml"
    scan_grid_file = curve_directory / "scan_grid.ecsv"

    base_label = args.get("label") or "bias_curve"
    curve_label = f"{base_label}_{curve_name}"

    _write_scan_config(
        curve_name=curve_name,
        scan_config_file=scan_config_file,
        label=curve_label,
        telescopes=args["telescopes"],
        args=args,
    )

    workflow_config_file = _write_full_workflow_config(
        curve_name=curve_name,
        args=args,
        primary=primary,
        energy_range=energy_range,
        base_grid_file=base_grid_file,
        scan_config_file=scan_config_file,
        scan_grid_file=scan_grid_file,
        label=curve_label,
        curve_directory=curve_directory,
    )

    _run_workflow(workflow_config_file, args)


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

    output_root = Path(args.get("output_path") or ".").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for curve_name, curve_definition in _CURVE_DEFINITIONS.items():
        _generate_curve_submissions(
            curve_name=curve_name,
            curve_definition=curve_definition,
            args=args,
            output_root=output_root,
        )
