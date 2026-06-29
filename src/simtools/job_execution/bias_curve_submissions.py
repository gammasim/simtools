r"""Generate scan grids for NSB (gamma) and proton bias curves.

This module generates two independent bias-curve scan-grid workflows:

- NSB curve:
  Uses gamma primary, a configurable low-energy range, and NSB-specific model
  overwrites.
- Proton curve:
  Uses proton primary, a configurable energy range, and proton trigger overwrites.

For each curve, this module writes a scan configuration and a simtools workflow
configuration, then executes only the backend-neutral grid-generation steps through
``simtools_runner.run_applications``.

The NSB overwrite always sets ``min_photons`` and ``min_photoelectrons`` to zero.
Both curves set ``nsb_scaling_factor`` because night-sky background affects the
trigger response in both simulations.
"""

import logging

from simtools.io import ascii_handler
from simtools.model.telescope_model import TelescopeModel
from simtools.runners import simtools_runner

_logger = logging.getLogger(__name__)

_PRODUCTION_GRID_ARGS = [
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


def _threshold_param_name(args):
    """Return the threshold parameter selected by the telescope model."""
    telescope_model = TelescopeModel(
        site=args["site"],
        telescope_name=args["telescope"],
        model_version=args["model_version"],
    )
    if telescope_model.get_parameter_value("default_trigger") == "AnalogSum":
        return "asum_threshold"
    return "dsum_threshold"


def _threshold_values(threshold_param, trigger_thresholds=None):
    """Return configured thresholds or defaults for the threshold parameter."""
    if trigger_thresholds is not None:
        if not trigger_thresholds:
            raise ValueError("trigger_thresholds must contain at least one value.")
        return trigger_thresholds

    if threshold_param == "asum_threshold":
        return [*range(220, 310, 10), *range(320, 361, 20)]
    return list(range(22, 31))


def _threshold_label_prefix(threshold_param):
    """Return a compact prefix for threshold scan labels."""
    return threshold_param.removesuffix("_threshold")


def _parameter_scan_entry(telescope, threshold_param, trigger_thresholds=None):
    """Build the parameter-scan entry for the telescope trigger threshold."""
    return {
        "name": threshold_param,
        "path": f"changes.{telescope}.{threshold_param}",
        "values": _threshold_values(threshold_param, trigger_thresholds),
        "label": _threshold_label_prefix(threshold_param),
        "label_separator": "",
    }


def _nsb_scaling_change(nsb_scaling_factor):
    """Return the NSB scaling model-parameter change."""
    return {
        "nsb_scaling_factor": {
            "value": nsb_scaling_factor,
        }
    }


def _base_proton_overwrite(telescope, site, model_version, nsb_scaling_factor):
    """Build the proton base overwrite block before threshold insertion."""
    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for proton telescope trigger scan",
        "changes": {
            telescope: {},
            f"OBS-{site}": _nsb_scaling_change(nsb_scaling_factor),
        },
    }


def _base_nsb_overwrite(telescope, site, model_version, nsb_scaling_factor):
    """Build the NSB base overwrite block before threshold insertion."""
    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for NSB telescope trigger scan",
        "changes": {
            telescope: {
                "min_photons": {
                    "value": 0,
                },
                "min_photoelectrons": {
                    "value": 0,
                },
            },
            f"OBS-{site}": _nsb_scaling_change(nsb_scaling_factor),
        },
    }


def _base_overwrite(curve_name, telescope, args):
    """Build the curve-specific base overwrite block before threshold insertion."""
    nsb_scaling_factor = args["nsb_scaling_factor"]
    if curve_name == "nsb":
        return _base_nsb_overwrite(
            telescope, args["site"], args["model_version"], nsb_scaling_factor
        )

    if curve_name == "proton":
        return _base_proton_overwrite(
            telescope, args["site"], args["model_version"], nsb_scaling_factor
        )

    raise ValueError(f"Unsupported curve name '{curve_name}'.")


def _write_yaml(file_path, content):
    """Write YAML content to file."""
    ascii_handler.write_data_to_file(content, file_path, sort_keys=False)
    _logger.info("Wrote %s", file_path)


def _scan_config(curve_name, telescope, args):
    """Build the parameter-scan configuration for one curve."""
    return {
        "label": curve_name,
        "parameter_scan": {
            "overwrite": _base_overwrite(curve_name, telescope, args),
            "parameters": [
                _parameter_scan_entry(
                    telescope,
                    args["threshold_parameter"],
                    args.get("trigger_thresholds"),
                )
            ],
            "job_grid_updates": {"telescope": telescope},
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
            "array_layout_name": args["telescope"],
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


def _workflow_config(
    curve_name,
    curve_definition,
    args,
    base_grid_file,
    scan_config_file,
    scan_grid_file,
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


def _generate_curve_submissions(curve_name, curve_definition, args, io_handler):
    """Generate base grid, scan config, and scan grid for one curve."""
    curve_directory = io_handler.get_output_directory(sub_dir=curve_name)

    telescope = args["telescope"]
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
        ),
    )
    _run_workflow(workflow_file, args)


def _validate_required_args(args):
    """Validate arguments required by this module."""
    required_args = [
        "site",
        "model_version",
        "telescope",
        "azimuth_angle",
        "zenith_angle",
        "showers_per_run",
        "core_scatter",
        "view_cone",
        "number_of_runs",
    ]

    for key in required_args:
        if args.get(key) in (None, ""):
            raise ValueError(f"Missing required argument: --{key}")


def _curve_definitions(args):
    """Build curve definitions from configured energy ranges."""
    return {
        "nsb": {
            "primary": "gamma",
            "energy_range": args["nsb_energy_range"],
        },
        "proton": {
            "primary": "proton",
            "energy_range": args["proton_energy_range"],
        },
    }


def generate_bias_curve_submissions(args, io_handler):
    """Generate NSB and proton bias-curve scan grids from CLI args.

    Parameters
    ----------
    args : dict
        Application arguments for production-grid and bias-curve scan configuration.
    io_handler : IOHandler
        IOHandler for file output.
    """
    _validate_required_args(args)

    generation_args = {**args, "threshold_parameter": _threshold_param_name(args)}

    for curve_name, curve_definition in _curve_definitions(args).items():
        _generate_curve_submissions(
            curve_name=curve_name,
            curve_definition=curve_definition,
            args=generation_args,
            io_handler=io_handler,
        )
