r"""Generate scan grids for NSB and proton bias curves.

This module generates two independent bias-curve scan grids:

- NSB curve:
  Uses gamma primary, a configurable low-energy range, and NSB-specific model
  overwrites.
- Proton curve:
  Uses proton primary, a configurable energy range, and proton trigger overwrites.

For each curve, this module generates a production grid, writes a scan
configuration, and expands the grid with the configured threshold values.

The NSB overwrite always sets ``min_photons`` and ``min_photoelectrons`` to zero.
Both curves set ``nsb_scaling_factor`` because night-sky background affects the
trigger response in both simulations.
"""

from simtools.io import ascii_handler
from simtools.job_execution import parameter_scan_generator
from simtools.model.telescope_model import TelescopeModel
from simtools.production_configuration.simulation_jobs import generate_job_grid

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
_DEFAULT_ASUM_THRESHOLDS = [*range(220, 310, 10), *range(320, 361, 20)]
_DEFAULT_DSUM_THRESHOLDS = list(range(22, 31))


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
    """Return trigger thresholds expanded from ``(minimum, count, step)`` or defaults."""
    if trigger_thresholds is not None:
        if len(trigger_thresholds) != 3:
            raise ValueError(
                "trigger_thresholds must contain minimum threshold, number of "
                "thresholds, and step size."
            )

        minimum, number, step_size = trigger_thresholds
        if number < 1 or not float(number).is_integer():
            raise ValueError("Number of trigger thresholds must be a positive integer.")
        if step_size <= 0:
            raise ValueError("Trigger-threshold step size must be positive.")

        return [minimum + index * step_size for index in range(int(number))]

    return (
        _DEFAULT_ASUM_THRESHOLDS
        if threshold_param == "asum_threshold"
        else _DEFAULT_DSUM_THRESHOLDS
    )


def _parameter_scan_entry(telescope, threshold_param, trigger_thresholds=None):
    """Build the parameter-scan entry for the telescope trigger threshold."""
    return {
        "name": threshold_param,
        "path": f"changes.{telescope}.{threshold_param}",
        "values": _threshold_values(threshold_param, trigger_thresholds),
        "label": threshold_param.removesuffix("_threshold"),
        "label_separator": "",
    }


def _base_overwrite(curve_name, telescope, args):
    """Build the curve-specific base overwrite block before threshold insertion."""
    if curve_name not in ("nsb", "proton"):
        raise ValueError(f"Unsupported curve name '{curve_name}'.")

    telescope_changes = {}
    if curve_name == "nsb":
        telescope_changes = {
            "min_photons": {"value": 0},
            "min_photoelectrons": {"value": 0},
        }

    curve_label = "NSB" if curve_name == "nsb" else "proton"
    return {
        "model_version": args["model_version"],
        "model_update": "patch_update",
        "model_version_history": [args["model_version"]],
        "description": f"Tune for {curve_label} telescope trigger scan",
        "changes": {
            telescope: telescope_changes,
            f"OBS-{args['site']}": {"nsb_scaling_factor": {"value": args["nsb_scaling_factor"]}},
        },
    }


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


def _production_grid_configuration(args, curve_definition, curve_label):
    """Build configuration for production-grid generation."""
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
        }
    )
    return configuration


def _generate_curve_submissions(curve_name, curve_definition, args, io_handler):
    """Generate base grid, scan config, and scan grid for one curve."""
    curve_directory = io_handler.get_output_directory(sub_dir=curve_name)

    telescope = args["telescope"]
    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_config_file = curve_directory / "scan_config.yaml"
    scan_grid_file = curve_directory / "scan_grid.ecsv"

    ascii_handler.write_data_to_file(
        _scan_config(curve_name, telescope, args), scan_config_file, sort_keys=False
    )
    generate_job_grid(
        _production_grid_configuration(args, curve_definition, curve_name),
        base_grid_file,
    )
    parameter_scan_generator.expand_job_grid_with_scan(
        base_grid_file,
        scan_config_file,
        scan_grid_file,
    )


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
    """Generate NSB and proton bias-curve scan grids.

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
