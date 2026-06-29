r"""Generate scan grids for NSB (gamma) and proton bias curves.

This module generates two independent bias-curve scan-grid workflows:

- NSB curve:
  Uses gamma primary, fixed low-energy range, and NSB-specific model overwrites.
- Proton curve:
  Uses proton primary, fixed proton energy range, and proton trigger overwrites.

For each curve, this module writes a scan configuration and a simtools workflow
configuration, then executes only the backend-neutral grid-generation steps through
``simtools_runner.run_applications``.

The NSB overwrite always sets ``min_photons`` and ``min_photoelectrons`` to zero.
Both curves set ``nsb_scaling_factor`` because night-sky background affects the
trigger response in both simulations.
"""

import logging

from simtools.io import ascii_handler
from simtools.io.io_handler import IOHandler
from simtools.model.site_model import SiteModel
from simtools.runners import simtools_runner
from simtools.utils import names

_logger = logging.getLogger(__name__)

_PARAMETER_VERSION = "2.0.0"

_NSB_ENERGY_RANGE = "20 MeV 25 MeV"
_PROTON_ENERGY_RANGE = "2 GeV 2000 GeV"
_NSB_SCALING_FACTOR = 2

_ASUM_THRESHOLDS = [220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360]
_DSUM_THRESHOLDS = [22, 23, 24, 25, 26, 27, 28, 29, 30]

_TELESCOPE_THRESHOLD_PARAM = {
    "LST": "asum_threshold",
    "MST": "dsum_threshold",
}

_CURVE_DEFINITIONS = {
    "nsb": {"primary": "gamma", "energy_range": _NSB_ENERGY_RANGE},
    "proton": {"primary": "proton", "energy_range": _PROTON_ENERGY_RANGE},
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


def _resolve_telescope_from_layout(args):
    """Resolve the telescope name from a single-telescope array layout."""
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

    raw_telescope = layout_elements[0]

    try:
        telescope = names.validate_array_element_name(str(raw_telescope))
    except ValueError as exc:
        raise ValueError(
            f"Array layout '{args['array_layout_name']}' resolved to invalid telescope "
            f"'{raw_telescope}'."
        ) from exc

    return telescope


def _threshold_param_name(telescope):
    """Return the trigger-threshold parameter name for a telescope."""
    if not telescope:
        raise ValueError("Cannot determine threshold parameter for empty telescope name.")

    # Extract telescope type (e.g., "LSTN-01" -> "LST", "MSTS-05" -> "MST")
    telescope_type = telescope.split("-")[0][:3].upper()

    try:
        return _TELESCOPE_THRESHOLD_PARAM[telescope_type]
    except KeyError as exc:
        raise ValueError(
            f"Cannot determine threshold parameter for telescope '{telescope}'. "
            f"Supported telescope types: {list(_TELESCOPE_THRESHOLD_PARAM.keys())}."
        ) from exc


def _threshold_values_for_telescope(telescope):
    """Return the threshold scan values for a telescope."""
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


def _nsb_scaling_change():
    """Return the NSB scaling model-parameter change."""
    return {
        "nsb_scaling_factor": {
            "version": _PARAMETER_VERSION,
            "value": _NSB_SCALING_FACTOR,
        }
    }


def _base_proton_overwrite(telescope, site, model_version):
    """Build the proton base overwrite block before threshold insertion."""
    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for proton telescope trigger scan",
        "changes": {
            telescope: {},
            f"OBS-{site}": _nsb_scaling_change(),
        },
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
            f"OBS-{site}": _nsb_scaling_change(),
        },
    }


def _base_overwrite(curve_name, telescope, args):
    """Build the curve-specific base overwrite block before threshold insertion."""
    if curve_name == "nsb":
        return _base_nsb_overwrite(telescope, args["site"], args["model_version"])

    if curve_name == "proton":
        return _base_proton_overwrite(telescope, args["site"], args["model_version"])

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
            "parameters": [_parameter_scan_entry(telescope)],
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
        "array_layout_name",
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


def generate_bias_curve_submissions(args):
    """Generate NSB and proton bias-curve scan grids from CLI args.

    Parameters
    ----------
    args : dict
        Application arguments for production-grid and bias-curve scan configuration.
    """
    _validate_required_args(args)

    telescope = _resolve_telescope_from_layout(args)
    generation_args = {**args, "telescope": telescope}

    io_handler = IOHandler()
    io_handler.set_paths(output_path=args.get("output_path"))

    for curve_name, curve_definition in _CURVE_DEFINITIONS.items():
        _generate_curve_submissions(
            curve_name=curve_name,
            curve_definition=curve_definition,
            args=generation_args,
            io_handler=io_handler,
        )
