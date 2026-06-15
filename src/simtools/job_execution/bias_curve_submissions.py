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

For each threshold value, one overwrite file is generated. The base job grid is
then expanded manually so that each threshold corresponds to one overwrite file.
This avoids the cartesian-product behaviour of the generic scan generator.

The NSB overwrite always sets ``min_photons`` and ``min_photoelectrons`` to zero
and resets ``nsb_scaling_factor`` to 2. The proton overwrite does not touch
these parameters.

No external overwrite templates are used; overwrite YAML files are generated
dynamically from scratch.
"""

import logging
import subprocess
from pathlib import Path

import yaml

from simtools.job_execution import htcondor_script_generator
from simtools.model.site_model import SiteModel
from simtools.production_configuration.job_grid_io import read_job_grid, serialize_job_grid

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


def _run_command(command, working_directory):
    """Run a shell command and raise a readable error on failure."""
    _logger.info(f"Running command: {' '.join(str(argument) for argument in command)}")

    try:
        subprocess.run(command, cwd=working_directory, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed: {' '.join(str(argument) for argument in command)}"
        ) from exc


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


def _build_proton_overwrite(telescopes, threshold, model_version):
    """Build overwrite YAML content for one proton threshold scan point."""
    threshold_value = _format_threshold_value(threshold)

    changes = {}

    for telescope in telescopes:
        threshold_param = _threshold_param_name(telescope)
        changes[telescope] = {
            threshold_param: {
                "version": _PARAMETER_VERSION,
                "value": threshold_value,
            }
        }

    return {
        "model_version": model_version,
        "model_update": "patch_update",
        "model_version_history": [model_version],
        "description": "Tune for proton telescope trigger scan",
        "changes": changes,
    }


def _build_nsb_overwrite(telescopes, threshold, site, model_version):
    """Build overwrite YAML content for one NSB threshold scan point."""
    threshold_value = _format_threshold_value(threshold)

    changes = {}

    for telescope in telescopes:
        threshold_param = _threshold_param_name(telescope)
        changes[telescope] = {
            "min_photons": {
                "version": _PARAMETER_VERSION,
                "value": 0,
            },
            "min_photoelectrons": {
                "version": _PARAMETER_VERSION,
                "value": 0,
            },
            threshold_param: {
                "version": _PARAMETER_VERSION,
                "value": threshold_value,
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


def _build_overwrite_content(curve_name, telescopes, threshold, args):
    """Build overwrite YAML content for one curve and one threshold value."""
    if curve_name == "nsb":
        return _build_nsb_overwrite(
            telescopes=telescopes,
            threshold=threshold,
            site=args["site"],
            model_version=args["model_version"],
        )

    if curve_name == "proton":
        return _build_proton_overwrite(
            telescopes=telescopes,
            threshold=threshold,
            model_version=args["model_version"],
        )

    raise ValueError(f"Unsupported curve name '{curve_name}'.")


def _generate_overwrite_files(curve_name, telescopes, args, curve_directory, label):
    """Write one overwrite YAML file per threshold value.

    Returns
    -------
    list[tuple[Path, str]]
        Pairs of overwrite file path and scan label.
    """
    if len(telescopes) != 1:
        raise ValueError(
            f"Bias-curve overwrite generation expects exactly one telescope; got {len(telescopes)}."
        )

    telescope = telescopes[0]
    threshold_param = _threshold_param_name(telescope)
    threshold_values = _threshold_values_for_telescope(telescope)

    _logger.info(
        f"Generating {curve_name} overwrite files for {telescope} using "
        f"{threshold_param}: {threshold_values}"
    )

    overwrite_dir = curve_directory / "overwrite_files"
    overwrite_dir.mkdir(parents=True, exist_ok=True)

    overwrite_files_and_labels = []

    for threshold in threshold_values:
        threshold_value = _format_threshold_value(threshold)

        content = _build_overwrite_content(
            curve_name=curve_name,
            telescopes=telescopes,
            threshold=threshold_value,
            args=args,
        )

        scan_label = f"{threshold_param}_{threshold_value}"
        overwrite_file = overwrite_dir / f"overwrite_{label}_{scan_label}.yaml"

        with open(overwrite_file, "w", encoding="utf-8") as file_handle:
            yaml.safe_dump(content, file_handle, sort_keys=False)

        overwrite_files_and_labels.append((overwrite_file, scan_label))

    return overwrite_files_and_labels


def _build_scan_grid(base_grid_file, overwrite_files_and_labels, scan_grid_file):
    """Expand a base job grid with one row set per overwrite file."""
    base_rows, metadata = read_job_grid(base_grid_file)

    expanded_rows = []

    for overwrite_file, scan_label in overwrite_files_and_labels:
        for row in base_rows:
            new_row = dict(row)
            new_row["overwrite_model_parameters"] = str(overwrite_file)
            new_row["scan_label"] = scan_label
            expanded_rows.append(new_row)

    serialize_job_grid(expanded_rows, scan_grid_file, metadata=metadata)
    _logger.info(f"Scan grid with {len(expanded_rows)} rows written to '{scan_grid_file}'.")


def _simulation_to_cli_args(args, primary, energy_range, output_file, label):
    """Build CLI command arguments for simtools-production-generate-grid."""
    command = ["simtools-production-generate-grid"]

    for key in _SIMULATION_CLI_ARGS:
        value = args.get(key)
        if value is not None:
            command.extend([f"--{key}", str(value)])

    command.extend(
        [
            "--primary",
            primary,
            "--energy_range",
            energy_range,
            "--label",
            label,
            "--output_file",
            str(output_file),
        ]
    )

    return command


def _build_htcondor_args(label, curve_directory, scan_grid_file, args):
    """Build args for htcondor_script_generator.generate_submission_script."""
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


def _generate_curve_submissions(curve_name, curve_definition, args, output_root):
    """Generate grid, overwrite files, scan grid, and HTCondor scripts for one curve."""
    primary = curve_definition["primary"]
    energy_range = curve_definition["energy_range"]

    curve_directory = output_root / curve_name
    curve_directory.mkdir(parents=True, exist_ok=True)

    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_grid_file = curve_directory / "scan_grid.ecsv"

    base_label = args.get("label") or "bias_curve"
    curve_label = f"{base_label}_{curve_name}"

    _run_command(
        _simulation_to_cli_args(
            args=args,
            primary=primary,
            energy_range=energy_range,
            output_file=base_grid_file,
            label=curve_label,
        ),
        output_root,
    )

    overwrite_files_and_labels = _generate_overwrite_files(
        curve_name=curve_name,
        telescopes=args["telescopes"],
        args=args,
        curve_directory=curve_directory,
        label=curve_label,
    )

    _build_scan_grid(
        base_grid_file=base_grid_file,
        overwrite_files_and_labels=overwrite_files_and_labels,
        scan_grid_file=scan_grid_file,
    )

    htcondor_script_generator.generate_submission_script(
        _build_htcondor_args(
            label=curve_label,
            curve_directory=curve_directory,
            scan_grid_file=scan_grid_file,
            args=args,
        )
    )


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
