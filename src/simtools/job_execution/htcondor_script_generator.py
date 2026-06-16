"""
HTCondor script generator for simulation production.

Generates three files in the specified output directory:
- '.condor': HTCondor submit file with queue-from syntax.
- 'params.txt': Parameters file consumed by the submit file.
- 'submit.sh': Executable script that runs the simulation command with parameters

"""

import logging
import re
from pathlib import Path

import astropy.units as u

from simtools.production_configuration.job_grid_io import read_job_grid

_logger = logging.getLogger(__name__)

_PARAMS_FIELDS = [
    "apptainer_label",
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "energy_min_value",
    "energy_min_unit",
    "energy_max_value",
    "energy_max_unit",
    "cores_per_shower",
    "core_scatter_max_value",
    "core_scatter_max_unit",
    "view_cone_min_value",
    "view_cone_min_unit",
    "view_cone_max_value",
    "view_cone_max_unit",
    "showers_per_run",
    "model_version",
    "array_layout_name",
    "corsika_le_interaction",
    "corsika_he_interaction",
    "run_number",
    "pack_for_grid_register",
]

_OPTIONAL_QUEUE_FIELDS = ("overwrite_model_parameters", "scan_label")

_REQUIRED_JOB_GRID_METADATA = ("site", "simulation_software")


def _resolve_apptainer_images(apptainer_image_arg):
    """
    Resolve and validate apptainer image configuration.

    Parameters
    ----------
    apptainer_image_arg: str or dict
        Either a string path to a single apptainer image or a dictionary
        mapping labels to image paths.

    Returns
    -------
    dict
        Dictionary mapping labels to resolved Path objects for apptainer images.
    """
    if not apptainer_image_arg:
        raise ValueError("Missing required apptainer_image path.")

    if isinstance(apptainer_image_arg, str):
        apptainer_image_arg = {"default": apptainer_image_arg}

    if not isinstance(apptainer_image_arg, dict):
        raise TypeError("apptainer_image must be a string path or a label-to-path dictionary.")

    resolved = {}
    for label, path in apptainer_image_arg.items():
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Apptainer image file not found: {image_path}")
        resolved[str(label)] = image_path

    if not resolved:
        raise ValueError("At least one apptainer image label/path must be configured.")

    return resolved


def _format_quantity(value, default_unit=None, convert_to=None):
    """Format scalar or Quantity value."""
    if isinstance(value, u.Quantity):
        if convert_to is not None:
            value = value.to(convert_to)
        return f"{value.value}", f"{value.unit}"

    return f"{value}", str(default_unit) if default_unit else None


def _format_param_value(value, field_name):
    """Format a value or Quantity for params file output."""
    if value is None:
        if field_name in _OPTIONAL_QUEUE_FIELDS:
            return ""
        raise ValueError(f"Missing required value for field '{field_name}'.")

    if field_name in ("apptainer_label", "pack_for_grid_register", "overwrite_model_parameters"):
        return _sanitize_label_for_params(value)

    if field_name == "cores_per_shower":
        return f"{int(value)}"

    quantity_fields = {
        "energy_min_value": (u.GeV, None),
        "energy_max_value": (u.GeV, None),
        "core_scatter_max_value": (u.m, u.m),
        "view_cone_min_value": (u.deg, u.deg),
        "view_cone_max_value": (u.deg, u.deg),
    }
    if field_name in quantity_fields:
        default_unit, convert_to = quantity_fields[field_name]
        return _format_quantity(value, default_unit=default_unit, convert_to=convert_to)

    if field_name in ("azimuth_angle", "zenith_angle"):
        if isinstance(value, u.Quantity):
            value = value.to(u.deg).value
        return f"{value}"

    return f"{value}"


def _sanitize_label_for_filename(label):
    """Sanitize image labels for use in file names."""
    label_string = str(label).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in ["-", "_", "."] else "_" for ch in label_string)


def _sanitize_label_for_params(label):
    """Sanitize image labels for whitespace-separated params files."""
    return re.sub(r"\s+", "_", str(label).strip())


def _group_job_specs_by_label(job_specs):
    """Group job specs by apptainer image label."""
    grouped = {}
    for job_spec in job_specs:
        label = job_spec["image_label"]
        grouped.setdefault(label, []).append(job_spec)
    return grouped


def _write_params_file(params_file_path, label_job_specs, params_fields=None):
    """Write parameter file consumed by HTCondor queue-from syntax."""
    params_fields = params_fields or _PARAMS_FIELDS
    with open(params_file_path, "w", encoding="utf-8") as params_file_handle:
        for job_spec in label_job_specs:
            energy_min_value, energy_min_unit = _format_param_value(
                job_spec["energy_min"], "energy_min_value"
            )
            energy_max_value, energy_max_unit = _format_param_value(
                job_spec["energy_max"], "energy_max_value"
            )
            cores_per_shower = _format_param_value(job_spec["cores_per_shower"], "cores_per_shower")
            core_scatter_max_value, core_scatter_max_unit = _format_param_value(
                job_spec["core_scatter_max"], "core_scatter_max_value"
            )
            view_cone_min_value, view_cone_min_unit = _format_param_value(
                job_spec["view_cone_min"], "view_cone_min_value"
            )
            view_cone_max_value, view_cone_max_unit = _format_param_value(
                job_spec["view_cone_max"], "view_cone_max_value"
            )

            row = [
                _format_param_value(job_spec["image_label"], "apptainer_label"),
                _format_param_value(job_spec["primary"], "primary"),
                _format_param_value(job_spec["azimuth_angle"], "azimuth_angle"),
                _format_param_value(job_spec["zenith_angle"], "zenith_angle"),
                energy_min_value,
                energy_min_unit,
                energy_max_value,
                energy_max_unit,
                cores_per_shower,
                core_scatter_max_value,
                core_scatter_max_unit,
                view_cone_min_value,
                view_cone_min_unit,
                view_cone_max_value,
                view_cone_max_unit,
                _format_param_value(job_spec["showers_per_run"], "showers_per_run"),
                _format_param_value(job_spec["model_version"], "model_version"),
                _format_param_value(job_spec["array_layout_name"], "array_layout_name"),
                _format_param_value(job_spec["corsika_le_interaction"], "corsika_le_interaction"),
                _format_param_value(job_spec["corsika_he_interaction"], "corsika_he_interaction"),
                _format_param_value(job_spec["run_number"], "run_number"),
                _format_param_value(job_spec["pack_for_grid_register"], "pack_for_grid_register"),
            ]

            if "overwrite_model_parameters" in params_fields:
                row.append(
                    _format_param_value(
                        job_spec.get("overwrite_model_parameters"), "overwrite_model_parameters"
                    )
                )
            if "scan_label" in params_fields:
                row.append(_format_param_value(job_spec.get("scan_label"), "scan_label"))

            params_file_handle.write(" ".join(row) + "\n")


def generate_submission_script(args_dict):
    """
    Generate the HT Condor submission script.

    Parameters
    ----------
    args_dict: dict
        Arguments dictionary.
    """
    apptainer_images = _resolve_apptainer_images(args_dict["apptainer_image"])
    job_specs, job_grid_metadata = build_job_specs(args_dict, list(apptainer_images.keys()))
    grouped_job_specs = _group_job_specs_by_label(job_specs)
    params_fields = list(_PARAMS_FIELDS)
    for field in _OPTIONAL_QUEUE_FIELDS:
        if any(job_spec.get(field) not in (None, "") for job_spec in job_specs):
            params_fields.append(field)

    work_dir = Path(args_dict["output_path"])
    htcondor_log_path = Path(
        args_dict["htcondor_log_path"]
        if args_dict.get("htcondor_log_path")
        else work_dir / "htcondor_logs"
    )
    htcondor_dirs = {
        "log": htcondor_log_path / "log",
        "error": htcondor_log_path / "error",
        "output": htcondor_log_path / "output",
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    for subdir in htcondor_dirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    submit_file_name = "simulate_prod.submit"
    _logger.info(f"Generating HT Condor submission scripts (path: {work_dir})")
    submit_args = {**job_grid_metadata, **args_dict}

    for label, label_job_specs in grouped_job_specs.items():
        suffix = (
            ""
            if len(grouped_job_specs) == 1 and label == "default"
            else f".{_sanitize_label_for_filename(label)}"
        )
        condor_file_name = f"{submit_file_name}{suffix}.condor"
        params_file_name = f"{submit_file_name}{suffix}.params.txt"

        _write_params_file(
            work_dir / params_file_name, label_job_specs, params_fields=params_fields
        )

        with open(work_dir / condor_file_name, "w", encoding="utf-8") as submit_file_handle:
            submit_file_handle.write(
                _get_submit_file(
                    f"{submit_file_name}.sh",
                    apptainer_images[label],
                    args_dict["priority"],
                    params_file_name,
                    htcondor_dirs=htcondor_dirs,
                    params_fields=params_fields,
                )
            )

    with open(work_dir / f"{submit_file_name}.sh", "w", encoding="utf-8") as submit_script_handle:
        submit_script_handle.write(_get_submit_script(submit_args, params_fields=params_fields))

    Path(work_dir / f"{submit_file_name}.sh").chmod(0o755)


def _get_submit_file(
    executable, apptainer_image, priority, params_file_name, htcondor_dirs, params_fields=None
):
    """
    Return HTCondor submit file.

    Database access variables are passed through the environment file.

    Parameters
    ----------
    executable: str
        Name of the executable script.
    apptainer_image: Path
        Path to the Apptainer image.
    priority: int
        Priority of the job.
    params_file_name: str
        Name of the params file for queue-from submission.
    htcondor_dirs: dict
        Directory mapping with HTCondor files locations. Expected keys are
        ``log``, ``error``, and ``output``.

    Returns
    -------
    str
        HTCondor submit file content.
    """
    params_fields = params_fields or _PARAMS_FIELDS
    arguments_string = "$(process) env.txt " + " ".join(f"$({field})" for field in params_fields)
    queue_string = ",".join(params_fields)

    return f"""universe = container
container_image = {apptainer_image}
transfer_container = false

executable = {executable}
error      = {htcondor_dirs["error"]}/err.$(cluster)_$(process)
output     = {htcondor_dirs["output"]}/out.$(cluster)_$(process)
log        = {htcondor_dirs["log"]}/log.$(cluster)_$(process)

priority = {priority}
arguments = "{arguments_string}"

queue {queue_string} from {params_file_name}
"""


def _get_submit_script(args_dict, params_fields=None):
    """
    Return HTCondor submit script.

    Parameters
    ----------
    args_dict: dict
        Arguments dictionary.

    Returns
    -------
    str
        HTCondor submit script content.
    """
    params_fields = params_fields or _PARAMS_FIELDS

    # Map params fields to bash positional indices ($3, $4, etc.)
    # Indices 1-2 are reserved for: $1=process_id, $2=env_file
    bash_indices = {}
    for i, field in enumerate(params_fields):
        idx = 3 + i
        bash_indices[field] = f"${{{idx}}}"

    label = args_dict["label"] if args_dict["label"] else "simulate-prod"
    run_number_offset = args_dict.get("run_number_offset", 0)

    energy_range_string = (
        f'"{bash_indices["energy_min_value"]} {bash_indices["energy_min_unit"]} '
        f'{bash_indices["energy_max_value"]} {bash_indices["energy_max_unit"]}"'
    )
    core_scatter_string = (
        f'"{bash_indices["cores_per_shower"]} {bash_indices["core_scatter_max_value"]} '
        f'{bash_indices["core_scatter_max_unit"]}"'
    )
    view_cone_string = (
        f'"{bash_indices["view_cone_min_value"]} {bash_indices["view_cone_min_unit"]} '
        f"{bash_indices['view_cone_max_value']} "
        f'{bash_indices["view_cone_max_unit"]}"'
    )
    energy_range_tag = (
        f"erange-{bash_indices['energy_min_value']}{bash_indices['energy_min_unit']}-"
        f"{bash_indices['energy_max_value']}{bash_indices['energy_max_unit']}"
    )

    scan_label_block = ""
    if "scan_label" in params_fields:
        scan_label_block = (
            f'scan_label="{bash_indices["scan_label"]}"\n'
            'if [ -n "$scan_label" ]; then\n'
            '    job_label="${job_label}_${scan_label}"\n'
            "fi\n"
        )

    overwrite_parameters_block = ""
    overwrite_parameters_argument = ""
    if "overwrite_model_parameters" in params_fields:
        overwrite_parameters_block = (
            f'overwrite_model_parameters="{bash_indices["overwrite_model_parameters"]}"\n'
            "overwrite_model_parameters_args=()\n"
            'if [ -n "$overwrite_model_parameters" ]; then\n'
            "    overwrite_model_parameters_args+=(--overwrite_model_parameters "
            '"$overwrite_model_parameters")\n'
            "fi\n"
        )
        overwrite_parameters_argument = '    "${overwrite_model_parameters_args[@]}" \\\n'

    return f"""#!/usr/bin/env bash

# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"
apptainer_label="{bash_indices["apptainer_label"]}"
primary="{bash_indices["primary"]}"
model_version="{bash_indices["model_version"]}"
array_layout_name="{bash_indices["array_layout_name"]}"
corsika_le_interaction="{bash_indices["corsika_le_interaction"]}"
corsika_he_interaction="{bash_indices["corsika_he_interaction"]}"
run_number="{bash_indices["run_number"]}"
pack_for_grid_register="{bash_indices["pack_for_grid_register"]}"
energy_range_tag="{energy_range_tag}"
job_label="{label}_${{corsika_he_interaction}}-${{corsika_le_interaction}}_${{energy_range_tag}}"
{scan_label_block}{overwrite_parameters_block}

simtools-simulate-prod \\
    --simulation_software {args_dict["simulation_software"]} \\
    --label "$job_label" \\
    --model_version "$model_version" \\
    --site {args_dict["site"]} \\
    --array_layout_name "$array_layout_name" \\
    --primary "$primary" \\
    --azimuth_angle "{bash_indices["azimuth_angle"]}" \\
    --zenith_angle "{bash_indices["zenith_angle"]}" \\
    --showers_per_run "{bash_indices["showers_per_run"]}" \\
    --energy_range {energy_range_string} \\
    --core_scatter {core_scatter_string} \\
    --view_cone {view_cone_string} \\
    --corsika_le_interaction "$corsika_le_interaction" \\
    --corsika_he_interaction "$corsika_he_interaction" \\
    --run_number "$run_number" \\
    --run_number_offset {run_number_offset} \\
    --save_reduced_event_lists \\
{overwrite_parameters_argument}    --output_path /tmp/simtools-output \\
    --log_level {args_dict["log_level"]} \\
    --pack_for_grid_register "$pack_for_grid_register"
"""


def build_job_specs(args_dict, image_labels):
    """Build backend-agnostic job specs from comparison and production grids."""
    base_pack_dir = args_dict.get("simulation_output") or "simtools-output"
    normalized_rows, job_grid_metadata = read_job_grid(args_dict["job_grid_file"])

    missing_metadata = [
        key
        for key in _REQUIRED_JOB_GRID_METADATA
        if key not in job_grid_metadata or job_grid_metadata.get(key) in (None, "")
    ]
    if missing_metadata:
        missing_keys = ", ".join(missing_metadata)
        raise ValueError(
            "Job grid metadata is missing required field(s): "
            f"{missing_keys}. Regenerate the job grid with "
            "simtools-production-generate-grid so metadata includes these values."
        )

    job_specs = []
    for label in image_labels:
        for row in normalized_rows:
            job_spec = {
                "image_label": str(label),
                **row,
                "pack_for_grid_register": f"{base_pack_dir}/{label!s}",
            }
            if row.get("scan_label") not in (None, ""):
                job_spec["scan_label"] = row["scan_label"]
            if row.get("overwrite_model_parameters") not in (None, ""):
                job_spec["overwrite_model_parameters"] = row["overwrite_model_parameters"]
            job_specs.append(job_spec)
    return job_specs, job_grid_metadata
