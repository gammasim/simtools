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

from simtools.layout.array_layout_utils import resolve_array_layout_name
from simtools.production_configuration.build_grid import build_simulation_jobs

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
    "core_scatter_max_value",
    "core_scatter_max_unit",
    "view_cone_max_value",
    "view_cone_max_unit",
    "nshow",
    "model_version",
    "array_layout_name",
    "corsika_le_interaction",
    "corsika_he_interaction",
    "run_number",
    "pack_for_grid_register",
]


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
        raise ValueError(f"Missing required value for field '{field_name}'.")

    if field_name in ("apptainer_label", "pack_for_grid_register"):
        return _sanitize_label_for_params(value)

    if field_name in ("energy_min_value", "energy_max_value"):
        return _format_quantity(value, default_unit=u.GeV)

    if field_name == "core_scatter_max_value":
        return _format_quantity(
            value,
            default_unit=u.m,
            convert_to=u.m,
        )

    if field_name == "view_cone_max_value":
        return _format_quantity(
            value,
            default_unit=u.deg,
            convert_to=u.deg,
        )

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


def _write_params_file(params_file_path, label_job_specs):
    """Write parameter file consumed by HTCondor queue-from syntax."""
    with open(params_file_path, "w", encoding="utf-8") as params_file_handle:
        for job_spec in label_job_specs:
            array_layout_name = resolve_array_layout_name(
                job_spec["array_layout_name"], job_spec["model_version"]
            )

            energy_min_value, energy_min_unit = _format_param_value(
                job_spec["energy_min"], "energy_min_value"
            )
            energy_max_value, energy_max_unit = _format_param_value(
                job_spec["energy_max"], "energy_max_value"
            )
            core_scatter_max_value, core_scatter_max_unit = _format_param_value(
                job_spec["core_scatter_max"], "core_scatter_max_value"
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
                core_scatter_max_value,
                core_scatter_max_unit,
                view_cone_max_value,
                view_cone_max_unit,
                _format_param_value(job_spec["nshow"], "nshow"),
                _format_param_value(job_spec["model_version"], "model_version"),
                _format_param_value(array_layout_name, "array_layout_name"),
                _format_param_value(job_spec["corsika_le_interaction"], "corsika_le_interaction"),
                _format_param_value(job_spec["corsika_he_interaction"], "corsika_he_interaction"),
                _format_param_value(job_spec["run_number"], "run_number"),
                _format_param_value(job_spec["pack_for_grid_register"], "pack_for_grid_register"),
            ]
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
    job_specs = build_job_specs(args_dict, list(apptainer_images.keys()))
    grouped_job_specs = _group_job_specs_by_label(job_specs)

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

    for label, label_job_specs in grouped_job_specs.items():
        suffix = (
            ""
            if len(grouped_job_specs) == 1 and label == "default"
            else f".{_sanitize_label_for_filename(label)}"
        )
        condor_file_name = f"{submit_file_name}{suffix}.condor"
        params_file_name = f"{submit_file_name}{suffix}.params.txt"

        _write_params_file(work_dir / params_file_name, label_job_specs)

        with open(work_dir / condor_file_name, "w", encoding="utf-8") as submit_file_handle:
            submit_file_handle.write(
                _get_submit_file(
                    f"{submit_file_name}.sh",
                    apptainer_images[label],
                    args_dict["priority"],
                    params_file_name,
                    htcondor_dirs=htcondor_dirs,
                )
            )

    with open(work_dir / f"{submit_file_name}.sh", "w", encoding="utf-8") as submit_script_handle:
        submit_script_handle.write(_get_submit_script(args_dict))

    Path(work_dir / f"{submit_file_name}.sh").chmod(0o755)


def _get_submit_file(executable, apptainer_image, priority, params_file_name, htcondor_dirs):
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
    arguments_string = "$(process) env.txt " + " ".join(f"$({field})" for field in _PARAMS_FIELDS)
    queue_string = ",".join(_PARAMS_FIELDS)

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


def _get_submit_script(args_dict):
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
    # Map _PARAMS_FIELDS to bash positional indices ($3, $4, etc.)
    # Indices 1-2 are reserved for: $1=process_id, $2=env_file
    bash_indices = {}
    for i, field in enumerate(_PARAMS_FIELDS):
        idx = 3 + i
        bash_indices[field] = f"${{{idx}}}"

    core_scatter = args_dict["core_scatter"]
    n_core_scatter = core_scatter[0]
    view_cone = args_dict["view_cone"]
    view_cone_min = view_cone[0].to(u.deg).value

    label = args_dict["label"] if args_dict["label"] else "simulate-prod"
    run_number_offset_arg = args_dict["run_number_offset"]
    run_number_offset = 0 if run_number_offset_arg is None else run_number_offset_arg

    energy_range_string = (
        f'"{bash_indices["energy_min_value"]} {bash_indices["energy_min_unit"]} '
        f'{bash_indices["energy_max_value"]} {bash_indices["energy_max_unit"]}"'
    )
    core_scatter_string = (
        f'"{n_core_scatter} {bash_indices["core_scatter_max_value"]} '
        f'{bash_indices["core_scatter_max_unit"]}"'
    )
    view_cone_string = (
        f'"{view_cone_min} deg {bash_indices["view_cone_max_value"]} '
        f'{bash_indices["view_cone_max_unit"]}"'
    )
    energy_range_tag = (
        f"erange-{bash_indices['energy_min_value']}{bash_indices['energy_min_unit']}-"
        f"{bash_indices['energy_max_value']}{bash_indices['energy_max_unit']}"
    )

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

simtools-simulate-prod \\
    --simulation_software {args_dict["simulation_software"]} \\
    --label "$job_label" \\
    --model_version "$model_version" \\
    --site {args_dict["site"]} \\
    --array_layout_name "$array_layout_name" \\
    --primary "$primary" \\
    --azimuth_angle "{bash_indices["azimuth_angle"]}" \\
    --zenith_angle "{bash_indices["zenith_angle"]}" \\
    --nshow "{bash_indices["nshow"]}" \\
    --energy_range {energy_range_string} \\
    --core_scatter {core_scatter_string} \\
    --view_cone {view_cone_string} \\
    --corsika_le_interaction "$corsika_le_interaction" \\
    --corsika_he_interaction "$corsika_he_interaction" \\
    --run_number "$run_number" \\
    --run_number_offset {run_number_offset} \\
    --save_reduced_event_lists \\
    --output_path /tmp/simtools-output \\
    --log_level {args_dict["log_level"]} \\
    --pack_for_grid_register "$pack_for_grid_register"
"""


def build_job_specs(args_dict, image_labels):
    """Build backend-agnostic job specs from comparison and production grids."""
    base_pack_dir = args_dict.get("simulation_output") or "simtools-output"
    normalized_rows = build_simulation_jobs(args_dict)

    job_specs = []
    for label in image_labels:
        for row in normalized_rows:
            job_specs.append(
                {
                    "image_label": str(label),
                    **row,
                    "pack_for_grid_register": f"{base_pack_dir}/{label!s}",
                }
            )
    return job_specs
