"""
HTCondor script generator for simulation production.

Generates three files in the specified output directory:
- '.condor': HTCondor submit file with queue-from syntax.
- 'params.txt': Parameters file consumed by the submit file.
- 'submit.sh': Executable script that runs the simulation command with parameters

"""

import ast
import itertools
import logging
from pathlib import Path

import astropy.units as u

import simtools.version as simtools_version

_logger = logging.getLogger(__name__)

_GRID_AXES = [
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "model_version",
    "corsika_le_interaction",
    "corsika_he_interaction",
]

_PARAMS_FIELDS = [
    "apptainer_label",
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "energy_min_value",
    "energy_min_unit",
    "energy_max_value",
    "energy_max_unit",
    "model_version",
    "array_layout_name",
    "corsika_le_interaction",
    "corsika_he_interaction",
    "run_number",
    "pack_for_grid_register",
]


def _normalize_to_list(value):
    """Normalize scalar values to lists of length one."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_grid_axes(args_dict):
    """Return normalized grid axes for cartesian product expansion."""
    return {
        axis: _normalize_to_list(args_dict[axis])
        if axis in args_dict and args_dict[axis] is not None
        else [None]
        for axis in _GRID_AXES
    }


def _normalize_energy_ranges(energy_range):
    """Normalize energy range argument to a list of (e_min, e_max) pairs."""
    if isinstance(energy_range, tuple) and len(energy_range) == 2:
        return [energy_range]

    if isinstance(energy_range, list):
        if len(energy_range) == 2 and all(hasattr(item, "to") for item in energy_range):
            return [(energy_range[0], energy_range[1])]
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in energy_range):
            return [tuple(item) for item in energy_range]

    raise ValueError(
        "energy_range must be one pair (e_min, e_max) or a list of (e_min, e_max) pairs."
    )


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
    if apptainer_image_arg is None:
        raise ValueError("Missing required apptainer_image path.")

    if isinstance(apptainer_image_arg, str):
        if not apptainer_image_arg.strip():
            raise ValueError("Missing required apptainer_image path.")
        image_path = Path(apptainer_image_arg)
        if not image_path.is_file():
            raise FileNotFoundError(f"Apptainer image file not found: {image_path}")
        return {"default": image_path}

    if isinstance(apptainer_image_arg, dict):
        resolved = {}
        for label, path in apptainer_image_arg.items():
            image_path = Path(path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Apptainer image file not found: {image_path}")
            resolved[str(label)] = image_path
        if not resolved:
            raise ValueError("At least one apptainer image label/path must be configured.")
        return resolved

    raise ValueError("apptainer_image must be either a string path or a label-to-path dictionary.")


def _format_param_value(value, field_name):
    """
    Format a value for params file output.

    For energy fields, returns (value_str, unit_str) tuple.
    For other fields, returns value_str.

    Parameters
    ----------
    value : any
        Value to format.
    field_name : str
        Field name to determine formatting and default units.

    Returns
    -------
    str or tuple
        Formatted value(s).
    """
    if value is None:
        raise ValueError(f"Missing required value for field '{field_name}'.")

    if field_name in ("energy_min_value", "energy_max_value"):
        if isinstance(value, u.Quantity):
            return f"{value.value}", f"{value.unit}"
        return f"{value}", str(u.GeV)

    if field_name in ("azimuth_angle", "zenith_angle"):
        if isinstance(value, u.Quantity):
            return f"{value.to(u.deg).value}"
        return f"{value}"

    return f"{value}"


def _sanitize_label_for_filename(label):
    """Sanitize image labels for use in file names."""
    label_string = str(label).strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in ["-", "_", "."] else "_" for ch in label_string)


def _resolve_array_layout_name(array_layout_name, model_version):
    """Resolve array layout configuration for a specific model version."""
    if isinstance(array_layout_name, list) and len(array_layout_name) == 1:
        array_layout_name = array_layout_name[0]

    # Configurator/_fill_config stringifies dict values when rebuilding argparse arguments.
    if isinstance(array_layout_name, str) and array_layout_name.strip().startswith("{"):
        try:
            parsed_layout = ast.literal_eval(array_layout_name)
            if isinstance(parsed_layout, dict):
                array_layout_name = parsed_layout
        except (SyntaxError, ValueError):
            return array_layout_name

    if not isinstance(array_layout_name, dict) or list(array_layout_name) != ["by_version"]:
        return array_layout_name

    resolved = simtools_version.resolve_by_version(
        {"array_layout_name": array_layout_name}, model_version
    )
    return resolved["array_layout_name"]


def _build_job_specs(args_dict, image_labels):
    """Build backend-agnostic job specs from comparison and production grids."""
    grid_axes = _normalize_grid_axes(args_dict)
    energy_ranges = _normalize_energy_ranges(args_dict["energy_range"])
    base_pack_dir = args_dict.get("simulation_output") or "simtools-output"

    combinations = list(
        itertools.product(
            grid_axes["primary"],
            grid_axes["azimuth_angle"],
            grid_axes["zenith_angle"],
            grid_axes["model_version"],
            grid_axes["corsika_le_interaction"],
            grid_axes["corsika_he_interaction"],
            energy_ranges,
        )
    )

    number_of_runs = args_dict.get("number_of_runs", 1)
    run_number = int(args_dict.get("run_number") or 1)

    job_specs = []
    for label in image_labels:
        row_index = 0
        for (
            primary,
            azimuth,
            zenith,
            model_version,
            corsika_le,
            corsika_he,
            energy_range_pair,
        ) in combinations:
            for _ in range(number_of_runs):
                job_specs.append(
                    {
                        "apptainer_label": str(label),
                        "primary": primary,
                        "azimuth_angle": azimuth,
                        "zenith_angle": zenith,
                        "model_version": model_version,
                        "array_layout_name": args_dict.get("array_layout_name"),
                        "corsika_le_interaction": corsika_le,
                        "corsika_he_interaction": corsika_he,
                        "energy_min": energy_range_pair[0],
                        "energy_max": energy_range_pair[1],
                        "pack_for_grid_register": f"{base_pack_dir}/{label}",
                        "run_number": run_number + row_index,
                    }
                )
                row_index += 1
    return job_specs


def _group_job_specs_by_label(job_specs):
    """Group job specs by apptainer image label."""
    grouped = {}
    for job_spec in job_specs:
        label = job_spec["apptainer_label"]
        grouped.setdefault(label, []).append(job_spec)
    return grouped


def _write_params_file(params_file_path, label_job_specs):
    """Write parameter file consumed by HTCondor queue-from syntax."""
    with open(params_file_path, "w", encoding="utf-8") as params_file_handle:
        for job_spec in label_job_specs:
            array_layout_name = _resolve_array_layout_name(
                job_spec["array_layout_name"], job_spec["model_version"]
            )

            energy_min_value, energy_min_unit = _format_param_value(
                job_spec["energy_min"], "energy_min_value"
            )
            energy_max_value, energy_max_unit = _format_param_value(
                job_spec["energy_max"], "energy_max_value"
            )

            row = [
                _format_param_value(job_spec["apptainer_label"], "apptainer_label"),
                _format_param_value(job_spec["primary"], "primary"),
                _format_param_value(job_spec["azimuth_angle"], "azimuth_angle"),
                _format_param_value(job_spec["zenith_angle"], "zenith_angle"),
                energy_min_value,
                energy_min_unit,
                energy_max_value,
                energy_max_unit,
                _format_param_value(job_spec["model_version"], "model_version"),
                _format_param_value(array_layout_name, "array_layout_name"),
                _format_param_value(job_spec["corsika_le_interaction"], "corsika_le_interaction"),
                _format_param_value(job_spec["corsika_he_interaction"], "corsika_he_interaction"),
                _format_param_value(job_spec["run_number"], "run_number"),
                job_spec["pack_for_grid_register"],
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
    job_specs = _build_job_specs(args_dict, list(apptainer_images.keys()))
    grouped_job_specs = _group_job_specs_by_label(job_specs)

    work_dir = Path(args_dict["output_path"])
    htcondor_log_path = Path(
        args_dict["htcondor_log_path"]
        if args_dict.get("htcondor_log_path")
        else work_dir / "htcondor_logs"
    )
    log_dir = htcondor_log_path / "log"
    error_dir = htcondor_log_path / "error"
    output_dir = htcondor_log_path / "output"
    work_dir.mkdir(parents=True, exist_ok=True)
    for subdir in (log_dir, error_dir, output_dir):
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
                    log_dir=log_dir,
                    error_dir=error_dir,
                    output_dir=output_dir,
                )
            )

    with open(work_dir / f"{submit_file_name}.sh", "w", encoding="utf-8") as submit_script_handle:
        submit_script_handle.write(_get_submit_script(args_dict))

    Path(work_dir / f"{submit_file_name}.sh").chmod(0o755)


def _get_submit_file(
    executable, apptainer_image, priority, params_file_name, log_dir, error_dir, output_dir
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
    log_dir: Path
        Directory for HTCondor log files.
    error_dir: Path
        Directory for HTCondor error files.
    output_dir: Path
        Directory for HTCondor output files.

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
error      = {error_dir}/err.$(cluster)_$(process)
output     = {output_dir}/out.$(cluster)_$(process)
log        = {log_dir}/log.$(cluster)_$(process)

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
    core_scatter_string = f'"{core_scatter[0]} {core_scatter[1].to(u.m).value} m"'
    view_cone = args_dict["view_cone"]
    view_cone_string = f'"{view_cone[0].to(u.deg)} {view_cone[1].to(u.deg)}"'

    label = args_dict["label"] if args_dict["label"] else "simulate-prod"
    run_number_offset = args_dict["run_number_offset"] or 1

    azimuth_angle_idx = bash_indices["azimuth_angle"]
    zenith_angle_idx = bash_indices["zenith_angle"]
    energy_min_value_idx = bash_indices["energy_min_value"]
    energy_min_unit_idx = bash_indices["energy_min_unit"]
    energy_max_value_idx = bash_indices["energy_max_value"]
    energy_max_unit_idx = bash_indices["energy_max_unit"]
    model_version_idx = bash_indices["model_version"]
    array_layout_name_idx = bash_indices["array_layout_name"]
    corsika_le_interaction_idx = bash_indices["corsika_le_interaction"]
    corsika_he_interaction_idx = bash_indices["corsika_he_interaction"]
    run_number_idx = bash_indices["run_number"]
    pack_for_grid_register_idx = bash_indices["pack_for_grid_register"]

    energy_range_string = (
        f'"{energy_min_value_idx} {energy_min_unit_idx} '
        f'{energy_max_value_idx} {energy_max_unit_idx}"'
    )
    energy_range_tag = (
        f"erange-{energy_min_value_idx}{energy_min_unit_idx}-"
        f"{energy_max_value_idx}{energy_max_unit_idx}"
    )

    return f"""#!/usr/bin/env bash

# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"
apptainer_label="{bash_indices["apptainer_label"]}"
primary="{bash_indices["primary"]}"
model_version="{model_version_idx}"
array_layout_name="{array_layout_name_idx}"
corsika_le_interaction="{corsika_le_interaction_idx}"
corsika_he_interaction="{corsika_he_interaction_idx}"
run_number="{run_number_idx}"
pack_for_grid_register="{pack_for_grid_register_idx}"
energy_range_tag="{energy_range_tag}"
job_label="{label}_${{corsika_he_interaction}}-${{corsika_le_interaction}}_${{energy_range_tag}}"

simtools-simulate-prod \\
    --simulation_software {args_dict["simulation_software"]} \\
    --label "$job_label" \\
    --model_version "$model_version" \\
    --site {args_dict["site"]} \\
    --array_layout_name "$array_layout_name" \\
    --primary "$primary" \\
    --azimuth_angle "{azimuth_angle_idx}" \\
    --zenith_angle "{zenith_angle_idx}" \\
    --nshow {args_dict["nshow"]} \\
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
