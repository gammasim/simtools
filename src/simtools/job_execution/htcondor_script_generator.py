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

from simtools.production_configuration.job_grid_io import JOB_GRID_SCHEMA, read_job_grid
from simtools.utils.value_conversion import format_quantity

_logger = logging.getLogger(__name__)
_PARAMS_FIELDS = [
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "energy_min",
    "energy_max",
    "cores_per_shower",
    "core_scatter_max",
    "view_cone_min",
    "view_cone_max",
    "showers_per_run",
    "model_version",
    "array_layout_name",
    "corsika_le_interaction",
    "corsika_he_interaction",
    "run_number",
    "grid_output_path",
]
_OPTIONAL_QUEUE_FIELDS = (
    "corsika_hadronic_transition_energy",
    "overwrite_model_parameters",
    "scan_label",
    "telescope",
)

_PARAMS_JOB_SPEC_FIELDS = {field: field for field in _PARAMS_FIELDS}

_PARAM_QUANTITY_UNITS = {
    field: JOB_GRID_SCHEMA.column_units[field]
    for field in (*_PARAMS_FIELDS, *_OPTIONAL_QUEUE_FIELDS)
    if field in JOB_GRID_SCHEMA.column_units
}

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


def _format_param_value(value, field_name):
    """Format a value or Quantity for params file output."""
    if value is None:
        if field_name in _OPTIONAL_QUEUE_FIELDS:
            return ""
        raise ValueError(f"Missing required value for field '{field_name}'.")

    if field_name in ("grid_output_path", "scan_label"):
        return _sanitize_label_for_params(value)

    if field_name == "cores_per_shower":
        return f"{int(value)}"

    if field_name in _PARAM_QUANTITY_UNITS:
        return format_quantity(value, _PARAM_QUANTITY_UNITS[field_name])

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


def _write_params_file(params_file_path, label_job_specs, params_fields):
    """Write parameter file consumed by HTCondor queue-from syntax."""
    with open(params_file_path, "w", encoding="utf-8") as params_file_handle:
        for job_spec in label_job_specs:
            row = [
                _format_param_value(job_spec[_PARAMS_JOB_SPEC_FIELDS[field]], field)
                for field in _PARAMS_FIELDS
            ]

            for field in _OPTIONAL_QUEUE_FIELDS:
                if field in params_fields:
                    value = _format_param_value(job_spec.get(field), field)
                    row.append(
                        f'"{value}"'
                        if isinstance(value, str) and (not value or re.search(r"\s", value))
                        else value
                    )

            params_file_handle.write(" ".join(row) + "\n")


def _format_multiline_command(command_parts):
    """Return a shell command body with line continuations."""
    command_lines = []
    for index, part in enumerate(command_parts):
        line_end = " \\" if index < len(command_parts) - 1 else ""
        command_lines.append(f"    {part}{line_end}")
    return command_lines


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
    executable, apptainer_image, priority, params_file_name, htcondor_dirs, params_fields
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
    params_fields : list
        List of parameter fields to include in the submit file.

    Returns
    -------
    str
        HTCondor submit file content.
    """
    arguments_string = "env.txt " + " ".join(f"$({field})" for field in params_fields)
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
    params_fields : list, optional
        List of parameter fields to include in the submit script.

    Returns
    -------
    str
        HTCondor submit script content.
    """
    params_fields = params_fields or _PARAMS_FIELDS
    bash_indices = {field: f"${{{index}}}" for index, field in enumerate(params_fields, start=2)}

    label = args_dict["label"] if args_dict["label"] else "simulate-prod"

    energy_unit = _PARAM_QUANTITY_UNITS["energy_min"]
    core_scatter_unit = _PARAM_QUANTITY_UNITS["core_scatter_max"]
    view_cone_unit = _PARAM_QUANTITY_UNITS["view_cone_min"]
    energy_range_string = (
        f'"{bash_indices["energy_min"]} {energy_unit} {bash_indices["energy_max"]} {energy_unit}"'
    )
    core_scatter_string = (
        f'"{bash_indices["cores_per_shower"]} {bash_indices["core_scatter_max"]} '
        f'{core_scatter_unit}"'
    )
    view_cone_string = (
        f'"{bash_indices["view_cone_min"]} {view_cone_unit} '
        f'{bash_indices["view_cone_max"]} {view_cone_unit}"'
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
        overwrite_parameters_argument = '"${overwrite_model_parameters_args[@]}"'

    telescope_block = ""
    telescope_argument = ""
    if "telescope" in params_fields:
        telescope_block = (
            f'telescope="{bash_indices["telescope"]}"\n'
            "telescope_args=()\n"
            'if [ -n "$telescope" ]; then\n'
            '    telescope_args+=(--telescope "$telescope")\n'
            "fi\n"
        )
        telescope_argument = '"${telescope_args[@]}"'

    job_label = (
        f"{label}_{bash_indices['corsika_he_interaction']}_"
        f"{bash_indices['corsika_le_interaction']}_"
        f"{bash_indices['energy_min']}{energy_unit}-"
        f"{bash_indices['energy_max']}{energy_unit}"
    )

    command_parts = [
        '--label "$job_label"',
        f"--simulation_software {args_dict['simulation_software']}",
        f"--site {args_dict['site']}",
        f"--log_level {args_dict['log_level']}",
    ]

    for field in (
        "model_version",
        "array_layout_name",
        "primary",
        "azimuth_angle",
        "zenith_angle",
        "showers_per_run",
        "corsika_le_interaction",
        "corsika_he_interaction",
        "run_number",
    ):
        command_parts.append(f'--{field} "{bash_indices[field]}"')

    if "corsika_hadronic_transition_energy" in params_fields:
        command_parts.append(
            "--corsika_hadronic_transition_energy "
            f'"{bash_indices["corsika_hadronic_transition_energy"]}"'
        )

    command_parts.extend(
        [
            f"--energy_range {energy_range_string}",
            f"--core_scatter {core_scatter_string}",
            f"--view_cone {view_cone_string}",
            f"--run_number_offset {args_dict.get('run_number_offset', 0)}",
            "--save_reduced_event_lists",
        ]
    )
    if args_dict.get("save_file_lists"):
        command_parts.append("--save_file_lists")
    if telescope_argument:
        command_parts.append(telescope_argument.rstrip())
    if overwrite_parameters_argument:
        command_parts.append(overwrite_parameters_argument.rstrip())
    command_parts.extend(
        [
            "--output_path /tmp/simtools-output",
            f'--grid_output_path "{bash_indices["grid_output_path"]}"',
        ]
    )

    script_lines = [
        "#!/usr/bin/env bash",
        "",
        "# Load environment variables (for DB access)",
        'set -a; source "$1"',
        f'job_label="{job_label}"',
        scan_label_block.rstrip(),
        overwrite_parameters_block.rstrip(),
        telescope_block.rstrip(),
        "",
        "simtools-simulate-prod \\",
        *_format_multiline_command(command_parts),
    ]
    script_lines.append("")

    return "\n".join(line for line in script_lines if line != "")


def _validate_job_grid_metadata(job_grid_metadata):
    """Validate required job-grid metadata."""
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


def _add_optional_job_spec_field(job_spec, field_name, value):
    """Add an optional field to a job spec when it has a usable value."""
    if value not in (None, ""):
        job_spec[field_name] = value


def _build_job_spec(row, label, base_pack_dir, args_dict):
    """Build one job spec from one job-grid row and one image label."""
    job_spec = {
        "image_label": str(label),
        **row,
        "grid_output_path": f"{base_pack_dir}/{label!s}",
    }

    _add_optional_job_spec_field(
        job_spec,
        "telescope",
        row.get("telescope") or args_dict.get("telescope"),
    )
    _add_optional_job_spec_field(job_spec, "scan_label", row.get("scan_label"))
    _add_optional_job_spec_field(
        job_spec,
        "overwrite_model_parameters",
        row.get("overwrite_model_parameters"),
    )

    return job_spec


def build_job_specs(args_dict, image_labels):
    """Build backend-agnostic job specs from comparison and production grids."""
    base_pack_dir = args_dict.get("simulation_output") or "simtools-output"
    normalized_rows, job_grid_metadata = read_job_grid(args_dict["job_grid_file"])

    _validate_job_grid_metadata(job_grid_metadata)

    job_specs = [
        _build_job_spec(row, label, base_pack_dir, args_dict)
        for label in image_labels
        for row in normalized_rows
    ]
    return job_specs, job_grid_metadata
