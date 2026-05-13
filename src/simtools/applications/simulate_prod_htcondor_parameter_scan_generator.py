#!/usr/bin/python3

r"""
Generate HTCondor submission files for parameter scans using overwrite files.

This tool creates multiple simulation jobs with different parameter values by generating
customized overwrite YAML files. For each parameter value, it generates:
- An overwrite YAML file with the specified parameter set to that value
- A submission shell script (.sh) that runs simulate-prod with the overwrite file
- A HTCondor submit file (.condor)

This is useful for systematic parameter studies such as NSB threshold scans, trigger
parameter variations, or any other model parameter sensitivity studies.

Command line arguments
----------------------
output_path (str, required)
    Directory where the HTCondor submission files are written.
apptainer_image (str or dict, required)
    Apptainer image to use for the simulation.
parameter_values (list of float, required)
    List of parameter values to scan (e.g., [10, 20, 30, 50, 100]).
overwrite_template (str, required)
    Path to template overwrite YAML file. The parameter will be set to each value.
parameter_path (str, required)
    Dot-separated path to the parameter in the overwrite file.

    Examples
--------
    - 'changes.LSTN-02.asum_threshold' for trigger threshold
    - 'changes.LSTN-02.quantum_efficiency' for QE scaling
    - 'changes.OBS-North.nsb_scaling_factor' for NSB scaling
parameter_name (str, optional)
    Short name for the parameter used in file naming (default: extracted from parameter_path).
    Example: 'threshold', 'qe', 'nsb'
number_of_runs (int, required)
    Number of runs to be simulated per parameter value.
priority (int, optional)
    Job priority (default: 1).

(all other command line arguments are identical to those of :ref:`simulate_prod`).

Examples
--------
NSB trigger threshold scan:

.. code-block:: console

    simtools-simulate-prod-htcondor-parameter-scan-generator \\
        --output_path ./htcondor_threshold_scan \\
        --apptainer_image /path/to/simtools.sif \\
        --parameter_values 10 20 30 50 100 \\
        --parameter_path changes.LSTN-02.asum_threshold \\
        --parameter_name threshold \\
        --overwrite_template overwrite_template.yaml \\
        --simulation_software corsika_sim_telarray \\
        --site North \\
        --array_layout_name 1lst \\
        --primary gamma \\
        --zenith_angle 20 \\
        --number_of_runs 5

NSB scaling factor scan:

.. code-block:: console

    simtools-simulate-prod-htcondor-parameter-scan-generator \\
        --output_path ./htcondor_nsb_scan \\
        --apptainer_image /path/to/simtools.sif \\
        --parameter_values 0.5 1.0 1.5 2.0 \\
        --parameter_path changes.OBS-North.nsb_scaling_factor \\
        --parameter_name nsb \\
        --overwrite_template overwrite_template.yaml \\
        --number_of_runs 10

"""

import logging
from pathlib import Path

import yaml

from simtools.application_control import build_application
from simtools.job_execution import htcondor_script_generator

_logger = logging.getLogger(__name__)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--parameter_values",
        help="List of parameter values to scan.",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--overwrite_template",
        help="Path to template overwrite YAML file.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parameter_path",
        help=(
            "Dot-separated path to parameter in overwrite file "
            "(e.g., 'changes.LSTN-02.asum_threshold')."
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parameter_name",
        help="Short name for parameter used in file naming (default: last part of parameter_path).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--number_of_runs",
        help="Number of runs to be simulated per parameter value.",
        type=int,
        required=True,
        default=1,
    )
    parser.add_argument(
        "--priority",
        help="Job priority.",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--htcondor_log_path",
        help="Directory for HTCondor output files (default: output_path/htcondor_logs).",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--simulation_output",
        help="Output path for simulation data (default: ./simtools-output).",
        type=str,
        required=False,
        default="./simtools-output",
    )


def _set_nested_value(data, path_parts, value, version="2.0.0"):
    """
    Set a value in a nested dictionary using a path.

    Parameters
    ----------
    data : dict
        Dictionary to modify.
    path_parts : list
        List of keys representing the path.
    value : any
        Value to set.
    version : str
        Version string for the parameter.
    """
    current = data
    for key in path_parts[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Get existing version if available
    if isinstance(current.get(path_parts[-1]), dict):
        existing_version = current[path_parts[-1]].get("version", version)
    else:
        existing_version = version

    current[path_parts[-1]] = {"version": existing_version, "value": value}


def _format_value_for_filename(value):
    """Format a parameter value for use in filenames."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _generate_overwrite_files(args_dict, work_dir):
    """
    Generate overwrite YAML files for each parameter value.

    Parameters
    ----------
    args_dict : dict
        Arguments dictionary.
    work_dir : Path
        Working directory for output files.

    Returns
    -------
    dict
        Mapping from parameter values to overwrite file paths.
    """
    template_path = Path(args_dict["overwrite_template"])
    if not template_path.exists():
        raise FileNotFoundError(f"Overwrite template file not found: {template_path}")

    with open(template_path, encoding="utf-8") as f:
        template_data = yaml.safe_load(f)

    parameter_path = args_dict["parameter_path"]
    path_parts = parameter_path.split(".")
    parameter_name = args_dict.get("parameter_name") or path_parts[-1]

    parameter_files = {}

    for param_value in args_dict["parameter_values"]:
        overwrite_data = template_data.copy()

        # Determine value type (int or float)
        if isinstance(param_value, float) and param_value.is_integer():
            value = int(param_value)
        else:
            value = param_value

        _set_nested_value(overwrite_data, path_parts, value)
        overwrite_data["description"] = (
            f"Parameter scan - {parameter_name}={_format_value_for_filename(param_value)}"
        )

        value_str = _format_value_for_filename(param_value)
        overwrite_file = work_dir / f"overwrite_{parameter_name}_{value_str}.yaml"
        with open(overwrite_file, "w", encoding="utf-8") as f:
            yaml.dump(overwrite_data, f, default_flow_style=False, sort_keys=False)

        parameter_files[param_value] = overwrite_file
        _logger.info(
            f"Generated overwrite file for {parameter_name}={param_value}: {overwrite_file}"
        )

    return parameter_files


def _setup_directories(args_dict):
    """Set up working and log directories."""
    work_dir = Path(args_dict["output_path"])
    work_dir.mkdir(parents=True, exist_ok=True)

    htcondor_log_path = Path(
        args_dict["htcondor_log_path"]
        if args_dict.get("htcondor_log_path")
        else work_dir / "htcondor_logs"
    )
    log_dir = htcondor_log_path / "log"
    error_dir = htcondor_log_path / "error"
    output_dir = htcondor_log_path / "output"

    for subdir in (log_dir, error_dir, output_dir):
        subdir.mkdir(parents=True, exist_ok=True)

    return work_dir, log_dir, error_dir, output_dir


def _generate_parameter_submission_scripts(args_dict, parameter_files):
    """
    Generate HTCondor submission scripts for parameter scans.

    Parameters
    ----------
    args_dict : dict
        Arguments dictionary.
    parameter_files : dict
        Mapping from parameter values to overwrite file paths.
    """
    work_dir, log_dir, error_dir, output_dir = _setup_directories(args_dict)

    # pylint: disable=protected-access
    apptainer_images = htcondor_script_generator._resolve_apptainer_images(
        args_dict["apptainer_image"]
    )
    apptainer_image = next(iter(apptainer_images.values()))

    parameter_path = args_dict["parameter_path"]
    path_parts = parameter_path.split(".")
    parameter_name = args_dict.get("parameter_name") or path_parts[-1]

    label = args_dict.get("label", f"{parameter_name}_scan")
    number_of_runs = args_dict.get("number_of_runs", 1)

    for param_value, overwrite_file in parameter_files.items():
        value_str = _format_value_for_filename(param_value)
        executable_name = f"simulate_prod_{label}_{value_str}.submit.sh"

        submit_sh_file = work_dir / executable_name
        submit_script = htcondor_script_generator._get_submit_script(args_dict)  # pylint: disable=protected-access
        submit_script += f" \\\n    --overwrite_model_parameters {overwrite_file.absolute()}\n"

        with open(submit_sh_file, "w", encoding="utf-8") as f:
            f.write(submit_script)
        submit_sh_file.chmod(0o755)

        condor_file = work_dir / f"simulate_prod_{label}_{value_str}.submit.condor"
        condor_content = htcondor_script_generator._get_submit_file(  # pylint: disable=protected-access
            executable_name,
            apptainer_image,
            args_dict["priority"],
            f"dummy_params_{value_str}.txt",
            log_dir=log_dir,
            error_dir=error_dir,
            output_dir=output_dir,
        )
        condor_content = condor_content.rsplit("queue", 1)[0] + f"queue {number_of_runs}\n"

        with open(condor_file, "w", encoding="utf-8") as f:
            f.write(condor_content)

        _logger.info(f"Generated submission files for {parameter_name}={param_value}")


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": False,
            "simulation_model": ["site", "layout", "model_version"],
            "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
        },
    )

    work_dir = Path(app_context.args["output_path"])
    parameter_files = _generate_overwrite_files(app_context.args, work_dir)
    _generate_parameter_submission_scripts(app_context.args, parameter_files)

    parameter_name = (
        app_context.args.get("parameter_name") or app_context.args["parameter_path"].split(".")[-1]
    )
    _logger.info(f"Parameter scan submission scripts for {parameter_name} generated in {work_dir}")


if __name__ == "__main__":
    main()
