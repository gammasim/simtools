"""HT Condor script generator for simulation production."""

import logging
from pathlib import Path

import astropy.units as u

_logger = logging.getLogger(__name__)


def generate_submission_script(args_dict):
    """
    Generate the HT Condor submission script.

    Parameters
    ----------
    args_dict: dict
        Arguments dictionary.
    """
    work_dir = Path(args_dict["output_path"])
    log_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    submit_file_name = "simulate_prod.submit"
    _logger.info(f"Generating HT Condor submission scripts (path: {work_dir})")

    with open(work_dir / f"{submit_file_name}.condor", "w", encoding="utf-8") as submit_file_handle:
        submit_file_handle.write(
            _get_submit_file(
                f"{submit_file_name}.sh",
                args_dict["apptainer_image"],
                args_dict["priority"],
                +args_dict["number_of_runs"],
            )
        )

    with open(work_dir / f"{submit_file_name}.sh", "w", encoding="utf-8") as submit_script_handle:
        submit_script_handle.write(_get_submit_script(args_dict))

    Path(work_dir / f"{submit_file_name}.sh").chmod(0o755)


def _get_submit_file(executable, apptainer_image, priority, n_jobs):
    """
    Return HT Condor submit file.

    Database access variables are passed through the environment file.

    Parameters
    ----------
    executable: str
        Name of the executable script.
    apptainer_image: str
        Path to the Apptainer image.
    priority: int
        Priority of the job.
    n_jobs: int
        Number of jobs to queue.

    Returns
    -------
    str
        HT Condor submit file content.
    """
    return f"""universe = container
container_image = {apptainer_image}
transfer_container = false

executable = {executable}
error      = logs/err.$(cluster)_$(process)
output     = logs/out.$(cluster)_$(process)
log        = logs/log.$(cluster)_$(process)

priority = {priority}
arguments = "$(process) env.txt"

queue {n_jobs}
"""


def _get_submit_script(args_dict):
    """
    Return HT Condor submit script.

    Parameters
    ----------
    args_dict: dict
        Arguments dictionary.

    Returns
    -------
    str
        HT Condor submit script content.
    """
    azimuth_angle_string = f"{args_dict['azimuth_angle'].to(u.deg).value}"
    zenith_angle_string = f"{args_dict['zenith_angle'].to(u.deg).value}"
    energy_range = args_dict["energy_range"]
    energy_range_string = (
        f'"{energy_range[0].to(u.GeV).value} GeV {energy_range[1].to(u.GeV).value} GeV"'
    )
    core_scatter = args_dict["core_scatter"]
    core_scatter_string = f'"{core_scatter[0]} {core_scatter[1].to(u.m).value} m"'
    view_cone = args_dict["view_cone"]
    view_cone_string = f'"{view_cone[0].to(u.deg).value} deg {view_cone[1].to(u.deg).value} deg"'

    label = args_dict["label"] if args_dict["label"] else "simulate-prod"

    array_layout_name = (
        args_dict["array_layout_name"][0]
        if isinstance(args_dict["array_layout_name"], list)
        and len(args_dict["array_layout_name"]) == 1
        else args_dict["array_layout_name"]
    )

    run_number_offset = args_dict["run_number_offset"] or 1

    return f"""#!/usr/bin/env bash

# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"

simtools-simulate-prod \\
    --simulation_software {args_dict["simulation_software"]} \\
    --label {label} \\
    --model_version {args_dict["model_version"]} \\
    --site {args_dict["site"]} \\
    --array_layout_name {array_layout_name} \\
    --primary {args_dict["primary"]} \\
    --azimuth_angle {azimuth_angle_string} \\
    --zenith_angle {zenith_angle_string} \\
    --nshow {args_dict["nshow"]} \\
    --energy_range {energy_range_string} \\
    --core_scatter {core_scatter_string} \\
    --view_cone {view_cone_string} \\
    --run_number $((process_id)) \\
    --run_number_offset {run_number_offset} \\
    --number_of_runs 1 \\
    --data_directory /tmp/simtools-data \\
    --output_path /tmp/simtools-output \\
    --log_level {args_dict["log_level"]} \\
    --pack_for_grid_register simtools-output
"""
