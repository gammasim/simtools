"""Script backend for simulation production job generation."""

import logging
import shlex
import subprocess
from pathlib import Path

import astropy.units as u

from simtools.job_execution.htcondor_script_generator import build_job_specs

_logger = logging.getLogger(__name__)


def generate_script_jobs(args_dict):
    """
    Generate local shell scripts from a production job grid.

    Parameters
    ----------
    args_dict: dict
        Arguments dictionary.
    """
    if args_dict.get("run_script") and args_dict.get("job_grid_line") is None:
        raise ValueError("--run_script requires --job_grid_line for the script backend.")

    work_dir = Path(args_dict["output_path"])
    work_dir.mkdir(parents=True, exist_ok=True)

    selected_job_specs, job_grid_metadata = build_job_specs(args_dict, ["default"])
    submit_args = {**job_grid_metadata, **args_dict}

    first_line = args_dict.get("job_grid_line") or 1
    script_paths = []
    for line_number, job_spec in enumerate(selected_job_specs, start=first_line):
        script_path = work_dir / f"simulate_prod.job_{line_number}.sh"
        script_path.write_text(_get_job_script(submit_args, job_spec), encoding="utf-8")
        script_path.chmod(0o755)
        script_paths.append(script_path)

    _logger.info(f"Generated {len(script_paths)} production job script(s) in {work_dir}")

    if args_dict.get("run_script"):
        subprocess.run([str(script_paths[0])], check=True)


def _get_job_script(args_dict, job_spec):
    """Return executable shell script for a single production job."""
    command = _get_simulate_prod_command(args_dict, job_spec)
    return f"""#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

if [ -f "${{script_dir}}/env.txt" ]; then
    set -a
    source "${{script_dir}}/env.txt"
    set +a
fi

{command}
"""


def _get_simulate_prod_command(args_dict, job_spec):
    """Return simtools-simulate-prod command for a single job spec."""
    label = args_dict["label"] if args_dict.get("label") else "simulate-prod"
    run_number_offset = args_dict.get("run_number_offset", 0)

    energy_min_value, energy_min_unit = _quantity_pair(job_spec["energy_min"], u.GeV)
    energy_max_value, energy_max_unit = _quantity_pair(job_spec["energy_max"], u.GeV)
    cores_per_shower = str(int(job_spec["cores_per_shower"]))
    core_scatter_max_value, core_scatter_max_unit = _quantity_pair(
        job_spec["core_scatter_max"], u.m, convert_to=u.m
    )
    view_cone_min_value, view_cone_min_unit = _quantity_pair(
        job_spec["view_cone_min"], u.deg, convert_to=u.deg
    )
    view_cone_max_value, view_cone_max_unit = _quantity_pair(
        job_spec["view_cone_max"], u.deg, convert_to=u.deg
    )

    energy_range_tag = (
        f"erange-{energy_min_value}{energy_min_unit}-{energy_max_value}{energy_max_unit}"
    )
    job_label = (
        f"{label}_{job_spec['corsika_he_interaction']}-{job_spec['corsika_le_interaction']}_"
        f"{energy_range_tag}"
    )

    args = [
        "simtools-simulate-prod",
        "--simulation_software",
        args_dict["simulation_software"],
        "--label",
        job_label,
        "--model_version",
        job_spec["model_version"],
        "--site",
        args_dict["site"],
        "--array_layout_name",
        job_spec["array_layout_name"],
        "--primary",
        job_spec["primary"],
        "--azimuth_angle",
        _angle_value(job_spec["azimuth_angle"]),
        "--zenith_angle",
        _angle_value(job_spec["zenith_angle"]),
        "--showers_per_run",
        str(job_spec["showers_per_run"]),
        "--energy_range",
        f"{energy_min_value} {energy_min_unit} {energy_max_value} {energy_max_unit}",
        "--core_scatter",
        f"{cores_per_shower} {core_scatter_max_value} {core_scatter_max_unit}",
        "--view_cone",
        f"{view_cone_min_value} {view_cone_min_unit} {view_cone_max_value} {view_cone_max_unit}",
        "--corsika_le_interaction",
        job_spec["corsika_le_interaction"],
        "--corsika_he_interaction",
        job_spec["corsika_he_interaction"],
        "--run_number",
        str(job_spec["run_number"]),
        "--run_number_offset",
        str(run_number_offset),
        "--save_reduced_event_lists",
        "--output_path",
        "/tmp/simtools-output",
        "--log_level",
        args_dict["log_level"],
        "--pack_for_grid_register",
        job_spec["pack_for_grid_register"],
    ]
    line_continuation = " \\\n    "
    return line_continuation.join(shlex.quote(str(arg)) for arg in args)


def _quantity_pair(value, default_unit, convert_to=None):
    """Return value and unit strings for a scalar or astropy Quantity."""
    if isinstance(value, u.Quantity):
        if convert_to is not None:
            value = value.to(convert_to)
        return f"{value.value}", f"{value.unit}"

    return f"{value}", f"{default_unit}"


def _angle_value(value):
    """Return angle value in degrees without unit."""
    if isinstance(value, u.Quantity):
        value = value.to(u.deg).value
    return f"{value}"
