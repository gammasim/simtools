"""
Parameter scan generator for HTCondor submissions.

Generates HTCondor submission files for parameter scans that run simulate-prod
with different overwrite YAML files for each parameter combination.
"""

import itertools
import logging
from pathlib import Path

import yaml

from simtools.job_execution.htcondor_script_generator import _resolve_apptainer_images

_logger = logging.getLogger(__name__)


def _set_nested_value(data, path_parts, value):
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
    """
    current = data
    for key in path_parts[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[path_parts[-1]] = {"value": value}


def _generate_overwrite_file(template_path, param_combo, combo_name, work_dir, label):
    """
    Generate overwrite YAML file for a parameter combination.

    Parameters
    ----------
    template_path : Path
        Path to overwrite template file.
    param_combo : dict
        Dictionary of parameter_name: (path, value) pairs.
    combo_name : str
        String representation for filename.
    work_dir : Path
        Output directory.
    label : str
        Label for the scan.

    Returns
    -------
    Path
        Path to generated overwrite file.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Overwrite template file not found: {template_path}")

    with open(template_path, encoding="utf-8") as f:
        template_data = yaml.safe_load(f)

    param_descriptions = []
    for param_name, (param_path, param_value) in param_combo.items():
        path_parts = param_path.split(".")

        if isinstance(param_value, float) and param_value.is_integer():
            value = int(param_value)
        else:
            value = param_value

        _set_nested_value(template_data, path_parts, value)
        param_descriptions.append(f"{param_name}={param_value}")

    template_data["description"] = f"Parameter scan - {', '.join(param_descriptions)}"

    overwrite_file = work_dir / f"overwrite_{label}_{combo_name}.yaml"
    with open(overwrite_file, "w", encoding="utf-8") as f:
        yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)

    _logger.debug(f"Generated overwrite file: {overwrite_file}")
    return overwrite_file


def _parse_parameter_scan_config(param_scan):
    """
    Parse parameter scan configuration.

    Parameters
    ----------
    param_scan : dict
        Parameter scan configuration section.

    Returns
    -------
    tuple
        (list of parameter specs, template_path)
    """
    template_path = Path(param_scan["overwrite_template"])

    params = []
    for param_spec in param_scan["parameters"]:
        params.append(
            {
                "name": param_spec["name"],
                "path": param_spec["path"],
                "values": param_spec["values"],
            }
        )

    return params, template_path


def _generate_parameter_combinations(param_specs):
    """
    Generate all combinations of parameter values (cartesian product).

    Parameters
    ----------
    param_specs : list
        List of parameter specifications.

    Returns
    -------
    list
        List of parameter combinations.
    """
    param_names = [p["name"] for p in param_specs]
    param_paths = [p["path"] for p in param_specs]
    value_lists = [p["values"] for p in param_specs]

    combinations = []
    for value_combo in itertools.product(*value_lists):
        combo = {}
        combo_name_parts = []
        for name, path, value in zip(param_names, param_paths, value_combo):
            combo[name] = (path, value)
            combo_name_parts.append(f"{name}_{value}")

        combinations.append(
            {
                "combo": combo,
                "name": "_".join(combo_name_parts),
            }
        )

    return combinations


def _generate_submit_script(sim_params):
    """
    Generate shell script that runs simulate-prod.

    Parameters
    ----------
    sim_params : dict
        Simulation parameters.

    Returns
    -------
    str
        Shell script content.
    """
    label = sim_params.get("label", "param_scan")

    script = f"""#!/usr/bin/env bash

OVERWRITE_FILE="$1"
RUN_NUMBER="$2"
COMBO_LABEL="$3"

if [ -z "$OVERWRITE_FILE" ] || [ -z "$RUN_NUMBER" ]; then
    echo "Error: Missing arguments"
    echo "Usage: $0 <overwrite_file.yaml> <run_number> <combo_label>"
    exit 1
fi

set -a; source env.txt; set +a

# Construct full label with combo-specific parameters
FULL_LABEL="{label}_$COMBO_LABEL"

simtools-simulate-prod \\
    --simulation_software {sim_params["simulation_software"]} \\
    --site {sim_params["site"]} \\
    --model_version {sim_params["model_version"]} \\
    --array_layout_name {sim_params["array_layout_name"]} \\
    --primary {sim_params["primary"]} \\
    --azimuth_angle {sim_params["azimuth_angle"]} \\
    --zenith_angle {sim_params["zenith_angle"]} \\
    --nshow {sim_params["nshow"]} \\
    --energy_range "{sim_params["energy_range"]}" \\
    --core_scatter "{sim_params["core_scatter"]}" \\
    --view_cone "{sim_params["view_cone"]}" \\
    --run_number "$RUN_NUMBER" \\
    --corsika_le_interaction {sim_params["corsika_le_interaction"]} \\
    --corsika_he_interaction {sim_params["corsika_he_interaction"]} \\
    --label "$FULL_LABEL" \\
    --overwrite_model_parameters "$OVERWRITE_FILE" \\
    --output_path {sim_params.get("output_path", "/tmp/simtools-output")}"""

    if sim_params.get("run_number_offset"):
        script += f" \\\n    --run_number_offset {sim_params['run_number_offset']}"
    if sim_params.get("save_reduced_event_lists"):
        script += " \\\n    --save_reduced_event_lists"
    if sim_params.get("pack_for_grid_register"):
        script += f" \\\n    --pack_for_grid_register {sim_params['pack_for_grid_register']}"

    script += "\n"
    return script


def _generate_condor_submit_file(script_name, apptainer_image, priority, param_file, work_dir):
    """
    Generate HTCondor submit file.

    Parameters
    ----------
    script_name : str
        Name of the executable script.
    apptainer_image : Path
        Path to Apptainer image.
    priority : int
        Job priority.
    param_file : str
        Name of parameter file.
    work_dir : Path
        Working directory.

    Returns
    -------
    str
        HTCondor submit file content.
    """
    log_dir = work_dir / "htcondor_logs"
    for subdir in ["log", "error", "output"]:
        (log_dir / subdir).mkdir(parents=True, exist_ok=True)

    return f"""universe = container
container_image = {apptainer_image}
transfer_container = false

executable = {script_name}
arguments = $(overwrite_file) $(run_number) $(combo_label)
error = htcondor_logs/error/err.$(cluster)_$(process)
output = htcondor_logs/output/out.$(cluster)_$(process)
log = htcondor_logs/log/log.$(cluster)_$(process)

priority = {priority}

queue overwrite_file,run_number,combo_label from {param_file}
"""


def generate_parameter_scan_htcondor(config_path):
    """
    Generate HTCondor submission files for parameter scan.

    Parameters
    ----------
    config_path : str or Path
        Path to configuration file.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sim_params = config["simulation"]
    param_scan = config["parameter_scan"]
    htcondor_config = config["htcondor"]

    param_specs, template_path = _parse_parameter_scan_config(param_scan)
    param_combinations = _generate_parameter_combinations(param_specs)

    output_path = Path(htcondor_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    apptainer_images = _resolve_apptainer_images(htcondor_config["apptainer_image"])
    apptainer_image = next(iter(apptainer_images.values()))

    priority = htcondor_config.get("priority", 1)
    number_of_runs = sim_params.get("number_of_runs", 1)
    base_run_number = sim_params.get("run_number", 1)
    label = sim_params.get("label", "param_scan")

    if len(param_specs) == 1:
        _logger.info(
            f"Single parameter: {param_specs[0]['name']} ({len(param_specs[0]['values'])} values)"
        )
    else:
        combo_str = " x ".join(f"{len(s['values'])}" for s in param_specs)
        _logger.info(f"Multi-parameter: {' x '.join(s['name'] for s in param_specs)}")
        _logger.info(f"  {combo_str} = {len(param_combinations)} combinations")

    job_specs = []
    for combo_spec in param_combinations:
        _logger.info(f"Processing: {combo_spec['name']}")

        overwrite_file = _generate_overwrite_file(
            template_path, combo_spec["combo"], combo_spec["name"], output_path, label
        )

        for run_idx in range(number_of_runs):
            job_specs.append(
                (overwrite_file.absolute(), base_run_number + run_idx, combo_spec["name"])
            )

    params_file = output_path / f"scan_parameters_{label}.txt"
    with open(params_file, "w", encoding="utf-8") as f:
        for overwrite_file, run_number, combo_label in job_specs:
            f.write(f"{overwrite_file}, {run_number}, {combo_label}\n")

    script_name = f"simulate_prod_scan_{label}.sh"
    script_path = output_path / script_name
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(_generate_submit_script(sim_params))
    script_path.chmod(0o755)

    condor_file = output_path / f"simulate_prod_scan_{label}.condor"
    with open(condor_file, "w", encoding="utf-8") as f:
        f.write(
            _generate_condor_submit_file(
                script_name, apptainer_image, priority, params_file.name, output_path
            )
        )

    _logger.info("Parameter scan generation complete!")
    _logger.info(f"Output: {output_path}")
    _logger.info(f"  - {len(param_combinations)} overwrite files")
    _logger.info(
        f"  - {len(job_specs)} total jobs "
        f"({len(param_combinations)} combinations x {number_of_runs} runs)"
    )
