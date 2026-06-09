r"""
Parameter scan grid generator.

Expands an existing production job grid with parameter scan combinations.
For each cartesian combination of scan parameters, one overwrite YAML file is
generated and each base grid row is duplicated with the overwrite file path
and a scan label attached.

Use with the three-step workflow::

    simtools-production-generate-grid --output_file base_grid.ecsv ...
    simtools-generate-parameter-scan-grid \\
        --job_grid_file base_grid.ecsv \\
        --scan_config scan.yaml \\
        --output_file scan_grid.ecsv
    simtools-simulate-prod-htcondor-generator \\
        --job_grid_file scan_grid.ecsv \\
        --output_path htcondor_submit \\
        --apptainer_image ...
"""

import itertools
import logging
from pathlib import Path

import yaml

from simtools.production_configuration.job_grid_io import read_job_grid, serialize_job_grid

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


def expand_job_grid_with_scan(base_grid_file, scan_config_path, output_file):
    """
    Expand a production job grid with parameter scan combinations.

    Reads a base job grid, generates one overwrite YAML file per scan parameter
    combination, and writes a new grid where each base row is duplicated for
    every combination with ``overwrite_model_parameters`` and ``scan_label``
    columns added.

    Parameters
    ----------
    base_grid_file : str or Path
        Base job grid ECSV file from ``simtools-production-generate-grid``.
    scan_config_path : str or Path
        Path to parameter scan YAML configuration.
    output_file : str or Path
        Output path for the expanded scan grid ECSV.
    """
    scan_config_path = Path(scan_config_path)
    output_file = Path(output_file)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(scan_config_path, encoding="utf-8") as f:
        scan_config = yaml.safe_load(f)

    label = scan_config.get("label", "scan")
    param_specs, template_path = _parse_parameter_scan_config(scan_config["parameter_scan"])
    param_combinations = _generate_parameter_combinations(param_specs)

    base_rows, metadata = read_job_grid(base_grid_file)
    _logger.info(
        f"Expanding {len(base_rows)} base rows with {len(param_combinations)} scan combinations."
    )

    expanded_rows = []
    for combo_spec in param_combinations:
        overwrite_file = _generate_overwrite_file(
            template_path, combo_spec["combo"], combo_spec["name"], output_dir, label
        )
        for row in base_rows:
            new_row = dict(row)
            new_row["overwrite_model_parameters"] = str(overwrite_file)
            new_row["scan_label"] = combo_spec["name"]
            expanded_rows.append(new_row)

    serialize_job_grid(expanded_rows, output_file, metadata=metadata)
    _logger.info(f"Scan grid with {len(expanded_rows)} rows written to '{output_file}'.")
