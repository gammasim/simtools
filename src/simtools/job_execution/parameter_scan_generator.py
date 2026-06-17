r"""
Parameter scan grid generator.

Expands an existing production job grid with parameter scan combinations.
For each cartesian combination of scan parameters, one overwrite YAML file is
created dynamically from the inline ``overwrite`` block in the scan configuration,
and each base grid row is duplicated with the overwrite file path and a scan
label attached.
"""

import itertools
import logging
from copy import deepcopy
from pathlib import Path

import yaml

from simtools.production_configuration.job_grid_io import read_job_grid, serialize_job_grid

_logger = logging.getLogger(__name__)


def _format_scan_value(value):
    """Return scan values in a stable YAML/filename-friendly representation."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _format_value_for_name(value):
    """Return a scan value string that is safe enough for generated filenames."""
    value = _format_scan_value(value)
    return str(value).replace(" ", "").replace("/", "-")


def _set_nested_parameter(data, path_parts, value, version=None):
    """Set a model-parameter value in a nested dictionary using a dotted path.

    The path points to the model-parameter node, e.g.
    ``changes.LSTN-01.asum_threshold``. If the node already exists as a
    dictionary, its ``value`` field is updated and existing metadata is kept.
    If the node does not exist, it is created as ``{version, value}`` when a
    version is given, otherwise as ``{value}``.
    """
    current = data
    for key in path_parts[:-1]:
        if key not in current or current[key] is None:
            current[key] = {}
        if not isinstance(current[key], dict):
            raise TypeError(
                f"Cannot set nested parameter at '{'.'.join(path_parts)}'; "
                f"intermediate key '{key}' is not a dictionary."
            )
        current = current[key]

    final_key = path_parts[-1]
    parameter_entry = current.get(final_key)

    if parameter_entry is None:
        parameter_entry = {}
        current[final_key] = parameter_entry

    if not isinstance(parameter_entry, dict):
        parameter_entry = {}
        current[final_key] = parameter_entry

    if version is not None:
        parameter_entry["version"] = version
    parameter_entry["value"] = _format_scan_value(value)


def _build_overwrite_data(overwrite_base, param_combo):
    """Build overwrite YAML content for one parameter combination."""
    overwrite_data = deepcopy(overwrite_base)
    param_descriptions = []

    for param_name, param in param_combo.items():
        param_value = _format_scan_value(param["value"])
        path_parts = param["path"].split(".")
        _set_nested_parameter(
            overwrite_data,
            path_parts,
            param_value,
            version=param.get("version"),
        )
        param_descriptions.append(f"{param_name}={param_value}")

    base_description = overwrite_data.get("description", "Parameter scan")
    overwrite_data["description"] = f"{base_description} - {', '.join(param_descriptions)}"
    return overwrite_data


def _generate_overwrite_file(overwrite_base, param_combo, combo_name, work_dir, label):
    """Generate overwrite YAML file for one parameter combination."""
    overwrite_data = _build_overwrite_data(overwrite_base, param_combo)

    overwrite_file = work_dir / f"overwrite_{label}_{combo_name}.yaml"
    with open(overwrite_file, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(overwrite_data, file_handle, default_flow_style=False, sort_keys=False)

    _logger.debug(f"Generated overwrite file: {overwrite_file}")
    return overwrite_file


def _parse_parameter_scan_config(param_scan):
    """Parse parameter scan configuration.

    The configuration must contain an inline ``overwrite`` dictionary. External
    overwrite template files are intentionally not supported.
    """
    if "overwrite" not in param_scan:
        raise KeyError("Parameter scan configuration requires 'overwrite'.")

    overwrite_base = param_scan["overwrite"] or {}

    if not isinstance(overwrite_base, dict):
        raise TypeError("Parameter scan configuration field 'overwrite' must be a dictionary.")

    params = []
    for param_spec in param_scan["parameters"]:
        params.append(
            {
                "name": param_spec["name"],
                "path": param_spec["path"],
                "values": param_spec["values"],
                "version": param_spec.get("version"),
            }
        )

    return params, overwrite_base


def _generate_parameter_combinations(param_specs):
    """Generate all cartesian combinations of parameter values."""
    param_names = [p["name"] for p in param_specs]
    value_lists = [p["values"] for p in param_specs]

    combinations = []
    for value_combo in itertools.product(*value_lists):
        combo = {}
        combo_name_parts = []
        for param_spec, name, value in zip(param_specs, param_names, value_combo):
            scan_value = _format_scan_value(value)
            combo[name] = {
                "path": param_spec["path"],
                "value": scan_value,
                "version": param_spec.get("version"),
            }
            combo_name_parts.append(f"{name}_{_format_value_for_name(scan_value)}")

        combinations.append(
            {
                "combo": combo,
                "name": "_".join(combo_name_parts),
            }
        )

    return combinations


def expand_job_grid_with_scan(base_grid_file, scan_config_path, output_file):
    """Expand a production job grid with parameter scan combinations.

    Reads a base job grid, dynamically generates one overwrite YAML file per
    scan parameter combination, and writes a new grid where each base row is
    duplicated for every combination with ``overwrite_model_parameters`` and
    ``scan_label`` columns added.
    """
    scan_config_path = Path(scan_config_path)
    output_file = Path(output_file)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(scan_config_path, encoding="utf-8") as file_handle:
        scan_config = yaml.safe_load(file_handle)

    label = scan_config.get("label", "scan")
    param_specs, overwrite_base = _parse_parameter_scan_config(scan_config["parameter_scan"])
    param_combinations = _generate_parameter_combinations(param_specs)

    base_rows, metadata = read_job_grid(base_grid_file)
    _logger.info(
        f"Expanding {len(base_rows)} base rows with {len(param_combinations)} scan combinations."
    )

    expanded_rows = []
    for combo_spec in param_combinations:
        overwrite_file = _generate_overwrite_file(
            overwrite_base, combo_spec["combo"], combo_spec["name"], output_dir, label
        )
        for row in base_rows:
            new_row = dict(row)
            new_row["overwrite_model_parameters"] = str(overwrite_file)
            new_row["scan_label"] = combo_spec["name"]
            expanded_rows.append(new_row)

    serialize_job_grid(expanded_rows, output_file, metadata=metadata)
    _logger.info(f"Scan grid with {len(expanded_rows)} rows written to '{output_file}'.")
