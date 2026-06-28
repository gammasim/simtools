r"""
Parameter scan grid generator.

Expands an existing production job grid with parameter scan combinations.
For each cartesian combination of scan parameters, one overwrite YAML file is
created dynamically from the inline ``overwrite`` block in the scan configuration,
and each base grid row is duplicated with the overwrite file path, scan
label, and optional fixed job-grid updates attached.
"""

import itertools
import logging
from copy import deepcopy
from pathlib import Path

from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.production_configuration.job_grid_io import read_job_grid, serialize_job_grid
from simtools.utils import general

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
    schema.validate_dict_using_schema(
        overwrite_data,
        schema_file="simulation_models_info.schema.yml",
        ignore_software_version=True,
        offline=True,
    )

    safe_label = _format_value_for_name(label)
    overwrite_file = work_dir / f"overwrite_{safe_label}_{combo_name}.yaml"
    ascii_handler.write_data_to_file(overwrite_data, overwrite_file, sort_keys=False)

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
        values = general.ensure_list(param_spec["values"])
        if not values:
            raise ValueError("'values' must contain at least one scan value.")
        params.append(
            {
                "name": param_spec["name"],
                "path": param_spec["path"],
                "values": values,
                "version": param_spec.get("version"),
                "label": param_spec.get("label", param_spec["name"]),
                "label_separator": param_spec.get("label_separator", "_"),
            }
        )

    job_grid_updates = param_scan.get("job_grid_updates") or {}
    if not isinstance(job_grid_updates, dict):
        raise TypeError(
            "Parameter scan configuration field 'job_grid_updates' must be a dictionary."
        )

    return params, overwrite_base, job_grid_updates


def _combo_name_part(param_spec, value):
    """Return the label component for a single scan parameter value."""
    scan_label = _format_value_for_name(param_spec.get("label", param_spec["name"]))
    scan_value = _format_value_for_name(value)
    return f"{scan_label}{param_spec.get('label_separator', '_')}{scan_value}"


def _generate_parameter_combinations(param_specs):
    """Generate all cartesian combinations of parameter values."""
    value_lists = [p["values"] for p in param_specs]

    combinations = []
    for value_combo in itertools.product(*value_lists):
        combo = {}
        combo_name_parts = []
        for param_spec, value in zip(param_specs, value_combo):
            scan_value = _format_scan_value(value)
            combo[param_spec["name"]] = {
                "path": param_spec["path"],
                "value": scan_value,
                "version": param_spec.get("version"),
            }
            combo_name_parts.append(_combo_name_part(param_spec, scan_value))

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

    Parameters
    ----------
    base_grid_file : str or Path
        Path to the base job grid file (ECSV format).
    scan_config_path : str or Path
        Path to the parameter scan configuration file (YAML format).
    output_file : str or Path
        Path to the output file where the expanded grid will be written.

    Returns
    -------
    None
        Writes the expanded grid to the specified output file.

    Raises
    ------
    FileNotFoundError
        If the base grid file or scan configuration file does not exist.
    KeyError
        If the scan configuration is missing required fields (e.g., 'overwrite').
    TypeError
        If the scan configuration fields are not of the expected type.
    """
    scan_config_path = Path(scan_config_path)
    output_file = Path(output_file)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_config = schema.validate_dict_using_schema(
        ascii_handler.collect_data_from_file(scan_config_path),
        schema_file="parameter_scan_config.schema.yml",
        ignore_software_version=True,
        offline=True,
    )
    schema.validate_dict_using_schema(
        scan_config["parameter_scan"]["overwrite"],
        schema_file="simulation_models_info.schema.yml",
        ignore_software_version=True,
        offline=True,
    )

    label = scan_config["label"]
    param_specs, overwrite_base, job_grid_updates = _parse_parameter_scan_config(
        scan_config["parameter_scan"]
    )
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
            new_row.update(job_grid_updates)
            expanded_rows.append(new_row)

    serialize_job_grid(expanded_rows, output_file, metadata=metadata)
    _logger.info(f"Scan grid with {len(expanded_rows)} rows written to '{output_file}'.")
