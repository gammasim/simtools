r"""
Parameter scan grid generator.

Expands an existing production job grid with parameter scan combinations.
Parameter sets are stored as compact references in the ECSV metadata and referenced
by a model_parameter_set column, with the actual override dictionaries embedded once
in the ECSV metadata.

For each cartesian combination of scan parameters, a parameter set is built
from the inline ``overwrite`` block in the scan configuration, and each base grid
row is duplicated with the parameter set reference, scan label, and optional
fixed job-grid updates attached.

Example metadata structure:
# meta:
#   model_parameter_sets:
#     lst_asum220:
#       LSTN-01:
#         asum_threshold: {value: 220}
#     mst_asum150:
#       MSTN-01:
#         asum_threshold: {value: 150}
#       OBS-North:
#         nsb_scaling_factor: {value: 2.0}
"""

import itertools
import logging
from copy import deepcopy
from pathlib import Path

from astropy.table import Table

from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.production_configuration.job_grid_io import (
    _ECSV_FORMAT,
    read_job_grid,
    serialize_job_grid,
)
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


def _build_parameter_set_name(param_combo, param_specs):
    """Build a unique name for a parameter set based on the combination."""
    name_parts = []
    for param_spec in param_specs:
        param_name = param_spec["name"]
        if param_name in param_combo:
            param_value = param_combo[param_name]["value"]
            scan_label = _format_value_for_name(param_spec.get("label", param_name))
            scan_value = _format_value_for_name(param_value)
            separator = param_spec.get("label_separator", "_")
            name_parts.append(f"{scan_label}{separator}{scan_value}")
    return "_".join(name_parts)


def _extract_changes_from_overwrite(overwrite_data):
    """Extract the changes section from overwrite data, removing non-change fields."""
    return overwrite_data.get("changes", {})


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


def _build_parameter_sets(param_combinations, overwrite_base, param_specs):
    """Build parameter sets dictionary for embedding in ECSV metadata.

    Creates a mapping from parameter set names to their actual override dictionaries,
    suitable for storage in ECSV metadata.

    Parameters
    ----------
    param_combinations : list
        List of parameter combinations from _generate_parameter_combinations
    overwrite_base : dict
        Base overwrite configuration
    param_specs : list
        Parameter specifications

    Returns
    -------
    dict
        Dictionary mapping parameter set names to their changes dictionaries
    """
    parameter_sets = {}

    for combo_spec in param_combinations:
        # Build the parameter set name
        param_set_name = _build_parameter_set_name(combo_spec["combo"], param_specs)

        # Build the overwrite data for this combination
        overwrite_data = _build_overwrite_data(overwrite_base, combo_spec["combo"])

        # Extract just the changes section for storage in metadata
        changes = _extract_changes_from_overwrite(overwrite_data)

        # Store the changes dictionary indexed by the parameter set name
        parameter_sets[param_set_name] = changes

    return parameter_sets


def expand_job_grid_with_scan(base_grid_file, scan_config_path, output_file):
    """Expand a production job grid with parameter scan combinations.

    Reads a base job grid, builds parameter sets from the scan configuration,
    stores them in the ECSV metadata, and writes a new grid where each base row
    is duplicated for every combination with ``model_parameter_set`` and ``scan_label``
    columns added.

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
    """
    scan_config_path = Path(scan_config_path)
    output_file = Path(output_file)
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    scan_config = schema.validate_dict_using_schema(
        ascii_handler.collect_data_from_file(scan_config_path),
        schema_file="parameter_scan_config.schema.yml",
    )

    # Validate the overwrite section against simulation_models_info schema
    if "parameter_scan" in scan_config and "overwrite" in scan_config["parameter_scan"]:
        schema.validate_dict_using_schema(
            scan_config["parameter_scan"]["overwrite"],
            schema_file="simulation_models_info.schema.yml",
        )

    param_specs, overwrite_base, job_grid_updates = _parse_parameter_scan_config(
        scan_config["parameter_scan"]
    )
    param_combinations = _generate_parameter_combinations(param_specs)

    base_rows, metadata = read_job_grid(base_grid_file)
    _logger.info(
        f"Expanding {len(base_rows)} base rows with {len(param_combinations)} scan combinations."
    )

    # Build parameter sets for metadata
    parameter_sets = _build_parameter_sets(param_combinations, overwrite_base, param_specs)

    # Add parameter sets to metadata
    if metadata is None:
        metadata = {}
    metadata["model_parameter_sets"] = parameter_sets

    expanded_rows = []
    _logger.info(
        f"About to expand {len(base_rows)} base rows with {len(param_combinations)} combinations"
    )
    for combo_spec in param_combinations:
        # Build the parameter set name for this combination
        param_set_name = _build_parameter_set_name(combo_spec["combo"], param_specs)
        _logger.info(f"Processing combination: {param_set_name}")

        combo_rows = []
        for row in base_rows:
            new_row = dict(row)
            new_row["model_parameter_set"] = param_set_name
            new_row["scan_label"] = combo_spec["name"]
            new_row.update(job_grid_updates)
            combo_rows.append(new_row)

        _logger.info(f"Combination {param_set_name} generated {len(combo_rows)} rows")
        expanded_rows.extend(combo_rows)

    _logger.info(f"Total expanded rows: {len(expanded_rows)}")

    serialize_job_grid(expanded_rows, output_file, metadata=metadata)
    try:
        _clean_scan_grid_metadata(output_file)
    except OSError as e:
        _logger.warning(f"Failed to clean scan grid metadata: {e}")
    _logger.info(f"Scan grid with {len(expanded_rows)} rows written to '{output_file}'.")


def _clean_scan_grid_metadata(output_file):
    """Remove unwanted metadata from scan grid file, keeping only essential fields."""
    output_path = Path(output_file)

    # Read the file
    table = Table.read(output_path, format=_ECSV_FORMAT)

    # Create clean metadata with only essential fields
    clean_meta = {}
    for key in ["model_parameter_sets", "site", "simulation_software"]:
        if key in table.meta:
            clean_meta[key] = table.meta[key]

    # Update table metadata
    table.meta = clean_meta

    # Write back to file
    table.write(output_path, format=_ECSV_FORMAT, overwrite=True)
