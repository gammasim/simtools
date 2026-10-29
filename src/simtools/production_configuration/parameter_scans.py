"""
Parameter scan grids and NSB/proton bias-curve scan submissions.

A parameter scan expands a production job grid with the cartesian product of
scan values. Each scan combination is stored once as a named parameter set in the
ECSV metadata (``model_parameter_sets``); every expanded row references its set
via ``model_parameter_set`` and carries the combination name in ``scan_label``.

Bias curves use two scans (NSB and proton), each with a production grid, a scan
configuration, and the expanded scan grid. The NSB overwrite sets
``min_photons`` and ``min_photoelectrons`` to zero; both curves set
``nsb_scaling_factor``.
"""

import itertools
import logging
from copy import deepcopy
from pathlib import Path

from astropy.table import Table

from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.layout.array_layout_utils import resolve_array_layout_name
from simtools.model.telescope_model import TelescopeModel
from simtools.production_configuration.grid import generate_job_grid
from simtools.production_configuration.jobs_io import read_job_grid, serialize_job_grid
from simtools.utils.value_conversion import ensure_list

_ECSV_FORMAT = "ascii.ecsv"
_KEPT_METADATA = (
    "model_parameter_sets",
    "site",
    "simulation_software",
    "job_grid_summary",
    "job_grid_format_version",
)
_PRODUCTION_GRID_ARGS = (
    "site",
    "model_version",
    "simulation_software",
    "azimuth_angle",
    "zenith_angle",
    "showers_per_run",
    "core_scatter",
    "view_cone",
    "number_of_runs",
    "corsika_le_interaction",
    "corsika_he_interaction",
    "corsika_hadronic_transition_energy",
)
_REQUIRED_ARGS = (
    "site",
    "model_version",
    "array_layout_name",
    "azimuth_angle",
    "zenith_angle",
    "showers_per_run",
    "core_scatter",
    "view_cone",
    "number_of_runs",
)
# curve name -> (primary, application argument holding the energy range)
_CURVES = {
    "nsb": ("gamma", "nsb_energy_range"),
    "proton": ("proton", "proton_energy_range"),
}
_DEFAULT_THRESHOLDS = {
    "asum_threshold": [*range(220, 310, 10), *range(320, 361, 20)],
    "dsum_threshold": list(range(22, 31)),
    "discriminator_threshold": list(range(0, 10)),
}
_logger = logging.getLogger(__name__)


def _format_scan_value(value):
    """Return integral floats as ints for stable YAML and file names."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _format_value_for_name(value):
    """Return a scan value string that is safe for generated names."""
    return str(_format_scan_value(value)).replace(" ", "").replace("/", "-")


def _set_nested_parameter(data, path_parts, value, version=None):
    """Set a model-parameter value at a dotted path, keeping existing metadata.

    Missing intermediate nodes are created as dictionaries. The parameter node is
    created as ``{version, value}`` (``version`` only if given) or updated in place.
    """
    current = data
    for key in path_parts[:-1]:
        if current.get(key) is None:
            current[key] = {}
        if not isinstance(current[key], dict):
            raise TypeError(
                f"Cannot set parameter at '{'.'.join(path_parts)}': '{key}' is not a dict."
            )
        current = current[key]
    entry = current.get(path_parts[-1])
    if not isinstance(entry, dict):
        entry = current[path_parts[-1]] = {}
    if version is not None:
        entry["version"] = version
    entry["value"] = _format_scan_value(value)


def _parameter_sets(params, overwrite_base):
    """Return ``(name, changes)`` for every cartesian combination of scan values."""
    parameter_sets = []
    for values in itertools.product(*(param["values"] for param in params)):
        overwrite = deepcopy(overwrite_base)
        name_parts = []
        for param, value in zip(params, values):
            value = _format_scan_value(value)
            _set_nested_parameter(
                overwrite, param["path"].split("."), value, version=param["version"]
            )
            label = _format_value_for_name(param["label"])
            name_parts.append(f"{label}{param['label_separator']}{_format_value_for_name(value)}")
        parameter_sets.append(("_".join(name_parts), overwrite.get("changes", {})))
    return parameter_sets


def _parse_parameters(param_scan):
    """Return normalized scan parameters with defaults filled in."""
    return [
        {
            "path": spec["path"],
            "values": ensure_list(spec["values"]),
            "version": spec.get("version"),
            "label": spec.get("label", spec["name"]),
            "label_separator": spec.get("label_separator", "_"),
        }
        for spec in param_scan["parameters"]
    ]


def expand_job_grid_with_scan(base_grid_file, scan_config_path, output_file):
    """Expand a production job grid with parameter scan combinations.

    Parameters
    ----------
    base_grid_file : str or Path
        Base job grid file (ECSV format).
    scan_config_path : str or Path
        Parameter scan configuration file (YAML format).
    output_file : str or Path
        Output file for the expanded grid.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scan_config = schema.validate_dict_using_schema(
        ascii_handler.collect_data_from_file(scan_config_path),
        schema_file="parameter_scan_config.schema.yml",
    )
    param_scan = scan_config["parameter_scan"]
    schema.validate_dict_using_schema(
        param_scan["overwrite"], schema_file="simulation_models_info.schema.yml"
    )
    params = _parse_parameters(param_scan)
    job_grid_updates = param_scan.get("job_grid_updates") or {}
    parameter_sets = _parameter_sets(params, param_scan["overwrite"])

    base_rows, metadata = read_job_grid(base_grid_file)
    _logger.info(
        f"Expanding {len(base_rows)} base rows with {len(parameter_sets)} scan combinations."
    )
    metadata = {**(metadata or {}), "model_parameter_sets": dict(parameter_sets)}
    expanded_rows = [
        {**row, "model_parameter_set": name, "scan_label": name, **job_grid_updates}
        for name, _ in parameter_sets
        for row in base_rows
    ]
    serialize_job_grid(expanded_rows, output_file, metadata=metadata)
    try:
        _clean_scan_grid_metadata(output_file)
    except OSError as e:
        _logger.warning(f"Failed to clean scan grid metadata: {e}")
    _logger.info(f"Scan grid with {len(expanded_rows)} rows written to '{output_file}'.")


def _clean_scan_grid_metadata(output_file):
    """Keep only the essential metadata fields in a scan grid file."""
    table = Table.read(output_file, format=_ECSV_FORMAT)
    table.meta = {key: table.meta[key] for key in _KEPT_METADATA if key in table.meta}
    table.write(output_file, format=_ECSV_FORMAT, overwrite=True)


def _threshold_values(threshold_param, trigger_thresholds=None):
    """Return trigger thresholds from ``(minimum, count, step)`` or the defaults."""
    if trigger_thresholds is None:
        return _DEFAULT_THRESHOLDS.get(threshold_param, _DEFAULT_THRESHOLDS["dsum_threshold"])
    if len(trigger_thresholds) != 3:
        raise ValueError(
            "trigger_thresholds must contain minimum threshold, number of thresholds, "
            "and step size."
        )
    minimum, number, step_size = trigger_thresholds
    if number < 1 or not float(number).is_integer():
        raise ValueError("Number of trigger thresholds must be a positive integer.")
    if step_size <= 0:
        raise ValueError("Trigger-threshold step size must be positive.")
    return [minimum + index * step_size for index in range(int(number))]


def _threshold_param_name(args, telescope):
    """Return the threshold parameter selected by the telescope's default trigger."""
    telescope_model = TelescopeModel(
        site=args["site"], telescope_name=telescope, model_version=args["model_version"]
    )
    default_trigger = telescope_model.get_parameter_value("default_trigger")
    if default_trigger == "AnalogSum":
        return "asum_threshold"
    if default_trigger == "Majority":
        return "discriminator_threshold"
    return "dsum_threshold"


def _scan_config(curve_name, telescope, args):
    """Build the parameter-scan configuration for one curve."""
    nsb = curve_name == "nsb"
    threshold_param = args["threshold_parameter"]
    overwrite = {
        "model_version": args["model_version"],
        "model_update": "patch_update",
        "model_version_history": [args["model_version"]],
        "description": f"Tune for {'NSB' if nsb else 'proton'} telescope trigger scan",
        "changes": {
            telescope: {"min_photons": {"value": 0}, "min_photoelectrons": {"value": 0}}
            if nsb
            else {},
            f"OBS-{args['site']}": {"nsb_scaling_factor": {"value": args["nsb_scaling_factor"]}},
        },
    }
    return {
        "label": curve_name,
        "parameter_scan": {
            "overwrite": overwrite,
            "parameters": [
                {
                    "name": threshold_param,
                    "path": f"changes.{telescope}.{threshold_param}",
                    "values": _threshold_values(threshold_param, args.get("trigger_thresholds")),
                    "label": threshold_param.removesuffix("_threshold"),
                    "label_separator": "",
                }
            ],
        },
    }


def _generate_scan_grid(curve_name, curve_definition, args, telescope, io_handler):
    """Generate the base grid, scan configuration, and scan grid for one curve."""
    curve_directory = io_handler.get_output_directory(sub_dir=curve_name)
    base_grid_file = curve_directory / "base_grid.ecsv"
    scan_config_file = curve_directory / "scan_config.yaml"
    scan_grid_file = curve_directory / "scan_grid.ecsv"

    ascii_handler.write_data_to_file(
        _scan_config(curve_name, telescope, args), scan_config_file, sort_keys=False
    )
    primary, energy_range_arg = curve_definition
    production_configuration = {
        key: args[key] for key in _PRODUCTION_GRID_ARGS if args.get(key) not in (None, "")
    }
    production_configuration.update(
        {
            "array_layout_name": args["array_layout_name"],
            "primary": primary,
            "energy_range": args[energy_range_arg],
            "label": curve_name,
        }
    )
    generate_job_grid(production_configuration, base_grid_file)
    expand_job_grid_with_scan(base_grid_file, scan_config_file, scan_grid_file)


def generate_scan_grids(args, io_handler):
    """Generate the NSB and proton bias-curve scan grids.

    Parameters
    ----------
    args : dict
        Application arguments for the production grid and scan configuration.
    io_handler : OutputPaths
        Output paths providing the output directories.
    """
    missing = [key for key in _REQUIRED_ARGS if args.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required argument: --{missing[0]}")

    telescope = resolve_array_layout_name(args["array_layout_name"], args["model_version"])
    generation_args = {**args, "threshold_parameter": _threshold_param_name(args, telescope)}
    for curve_name, curve_definition in _CURVES.items():
        _generate_scan_grid(curve_name, curve_definition, generation_args, telescope, io_handler)
