"""Read and write executable job grids for production preparation."""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy import units as u
from astropy.table import Table

from simtools.constants import SCHEMA_PATH, SCHEMA_URL
from simtools.data_model import validate_data
from simtools.io.ascii_handler import collect_data_from_file
from simtools.job_execution.job import JobSpec
from simtools.utils.value_conversion import get_value_as_quantity, get_value_in_unit

logger = logging.getLogger(__name__)

_ECSV_SUFFIX = ".ecsv"
_ECSV_FORMAT = "ascii.ecsv"
_JOB_GRID_SCHEMA_FILE = "job_grid_density.schema.yml"
_OPTIONAL_STRING_FIELDS = ("overwrite_model_parameters", "scan_label", "model_parameter_set")
SIMULATE_PROD_JOB_GRID_EXCLUSIVE_FIELDS = frozenset(
    """primary azimuth_angle zenith_angle energy_range core_scatter view_cone showers_per_run
    model_version array_layout_name corsika_le_interaction corsika_he_interaction
    corsika_hadronic_transition_energy run_number run_number_offset site simulation_software
    """.split()
)
_SIMULATE_PROD_PATH_FIELDS = (
    "corsika_file grid_output_path sim_telarray_path corsika_path corsika_interaction_table_path "
    "simulation_models_path simulation_models_git_path model_path overwrite_model_parameters"
).split()
_SIMULATE_PROD_CONTROLLER_FIELDS = frozenset(
    """activity_id application_label backend backend_config config data_search_path env_file
    job_grid_file job_grid_row list_available_corsika_models""".split()
)


def _load_job_grid_schema():
    """Load the job-grid format definition (version, columns, units, dtypes) from YAML."""
    schema = collect_data_from_file(SCHEMA_PATH / _JOB_GRID_SCHEMA_FILE)
    if isinstance(schema, list):
        schema = schema[0]
    columns = next(item for item in schema["data"] if item["type"] == "data_table")["table_columns"]
    return SimpleNamespace(
        version=schema["schema_version"],
        columns=tuple(column["name"] for column in columns),
        optional_columns=tuple(column["name"] for column in columns if not column.get("required")),
        column_units={c["name"]: u.Unit(c["unit"]) for c in columns if c.get("unit")},
        column_dtypes={
            c["name"]: str if c["type"] == "string" else np.dtype(c["type"]) for c in columns
        },
    )


JOB_GRID_SCHEMA = _load_job_grid_schema()
JOB_GRID_COLUMNS = [
    name for name in JOB_GRID_SCHEMA.columns if name not in JOB_GRID_SCHEMA.optional_columns
]


def _cast_scalar(value, dtype):
    """Cast a scalar using its schema dtype and return a Python value."""
    return str(value) if dtype is str else dtype.type(value).item()


def _serialize_job_row(job_row):
    """Serialize one job row to the on-disk schema."""
    serialized_row = {}
    for name in JOB_GRID_SCHEMA.columns:
        value = job_row.get(name)
        if value is None:
            if name in JOB_GRID_SCHEMA.optional_columns:
                continue
            if name not in job_row:
                raise KeyError(name)
            raise TypeError(f"Missing required value for field '{name}'.")
        if name in _OPTIONAL_STRING_FIELDS:
            serialized_row[name] = str(value)
        elif name in JOB_GRID_SCHEMA.column_units:
            unit = JOB_GRID_SCHEMA.column_units[name]
            serialized_row[name] = float(get_value_in_unit(value, unit))
        else:
            serialized_row[name] = _cast_scalar(value, JOB_GRID_SCHEMA.column_dtypes[name])
    return serialized_row


def _deserialize_job_row(serialized_row, column_units=None):
    """Deserialize one stored row to the in-memory job-row schema."""
    column_units = column_units or {}
    job_row = {}
    for name in JOB_GRID_SCHEMA.columns:
        value = serialized_row.get(name)
        if value is None or np.ma.is_masked(value):
            continue
        if JOB_GRID_SCHEMA.column_dtypes[name] is str and not str(value).strip():
            continue
        if name in JOB_GRID_SCHEMA.column_units:
            unit = JOB_GRID_SCHEMA.column_units[name]
            source_unit = column_units.get(name) or unit
            job_row[name] = get_value_as_quantity(float(value), source_unit).to(unit)
        else:
            job_row[name] = _cast_scalar(value, JOB_GRID_SCHEMA.column_dtypes[name])
    return job_row


def _validate_job_grid_table(table):
    """Restore integer dtypes lost in ECSV storage, then validate against the schema."""
    if len(table) == 0:
        return table
    table = table.copy(copy_data=True)
    for name, dtype in JOB_GRID_SCHEMA.column_dtypes.items():
        if name not in table.colnames or not np.issubdtype(dtype, np.integer):
            continue
        column = table[name]
        values = np.asarray(column.data)
        if (
            np.issubdtype(column.dtype, np.floating)
            and np.all(np.isfinite(values))
            and np.array_equal(values, np.round(values))
        ):
            table[name] = np.round(values).astype(dtype, copy=False)
    return validate_data.DataValidator(
        schema_file=SCHEMA_PATH / _JOB_GRID_SCHEMA_FILE, data_table=table
    ).validate_and_transform()


def serialize_job_grid(job_rows, output_file, metadata=None):
    """
    Serialize executable job rows to ECSV output.

    Parameters
    ----------
    job_rows : list[dict]
        Job rows in the in-memory schema.
    output_file : str or Path
        Output file path. Must use the ``.ecsv`` suffix.
    metadata : dict, optional
        Metadata to store alongside the rows.
    """
    output_path = Path(output_file)
    serialized_rows = [_serialize_job_row(job_row) for job_row in job_rows]
    metadata = metadata.copy() if metadata else {}
    model_metadata = metadata.setdefault("cta", {}).setdefault("product", {}).setdefault("data", {})
    model_metadata.setdefault("model", {})["url"] = SCHEMA_URL + "/" + _JOB_GRID_SCHEMA_FILE
    metadata["job_grid_format_version"] = JOB_GRID_SCHEMA.version
    metadata["job_grid_summary"] = build_job_grid_summary(job_rows)

    if output_path.suffix.lower() != _ECSV_SUFFIX:
        raise ValueError("Job grid output file must use the '.ecsv' extension.")

    optional_columns = tuple(
        column
        for column in JOB_GRID_SCHEMA.optional_columns
        if not serialized_rows or any(column in row for row in serialized_rows)
    )
    output_columns = (*JOB_GRID_COLUMNS, *optional_columns)
    output_table = Table(
        meta=metadata,
        rows=[
            tuple(
                row.get(column, "" if column in _OPTIONAL_STRING_FIELDS else None)
                for column in output_columns
            )
            for row in serialized_rows
        ],
        names=output_columns,
        dtype=[JOB_GRID_SCHEMA.column_dtypes[column] for column in output_columns],
    )
    for name in set(output_columns) & set(JOB_GRID_SCHEMA.column_units):
        output_table[name].unit = JOB_GRID_SCHEMA.column_units[name]
    output_table = _validate_job_grid_table(output_table)
    logger.info(f"Writing job grid with {len(job_rows)} rows to '{output_path}'.")
    output_table.write(output_path, format=_ECSV_FORMAT, overwrite=True)


def read_job_grid(input_file):
    """
    Read executable job rows from ECSV input.

    Parameters
    ----------
    input_file : str or Path
        Input file path.

    Returns
    -------
    tuple[list[dict], dict]
        Deserialized job rows and metadata.
    """
    input_path = Path(input_file)
    if input_path.suffix.lower() != _ECSV_SUFFIX:
        raise ValueError("Job grid input file must use the '.ecsv' extension.")

    table = _validate_job_grid_table(Table.read(input_path, format=_ECSV_FORMAT))
    column_units = {name: table[name].unit for name in table.colnames}
    rows = [
        _deserialize_job_row({name: row[name] for name in table.colnames}, column_units)
        for row in table
    ]
    return rows, dict(table.meta)


def _deep_merge_dicts(dict1, dict2):
    """Deep merge two dictionaries, with dict2 taking precedence."""
    result = dict1.copy()
    for key, value in dict2.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def job_grid_row_to_simulate_prod_args(job_row, metadata=None):
    """
    Convert an in-memory job grid row to simulate_prod argument format.

    Callers are responsible for not combining these production-defining values ambiguously
    with independently supplied production arguments.

    Parameters
    ----------
    job_row : dict
        A single deserialized job row as returned by :func:`read_job_grid`.
    metadata : dict, optional
        Job grid metadata as returned alongside the rows.  When provided,
        ``site`` and ``simulation_software`` are included in the result.

    Returns
    -------
    dict
        Argument dictionary compatible with ``simulate_prod`` ``args_dict`` keys.
    """
    args = {
        "primary": job_row["primary"],
        "azimuth_angle": job_row["azimuth_angle"],
        "zenith_angle": job_row["zenith_angle"],
        "energy_range": (job_row["energy_min"], job_row["energy_max"]),
        "core_scatter": (int(job_row["cores_per_shower"]), job_row["core_scatter_max"]),
        "view_cone": (job_row["view_cone_min"], job_row["view_cone_max"]),
        "showers_per_run": int(job_row["showers_per_run"]),
        "model_version": job_row["model_version"],
        "array_layout_name": job_row["array_layout_name"],
        "corsika_le_interaction": job_row["corsika_le_interaction"],
        "corsika_he_interaction": job_row["corsika_he_interaction"],
        "run_number": int(job_row["run_number"]),
        "run_number_offset": 0,
    }
    if job_row.get("corsika_hadronic_transition_energy") is not None:
        args["corsika_hadronic_transition_energy"] = job_row["corsika_hadronic_transition_energy"]
    args.update({key: job_row[key] for key in ("ha", "dec") if job_row.get(key) is not None})

    metadata = metadata or {}
    args.update(
        {key: metadata[key] for key in ("site", "simulation_software") if metadata.get(key)}
    )
    if job_row.get("scan_label"):
        args["scan_label"] = job_row["scan_label"]

    overwrites = job_row.get("overwrite_model_parameters")
    parameter_set = job_row.get("model_parameter_set")
    parameter_sets = metadata.get("model_parameter_sets")
    if parameter_set and parameter_sets:
        if parameter_set in parameter_sets:
            parameter_overwrites = parameter_sets[parameter_set]
            if overwrites and isinstance(overwrites, dict):
                parameter_overwrites = _deep_merge_dicts(overwrites, parameter_overwrites)
            args["overwrite_model_parameters"] = parameter_overwrites
    elif overwrites:
        args["overwrite_model_parameters"] = overwrites
    return args


def build_simulate_prod_job_specs(args_dict, rows, parser, metadata=None):
    """Build backend-neutral command jobs for production-grid rows.

    Parameters
    ----------
    args_dict : dict
        Controller application arguments.
    rows : list[dict]
        Deserialized production-grid rows.
    parser : argparse.ArgumentParser
        Parser defining the nested ``simulate_prod`` command.
    metadata : dict, optional
        Production-grid metadata.

    Returns
    -------
    list[JobSpec]
        Ordered command jobs that force local execution inside each worker.
    """
    metadata = metadata or {}
    supported = parser._option_string_actions  # pylint: disable=protected-access
    jobs = []
    for index, row in enumerate(rows):
        job_name = f"job-{index + 1:06d}"
        job_args = {
            key: value
            for key, value in args_dict.items()
            if key not in _SIMULATE_PROD_CONTROLLER_FIELDS and f"--{key}" in supported
        }
        job_args.update(job_grid_row_to_simulate_prod_args(row, metadata))
        job_args = {key: value for key, value in job_args.items() if f"--{key}" in supported}

        output_path = Path(args_dict["output_path"]).expanduser().resolve() / job_name
        job_args["output_path"] = str(output_path)
        mount_paths = [output_path]
        output_paths = [output_path]
        if job_args.get("grid_output_path"):
            grid_output_path = Path(job_args["grid_output_path"]) / job_name
            job_args["grid_output_path"] = str(grid_output_path)
            mount_paths.append(grid_output_path)
            output_paths.append(grid_output_path)
        if row.get("scan_label"):
            label = job_args.get("label") or "simulate-prod"
            job_args["label"] = f"{label}_{row['scan_label']}"
            job_args.pop("scan_label", None)
        if job_args.get("simulation_models_path"):
            job_args.pop("simulation_models_git_path", None)
            job_args.pop("simulation_models_git_revision", None)
        mount_paths.extend(_normalize_simulate_prod_paths(job_args, args_dict))

        command = [*_simulate_prod_local_command(), *_job_arguments_to_cli(job_args, parser)]
        jobs.append(
            JobSpec(
                job_name,
                index,
                command=tuple(command),
                mount_paths=tuple(mount_paths),
                output_paths=tuple(output_paths),
            )
        )
    return jobs


def _simulate_prod_local_command():
    """Return the interpreter and module prefix for a nested local simulate_prod command."""
    return (sys.executable, "-m", "simtools.applications.simulate_prod", "--backend", "local")


def _bool_option_tokens(option, value, action, action_name):
    """Return the CLI tokens for a boolean flag."""
    if action_name == "BooleanOptionalAction" and not value:
        return [action.option_strings[1]]
    if value != (action_name == "_StoreFalseAction"):
        return [option]
    return []


def _append_option_tokens(option, value, action):
    """Return the CLI tokens for an append-style option, one option per item."""
    grouped = action.nargs is not None and value and isinstance(value[0], list)
    items = [tuple(map(str, g)) for g in value] if grouped else [(str(i),) for i in value]
    return [token for item in items for token in (option, *item)]


def _option_tokens(option, value, action, action_name):
    """Return the CLI tokens for one option value."""
    if isinstance(value, tuple):
        value = " ".join(map(str, value))
    if isinstance(value, bool):
        return _bool_option_tokens(option, value, action, action_name)
    if isinstance(value, dict):
        return [option, json.dumps(value)]
    if isinstance(value, list) and action_name == "_AppendAction":
        return _append_option_tokens(option, value, action)
    values = list(map(str, value)) if isinstance(value, list) else [str(value)]
    return [option, *values]


def _job_arguments_to_cli(job_args, parser):
    """Encode job arguments as CLI tokens using each option's child parser action."""
    tokens = []
    for key, value in job_args.items():
        if value is None:
            continue
        option = f"--{key}"
        action = parser._option_string_actions.get(option)  # pylint: disable=protected-access
        action_name = action.__class__.__name__ if action is not None else None
        tokens.extend(_option_tokens(option, value, action, action_name))
    return tokens


def _normalize_simulate_prod_paths(job_args, args_dict):
    """Normalize forwarded paths in place and return the input paths to mount in jobs."""
    is_local = args_dict.get("backend", "local") == "local"
    sources = args_dict.get("_metadata_configuration_sources", {})
    mount_paths = []
    for key in _SIMULATE_PROD_PATH_FIELDS:
        # Dict-valued overwrites are override values, not paths, and are always forwarded.
        if isinstance(job_args.get(key), dict):
            continue
        if not is_local and sources.get(key) not in {"cli", "yaml"}:
            job_args.pop(key, None)
        elif job_args.get(key):
            job_args[key] = str(Path(job_args[key]).expanduser().resolve())
            if key != "grid_output_path":
                path = Path(job_args[key])
                is_file_path = key in {"corsika_file", "overwrite_model_parameters"}
                mount_paths.append(path.parent if is_file_path else path)
    return mount_paths


def resolve_job_grid_arguments(args_dict, config_sources, parser):
    """Merge selected job-grid row values into args after rejecting ambiguous input."""
    explicit_keys = {key for key, source in config_sources.items() if source in {"cli", "yaml"}}
    job_grid_row_is_explicit = "job_grid_row" in explicit_keys

    if not args_dict.get("job_grid_file"):
        if job_grid_row_is_explicit:
            parser.error("'--job_grid_row' requires '--job_grid_file'.")
        if not args_dict.get("array_layout_name"):
            parser.error("the following argument is required: --array_layout_name")
        _validate_simulation_arguments(args_dict, parser)
        return

    conflicting_keys = sorted(explicit_keys & SIMULATE_PROD_JOB_GRID_EXCLUSIVE_FIELDS)
    if conflicting_keys:
        parser.error(
            "'--job_grid_file' cannot be combined with explicit production parameter(s): "
            + ", ".join(conflicting_keys)
        )
    rows, metadata = read_job_grid(args_dict["job_grid_file"])
    if not rows:
        parser.error("Job grid contains no rows to process.")
    missing_layout_rows = [i + 1 for i, row in enumerate(rows) if not row.get("array_layout_name")]
    if missing_layout_rows:
        parser.error(
            "Job grid row(s) missing array_layout_name: " + ", ".join(map(str, missing_layout_rows))
        )
    is_parameter_scan = any(row.get("model_parameter_set") for row in rows)

    is_local = args_dict.get("backend", "local") == "local"
    selected_row = None
    if is_local or job_grid_row_is_explicit:
        selected_row, rows = _select_rows(
            args_dict, rows, is_parameter_scan, job_grid_row_is_explicit, parser
        )

    if not is_local or selected_row is None:
        if is_local:
            logger.info(f"Setting _job_grid_rows to {len(rows)} rows for parameter scan")
        args_dict["_job_grid_rows"] = rows
        args_dict["_job_grid_metadata"] = metadata
    else:
        args_dict.update(job_grid_row_to_simulate_prod_args(selected_row, metadata))
        _validate_simulation_arguments(args_dict, parser)


def _select_rows(args_dict, rows, is_parameter_scan, job_grid_row_is_explicit, parser):
    """Select rows based on job_grid_row and grid type."""
    row_index = args_dict.get("job_grid_row")
    if row_index is not None and (job_grid_row_is_explicit or not is_parameter_scan):
        if not 1 <= row_index <= len(rows):
            parser.error(
                f"Row index {row_index} is out of range for a grid with {len(rows)} row(s)."
            )
        selected_row = rows[row_index - 1]
        return selected_row, ([selected_row] if job_grid_row_is_explicit else rows)
    if is_parameter_scan:
        return None, rows
    return rows[0], [rows[0]]


def _validate_simulation_arguments(args_dict, parser):
    """Validate requirements that depend on the selected simulation software."""
    if "corsika" in args_dict["simulation_software"] and not args_dict.get("primary"):
        parser.error("the following argument is required for CORSIKA: --primary")


def format_quantity_field(rows, key, summary_unit=None, *, log_wording=False):
    """Format one quantity field across rows as a single value or a range.

    ``log_wording`` selects the sentence style used in log messages instead of the
    summary style used in the job-grid metadata.
    """
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return "is not available" if log_wording else "not available"
    quantities = u.Quantity(values)
    unit = summary_unit or quantities.max().unit
    low = quantities.min().to_value(unit)
    high = quantities.max().to_value(unit)
    if np.isclose(low, high):
        return f"is {high:.6g} {unit}" if log_wording else f"{high:.6g} {unit}"
    if log_wording:
        return f"ranges from {low:.6g} to {high:.6g} {unit}"
    return f"[{low:.6g}, {high:.6g}] {unit}"


def _summary_integer_text(values):
    """Format integer min/max as a single value or compact range."""
    min_value, max_value = min(values), max(values)
    if min_value == max_value:
        return f"{max_value:d}"
    return f"[{min_value:d}, {max_value:d}]"


def build_job_grid_summary(rows):
    """Build a user-facing summary of generated job-grid values."""
    if not rows:
        return {"simulation_rows": 0, "total_showers": 0}

    showers_per_run_values = [int(row["showers_per_run"]) for row in rows]
    configured_showers_per_run_values = [
        int(row.get("configured_showers_per_run", row["showers_per_run"])) for row in rows
    ]

    def text(key, summary_unit=None):
        return format_quantity_field(rows, key, summary_unit)

    return {
        "simulation_rows": len(rows),
        "energy_min_used": text("energy_min"),
        "energy_max_used": text("energy_max"),
        "energy_min_configured": text("configured_energy_min"),
        "energy_max_configured": text("configured_energy_max"),
        "energy_min_lookup_limit": text("energy_min_lookup_limit", u.GeV),
        "core_scatter_max_used": text("core_scatter_max"),
        "core_scatter_max_configured": text("configured_core_scatter_max"),
        "core_scatter_max_lookup_limit": text("lookup_core_scatter_max"),
        "view_cone_min_used": text("view_cone_min"),
        "view_cone_max_used": text("view_cone_max"),
        "view_cone_min_configured": text("configured_view_cone_min"),
        "view_cone_max_configured": text("configured_view_cone_max"),
        "view_cone_max_lookup_limit": text("lookup_view_cone_max"),
        "showers_per_run_min": min(showers_per_run_values),
        "showers_per_run_max": max(showers_per_run_values),
        "showers_per_run_used": _summary_integer_text(showers_per_run_values),
        "showers_per_run_configured_max": max(configured_showers_per_run_values),
        "showers_per_run_configured": _summary_integer_text(configured_showers_per_run_values),
        "total_showers": int(sum(showers_per_run_values)),
    }
