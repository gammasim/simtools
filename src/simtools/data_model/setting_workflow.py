"""Create and execute versioned workflows for simple model-parameter settings."""

from copy import deepcopy
from pathlib import Path

from simtools.constants import METADATA_JSON_SCHEMA, RUN_TIME_ENVIRONMENT_SCHEMA, SCHEMA_PATH
from simtools.data_model import schema
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.data_model.model_data_writer import ModelDataWriter
from simtools.io import ascii_handler
from simtools.runners import simtools_runner
from simtools.utils import general, names
from simtools.version import is_valid_semantic_version

_SUBMIT_APPLICATION = "simtools-submit-model-parameter-from-external"
_INPUT_PATH = "input/__SETTING_WORKFLOW__"
_OUTPUT_PATH = "output/__SETTING_WORKFLOW__"


def create_setting_workflow(args, model_reader=None):
    """Create validated input files, or reuse an identical setting.

    Parameters
    ----------
    args : dict
        Parameter, instrument, value, parameter_version, description, and optional
        site, source_url, workflow_runtime_file, output_path (repository root),
        model_parameter_schema_version, and standard contact arguments.
    model_reader : object, optional
        Reader used to check for an already submitted parameter version.

    Returns
    -------
    pathlib.Path
        Path to the prepared workflow configuration.

    Raises
    ------
    ValueError
        If inputs are invalid or this version has conflicting input files.
    """
    args = dict(args)
    args["instrument"] = names.validate_array_element_name(args["instrument"])
    if Path(args["parameter"]).name != args["parameter"] or args["parameter"] in {".", ".."}:
        raise ValueError("parameter must be a model parameter name, without path separators.")
    args["site"] = _setting_site(args)
    if not is_valid_semantic_version(args["parameter_version"], strict=False):
        raise ValueError("parameter_version must be a semantic version without a leading 'v'.")
    if not args["description"].strip():
        raise ValueError("A scientific description is required.")

    writer = ModelDataWriter(model_reader=model_reader)
    parameter = _validated_parameter(writer, args)
    args["model_parameter_schema_version"] = parameter["model_parameter_schema_version"]
    args["activity_id"] = general.get_uuid()
    config = _workflow_config(args)
    runtime = _runtime_configuration(args)
    metadata = _input_metadata(args)
    root = Path(args.get("output_path") or Path.cwd()).resolve()
    directory = root / "input" / args["instrument"] / args["parameter"]
    existing = _existing_setting(directory, args)
    if existing is not None:
        _check_matching_inputs(existing, config, metadata, writer, parameter)
        return existing

    writer.check_for_existing_parameter(
        args["parameter"], args["instrument"], args["parameter_version"]
    )
    activity_id = args["activity_id"]
    metadata["cta"]["activity"]["id"] = activity_id
    config_file = directory / activity_id / "config.yml"
    config_file.parent.mkdir(parents=True, exist_ok=False)
    ascii_handler.write_data_to_file(config, config_file, sort_keys=False)
    ascii_handler.write_data_to_file(metadata, config_file.with_name("input.meta.yml"))
    _write_runtime_file(runtime, config_file.parent)
    return config_file


def _setting_site(args):
    """Infer an unambiguous site and reject a conflicting explicit site."""
    inferred = names.get_site_from_array_element_name(args["instrument"])
    supplied = args.get("site")
    if supplied is not None:
        supplied = names.validate_site_name(supplied)
        allowed = general.ensure_list(inferred)
        if supplied not in allowed:
            raise ValueError(f"Site {supplied} does not match instrument {args['instrument']}.")
        return supplied
    if not isinstance(inferred, str):
        raise ValueError(f"Specify --site for instrument {args['instrument']}.")
    return inferred


def _validated_parameter(writer, args):
    """Validate simple values through the existing model data writer."""
    parameter_type = writer.get_parameter_type_for_schema(
        args["parameter"], args.get("model_parameter_schema_version")
    )
    if parameter_type in ("file", "dict"):
        raise ValueError("Use an existing derivation workflow for file or structured parameters.")
    parameter = writer.get_validated_parameter_dict(
        parameter_name=args["parameter"],
        value=args["value"],
        instrument=args["instrument"],
        parameter_version=args["parameter_version"],
        model_parameter_schema_version=args.get("model_parameter_schema_version"),
    )
    return ascii_handler.to_builtin(parameter)


def _workflow_config(args):
    """Build a portable workflow using the existing submission application."""
    configuration = {
        key: args[key]
        for key in (
            "instrument",
            "site",
            "parameter",
            "parameter_version",
            "value",
            "model_parameter_schema_version",
        )
    }
    configuration.update(
        {
            "input_meta": [f"{_INPUT_PATH}/input.meta.yml"],
            "output_path": f"{_OUTPUT_PATH}/",
            "check_parameter_version": True,
        }
    )
    config = {
        "applications": [{"application": _SUBMIT_APPLICATION, "configuration": configuration}],
        "schema_name": "application_workflow.metaschema",
        "schema_version": "0.5.0",
    }
    schema.validate_dict_using_schema(
        config, schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml", offline=True
    )
    return config


def _runtime_configuration(args):
    """Return the validated runtime configuration selected for a new workflow."""
    runtime_file = args.get("workflow_runtime_file")
    if runtime_file is None:
        return None
    runtime = ascii_handler.collect_data_from_file(runtime_file)
    schema.validate_dict_using_schema(
        runtime, schema_file=RUN_TIME_ENVIRONMENT_SCHEMA, offline=True
    )
    return runtime


def _write_runtime_file(runtime, workflow_directory):
    """Copy a validated runtime configuration into a new workflow directory."""
    if runtime is None:
        return
    ascii_handler.write_data_to_file(runtime, workflow_directory / "runtime.yml", sort_keys=False)


def _input_metadata(args):
    """Collect technical metadata and add the supplied scientific description."""
    metadata_args = {key: value for key, value in args.items() if key.startswith("user_")}
    metadata_args.update(
        {
            "instrument": args["instrument"],
            "site": args["site"],
            "label": "setting_workflow",
            "activity_id": args["activity_id"],
            "output_file": "config.yml",
            "output_file_format": "yaml",
        }
    )
    metadata = MetadataCollector(metadata_args).get_top_level_metadata()
    metadata["cta"]["product"]["description"] = args["description"].strip()
    if args.get("source_url"):
        metadata["cta"]["product"]["description"] += f" Source: {args['source_url']}"
    metadata["cta"]["product"]["data"]["model"].update(
        {"name": args["parameter"], "version": args["parameter_version"]}
    )
    metadata = ascii_handler.to_builtin(general.change_dict_keys_case(metadata, True))
    schema.validate_dict_using_schema(metadata, schema_file=METADATA_JSON_SCHEMA, offline=True)
    return metadata


def _existing_setting(directory, args):
    """Find a unique workflow already submitting this parameter version."""
    matches = []
    for config_file in sorted(directory.glob("*/config.yml")):
        config = ascii_handler.collect_data_from_file(config_file)
        for step in config.get("applications", []):
            configuration = step.get("configuration", {})
            if step.get("application") == _SUBMIT_APPLICATION and all(
                configuration.get(key) == args[key]
                for key in ("instrument", "parameter", "parameter_version")
            ):
                matches.append(config_file)
                break
    if len(matches) > 1:
        raise ValueError(f"Multiple workflows already set version {args['parameter_version']}.")
    return matches[0] if matches else None


def _check_matching_inputs(config_file, config, metadata, writer, parameter):
    """Reuse only unchanged configuration and scientific/contact metadata."""
    existing = ascii_handler.collect_data_from_file(config_file)
    comparison = deepcopy(config)
    previous = existing["applications"][0]["configuration"]
    # Compare validated values so equivalent units and numeric spellings can be reused.
    previous_parameter = _validated_parameter(writer, previous)
    previous["value"] = previous_parameter["value"]
    comparison["applications"][0]["configuration"]["value"] = parameter["value"]
    existing_metadata = ascii_handler.collect_data_from_file(
        config_file.with_name("input.meta.yml")
    )["cta"]
    expected = metadata["cta"]
    same_metadata = _setting_metadata(existing_metadata) == _setting_metadata(expected)
    if existing != comparison or previous_parameter != parameter or not same_metadata:
        raise ValueError(
            f"Conflicting inputs in {config_file}. Use a new parameter version; "
            "existing files are preserved."
        )


def _setting_metadata(metadata):
    """Compare inputs without regenerated execution IDs, times, and software details."""
    metadata = deepcopy(metadata)
    for key in ("id", "start", "end", "software"):
        metadata.get("activity", {}).pop(key, None)
    for key in ("id", "creation_time"):
        metadata.get("product", {}).pop(key, None)
    metadata.get("context", {}).pop("application_configuration", None)
    return metadata


def run_setting_workflow(config_file, args):
    """Execute a prepared setting, keeping previous outputs intact.

    Parameters
    ----------
    config_file : pathlib.Path
        Configuration returned by create_setting_workflow.
    args : dict
        Runner options, including model-source options and ignore_runtime_environment.

    Returns
    -------
    pathlib.Path
        Output directory used by this execution.
    """
    config_file = Path(config_file).resolve()
    relative = Path(*config_file.parent.parts[-3:])
    input_root = config_file.parents[3]
    root = input_root.parent
    output = root / "output" / relative
    rerun = output.exists()
    if rerun:
        output = output / "reruns" / general.get_uuid()
    runner_args = {
        **args,
        "config_file": str(config_file),
        "ignore_runtime_environment": args.get("ignore_runtime_environment", False),
        "ignore_existing_parameter_version": rerun,
    }
    runtime_file = config_file.with_name("runtime.yml")
    run_time = None
    if runtime_file.is_file() and not runner_args["ignore_runtime_environment"]:
        runtime_environment, run_time = simtools_runner.prepare_runtime_environment(runtime_file)
        runner_args["runtime_environment"] = runtime_environment
    simtools_runner.run_applications(
        runner_args,
        run_time=run_time,
        replacements={_INPUT_PATH: str(config_file.parent), _OUTPUT_PATH: str(output)},
    )
    return output
