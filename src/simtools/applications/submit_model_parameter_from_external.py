#!/usr/bin/python3
"""Submit a model parameter value and corresponding metadata through the command line."""

import json
from pathlib import Path

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import instrument
from simtools.data_model.json_validation import validate_finite_json_values

_VALUE_ARGUMENT = "value"

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "parameter", type=str, required=True, help="Parameter for simulation model"
    ),
    cli.ArgumentDefinition("instrument", type=instrument, required=True, help="Instrument name"),
    cli.SITE(required=True),
    cli.ArgumentDefinition("parameter_version", type=str, required=True, help="Parameter version"),
    cli.ArgumentDefinition(
        "model_parameter_schema_version",
        type=str,
        help="Model-parameter schema version to use for validation and value interpretation",
    ),
    cli.ArgumentDefinition(
        _VALUE_ARGUMENT,
        type=str,
        nargs="+",
        required=True,
        help=(
            "Model parameter value: a number, a number with a unit, or a list of values "
            "with units. Examples: --value=5, --value='5 km', --value='5 cm, 0.5 deg'"
        ),
    ),
    cli.ArgumentDefinition(
        "input_meta",
        help="meta data file(s) associated to input data (wildcards or list of files allowed)",
        type=str,
        nargs="+",
    ),
    cli.ArgumentDefinition(
        "check_parameter_version",
        help=(
            "Fail if the parameter version already exists in the model repository "
            "(skipped with --ignore_existing_parameter_version)."
        ),
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.IGNORE_EXISTING_PARAMETER_VERSION,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    needs=("model", "corsika", "output"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    model_parameter_schema_version = app_context.args.get("model_parameter_schema_version")
    value = app_context.args[_VALUE_ARGUMENT]
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    data_writer = writer.ModelDataWriter(context=app_context)
    parameter_type = data_writer.get_parameter_type_for_schema(
        app_context.args["parameter"],
        model_parameter_schema_version,
    )

    if parameter_type == "dict":
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Structured model-parameter values must be valid JSON") from exc
        validate_finite_json_values(value)

    if app_context.args.get("output_path"):
        output_path = app_context.paths.get_output_directory(
            sub_dir=app_context.args.get("parameter")
        )
    else:
        output_path = None

    writer.ModelDataWriter.write_model_parameter(
        parameter_name=app_context.args["parameter"],
        value=value,
        instrument=app_context.args["instrument"],
        parameter_version=app_context.args["parameter_version"],
        output_file=Path(
            app_context.args["parameter"] + "-" + app_context.args["parameter_version"] + ".json"
        ),
        output_path=output_path,
        metadata_input_dict=app_context.args,
        check_for_existing_parameter=app_context.args.get("check_parameter_version", False),
        model_parameter_schema_version=model_parameter_schema_version,
        context=app_context,
    )


if __name__ == "__main__":
    main()
