#!/usr/bin/python3
"""Submit a model parameter value and corresponding metadata through the command line."""

from pathlib import Path

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import instrument
from simtools.simtel import simtel_table_reader

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "parameter", type=str, required=True, help="Parameter for simulation model"
    ),
    cli.ArgumentDefinition("instrument", type=instrument, required=True, help="Instrument name"),
    cli.ArgumentDefinition("site", type=str, required=True, help="Site location"),
    cli.ArgumentDefinition("parameter_version", type=str, required=True, help="Parameter version"),
    cli.ArgumentDefinition(
        "model_parameter_schema_version",
        type=str,
        required=False,
        help="Model-parameter schema version to use for validation and value interpretation",
    ),
    cli.ArgumentDefinition(
        "value",
        type=str,
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
        required=False,
        nargs="+",
    ),
    cli.ArgumentDefinition(
        "value_data_path",
        help="Directory containing files referenced by table-valued parameters.",
        type=Path,
        default=None,
    ),
    cli.ArgumentDefinition(
        "check_parameter_version",
        help="Check if the parameter version exists in the database",
        action="store_true",
    ),
    cli.ArgumentDefinition(
        "meta_parameter",
        help=(
            "If set, treat the submitted parameter as a sim_telarray metaparameter (written into sim_telarray metadata)."
        ),
        action="store_true",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    model_parameter_schema_version = app_context.args.get("model_parameter_schema_version")
    value = app_context.args["value"]
    data_writer = writer.ModelDataWriter()
    parameter_type = data_writer.get_parameter_type_for_schema(
        app_context.args["parameter"],
        model_parameter_schema_version,
    )

    if parameter_type == "dict":
        value = simtel_table_reader.resolve_dict_parameter_value(
            value,
            app_context.args["parameter"],
            app_context.args.get("value_data_path"),
        )

    if app_context.args.get("output_path"):
        output_path = app_context.io_handler.get_output_directory(
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
        check_db_for_existing_parameter=app_context.args.get("check_parameter_version", False),
        model_parameter_schema_version=model_parameter_schema_version,
        meta_parameter=app_context.args.get("meta_parameter", False),
    )


if __name__ == "__main__":
    main()
