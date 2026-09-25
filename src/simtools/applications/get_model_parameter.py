#!/usr/bin/python3

"""Get a model parameter from a simulation-model repository or database."""

from pprint import pprint

from simtools.application.definition import ApplicationDefinition
from simtools.application.model_reader import require_model_reader
from simtools.configuration import arguments as cli
from simtools.io import ascii_handler

ARGUMENTS = (
    cli.ArgumentDefinition("parameter", help="Parameter name", type=str, required=True),
    cli.ArgumentDefinition(
        "output_file",
        help=(
            "Output file name for writing the parameter value, overriding the file-backed "
            "export name, or base name for ECSV export of dict-backed tables."
        ),
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "export_model_file",
        help=(
            "Export file-backed parameter data (or dict-backed table data as ECSV when "
            "--output_file is supplied)."
        ),
        action="store_true",
        required=False,
    ),
    cli.ArgumentDefinition(
        "export_model_file_as_table",
        help=(
            "Also export file-backed parameters as ECSV. Use with --export_model_file. "
            "This legacy option will be removed when file-backed parameters are replaced "
            "by table-backed ones."
        ),
        action="store_true",
        required=False,
    ),
)


def _is_row_table_dict(value):
    """Return whether a value uses the legacy row-table dictionary structure."""
    return isinstance(value, dict) and {"columns", "rows", "column_units"} <= value.keys()


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
    initialize_output=False,
)


def _export_parameter_file(app_context, model_reader, parameters):
    """Export a parameter payload and return the generated output paths."""
    parameter = app_context.args["parameter"]
    parameter_info = parameters[parameter]
    output_file = app_context.args.get("output_file")
    output_directory = app_context.io_handler.get_output_directory()

    if _is_row_table_dict(parameter_info.get("value")):
        if output_file is None:
            raise ValueError(
                "Use --output_file when exporting dict-backed parameters as an ECSV table."
            )
        table = model_reader.export_model_file(
            parameter=parameter,
            site=app_context.args["site"],
            array_element_name=app_context.args.get("telescope"),
            model_version=app_context.args.get("model_version"),
            parameter_version=app_context.args.get("parameter_version"),
            export_file_as_table=True,
            dest=output_directory,
        )
        table_file = app_context.io_handler.get_output_file(output_file).with_suffix(".ecsv")
        table.write(table_file, format="ascii.ecsv", overwrite=True)
        return [table_file]

    if not parameter_info.get("file"):
        raise ValueError(f"Parameter {parameter!r} does not reference a model file.")

    table = model_reader.export_model_file(
        parameter=parameter,
        site=app_context.args["site"],
        array_element_name=app_context.args.get("telescope"),
        model_version=app_context.args.get("model_version"),
        parameter_version=app_context.args.get("parameter_version"),
        export_file_as_table=app_context.args["export_model_file_as_table"],
        dest=output_directory,
    )
    source_file = output_directory / parameter_info["value"]
    model_output_file = (
        app_context.io_handler.get_output_file(output_file) if output_file else source_file
    )
    output_files = []
    if app_context.args["export_model_file"]:
        if model_output_file != source_file:
            model_output_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.rename(model_output_file)
        output_files.append(model_output_file)
    if app_context.args["export_model_file_as_table"]:
        table_output_file = model_output_file.with_suffix(".ecsv")
        table.write(table_output_file, format="ascii.ecsv", overwrite=True)
        output_files.append(table_output_file)
        if not app_context.args["export_model_file"] and source_file.exists():
            source_file.unlink()
    return output_files


def run(app_context):
    """Run the model-parameter retrieval using an initialized application context."""
    model_reader = require_model_reader(app_context.model_reader)
    parameters = model_reader.get_model_parameter(
        parameter=app_context.args["parameter"],
        site=app_context.args["site"],
        array_element_name=app_context.args.get("telescope"),
        parameter_version=app_context.args.get("parameter_version"),
        model_version=app_context.args.get("model_version"),
    )

    if app_context.args["export_model_file"] or app_context.args["export_model_file_as_table"]:
        for output_file in _export_parameter_file(app_context, model_reader, parameters):
            app_context.logger.info("Exported parameter output to %s", output_file)
        return

    parameter_data = parameters[app_context.args["parameter"]]
    if app_context.args["output_file"] is not None:
        data = dict(parameter_data)
        data.pop("_id", None)
        data.pop("entry_date", None)
        ascii_handler.write_data_to_file(
            data=data,
            output_file=app_context.io_handler.get_output_file(app_context.args["output_file"]),
        )
    else:
        pprint(parameter_data)


def main():
    """See CLI description."""
    run(APPLICATION.start())


if __name__ == "__main__":
    main()
