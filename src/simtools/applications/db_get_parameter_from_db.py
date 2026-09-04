#!/usr/bin/python3

"""Get a parameter entry from DB for a specific telescope or a site."""

from pprint import pprint

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.db import db_handler
from simtools.io import ascii_handler

_ARGUMENTS = (
    cli.ArgumentDefinition("parameter", help="Parameter name", type=str, required=True),
    cli.ArgumentDefinition(
        "output_file",
        help=(
            "Output file name for writing the DB entry, overriding file-backed export name, "
            "or base name for ECSV export of dict-backed tables."
        ),
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "export_model_file",
        help=(
            "Export parameter data (model files for file-backed parameters; ECSV for "
            "dict-backed table parameters)."
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


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
    initialize_model_reader=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    db = db_handler.DatabaseHandler()

    if app_context.args["export_model_file"] or app_context.args["export_model_file_as_table"]:
        output_files = db.export_parameter_data(
            parameter=app_context.args["parameter"],
            site=app_context.args["site"],
            array_element_name=app_context.args.get("telescope"),
            parameter_version=app_context.args.get("parameter_version"),
            model_version=app_context.args.get("model_version"),
            output_file=app_context.args.get("output_file"),
            export_model_file=app_context.args["export_model_file"],
            export_model_file_as_table=app_context.args["export_model_file_as_table"],
        )
        for output_file in output_files:
            app_context.logger.info(f"Exported parameter output to {output_file}")
        return

    pars = db.get_model_parameter(
        parameter=app_context.args["parameter"],
        site=app_context.args["site"],
        array_element_name=app_context.args.get("telescope"),
        parameter_version=app_context.args.get("parameter_version"),
        model_version=app_context.args.get("model_version"),
    )

    if app_context.args["output_file"] is not None:
        pars[app_context.args["parameter"]].pop("_id", None)
        pars[app_context.args["parameter"]].pop("entry_date", None)
        ascii_handler.write_data_to_file(
            data=pars[app_context.args["parameter"]],
            output_file=app_context.io_handler.get_output_file(app_context.args["output_file"]),
        )
    else:
        pprint(pars[app_context.args["parameter"]])


if __name__ == "__main__":
    main()
