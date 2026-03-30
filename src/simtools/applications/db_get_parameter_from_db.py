#!/usr/bin/python3

r"""
    Get a parameter entry from DB for a specific telescope or a site.

    The application supports three output modes:

    1. Print the database entry to stdout.
    2. Write the database entry to a JSON or YAML file using output_file.
    3. Export table-type model parameters using export_model_file.

    The export_model_file mode is type-dependent:

    - File-backed parameters are exported with their original file name from the database.
    - Dict-backed table parameters are exported as ECSV, using output_file as the base name.

    For file-backed parameters, export_model_file_as_table can be added to also write an
    ECSV representation next to the exported file.

    Command line arguments
    ----------------------
    parameter (str, required)
        Parameter name

    parameter_version (str, optional)
        Parameter version

    model_version (str, required)
        Model version

    site (str, required)
        South or North.

    telescope (str, optional)
        Telescope model name (e.g. LST-1, SST-D, ...)

    output_file (str, optional)
        Output file name for writing the database entry, or base file name for
        exporting dict-backed tables as ECSV.

    export_model_file (bool, optional)
        Export parameter data. File-backed parameters are written as model files.
        Embedded dict-typed table parameters are written as ECSV using output_file.

    export_model_file_as_table (bool, optional)
        Export file-backed parameters as astropy tables in addition to the
        original file export. Use together with export_model_file.

    Raises
    ------
    KeyError in case the parameter requested does not exist in the model parameters.

    Example
    -------
    Print the mirror_list parameter entry used for a given model_version.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter mirror_list \\
                --site North --telescope LSTN-01 \\
                --model_version 5.0.0

    Write the database entry for a parameter to a JSON file.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter array_element_position_ground \\
                --site North --telescope LSTN-01 \\
                --parameter_version 6.0.0 \\
                --output_file array_element_position_ground.json

    Export a file-backed parameter using the original file name stored in the database.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter mirror_list \\
                --site North --telescope LSTN-01 \\
                --parameter_version 1.0.0 \\
                --export_model_file

    Export a file-backed parameter and also write an ECSV table representation.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter mirror_reflectivity \\
                --site North --telescope LSTN-01 \\
                --model_version 6.0.2 \\
                --export_model_file --export_model_file_as_table

    Export a dict-backed table parameter as ECSV. The .ecsv suffix is added automatically.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter fadc_pulse_shape \\
                --site North --telescope LSTN-01 \\
                --parameter_version 2.0.0 \\
                --export_model_file --output_file fadc_pulse_shape

"""

from pprint import pprint

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io import ascii_handler


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=("Export a parameter entry from model parameter database."),
    )

    config.parser.add_argument("--parameter", help="Parameter name", type=str, required=True)
    config.parser.add_argument(
        "--output_file",
        help=(
            "Output file name for writing the DB entry, or base name for ECSV export of "
            "dict-backed tables."
        ),
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--export_model_file",
        help=(
            "Export parameter data. File-backed parameters are written as files; "
            "embedded dict-typed table parameters are written as ECSV using --output_file."
        ),
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--export_model_file_as_table",
        help=(
            "Also export file-backed parameters as ECSV. Use together with "
            "--export_model_file. "
            "(legacy option; as file-backed parameters will be replaced by table-backed ones, "
            "this option will be removed in the future)"
        ),
        action="store_true",
        required=False,
    )
    return config.initialize(
        db_config=True, simulation_model=["telescope", "parameter_version", "model_version"]
    )


def main():
    """Get a parameter entry from DB for a specific telescope or a site."""
    app_context = startup_application(_parse)

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
        pars[app_context.args["parameter"]].pop("_id")
        pars[app_context.args["parameter"]].pop("entry_date")
        ascii_handler.write_data_to_file(
            data=pars[app_context.args["parameter"]],
            output_file=app_context.io_handler.get_output_file(app_context.args["output_file"]),
        )
    else:
        pprint(pars[app_context.args["parameter"]])


if __name__ == "__main__":
    main()
