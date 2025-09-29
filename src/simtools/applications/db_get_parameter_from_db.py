#!/usr/bin/python3

r"""
    Get a parameter entry from DB for a specific telescope or a site.

    The application receives a parameter name, a site, a telescope (if applicable) and
    a version. Allow to print the parameter entry to screen or save it to a file.
    Parameter describing a table file can be written to disk or exported as an astropy table
    (if available).

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
        Output file name. If not given, print to stdout.

    export_model_file (bool, optional)
        Export model file (if parameter describes a file).

    export_model_file_as_table (bool, optional)
        Export model file as astropy table (if parameter describes a file).

    Raises
    ------
    KeyError in case the parameter requested does not exist in the model parameters.

    Example
    -------
    Get the mirror_list parameter used for a given model_version from the DB.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter mirror_list \\
                --site North --telescope LSTN-01 \\
                --model_version 5.0.0

    Get the mirror_list parameter using the parameter_version from the DB.
    Write the mirror list to disk.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter mirror_list \\
                --site North --telescope LSTN-01 \\
                --parameter_version 1.0.0 \\
                --export_model_file

"""

from pprint import pprint

from simtools.application_startup import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io import ascii_handler


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Get a parameter entry from DB for a specific telescope or a site. "
            "The application receives a parameter name, a site, a telescope (if applicable), "
            "and a version. It then prints out the parameter entry. "
        ),
    )

    config.parser.add_argument("--parameter", help="Parameter name", type=str, required=True)
    config.parser.add_argument(
        "--output_file",
        help="output file name (if not given: print to stdout)",
        type=str,
        required=False,
    )
    config.parser.add_argument(
        "--export_model_file",
        help="Export model file (if parameter describes a file)",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--export_model_file_as_table",
        help="Export model file as astropy table (if parameter describes a file)",
        action="store_true",
        required=False,
    )

    return config.initialize(
        db_config=True, simulation_model=["telescope", "parameter_version", "model_version"]
    )


def main():
    """Get a parameter entry from DB for a specific telescope or a site."""
    args_dict, db_config, logger, _io_handler = startup_application(_parse)

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    pars = db.get_model_parameter(
        parameter=args_dict["parameter"],
        site=args_dict["site"],
        array_element_name=args_dict.get("telescope"),
        parameter_version=args_dict.get("parameter_version"),
        model_version=args_dict.get("model_version"),
    )
    if args_dict["export_model_file"] or args_dict["export_model_file_as_table"]:
        table = db.export_model_file(
            parameter=args_dict["parameter"],
            site=args_dict["site"],
            array_element_name=args_dict["telescope"],
            parameter_version=args_dict.get("parameter_version"),
            model_version=args_dict.get("model_version"),
            export_file_as_table=args_dict["export_model_file_as_table"],
        )
        param_value = pars[args_dict["parameter"]]["value"]
        table_file = _io_handler.get_output_file(param_value)
        logger.info(f"Exported model file {param_value} to {table_file}")
        if table and table_file.suffix != ".ecsv":
            table.write(table_file.with_suffix(".ecsv"), format="ascii.ecsv", overwrite=True)
            logger.info(f"Exported model file {param_value} to {table_file.with_suffix('.ecsv')}")

    if args_dict["output_file"] is not None:
        pars[args_dict["parameter"]].pop("_id")
        pars[args_dict["parameter"]].pop("entry_date")
        ascii_handler.write_data_to_file(
            data=pars[args_dict["parameter"]],
            output_file=_io_handler.get_output_file(args_dict["output_file"]),
        )
    else:
        pprint(pars[args_dict["parameter"]])


if __name__ == "__main__":
    main()
