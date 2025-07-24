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

import json
import logging
from pathlib import Path
from pprint import pprint

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io import io_handler


def _parse():
    config = configurator.Configurator(
        description=(
            "Get a parameter entry from DB for a specific telescope or a site. "
            "The application receives a parameter name, a site, a telescope (if applicable), "
            "and a version. It then prints out the parameter entry. "
        )
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


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

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
        table_file = Path(io_handler.IOHandler().get_output_directory()) / f"{param_value}"
        logger.info(f"Exported model file {param_value} to {table_file}")
        if table and table_file.suffix != ".ecsv":
            table.write(table_file.with_suffix(".ecsv"), format="ascii.ecsv", overwrite=True)
            logger.info(f"Exported model file {param_value} to {table_file.with_suffix('.ecsv')}")

    if args_dict["output_file"] is not None:
        _output_file = (
            Path(io_handler.IOHandler().get_output_directory()) / args_dict["output_file"]
        )
        pars[args_dict["parameter"]].pop("_id")
        pars[args_dict["parameter"]].pop("entry_date")
        with open(_output_file, "w", encoding="utf-8") as json_file:
            json.dump(pars[args_dict["parameter"]], json_file, indent=4)
    else:
        pprint(pars[args_dict["parameter"]])


if __name__ == "__main__":
    main()
