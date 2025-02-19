#!/usr/bin/python3

r"""
    Get a parameter entry from DB for a specific telescope or a site.

    The application receives a parameter name, a site, a telescope (if applicable) and \
    a version. It then prints out the parameter entry.

    Command line arguments
    ----------------------
    parameter (str, required)
        Parameter name

    site (str, required)
        South or North.

    telescope (str, optional)
        Telescope model name (e.g. LST-1, SST-D, ...)

    log_level (str, optional)
        Log level to print.

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

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter mirror_list \\
                --site North --telescope LSTN-01 \\
                --parameter_version 1.0.0

"""

import json
import logging
from pathlib import Path
from pprint import pprint

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io_operations import io_handler
from simtools.utils import names


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

    return config.initialize(
        db_config=True, simulation_model=["telescope", "parameter_version", "model_version"]
    )


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    # get parameter using 'parameter_version'
    if args_dict["parameter_version"] is not None:
        pars = db.get_model_parameter(
            parameter=args_dict["parameter"],
            parameter_version=args_dict["parameter_version"],
            site=args_dict["site"],
            array_element_name=args_dict["telescope"],
        )
    # get parameter using 'model_version'
    elif args_dict["model_version"] is not None:
        pars = db.get_model_parameters(
            site=args_dict["site"],
            array_element_name=args_dict.get("telescope"),
            model_version=args_dict["model_version"],
            collection=names.get_collection_name_from_parameter_name(args_dict["parameter"]),
        )
    else:
        raise ValueError("Either 'parameter_version' or 'model_version' must be provided.")
    if args_dict["parameter"] not in pars:
        raise KeyError(f"The requested parameter, {args_dict['parameter']}, does not exist.")
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
