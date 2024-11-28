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
    Get the mirror_list parameter from the DB.

    .. code-block:: console

        simtools-db-get-parameter-from-db --parameter mirror_list \\
                --site North --telescope LSTN-01 \\
                --model_version 5.0.0

    Expected final print-out message:

    .. code-block:: console

        {'Applicable': True,
         'File': True,
         'Type': 'str',
         'Value': 'mirror_CTA-N-LST1_v2019-03-31.dat',
         'Version': '5.0.0',
         '_id': ObjectId('608834f257df2db2531b8e78'),
         'entry_date': datetime.datetime(2021, 4, 27, 15, 59, 46, tzinfo=<bson.tz_util.FixedOffset \
          object at 0x7f601dd51d80>)}

"""

import json
import logging
from pathlib import Path
from pprint import pprint

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.io_operations import io_handler


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
        "--db_collection",
        help="DB collection to which to add the file",
        default="telescopes",
        required=False,
    )
    config.parser.add_argument(
        "--output_file",
        help="output file name (if not given: print to stdout)",
        type=str,
        required=False,
    )

    return config.initialize(db_config=True, simulation_model="telescope")


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    if args_dict["db_collection"] == "configuration_sim_telarray":
        pars = db.get_model_parameters(
            args_dict["site"],
            args_dict["telescope"],
            args_dict["model_version"],
            collection="configuration_sim_telarray",
        )
    elif args_dict["db_collection"] == "configuration_corsika":
        pars = db.get_corsika_configuration_parameters(args_dict["model_version"])
    elif args_dict["telescope"] is not None:
        pars = db.get_model_parameters(
            args_dict["site"],
            args_dict["telescope"],
            args_dict["model_version"],
            collection="telescopes",
        )
    else:
        pars = db.get_site_parameters(args_dict["site"], args_dict["model_version"])
    if args_dict["parameter"] not in pars:
        raise KeyError(f"The requested parameter, {args_dict['parameter']}, does not exist.")
    if args_dict["output_file"] is not None:
        _io_handler = io_handler.IOHandler()
        pars[args_dict["parameter"]].pop("_id")
        pars[args_dict["parameter"]].pop("entry_date")
        _output_file = Path(_io_handler.get_output_directory()) / args_dict["output_file"]
        with open(_output_file, "w", encoding="utf-8") as json_file:
            json.dump(pars[args_dict["parameter"]], json_file, indent=4)
    else:
        print()
        pprint(pars[args_dict["parameter"]])
        print()


if __name__ == "__main__":
    main()
