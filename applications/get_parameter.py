#!/usr/bin/python3

import logging
from pprint import pprint

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import db_handler


def main():

    config = configurator.Configurator(
        description=(
            "Get a parameter entry from DB for a specific telescope. "
            "The application receives a parameter name and optionally a version. "
            "It then prints out the parameter entry. "
            "If no version is provided, the entries of the last 5 versions are printed."
        )
    )
    config.parser.add_argument("--parameter", help="Parameter name", type=str, required=True)
    args_dict, db_config = config.initialize(db_config=True, telescope_model=True)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongoDBConfig=db_config)

    if args_dict["model_version"] == "all":
        raise NotImplementedError("Printing last 5 versions is not implemented yet.")
    else:
        version = args_dict["model_version"]
    pars = db.getModelParameters(args_dict["site"], args_dict["telescope"], version)
    print()
    pprint(pars[args_dict["parameter"]])
    print()


if __name__ == "__main__":
    main()
