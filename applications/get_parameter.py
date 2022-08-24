#!/usr/bin/python3

import logging
from pprint import pprint

import simtools.config as cfg
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler

if __name__ == "__main__":

    parser = argparser.CommandLineParser(
        description=(
            "Get a parameter entry from DB for a specific telescope. "
            "The application receives a parameter name and optionally a version. "
            "It then prints out the parameter entry. "
            "If no version is provided, the entries of the last 5 versions are printed."
        )
    )
    parser.initialize_telescope_model_arguments()
    parser.add_argument("-p", "--parameter", help="Parameter name", type=str, required=True)
    parser.initialize_default_arguments()

    args = parser.parse_args()
    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    if not cfg.get("useMongoDB"):
        raise ValueError("This application works only with MongoDB and you asked not to use it")

    db = db_handler.DatabaseHandler()

    if args.model_version == "all":
        raise NotImplementedError("Printing last 5 versions is not implemented yet.")
    else:
        version = args.model_version
    pars = db.getModelParameters(args.site, args.telescope, version)
    print()
    pprint(pars[args.parameter])
    print()
