#!/usr/bin/python3

"""
    Summary
    -------
    Get a parameter entry from DB for a specific telescope or a site.
    The application receives a parameter name, a site, a telescope (if applicable) and \
    optionally a version. It then prints out the parameter entry.
    If no version is provided, the value of the released model is printed..

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

        simtools-get-parameter --parameter mirror_list --site North --telescope LST-1\
         --model_version prod5

    Expected final print-out message:

    .. code-block:: console

        {'Applicable': True,
         'File': True,
         'Type': 'str',
         'Value': 'mirror_CTA-N-LST1_v2019-03-31.dat',
         'Version': '2020-06-28',
         '_id': ObjectId('608834f257df2db2531b8e78'),
         'entry_date': datetime.datetime(2021, 4, 27, 15, 59, 46, tzinfo=<bson.tz_util.FixedOffset \
          object at 0x7f601dd51d80>)}

"""

import logging
from pprint import pprint

import simtools.utils.general as gen
from simtools import db_handler
from simtools.configuration import configurator


def main():
    config = configurator.Configurator(
        description=(
            "Get a parameter entry from DB for a specific telescope or a site. "
            "The application receives a parameter name a site, a telescope (if applicable), "
            " and optionally a version. It then prints out the parameter entry. "
            "If no version is provided, the value of the released model is printed. "
        )
    )
    config.parser.add_argument("--parameter", help="Parameter name", type=str, required=True)
    args_dict, db_config = config.initialize(db_config=True, telescope_model=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    if args_dict["telescope"] is not None:
        pars = db.get_model_parameters(
            args_dict["site"], args_dict["telescope"], args_dict["model_version"]
        )
    else:
        pars = db.get_site_parameters(args_dict["site"], args_dict["model_version"])
    if args_dict["parameter"] not in pars:
        raise KeyError(f"The requested parameter, {args_dict['parameter']}, does not exist.")
    print()
    pprint(pars[args_dict["parameter"]])
    print()


if __name__ == "__main__":
    main()
