#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to modify all non-optics parameters \
    in the MST-Structure entries in the DB to non-applicable.

    This application should not be used anymore by anyone.

    Therefore, no additional documentation about this applications will be given.

"""

import argparse
import logging

import yaml

import simtools.config as cfg
import simtools.util.general as gen
from simtools import db_handler


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Mark all non-structure related parameters in the MST-Structure "
            "DB entries as non-applicable. "
            "The definition of what is non-structure related is given in the "
            "input yaml file which should look like sections.yml "
            "(see reports repo)."
        )
    )
    parser.add_argument(
        "-s",
        "--sections",
        help="Provide a sections.yml file (see reports repo).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_file",
        help="gammasim-tools configuration file",
        required=False,
    )
    parser.add_argument(
        "-V",
        "--verbosity",
        dest="log_level",
        action="store",
        default="info",
        help="Log level to print (default is INFO)",
    )

    args = parser.parse_args()
    cfg.set_config_file_name(args.config_file)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args.log_level))

    with open(args.sections, "r") as stream:
        parameter_catogeries = yaml.safe_load(stream)

    non_optic_catagories = [
        "Readout electronics",
        "Trigger",
        "Photon conversion",
        "Camera",
        "Unnecessary",
    ]
    non_optic_parameters = list()
    for category in non_optic_catagories:
        for par_now in parameter_catogeries[category]:
            non_optic_parameters.append(par_now)

    versions = [
        "default",
        "2016-12-20",
        "prod3_compatible",
        "post_prod3_updates",
        "2018-11-07",
        "2019-02-22",
        "2019-05-13",
        "2019-11-20",
        "2019-12-30",
        "2020-02-26",
        "2020-06-28",
        "prod4",
    ]

    db = db_handler.DatabaseHandler()

    for version_now in versions:
        for site in ["North", "South"]:
            for par_now in non_optic_parameters:
                db.update_parameter_field(
                    db_name=db.DB_CTA_SIMULATION_MODEL,
                    telescope="{}-MST-Structure-D".format(site),
                    version=version_now,
                    parameter=par_now,
                    field="Applicable",
                    new_value=False,
                )
            pars = db.read_mongo_db(
                db_name=db.DB_CTA_SIMULATION_MODEL,
                telescope_model_name_db="{}-MST-Structure-D".format(site),
                model_version=version_now,
                run_location="",
                write_files=False,
                only_applicable=False,
            )
            for par_now in non_optic_parameters:
                if par_now in pars:
                    assert pars[par_now]["Applicable"] is False


if __name__ == "__main__":
    main()
