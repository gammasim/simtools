#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to modify all non-optics parameters \
    in the MST-Structure entries in the DB to non-applicable.

    This application should not be used anymore by anyone.

    Therefore, no additional documentation about this applications will be given.

"""

import logging
import argparse
import yaml

from simtools import db_handler
import simtools.config as cfg
import simtools.util.general as gen

if __name__ == "__main__":

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
        "-V",
        "--verbosity",
        dest="logLevel",
        action="store",
        default="info",
        help="Log level to print (default is INFO)",
    )

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    with open(args.sections, "r") as stream:
        parameterCatogeries = yaml.load(stream, Loader=yaml.FullLoader)

    nonOpticCatagories = ["Readout electronics", "Trigger", "Photon conversion", "Camera", "Unnecessary"]
    nonOpticParameters = list()
    for category in nonOpticCatagories:
        for parNow in parameterCatogeries[category]:
            nonOpticParameters.append(parNow)

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

    for versionNow in versions:
        for site in ["North", "South"]:
            for parNow in nonOpticParameters:
                db.updateParameterField(
                    dbName=db.DB_CTA_SIMULATION_MODEL,
                    telescope="{}-MST-Structure-D".format(site),
                    version=versionNow,
                    parameter=parNow,
                    field="Applicable",
                    newValue=False
                )
            pars = db.readMongoDB(
                dbName=db.DB_CTA_SIMULATION_MODEL,
                telescopeModelNameDB="{}-MST-Structure-D".format(site),
                modelVersion=versionNow,
                runLocation="",
                writeFiles=False,
                onlyApplicable=False
            )
            for parNow in nonOpticParameters:
                if parNow in pars:
                    assert pars[parNow]["Applicable"] is False
