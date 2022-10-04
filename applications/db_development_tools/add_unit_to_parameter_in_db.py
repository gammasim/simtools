#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to add a unit to parameters in the DB.

    This application should not be used by anyone but expert users and only in unusual cases.
    Therefore, no additional documentation about this applications will be given.

"""

import logging

import simtools.config as cfg
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler


def main():

    parser = argparser.CommandLineParser(description=("Add a unit field to a parameter in the DB."))
    parser.initialize_default_arguments(add_workflow_config=False)

    args = parser.parse_args()
    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # parsToUpdate = ["altitude"]
    parsToUpdate = ["ref_long", "ref_lat"]

    # units = ["m"]
    units = ["deg", "deg"]

    db = db_handler.DatabaseHandler()

    for site in ["North", "South"]:
        for parNow, unitNow in zip(parsToUpdate, units):
            allVersions = db.getAllVersions(
                dbName=db.DB_CTA_SIMULATION_MODEL,
                site=site,
                parameter=parNow,
                collectionName="sites",
            )
            for versionNow in allVersions:
                db.updateParameterField(
                    dbName=db.DB_CTA_SIMULATION_MODEL,
                    site=site,
                    version=versionNow,
                    parameter=parNow,
                    field="units",
                    newValue=unitNow,
                    collectionName="sites",
                )

                sitePars = db.getSiteParameters(site, versionNow)
                assert sitePars[parNow]["units"] == unitNow


if __name__ == "__main__":
    main()
