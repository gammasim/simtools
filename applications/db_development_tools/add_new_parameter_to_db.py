#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to add a new parameter to the sites collection in the DB.

    This application should not be used by anyone but expert users and not often.
    Therefore, no additional documentation about this applications will be given.

"""

import logging

import simtools.config as cfg
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler


def main():

    parser = argparser.CommandLineParser(
        description=("Add a new parameter to the sites collection in the DB.")
    )
    parser.initialize_default_arguments(add_workflow_config=False)

    args = parser.parse_args()
    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    epsgs = [32628, 32719]

    db = db_handler.DatabaseHandler()

    for site, epsgSite in zip(["North", "South"], epsgs):
        allVersions = db.getAllVersions(
            dbName=db.DB_CTA_SIMULATION_MODEL,
            site=site,
            parameter="altitude",
            collectionName="sites",
        )
        for versionNow in allVersions:
            db.addNewParameter(
                dbName=db.DB_CTA_SIMULATION_MODEL,
                site=site,
                parameter="EPSG",
                version=versionNow,
                value=epsgSite,
                collectionName="sites",
                Applicable=True,
                Type=str(int),
                File=False,
            )
            sitePars = db.getSiteParameters(site, versionNow)
            assert sitePars["EPSG"]["Value"] == epsgSite


if __name__ == "__main__":
    main()
