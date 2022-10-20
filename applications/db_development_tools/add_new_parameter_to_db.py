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
from simtools.util import names


def main():

    parser = argparser.CommandLineParser(
        description=("Add a new parameter to the sites collection in the DB.")
    )
    parser.initialize_default_arguments(add_workflow_config=False)

    args = parser.parse_args()
    cfg.setConfigFileName(args.configFile)

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # epsgs = [32628, 32719]
    parameter = {"camera_filter_incidence_angle": "sst_photon_incidence_angle_camera_window.ecsv"}

    db = db_handler.DatabaseHandler()

    telescopes = [
        "South-SST-Structure-D",
        "South-SST-Camera-D",
        "South-SST-ASTRI-D",
    ]

    for telescopeNow in telescopes:
        for parNow, parValue in parameter.items():
            allVersions = db.getAllVersions(
                dbName=db.DB_CTA_SIMULATION_MODEL,
                telescopeModelName="-".join(telescopeNow.split("-")[1:]),
                site=names.getSiteFromTelescopeName(telescopeNow),
                parameter="camera_config_file",  # Just a random parameter to get the versions
                collectionName="telescopes",
            )
            for versionNow in allVersions:
                db.addNewParameter(
                    dbName=db.DB_CTA_SIMULATION_MODEL,
                    telescope=telescopeNow,
                    parameter=parNow,
                    version=versionNow,
                    value=parValue,
                    collectionName="telescopes",
                    Applicable=True,
                    Type=str(str),
                    File=True,
                    filePrefix=(
                        "/Users/ogueta/work/cta/aswg/simulations/simulation-model/verification/"
                        "data-from-instrument-teams/SST-Structure/incidence-angle-focal-plane/"
                    ),
                )
                pars = db.readMongoDB(
                    dbName=db.DB_CTA_SIMULATION_MODEL,
                    telescopeModelNameDB=telescopeNow,
                    modelVersion=versionNow,
                    runLocation="./",
                    collectionName="telescopes",
                    writeFiles=False,
                )
                assert pars[parNow]["Value"] == parValue


if __name__ == "__main__":
    main()
