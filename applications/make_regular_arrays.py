#!/usr/bin/python3

"""
    Summary
    -------

    This application creates the layout array files (ECSV) of regular arrays
    with one telescope at the center of the array and with 4 telescopes
    in a square grid. These arrays are used for trigger rate simulations.

    The array layout files created should be available at the data/layout directory.

    Command line arguments
    ----------------------
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    Runtime < 1 min.

    .. code-block:: console

        python applications/make_regular_arrays.py
"""

import logging

import astropy.units as u

import simtools.config as cfg
import simtools.util.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler
from simtools.layout.layout_array import LayoutArray


if __name__ == "__main__":

    parser = argparser.CommandLineParser(
        description=(
            "Calculate and plot the PSF and eff. mirror area as a function of off-axis angle "
            "of the telescope requested."
        )
    )
    parser.initialize_default_arguments()

    args = parser.parse_args()
    if args.configFile:
        cfg.setConfigFileName(args.configFile)

    label = "make_regular_arrays"

    logger = logging.getLogger()
    logger.setLevel(gen.getLogLevelFromUser(args.logLevel))

    # Hardcoded parameters - should go to DB
    logger.warning("These hardcoded parameters should go into the DB")
    hardcodedPars = {
        "North": {
            "epsg": 32628,
            "corsikaObsLevel": 2158 * u.m,
            "corsikaSphereRadius": {
                "LST": 12.5 * u.m,
                "MST": 9.6 * u.m,
                "SST": 3.0 * u.m,
            },
            "corsikaSphereCenter": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        },
        "South": {
            "epsg": 32719,
            "corsikaObsLevel": 2147 * u.m,
            "corsikaSphereRadius": {
                "LST": 12.5 * u.m,
                "MST": 9.6 * u.m,
                "SST": 3.0 * u.m,
            },
            "corsikaSphereCenter": {"LST": 16 * u.m, "MST": 9 * u.m, "SST": 3.25 * u.m},
        },
    }  # hadcodedPars

    # Reading site parameters from DB
    db = db_handler.DatabaseHandler()

    siteParsDB = dict()
    sitePars = dict()
    for site in ["North", "South"]:
        siteParsDB[site] = db.getSiteParameters(
            site=site, modelVersion="prod3_compatible"
        )

        sitePars[site] = dict()
        sitePars[site]["centerLatitude"] = (
            float(siteParsDB[site]["ref_lat"]["Value"]) * u.deg
        )
        sitePars[site]["centerLongitude"] = (
            float(siteParsDB[site]["ref_long"]["Value"]) * u.deg
        )
        sitePars[site]["centerAltitude"] = (
            float(siteParsDB[site]["altitude"]["Value"]) * u.m
        )

        sitePars[site]["epsg"] = hardcodedPars[site]["epsg"]
        sitePars[site]["corsikaObsLevel"] = hardcodedPars[site]["corsikaObsLevel"]
        sitePars[site]["corsikaSphereCenter"] = hardcodedPars[site][
            "corsikaSphereCenter"
        ]
        sitePars[site]["corsikaSphereRadius"] = hardcodedPars[site][
            "corsikaSphereRadius"
        ]

    # Telescope distances for 4 tel square arrays
    # !HARDCODED
    telescopeDistance = {"LST": 57.5 * u.m, "MST": 70 * u.m, "SST": 80 * u.m}

    for site in ["South", "North"]:
        for arrayName in ["1SST", "4SST", "1MST", "4MST", "1LST", "4LST"]:
            logger.info("Processing array {}".format(arrayName))
            layout = LayoutArray.fromKwargs(
                label=label, name=site + "-" + arrayName, **sitePars[site]
            )

            telNameRoot = arrayName[1]
            telSize = arrayName[1:4]

            # Single telescope at the center
            if arrayName[0] == "1":
                layout.addTelescope(
                    telescopeName=telNameRoot + "-01",
                    posX=0 * u.m,
                    posY=0 * u.m,
                    posZ=0 * u.m,
                )
            # 4 telescopes in a regular square grid
            else:
                layout.addTelescope(
                    telescopeName=telNameRoot + "-01",
                    posX=telescopeDistance[telSize],
                    posY=telescopeDistance[telSize],
                    posZ=0 * u.m,
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + "-02",
                    posX=-telescopeDistance[telSize],
                    posY=telescopeDistance[telSize],
                    posZ=0 * u.m,
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + "-03",
                    posX=telescopeDistance[telSize],
                    posY=-telescopeDistance[telSize],
                    posZ=0 * u.m,
                )
                layout.addTelescope(
                    telescopeName=telNameRoot + "-04",
                    posX=-telescopeDistance[telSize],
                    posY=-telescopeDistance[telSize],
                    posZ=0 * u.m,
                )

            layout.convertCoordinates()
            layout.printTelescopeList()
            layout.exportTelescopeList()
