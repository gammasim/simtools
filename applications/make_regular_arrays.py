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
from pathlib import Path

import astropy.units as u

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import db_handler, io_handler
from simtools.layout.layout_array import LayoutArray


def main():

    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=(
            "This application creates the layout array files (ECSV) of regular arrays "
            "with one telescope at the center of the array and with 4 telescopes "
            "in a square grid. These arrays are used for trigger rate simulations. "
            "The array layout files created should be available at the data/layout directory."
        ),
    )
    args_dict, db_config = config.initialize(db_config=True)

    label = "make_regular_arrays"

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _io_handler = io_handler.IOHandler()

    corsikaPars = gen.collect_data_from_yaml_or_dict(
        _io_handler.get_input_data_file("corsika", "corsika_parameters.yml"), None
    )

    # Reading site parameters from DB
    db = db_handler.DatabaseHandler(mongoDBConfig=db_config)

    siteParsDB = dict()
    layoutCenterData = dict()
    corsikaTelescopeData = dict()
    for site in ["North", "South"]:
        siteParsDB[site] = db.get_site_parameters(site=site, modelVersion="prod5")

        layoutCenterData[site] = dict()
        layoutCenterData[site]["center_lat"] = float(siteParsDB[site]["ref_lat"]["Value"]) * u.deg
        layoutCenterData[site]["center_lon"] = float(siteParsDB[site]["ref_long"]["Value"]) * u.deg
        layoutCenterData[site]["center_alt"] = float(siteParsDB[site]["altitude"]["Value"]) * u.m
        # TEMPORARY TODO should go into DB
        layoutCenterData[site]["EPSG"] = corsikaPars["SITE_PARAMETERS"][site]["EPSG"][0]
        corsikaTelescopeData[site] = dict()
        corsikaTelescopeData[site]["corsika_obs_level"] = layoutCenterData[site]["center_alt"]
        corsikaTelescopeData[site]["corsika_sphere_center"] = corsikaPars["corsika_sphere_center"]
        corsikaTelescopeData[site]["corsika_sphere_radius"] = corsikaPars["corsika_sphere_radius"]

    # Telescope distances for 4 tel square arrays
    # !HARDCODED
    telescopeDistance = {"LST": 57.5 * u.m, "MST": 70 * u.m, "SST": 80 * u.m}

    for site in ["South", "North"]:
        for arrayName in ["1SST", "4SST", "1MST", "4MST", "1LST", "4LST"]:
            logger.info("Processing array {}".format(arrayName))
            layout = LayoutArray(
                label=label,
                name=site + "-" + arrayName,
                layoutCenterData=layoutCenterData[site],
                corsikaTelescopeData=corsikaTelescopeData[site],
            )

            telNameRoot = arrayName[1]
            telSize = arrayName[1:4]

            # Single telescope at the center
            if arrayName[0] == "1":
                layout.add_telescope(
                    telescopeName=telNameRoot + "-01",
                    crsName="corsika",
                    xx=0 * u.m,
                    yy=0 * u.m,
                    telCorsikaZ=0 * u.m,
                )
            # 4 telescopes in a regular square grid
            else:
                layout.add_telescope(
                    telescopeName=telNameRoot + "-01",
                    crsName="corsika",
                    xx=telescopeDistance[telSize],
                    yy=telescopeDistance[telSize],
                    telCorsikaZ=0 * u.m,
                )
                layout.add_telescope(
                    telescopeName=telNameRoot + "-02",
                    crsName="corsika",
                    xx=-telescopeDistance[telSize],
                    yy=telescopeDistance[telSize],
                    telCorsikaZ=0 * u.m,
                )
                layout.add_telescope(
                    telescopeName=telNameRoot + "-03",
                    crsName="corsika",
                    xx=telescopeDistance[telSize],
                    yy=-telescopeDistance[telSize],
                    telCorsikaZ=0 * u.m,
                )
                layout.add_telescope(
                    telescopeName=telNameRoot + "-04",
                    crsName="corsika",
                    xx=-telescopeDistance[telSize],
                    yy=-telescopeDistance[telSize],
                    telCorsikaZ=0 * u.m,
                )

            layout.convert_coordinates()
            layout.print_telescope_list()
            layout.export_telescope_list(crsName="corsika", corsikaZ=None)


if __name__ == "__main__":
    main()
