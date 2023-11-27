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
        Log level to print.

    Example
    -------
    Runtime < 10 s.

    .. code-block:: console

        simtools-make-regular-arrays

    The output is saved in simtools-output/make_regular_arrays.

    Expected final print-out message:

    .. code-block:: console

        INFO::layout_array(l608)::export_telescope_list::Exporting telescope list to /workdir/exter\
        nal/simtools/simtools-output/make_regular_arrays/layout/telescope_positions-North-4LS\
        T-corsika.ecsv
"""

import logging
from pathlib import Path

import astropy.units as u

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools import db_handler
from simtools.configuration import configurator
from simtools.io_operations import io_handler
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
    args_dict, db_config = config.initialize(db_config=True, output=True)

    label = "make_regular_arrays"

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _io_handler = io_handler.IOHandler()

    corsika_pars = gen.collect_data_from_yaml_or_dict(
        _io_handler.get_input_data_file("parameters", "corsika_parameters.yml"), None
    )

    # Reading site parameters from DB
    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    site_pars_db = {}
    layout_center_data = {}
    corsika_telescope_data = {}
    for site in ["North", "South"]:
        site_pars_db[site] = db.get_site_parameters(site=site, model_version="prod5")

        layout_center_data[site] = {}
        layout_center_data[site]["center_lat"] = (
            float(site_pars_db[site]["ref_lat"]["Value"]) * u.deg
        )
        layout_center_data[site]["center_lon"] = (
            float(site_pars_db[site]["ref_long"]["Value"]) * u.deg
        )
        layout_center_data[site]["center_alt"] = (
            float(site_pars_db[site]["altitude"]["Value"]) * u.m
        )
        layout_center_data[site]["EPSG"] = site_pars_db[site]["EPSG"]["Value"]
        corsika_telescope_data[site] = {}
        corsika_telescope_data[site]["corsika_obs_level"] = layout_center_data[site]["center_alt"]
        corsika_telescope_data[site]["corsika_sphere_center"] = corsika_pars[
            "corsika_sphere_center"
        ]
        corsika_telescope_data[site]["corsika_sphere_radius"] = corsika_pars[
            "corsika_sphere_radius"
        ]

    # Telescope distances for 4 tel square arrays
    # !HARDCODED
    telescope_distance = {"LST": 57.5 * u.m, "MST": 70 * u.m, "SST": 80 * u.m}

    for site in ["South", "North"]:
        for array_name in ["1SST", "4SST", "1MST", "4MST", "1LST", "4LST"]:
            logger.info(f"Processing array {array_name}")
            layout = LayoutArray(
                site=site,
                mongo_db_config=db_config,
                label=label,
                name=f"{site}-{array_name}",
                layout_center_data=layout_center_data[site],
                corsika_telescope_data=corsika_telescope_data[site],
            )

            tel_size = array_name[1:4]

            # Single telescope at the center
            if array_name[0] == "1":
                layout.add_telescope(
                    telescope_name=tel_size + "-01",
                    crs_name="corsika",
                    xx=0 * u.m,
                    yy=0 * u.m,
                    tel_corsika_z=0 * u.m,
                )
            # 4 telescopes in a regular square grid
            else:
                layout.add_telescope(
                    telescope_name=tel_size + "-01",
                    crs_name="corsika",
                    xx=telescope_distance[tel_size],
                    yy=telescope_distance[tel_size],
                    tel_corsika_z=0 * u.m,
                )
                layout.add_telescope(
                    telescope_name=tel_size + "-02",
                    crs_name="corsika",
                    xx=-telescope_distance[tel_size],
                    yy=telescope_distance[tel_size],
                    tel_corsika_z=0 * u.m,
                )
                layout.add_telescope(
                    telescope_name=tel_size + "-03",
                    crs_name="corsika",
                    xx=telescope_distance[tel_size],
                    yy=-telescope_distance[tel_size],
                    tel_corsika_z=0 * u.m,
                )
                layout.add_telescope(
                    telescope_name=tel_size + "-04",
                    crs_name="corsika",
                    xx=-telescope_distance[tel_size],
                    yy=-telescope_distance[tel_size],
                    tel_corsika_z=0 * u.m,
                )

            layout.convert_coordinates()
            layout.print_telescope_list()
            writer.ModelDataWriter.dump(
                args_dict=args_dict,
                metadata=None,
                product_data=layout.export_telescope_list_table(
                    crs_name="corsika",
                    corsika_z=False,
                ),
            )


if __name__ == "__main__":
    main()
