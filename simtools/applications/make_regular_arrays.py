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
    site (str, required)
        observatory site (e.g., North or South).
    model_version (str, optional)
        Model version to use (e.g., prod6). If not provided, the latest version is used.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    Runtime < 10 s.

    .. code-block:: console

        simtools-make-regular-arrays --site=North

"""

import logging
import os
from pathlib import Path

import astropy.units as u

import simtools.data_model.model_data_writer as writer
import simtools.utils.general as gen
from simtools import db_handler
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.layout.array_layout import ArrayLayout


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
    args_dict, db_config = config.initialize(db_config=True, site_model=True, output=True)

    label = "make_regular_arrays"

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _io_handler = io_handler.IOHandler()

    corsika_pars = gen.collect_data_from_file_or_dict(
        _io_handler.get_input_data_file("parameters", "corsika_parameters.yml"), None
    )

    # Reading site parameters from DB
    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    site_pars_db = db.get_site_parameters(
        site=args_dict["site"], model_version=args_dict["model_version"]
    )

    layout_center_data = {}
    layout_center_data["center_lat"] = float(site_pars_db["ref_lat"]["Value"]) * u.deg
    layout_center_data["center_lon"] = float(site_pars_db["ref_long"]["Value"]) * u.deg
    layout_center_data["center_alt"] = float(site_pars_db["altitude"]["Value"]) * u.m
    layout_center_data["EPSG"] = site_pars_db["EPSG"]["Value"]
    corsika_telescope_data = {}
    corsika_telescope_data["corsika_obs_level"] = layout_center_data["center_alt"]
    corsika_telescope_data["corsika_sphere_center"] = corsika_pars["corsika_sphere_center"]
    corsika_telescope_data["corsika_sphere_radius"] = corsika_pars["corsika_sphere_radius"]

    # Telescope distances for 4 tel square arrays
    # !HARDCODED
    telescope_distance = {"LST": 57.5 * u.m, "MST": 70 * u.m, "SST": 80 * u.m}

    for array_name in ["1SST", "4SST", "1MST", "4MST", "1LST", "4LST"]:
        logger.info(f"Processing array {array_name}")
        layout = ArrayLayout(
            site=args_dict["site"],
            mongo_db_config=db_config,
            label=label,
            name=f"{args_dict['site']}-{array_name}",
            layout_center_data=layout_center_data,
            corsika_telescope_data=corsika_telescope_data,
        )

        tel_size = array_name[1:4]

        # Single telescope at the center
        if array_name[0] == "1":
            layout.add_telescope(
                telescope_name=tel_size + "-01",
                crs_name="ground",
                xx=0 * u.m,
                yy=0 * u.m,
                tel_corsika_z=0 * u.m,
            )
        # 4 telescopes in a regular square grid
        else:
            layout.add_telescope(
                telescope_name=tel_size + "-01",
                crs_name="ground",
                xx=telescope_distance[tel_size],
                yy=telescope_distance[tel_size],
                tel_corsika_z=0 * u.m,
            )
            layout.add_telescope(
                telescope_name=tel_size + "-02",
                crs_name="ground",
                xx=-telescope_distance[tel_size],
                yy=telescope_distance[tel_size],
                tel_corsika_z=0 * u.m,
            )
            layout.add_telescope(
                telescope_name=tel_size + "-03",
                crs_name="ground",
                xx=telescope_distance[tel_size],
                yy=-telescope_distance[tel_size],
                tel_corsika_z=0 * u.m,
            )
            layout.add_telescope(
                telescope_name=tel_size + "-04",
                crs_name="ground",
                xx=-telescope_distance[tel_size],
                yy=-telescope_distance[tel_size],
                tel_corsika_z=0 * u.m,
            )

        layout.convert_coordinates()
        layout.print_telescope_list()
        output_file = args_dict.get("output_file", None)
        if output_file is not None:
            base_name, file_extension = os.path.splitext(output_file)
            output_file = f"{base_name}-{args_dict['site']}-{array_name}{file_extension}"
        writer.ModelDataWriter.dump(
            args_dict=args_dict,
            output_file=output_file,
            metadata=None,
            product_data=layout.export_telescope_list_table(
                crs_name="ground",
                corsika_z=False,
            ),
        )


if __name__ == "__main__":
    main()
