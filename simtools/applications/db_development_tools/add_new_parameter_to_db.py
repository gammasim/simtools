#!/usr/bin/python3

"""
    Summary
    -------
    This application is used to add a new parameter to the sites collection in the DB.

    This application should not be used by anyone but expert users and not often.
    Therefore, no additional documentation about this applications will be given.

"""

import logging

import astropy.units as u

import simtools.utils.general as gen
from simtools import db_handler
from simtools.configuration import configurator
from simtools.utils import names


def main():
    config = configurator.Configurator(
        description=("Add a new parameter to the sites collection in the DB.")
    )
    args_dict, db_config = config.initialize(db_config=True, telescope_model=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    # epsgs = [32628, 32719]
    parameter = {
        "mirror_panel_shape": 1,
        "mirror_panel_diameter": 84.6 * u.cm,
    }

    telescopes = [
        "South-SST-Structure-D",
    ]

    for telescope_now in telescopes:
        for par_now, par_value in parameter.items():
            all_versions = db.get_all_versions(
                db_name=db.DB_CTA_SIMULATION_MODEL,
                telescope_model_name="-".join(telescope_now.split("-")[1:]),
                site=names.get_site_from_telescope_name(telescope_now),
                parameter="camera_config_file",  # Just a random parameter to get the versions
                collection_name="telescopes",
            )
            for version_now in all_versions:
                db.add_new_parameter(
                    db_name=db.DB_CTA_SIMULATION_MODEL,
                    telescope=telescope_now,
                    parameter=par_now,
                    version=version_now,
                    value=par_value,
                    collection_name="telescopes",
                    Applicable=True,
                    File=False,
                    file_prefix="./",
                )
                pars = db.read_mongo_db(
                    db_name=db.DB_CTA_SIMULATION_MODEL,
                    telescope_model_name_db=telescope_now,
                    model_version=version_now,
                    run_location="./",
                    collection_name="telescopes",
                    write_files=False,
                )
                if isinstance(par_value, u.Quantity):
                    assert pars[par_now]["Value"] == par_value.value
                    assert pars[par_now]["units"] == par_value.unit.to_string()
                else:
                    assert pars[par_now]["Value"] == par_value


if __name__ == "__main__":
    main()
