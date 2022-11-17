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
import simtools.configuration.commandline_parser as argparser
import simtools.util.general as gen
from simtools import db_handler


def main():

    parser = argparser.CommandLineParser(description=("Add a unit field to a parameter in the DB."))
    parser.initialize_default_arguments(add_workflow_config=False)

    args = parser.parse_args()
    cfg.set_config_file_name(args.config_file)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args.log_level))

    # pars_to_update = ["altitude"]
    pars_to_update = ["ref_long", "ref_lat"]

    # units = ["m"]
    units = ["deg", "deg"]

    db = db_handler.DatabaseHandler()

    for site in ["North", "South"]:
        for par_now, unit_now in zip(pars_to_update, units):
            all_versions = db.get_all_versions(
                db_name=db.DB_CTA_SIMULATION_MODEL,
                site=site,
                parameter=par_now,
                collection_name="sites",
            )
            for version_now in all_versions:
                db.update_parameter_field(
                    db_name=db.DB_CTA_SIMULATION_MODEL,
                    site=site,
                    version=version_now,
                    parameter=par_now,
                    field="units",
                    new_value=unit_now,
                    collection_name="sites",
                )

                site_pars = db.get_site_parameters(site, version_now)
                assert site_pars[par_now]["units"] == unit_now


if __name__ == "__main__":
    main()
