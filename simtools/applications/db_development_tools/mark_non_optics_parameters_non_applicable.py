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

import yaml

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def initialize_config():
    config = configurator.Configurator(
        description=(
            "Mark all non-structure related parameters in the MST-Structure "
            "DB entries as non-applicable. "
            "The definition of what is non-structure related is given in the "
            "input yaml file which should look like sections.yml "
            "(see reports repo)."
        )
    )

    config.parser.add_argument(
        "--sections",
        help="Provide a sections.yml file (see reports repo).",
        type=str,
        required=True,
    )
    return config


def load_non_optic_parameters(sections_file):
    with open(sections_file, encoding="utf-8") as stream:
        parameter_categories = yaml.safe_load(stream)

    non_optic_categories = [
        "Readout electronics",
        "Trigger",
        "Photon conversion",
        "Camera",
        "Unnecessary",
    ]

    non_optic_parameters = []
    for category in non_optic_categories:
        non_optic_parameters.extend(parameter_categories[category])
    return non_optic_parameters


def process_site_version(db, db_config, non_optic_parameters, site, version):
    for par_now in non_optic_parameters:
        db.update_parameter_field(
            db_name=db_config["db_simulation_model"],
            telescope=f"{site}-MST-Structure-D",
            version=version,
            parameter=par_now,
            field="Applicable",
            new_value=False,
        )

    pars = db.read_mongo_db(
        db_name=db_config["db_simulation_model"],
        telescope_model_name=f"{site}-MST-Structure-D",
        model_version=version,
        run_location="",
        collection_name="telescope",
        write_files=False,
        only_applicable=False,
    )

    for par_now in non_optic_parameters:
        if par_now in pars:
            assert pars[par_now]["Applicable"] is False


def main():
    config = initialize_config()
    args_dict, db_config = config.initialize(db_config=True, simulation_model="telescope")
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    non_optic_parameters = load_non_optic_parameters(args_dict["sections"])

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

    for version in versions:
        for site in ["North", "South"]:
            process_site_version(db, db_config, non_optic_parameters, site, version)


if __name__ == "__main__":
    main()
