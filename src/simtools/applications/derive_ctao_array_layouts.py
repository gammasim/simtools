#!/usr/bin/python3

r"""
    Derive CTAO array layouts definitions using CTAO common identifiers repository.

    Retrieves the CTAO array layouts definitions from the CTAO common identifiers repository
    and synchronizes them with the model parameter definition in 'array_layouts' for all
    CTAO sites.

    Requires access to the CTAO common identifiers repository. Future versions of the
    common identifiers might be stored in a CTAO technical database.

    Command line arguments
    ----------------------
    site (str)
        CTAO site (North or South).
    parameter_version (str)
        Model parameter version.
    repository_url (str)
        URL or path of the CTAO common identifiers repository.
    repository_branch (str )
        Repository branch to use for CTAO common identifiers.
    updated_parameter_version (str)
        Updated parameter version.

    Example
    -------

    Derive the CTAO array layouts definitions for the North site using the CTAO common identifiers
    repository. Merge the retrieved array layouts with the model parameter definition in
    'array_layouts' (in this example for site North and parameter version 2.0.0). Write the
    updated array layouts and metadata using the parameter version indicated by the
    'updated_parameter_version' argument.

    .. code-block:: console

        simtools-derive-ctao-array-layouts --site North \
            --repository_url "https://gitlab.cta-observatory.org/" \
            "cta-computing/common/identifiers/-/raw/" \
            --repository_branch main \
            --site North --parameter_version 2.0.0 \
            --updated_parameter_version 3.0.0
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler
from simtools.layout.ctao_array_layouts import (
    merge_array_layouts,
    retrieve_array_layouts,
    write_array_layouts,
)


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description="Derive CTAO array layouts from CTAO common identifiers repository.",
    )
    config.parser.add_argument(
        "--repository_url",
        help="URL or path of the CTAO common identifiers repository.",
        type=str,
        default="https://gitlab.cta-observatory.org/cta-computing/common/identifiers/-/raw/",
    )
    config.parser.add_argument(
        "--repository_branch",
        help="Repository branch to use for CTAO common identifiers.",
        type=str,
        default="main",
        required=False,
    )
    config.parser.add_argument(
        "--updated_parameter_version",
        help="Updated parameter version.",
        type=str,
        required=False,
    )
    return config.initialize(
        db_config=True, output=True, simulation_model=["site", "parameter_version", "model_version"]
    )


def main():  # noqa: D103
    args_dict, db_config = _parse(Path(__file__).stem)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    ctao_array_layouts = retrieve_array_layouts(
        site=args_dict["site"],
        repository_url=args_dict["repository_url"],
        branch_name=args_dict["repository_branch"],
    )

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
    db_array_layouts = db.get_model_parameter(
        parameter="array_layouts",
        site=args_dict["site"],
        array_element_name=None,
        parameter_version=args_dict.get("parameter_version"),
        model_version=args_dict.get("model_version"),
    )
    db_array_layouts["array_layouts"].pop("_id", None)
    db_array_layouts["array_layouts"].pop("entry_date", None)
    logger.info(f"Layouts from DB: {db_array_layouts}")

    write_array_layouts(
        array_layouts=merge_array_layouts(db_array_layouts["array_layouts"], ctao_array_layouts),
        args_dict=args_dict,
        db_config=db_config,
    )


if __name__ == "__main__":
    main()
