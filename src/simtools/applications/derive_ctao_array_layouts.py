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
    repository_branch (str)
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

from simtools.application_control import build_application
from simtools.db import db_handler
from simtools.layout.array_layout_utils import (
    merge_array_layouts,
    retrieve_ctao_array_layouts,
    write_array_layouts,
)


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--repository_url",
        help="URL or path of the CTAO common identifiers repository.",
        type=str,
        default="https://gitlab.cta-observatory.org/cta-computing/common/identifiers/-/raw/",
    )
    parser.add_argument(
        "--repository_branch",
        help="Repository branch to use for CTAO common identifiers.",
        type=str,
        default="main",
        required=False,
    )
    parser.add_argument(
        "--updated_parameter_version",
        help="Updated parameter version.",
        type=str,
        required=False,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={
            "db_config": True,
            "output": True,
            "simulation_model": ["site", "parameter_version", "model_version"],
        },
    )

    ctao_array_layouts = retrieve_ctao_array_layouts(
        site=app_context.args["site"],
        repository_url=app_context.args["repository_url"],
        branch_name=app_context.args["repository_branch"],
    )

    db = db_handler.DatabaseHandler()
    db_array_layouts = db.get_model_parameter(
        parameter="array_layouts",
        site=app_context.args["site"],
        array_element_name=None,
        parameter_version=app_context.args.get("parameter_version"),
        model_version=app_context.args.get("model_version"),
    )
    db_array_layouts["array_layouts"].pop("_id", None)
    db_array_layouts["array_layouts"].pop("entry_date", None)
    app_context.logger.info(f"Layouts from model parameter database: {db_array_layouts}")

    write_array_layouts(
        array_layouts=merge_array_layouts(db_array_layouts["array_layouts"], ctao_array_layouts),
        args_dict=app_context.args,
    )


if __name__ == "__main__":
    main()
