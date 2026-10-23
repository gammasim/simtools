#!/usr/bin/python3

r"""Assemble a multi-version documentation site from released documentation builds.

Publishes all versions listed in a version switcher file (pydata-sphinx-theme format).
Each version is downloaded as a tarball of its built HTML site (e.g., from the GitLab
generic package registry) and extracted to ``<site_path>/<version>/``; the newest version
is also copied to the site root. Identical files are replaced by relative symlinks to a
single copy, and the deployed size is checked against a size budget.

The GitLab CI job token is read from the ``CI_JOB_TOKEN`` environment variable.
"""

import os
from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.reporting.docs_pages import assemble_pages, read_version_switcher

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "switcher_file",
        type=Path,
        required=True,
        help="Version switcher file listing the versions to publish.",
    ),
    cli.ArgumentDefinition(
        "site_path",
        type=Path,
        default=Path("public"),
        help="Directory of the assembled site (must be empty or not exist).",
    ),
    cli.ArgumentDefinition(
        "package_url",
        type=str,
        default=None,
        help="URL template of the documentation tarballs with a '{version}' placeholder.",
    ),
    cli.ArgumentDefinition(
        "size_budget_mb",
        type=float,
        default=1500.0,
        help="Maximum deployed size in MB.",
    ),
    cli.ArgumentDefinition(
        "validate_only",
        action="store_true",
        help="Only validate the version switcher file.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=_ARGUMENTS,
    setup_io_handler=False,
    resolve_sim_software_executables=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    args = app_context.args

    versions = read_version_switcher(args["switcher_file"])
    app_context.logger.info(f"Valid version switcher file listing {', '.join(versions)}")
    if args.get("validate_only"):
        return
    if not args.get("package_url"):
        raise ValueError("Missing required argument package_url.")

    assemble_pages(
        switcher_file=args["switcher_file"],
        output_path=args["site_path"],
        package_url=args["package_url"],
        size_budget_mb=args["size_budget_mb"],
        token=os.getenv("CI_JOB_TOKEN"),
    )


if __name__ == "__main__":
    main()
