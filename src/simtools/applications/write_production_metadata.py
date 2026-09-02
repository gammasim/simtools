#!/usr/bin/python3

"""Write or check production-job metadata manifests."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.production_configuration.production_metadata import write_production_metadata

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "production_path",
        help="Directory containing production job output directories.",
        type=str,
        required=True,
    ),
    cli.ArgumentDefinition(
        "job_grid_file",
        help="Authoritative job grid used to reconstruct resolved production configuration.",
        type=str,
        required=False,
    ),
    cli.ArgumentDefinition(
        "check",
        help="Validate existing manifests without writing files.",
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "overwrite",
        help="Overwrite existing metadata manifests in write mode.",
        action="store_true",
        default=False,
    ),
)


def _post_parse(args_dict, _config_sources, parser):
    """Validate write/check mode arguments."""
    if not args_dict.get("check") and not args_dict.get("job_grid_file"):
        parser.error(
            "'--job_grid_file' is required when writing production metadata; "
            "filenames alone are not an authoritative configuration source."
        )


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=_ARGUMENTS,
    initialize_output=False,
    post_parse=_post_parse,
)


def main():
    """Run the production metadata writer/checker."""
    app_context = APPLICATION.start()
    write_production_metadata(app_context.args)


if __name__ == "__main__":
    main()
