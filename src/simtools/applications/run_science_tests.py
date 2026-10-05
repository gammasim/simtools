#!/usr/bin/python3
"""Run the reusable release science-test catalogue."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.science_tests import run_release


def _post_parse(args_dict, _config_sources, _parser):
    """Keep validation-only commands free of log-file writes."""
    if args_dict.get("dry_run"):
        args_dict["disable_log_file"] = True


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        cli.ArgumentDefinition("release_dir", type=str, required=True),
        cli.ArgumentDefinition("context_file", type=str, required=True),
        cli.ArgumentDefinition("template_dir", type=str),
        cli.ArgumentDefinition("site", action="append", help="Select a required site."),
        cli.ArgumentDefinition("test", action="append", help="Select a named test."),
        cli.ArgumentDefinition("dry_run", action="store_true", help="Validate without writes."),
        cli.ArgumentDefinition(
            "allow_production",
            action="store_true",
            help="Allow explicitly selected production after grid review.",
        ),
    ),
    setup_io_handler=False,
    resolve_sim_software_executables=False,
    initialize_model_reader=False,
    excluded_standard_arguments=("test",),
    post_parse=_post_parse,
)


def main():
    """Parse the small release-runner interface and execute it."""
    args = APPLICATION.start().args
    run_release(
        args["release_dir"],
        context_file=args["context_file"],
        template_dir=args.get("template_dir"),
        sites=args.get("site"),
        tests=args.get("test"),
        dry_run=args.get("dry_run", False),
        allow_production=args.get("allow_production", False),
        application_args=args,
    )


if __name__ == "__main__":
    main()
