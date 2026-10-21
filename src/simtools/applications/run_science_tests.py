#!/usr/bin/python3
"""Run the reusable release science-test catalogue."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import argument_helpers
from simtools.configuration import arguments as cli
from simtools.science_tests.campaign import run_release, setup_release

RELEASE_DIR = "release_dir"
CONTEXT_FILE = "context_file"
TEMPLATE_DIR = "template_dir"
BASELINE_VERSION = "baseline_version"


def _post_parse(args_dict, _config_sources, _parser):
    """Keep validation-only commands free of log-file writes."""
    if args_dict.get("dry_run"):
        args_dict["disable_log_file"] = True


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        cli.ArgumentDefinition(
            RELEASE_DIR,
            type=str,
            required=True,
            help="Release directory containing release.yml and the site suites.",
        ),
        cli.ArgumentDefinition(
            CONTEXT_FILE,
            type=str,
            help=(
                "Context file with candidate, baseline, and production paths "
                "(default: <release_dir>/context.yml)."
            ),
        ),
        cli.ArgumentDefinition(
            TEMPLATE_DIR,
            type=str,
            help="Shared template directory (default: discovered above the release directory).",
        ),
        cli.ArgumentDefinition(
            BASELINE_VERSION, type=str, help="Baseline release label to write during --setup."
        ),
        cli.ArgumentDefinition(
            "site",
            action="extend",
            nargs="+",
            type=argument_helpers.site,
            help="Select one or more required sites; can be repeated.",
        ),
        cli.ArgumentDefinition(
            "test",
            action="extend",
            nargs="+",
            help=(
                "Select one or more tests to execute; can be repeated. Prerequisites must be "
                "selected explicitly."
            ),
        ),
        cli.ArgumentDefinition(
            "overwrite",
            action="store_true",
            help="Rebuild derived products, or archive and retry failed production after jobs end.",
        ),
        cli.ArgumentDefinition("dry_run", action="store_true", help="Validate without writes."),
        cli.ArgumentDefinition(
            "allow_production",
            action="store_true",
            help="Allow explicitly selected production after grid review.",
        ),
        cli.ArgumentDefinition(
            "preflight",
            action="store_true",
            help="Recommended: check paths and worker container access before production.",
        ),
        cli.ArgumentDefinition(
            "setup",
            action="store_true",
            help="Create a new release directory from the shared setup templates and exit.",
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
    if args.get("setup"):
        setup_release(
            args[RELEASE_DIR],
            template_dir=args.get(TEMPLATE_DIR),
            context_file=args.get(CONTEXT_FILE),
            baseline_version=args.get(BASELINE_VERSION),
            simulation_models_git_revision=args.get("simulation_models_git_revision"),
        )
        return
    context_file = args.get(CONTEXT_FILE) or str(Path(args[RELEASE_DIR]) / "context.yml")
    run_release(
        args[RELEASE_DIR],
        context_file=context_file,
        template_dir=args.get(TEMPLATE_DIR),
        sites=args.get("site"),
        tests=args.get("test"),
        dry_run=args.get("dry_run", False),
        allow_production=args.get("allow_production", False),
        preflight=args.get("preflight", False),
        overwrite=args.get("overwrite", False),
        application_args=args,
    )


if __name__ == "__main__":
    main()
