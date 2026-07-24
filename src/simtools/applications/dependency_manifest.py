#!/usr/bin/python3

"""Generate a dependency provenance manifest."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.dependencies import write_dependency_manifest, write_development_dependency_manifest

_DEFAULT_OUTPUT = Path("/opt/simtools/provenance/dependency-manifest.json")

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        cli.ArgumentDefinition("output_file", type=Path, default=_DEFAULT_OUTPUT),
        cli.ArgumentDefinition(
            "development",
            action="store_true",
            help="Generate provenance for an image where simtools is not installed.",
        ),
        cli.ArgumentDefinition(
            "project_file",
            type=Path,
            default=Path("pyproject.toml"),
            help="Project file used to discover direct Python dependencies in development mode.",
        ),
        cli.ArgumentDefinition(
            "build_option_files",
            type=Path,
            nargs="*",
            default=[],
            help="Build-options YAML files to merge in development mode.",
        ),
    ),
    setup_io_handler=False,
    resolve_sim_software_executables=False,
)


def main():
    """Write the requested provenance manifest."""
    args = APPLICATION.start().args
    if args["development"]:
        write_development_dependency_manifest(
            args["output_file"], args["project_file"], args["build_option_files"]
        )
        return
    write_dependency_manifest(args["output_file"])


if __name__ == "__main__":
    main()
