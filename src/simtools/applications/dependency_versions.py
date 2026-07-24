"""Export the validated simtools dependency version catalog."""

import argparse
import sys
from pathlib import Path

APPLICATION = None  # pylint: disable=invalid-name

if __package__:
    from simtools.application.definition import ApplicationDefinition
    from simtools.configuration import arguments as cli

    APPLICATION = ApplicationDefinition.for_module(  # pylint: disable=invalid-name
        __name__,
        arguments=(
            cli.ArgumentDefinition(
                "pyproject",
                help="Explicit project file (the repository is searched when omitted).",
                type=Path,
                default=None,
            ),
            cli.ArgumentDefinition(
                "format",
                help="Output format.",
                choices=("catalog", "github-output", "python-requirements", "summary"),
                default="catalog",
            ),
            cli.ArgumentDefinition(
                "extras",
                help="Optional dependency groups to include with python-requirements.",
                nargs="*",
                default=[],
            ),
        ),
        setup_io_handler=False,
        resolve_sim_software_executables=False,
    )


def main():
    """Export validated dependency configuration for automation."""
    if APPLICATION is None:
        raise RuntimeError("The dependency-versions application must be imported as simtools.")
    args = APPLICATION.start().args
    sys.stdout.write(
        _export_dependency_configuration(args["pyproject"], args["format"], args["extras"])
    )


def _main_standalone():
    """Export catalog data when this application is executed by a build workflow."""
    source_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source_root))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path)
    parser.add_argument(
        "--format",
        choices=("catalog", "github-output", "python-requirements", "summary"),
        default="catalog",
    )
    parser.add_argument("--extras", nargs="*", default=[])
    args = parser.parse_args()
    sys.stdout.write(_export_dependency_configuration(args.pyproject, args.format, args.extras))


def _export_dependency_configuration(pyproject_path, output_format, extras):
    """Load the library exporter without requiring the full application stack."""
    # pylint: disable=import-outside-toplevel
    from simtools.dependency_versions import export_dependency_configuration

    return export_dependency_configuration(pyproject_path, output_format, extras)


if __name__ == "__main__":
    _main_standalone()
