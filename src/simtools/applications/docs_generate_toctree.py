#!/usr/bin/python3

r"""Generate Sphinx toctree index files for the simulation model documentation.

Writes an ``index.md`` file with a toctree into each directory of the documentation source
tree listing its markdown reports, images, and subdirectories. An index containing
``<!-- simtools: keep-index -->`` is preserved so site-owned pages can define their layout.
Directories for static content, images, data files, and build output are ignored.
"""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.reporting.docs_toctree import generate_toctree

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "docs_path",
        type=Path,
        required=True,
        help="Root directory of the Sphinx documentation source.",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=_ARGUMENTS,
    needs=(),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    generate_toctree(app_context.args["docs_path"])
    app_context.logger.info(f"Toctree index files written to {app_context.args['docs_path']}")


if __name__ == "__main__":
    main()
