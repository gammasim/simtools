#!/usr/bin/python3

"""Compatibility alias that will be removed in the near future."""

from simtools.application.definition import ApplicationDefinition
from simtools.applications import get_array_layout
from simtools.configuration import arguments as cli

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *get_array_layout.ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.layout_selection_arguments(required=False),
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    app_context.logger.warning(
        "simtools-db-get-array-layouts-from-db will be removed in the near future; use "
        "simtools-get-array-layout instead."
    )
    get_array_layout.run(app_context)


if __name__ == "__main__":
    main()
