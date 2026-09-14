#!/usr/bin/python3

"""Compatibility alias that will be removed in the near future."""

from simtools.application.definition import ApplicationDefinition
from simtools.applications import get_model_parameter
from simtools.configuration import arguments as cli

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *get_model_parameter.ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    app_context.logger.warning(
        "simtools-db-get-parameter-from-db will be removed in the near future; use "
        "simtools-get-model-parameter instead."
    )
    get_model_parameter.run(app_context)


if __name__ == "__main__":
    main()
