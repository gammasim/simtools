#!/usr/bin/python3

"""Produce a markdown file with production version descriptions."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.reporting.docs_production_summary import write_production_summary_markdown

_ARGUMENTS = (cli.SIMULATION_MODELS_PATH(required=True),)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    output_file = app_context.args.get("output_file")
    if output_file is None:
        raise ValueError("Missing required argument output_file.")

    output_file_path = app_context.io_handler.get_output_file(output_file)
    write_production_summary_markdown(app_context.args["simulation_models_path"], output_file_path)

    app_context.logger.info(f"Production summary written to {output_file_path}")


if __name__ == "__main__":
    main()
