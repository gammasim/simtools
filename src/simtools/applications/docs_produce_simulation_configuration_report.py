#!/usr/bin/python3

r"""Produces a markdown file for a given simulation configuration."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.reporting.docs_auto_report_generator import ReportGenerator

_ARGUMENTS = (cli.ALL_MODEL_VERSIONS,)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.IGNORE_MISSING_DESIGN_MODEL,
        cli.SIMULATION_SOFTWARE,
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    output_path = app_context.io_handler.get_output_directory()

    report_generator = ReportGenerator(args=app_context.args, output_path=output_path)
    report_generator.auto_generate_simulation_configuration_reports()

    app_context.logger.info(
        f"Configuration reports for {app_context.args.get('simulation_software')} "
        "produced successfully."
    )
    app_context.logger.info(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
