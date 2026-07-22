#!/usr/bin/python3

r"""
Produce a model parameter report per array element.

The markdown reports include detailed information on each parameter,
comparing their values over various model versions.
Currently only implemented for telescopes.
"""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.reporting.docs_auto_report_generator import ReportGenerator
from simtools.reporting.docs_read_parameters import ReadParameters

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "all_telescopes", action="store_true", help="Produce reports for all telescopes."
    ),
    cli.ArgumentDefinition("all_sites", action="store_true", help="Produce reports for all sites."),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.IGNORE_MISSING_DESIGN_MODEL,
        cli.SITE,
        cli.TELESCOPE,
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    output_path = app_context.io_handler.get_output_directory()

    if any([app_context.args.get("all_telescopes"), app_context.args.get("all_sites")]):
        ReportGenerator(
            app_context.args,
            output_path,
        ).auto_generate_parameter_reports()

    else:
        ReadParameters(
            app_context.args,
            output_path,
        ).produce_model_parameter_reports()

        app_context.logger.info(
            f"Markdown report generated for {app_context.args['site']}"
            f"Telescope {app_context.args['telescope']}: {output_path}"
        )


if __name__ == "__main__":
    main()
