#!/usr/bin/python3

r"""
Produce a model parameter report per array element.

The markdown reports include detailed information on each parameter,
comparing their values over various model versions.
Currently only implemented for telescopes.
"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.reporting.docs_auto_report_generator import ReportGenerator
from simtools.reporting.docs_read_parameters import ReadParameters


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=("Produce a markdown report for model parameters."),
    )

    config.parser.add_argument(
        "--all_telescopes",
        action="store_true",
        help="Produce reports for all telescopes.",
    )

    config.parser.add_argument(
        "--all_sites", action="store_true", help="Produce reports for all sites."
    )

    return config.initialize(db_config=True, simulation_model=["site", "telescope"])


def main():
    """Produce a model parameter report per array element."""
    app_context = startup_application(_parse)
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
