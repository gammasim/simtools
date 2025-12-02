#!/usr/bin/python3

r"""
Produces a markdown file for a given array element, site, and model version.

The report includes detailed information on each parameter,
such as the parameter name, value, unit, description, and short description.
"""

from pathlib import Path

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
        "--all_model_versions",
        action="store_true",
        help="Produce reports for all model versions.",
    )

    config.parser.add_argument(
        "--all_sites", action="store_true", help="Produce reports for all sites."
    )

    config.parser.add_argument(
        "--observatory",
        action="store_true",
        help="Produce reports for an observatory at a given site.",
    )

    return config.initialize(
        db_config=True, simulation_model=["site", "telescope", "model_version"]
    )


def main():
    """Produce a markdown file for a given array element, site, and model version."""
    app_context = startup_application(_parse)
    output_path = app_context.io_handler.get_output_directory()

    if any(
        [
            app_context.args.get("all_telescopes"),
            app_context.args.get("all_sites"),
            app_context.args.get("all_model_versions"),
        ]
    ):
        ReportGenerator(app_context.args, output_path).auto_generate_array_element_reports()

    else:
        model_version = app_context.args["model_version"]
        ReadParameters(
            app_context.args, Path(output_path / f"{model_version}")
        ).produce_array_element_report()

        app_context.logger.info(
            f"Markdown report generated for {app_context.args['site']}"
            f" Telescope {app_context.args['telescope']} (v{model_version}):"
            f" {output_path}"
        )


if __name__ == "__main__":
    main()
