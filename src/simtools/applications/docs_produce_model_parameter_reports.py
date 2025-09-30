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
    args_dict, db_config, logger, _io_handler = startup_application(_parse)
    output_path = _io_handler.get_output_directory()

    if any([args_dict.get("all_telescopes"), args_dict.get("all_sites")]):
        ReportGenerator(
            db_config,
            args_dict,
            output_path,
        ).auto_generate_parameter_reports()

    else:
        ReadParameters(
            db_config,
            args_dict,
            output_path,
        ).produce_model_parameter_reports()

        logger.info(
            f"Markdown report generated for {args_dict['site']}"
            f"Telescope {args_dict['telescope']}: {output_path}"
        )


if __name__ == "__main__":
    main()
