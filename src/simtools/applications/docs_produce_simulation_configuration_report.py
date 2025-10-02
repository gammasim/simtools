#!/usr/bin/python3

r"""Produces a markdown file for a given simulation configuration."""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.io import io_handler
from simtools.reporting.docs_auto_report_generator import ReportGenerator
from simtools.utils import general as gen


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=("Produce a markdown report for model parameters."),
    )

    config.parser.add_argument(
        "--all_model_versions",
        action="store_true",
        help="Produce reports for all model versions.",
    )

    return config.initialize(
        db_config=True,
        simulation_model=["model_version"],
        simulation_configuration=["software"],
    )


def main():
    """Produce a markdown file for a given simulation configuration."""
    app_context = startup_application(_parse)

    output_path = app_context.io_handler.get_output_directory(
        f"{app_context.args.get('model_version')}"
    )

    report_generator = ReportGenerator(db_config=db_config, args=args, output_path=output_path)
    report_generator.auto_generate_simulation_configuration_reports()

    app_context.logger.info(
        f"Configuration reports for {app_context.args.get('simulation_software')} "
        "produced successfully."
    )
    app_context.logger.info(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
