#!/usr/bin/python3

r"""Produces a markdown file for a given simulation configuration."""

from simtools.application_control import build_application
from simtools.reporting.docs_auto_report_generator import ReportGenerator


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.initialize_application_arguments(["all_model_versions"])


def main():
    """Produce a markdown file for a given simulation configuration."""
    app_context = build_application(
        __file__,
        description="Produce a markdown report for model parameters.",
        add_arguments_function=_add_arguments,
        initialization_kwargs={
            "db_config": True,
            "simulation_model": ["model_version"],
            "simulation_configuration": ["software"],
        },
    )

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
