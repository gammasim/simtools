#!/usr/bin/python3

r"""Produces a markdown file for calibration reports."""

from simtools.application_control import build_application
from simtools.reporting.docs_auto_report_generator import ReportGenerator


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--all_model_versions",
        action="store_true",
        help="Produce reports for all model versions.",
    )


def main():
    """Produce a markdown file for calibration reports."""
    app_context = build_application(
        __file__,
        description="Produce a markdown report for calibration parameters.",
        add_arguments_function=_add_arguments,
        initialization_kwargs={"db_config": True, "simulation_model": ["model_version"]},
    )

    output_path = app_context.io_handler.get_output_directory()

    generator = ReportGenerator(args=app_context.args, output_path=output_path)
    generator.auto_generate_calibration_reports()

    if app_context.args.get("all_model_versions"):
        app_context.logger.info("Calibration reports for all model versions produced successfully.")
    else:
        app_context.logger.info(
            f"Calibration reports for model version {app_context.args.get('model_version')}"
            " produced successfully."
        )
    app_context.logger.info(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
