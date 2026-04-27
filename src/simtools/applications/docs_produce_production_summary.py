#!/usr/bin/python3

r"""Produce a markdown file with production version descriptions."""

from simtools.application_control import build_application
from simtools.reporting.docs_production_summary import write_production_summary_markdown


def main():
    """See CLI description."""
    app_context = build_application(
        initialization_kwargs={"output": True, "require_command_line": True},
    )

    output_file = app_context.args.get("output_file")
    if output_file is None:
        raise ValueError("Missing required argument output_file.")

    output_path = app_context.io_handler.get_output_file(output_file)
    write_production_summary_markdown(app_context.args["data_path"], output_path)

    app_context.logger.info(f"Production summary written to {output_path}")


if __name__ == "__main__":
    main()
