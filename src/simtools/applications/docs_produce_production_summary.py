#!/usr/bin/python3

r"""
Produce a markdown file with production version descriptions.

Reads ``info.yml`` files from the simulation-models productions directory
and writes a markdown table of production model versions and their short
descriptions.

Command line arguments
----------------------
data_path (str)
    Path to the simulation-models repository root.
output_path (str)
    Directory for the output file.
output_file (str)
    Output markdown file name.

Example
-------
.. code-block:: console

    simtools-docs-produce-production-summary \\
        --data_path ../simulation-models \\
        --output_path simtools-output/reports/productions \\
        --output_file production_version_descriptions.md

"""

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
