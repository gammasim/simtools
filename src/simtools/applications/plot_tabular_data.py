#!/usr/bin/python3
"""
Plot tabular data read from file or from model parameter database.

Uses a configuration file to define the data to be plotted and all
plotting details.

Command line arguments
----------------------
config_file (str, required)
    Configuration file name for plotting.
output_file (str, required)
    Output file name (without suffix).

Example
-------

Plot tabular data using a configuration file.

.. code-block:: console

    simtools-plot-tabular-data --plot_config config_file_name --output_file output_file_name

"""

import simtools.utils.general as gen
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.constants import PLOT_CONFIG_SCHEMA
from simtools.data_model import schema
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler
from simtools.visualization import plot_tables

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "plot_config",
        help="Plotting configuration file name.",
        type=str,
        required=True,
        default=None,
    ),
    cli.ArgumentDefinition(
        "table_data_path",
        help="Path to the data files (optional). Expect all files to be in the same directory.",
        type=str,
        default=None,
    ),
    cli.ArgumentDefinition(
        "output_file", help="Output file name (without suffix)", type=str, required=True
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.OVERWRITE_MODEL_PARAMETERS(),
        cli.SITE(),
        cli.TELESCOPE(),
        *cli.PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    plot_config = gen.convert_keys_in_dict_to_lowercase(
        schema.validate_dict_using_schema(
            ascii_handler.collect_data_from_file(app_context.args["plot_config"]),
            PLOT_CONFIG_SCHEMA,
        )
    )
    if "__SETTING_WORKFLOW__" in str(plot_config):
        setting_workflow = gen.extract_subdirectories_from_path(
            app_context.args["plot_config"],
            anchor="input",
        )
        plot_config = gen.replace_placeholders_recursively(
            plot_config,
            {"__SETTING_WORKFLOW__": setting_workflow},
        )

    plot_tables.plot(
        config=plot_config["plot"],
        output_file=app_context.io_handler.get_output_file(app_context.args["output_file"]),
        data_path=app_context.args.get("table_data_path"),
    )

    MetadataCollector.dump(
        app_context.args,
        app_context.io_handler.get_output_file(app_context.args["output_file"]),
        add_activity_name=True,
    )


if __name__ == "__main__":
    main()
