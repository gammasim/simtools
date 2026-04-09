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
from simtools.application_control import build_application
from simtools.constants import PLOT_CONFIG_SCHEMA
from simtools.data_model import schema
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler
from simtools.visualization import plot_tables


def _add_arguments(parser):
    """Register application-specific command line arguments."""
    parser.add_argument(
        "--plot_config",
        help="Plotting configuration file name.",
        type=str,
        required=True,
        default=None,
    )
    parser.add_argument(
        "--table_data_path",
        help="Path to the data files (optional). Expect all files to be in the same directory.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_file",
        help="Output file name (without suffix)",
        type=str,
        required=True,
    )


def main():
    """See CLI description."""
    app_context = build_application(
        usage="""simtools-plot-tabular-data --plot_config config_file_name "
                 --output_file output_file_name""",
        initialization_kwargs={"db_config": True, "simulation_model": ["telescope"]},
    )

    plot_config = gen.convert_keys_in_dict_to_lowercase(
        schema.validate_dict_using_schema(
            ascii_handler.collect_data_from_file(app_context.args["plot_config"]),
            PLOT_CONFIG_SCHEMA,
        )
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
