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

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.constants import PLOT_CONFIG_SCHEMA
from simtools.data_model import schema
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler, io_handler
from simtools.visualization import plot_tables


def _parse(label, description, usage):
    """
    Parse command line configuration.

    Parameters
    ----------
    label : str
        Label describing the application.
    description : str
        Description of the application.
    usage : str
        Example on how to use the application.

    Returns
    -------
    CommandLineParser
        Command line parser object.
    """
    config = configurator.Configurator(label=label, description=description, usage=usage)

    config.parser.add_argument(
        "--plot_config",
        help="Plotting configuration file name.",
        type=str,
        required=True,
        default=None,
    )
    config.parser.add_argument(
        "--output_file",
        help="Output file name (without suffix)",
        type=str,
        required=True,
    )
    return config.initialize(db_config=True, simulation_model=["telescope"])


def main():
    """Plot tabular data."""
    args_dict, db_config_ = _parse(
        label=Path(__file__).stem,
        description="Plots tabular data.",
        usage="""simtools-plot-tabular-data --plot_config config_file_name "
                 --output_file output_file_name""",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict.get("log_level", "INFO")))
    io_handler_instance = io_handler.IOHandler()

    plot_config = gen.convert_keys_in_dict_to_lowercase(
        schema.validate_dict_using_schema(
            ascii_handler.collect_data_from_file(args_dict["plot_config"]),
            PLOT_CONFIG_SCHEMA,
        )
    )

    plot_tables.plot(
        config=plot_config["plot"],
        output_file=io_handler_instance.get_output_file(args_dict["output_file"]),
        db_config=db_config_,
    )

    MetadataCollector.dump(
        args_dict,
        io_handler_instance.get_output_file(args_dict["output_file"]),
        add_activity_name=True,
    )


if __name__ == "__main__":
    main()
