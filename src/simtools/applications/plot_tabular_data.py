#!/usr/bin/python3

"""
Plot tabular data read from file or from model parameter database.

Uses a configuration file to define the data to be plotted and all
plotting details.

Command line arguments
----------------------
config_file (str, required)
    Configuration file name.

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler, legacy_data_handler
from simtools.visualization import visualize


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
    return config.initialize(db_config=True)


def main():
    """Plot tabular data."""
    args_dict, _ = _parse(
        label=Path(__file__).stem,
        description="Plots tabular data.",
        usage="""simtools-plot-tabular-data --plot_config config_file_name "
                 --output_file output_file_name""",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    io_handler_instance = io_handler.IOHandler()

    plot_config = gen.collect_data_from_file(args_dict["plot_config"])["CTA_SIMPIPE"]["PLOT"]
    # data normalization? (e.g. divide by max value)

    data = {}
    for _config in plot_config["DATA"]:
        table = legacy_data_handler.read_legacy_data_file(
            _config["FILE_NAME"],
            _config["TYPE"],
        )
        data[_config["LABEL"]] = gen.get_structure_array_from_table(
            table, [_config["COLUMN_X"], _config["COLUMN_Y"]]
        )

    fig = visualize.plot_1d(
        data,
        y_title="Response",
        title=plot_config["TITLE"],
        xscale=plot_config["AXIS"][0].get("SCALE", "linear"),
        yscale=plot_config["AXIS"][1].get("SCALE", "linear"),
        xlim=(plot_config["AXIS"][0].get("MIN"), plot_config["AXIS"][0].get("MAX")),
        ylim=(plot_config["AXIS"][1].get("MIN"), plot_config["AXIS"][1].get("MAX")),
    )

    visualize.save_figure(
        fig,
        io_handler_instance.get_output_file(args_dict["output_file"]),
    )


if __name__ == "__main__":
    main()
