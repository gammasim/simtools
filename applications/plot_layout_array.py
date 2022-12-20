#!/usr/bin/python3

"""
    Summary
    -------
    This application plots the layout array and saves into pdf and png files.

    It accepts as input the telescope list file, the name of the layout, a sequence of arguments
    with the telescope files or a sequence of arguments with the layout names.

    A rotation angle in degrees can be passed in case the array should be rotated before plotting.
    A sequence of arguments for the rotation angle is also permitted, in which case all of them
    are plotted and saved separately.


    Command line arguments
    ----------------------
    figure_name (str, optional)
        File name for the pdf output (without extension).
    telescope_list (str, optional)
        The telescopes file (.ecsv) with the array information.
    layout_array_name (str, optional)
        Name of the layout array.
    rotate_angle (float, optional)
        Angle to rotate the array before plotting (in degrees).
    show_tel_label (bool, optional)
        Shows the telescope labels in the plot.
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    .. code-block:: console

        python applications/plot_layout_array.py --figure_name northern_array_alpha \
        --layout_array_name test_layout

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from astropy import units as u

import simtools.util.general as gen
from simtools import io_handler
from simtools.configuration import configurator
from simtools.layout.layout_array import LayoutArray
from simtools.visualization.visualize import plot_array


def _parse(label):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing application.
    description: str
        Description of application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(
        label=label,
        description=("Plots layout array."),
    )

    config.parser.add_argument(
        "--figure_name",
        help="Name of the output figure to be saved into as a pdf.",
        type=str,
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--rotate_angle",
        help="Angle to rotate the array before plotting (in degrees).",
        type=str,
        nargs="+",
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--show_tel_label",
        help="Shows the telescope labels in the plot.",
        action="store_true",
        required=False,
        default=False,
    )
    input_group = config.parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--telescope_list",
        help="ECSV file with the list of telescopes",
        nargs="+",
        type=str,
        required=False,
        default=None,
    )
    input_group.add_argument(
        "--layout_array_name",
        help="Name of the layout array.",
        nargs="+",
        type=str,
        required=False,
        default=None,
    )

    return config.initialize(db_config=True)


def main():

    label = Path(__file__).stem
    args_dict, _ = _parse(label)
    io_handler_instance = io_handler.IOHandler()

    if args_dict["rotate_angle"] is None:
        rotate_angles = [0 * u.deg]
    else:
        rotate_angles = []
        for _, one_angle in enumerate(args_dict["rotate_angle"]):
            rotate_angles.append(float(one_angle) * u.deg)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict["telescope_list"] is not None:
        logger.info("Plotting array from telescope list file(s).")
        telescope_file = args_dict["telescope_list"]

    elif args_dict["layout_array_name"] is not None:
        logger.info("Plotting array from layout array name(s).")
        telescope_file = [
            io_handler_instance.get_input_data_file(
                "layout", f"telescope_positions-{one_array}.ecsv"
            )
            for _, one_array in enumerate(args_dict["layout_array_name"])
        ]

    for one_file in telescope_file:

        logger.debug(f"Processing: {one_file}.")
        for one_angle in rotate_angles:
            logger.debug(f"Processing: {one_angle}.")
            base_name = (Path(one_file).name).split(".")[0] + "_"
            if args_dict["figure_name"] is None:
                print(one_angle.to(u.deg))
                plot_file_name = (
                    "plot_layout_array_"
                    + base_name
                    + str((round((one_angle.to(u.deg).value))))
                    + "deg"
                )
            else:
                plot_file_name = args_dict["figure_name"]

            telescope_table = LayoutArray.read_telescope_list_file(one_file)
            telescopes_dict = LayoutArray.include_radius_into_telescope_table(telescope_table)
            fig_out = plot_array(
                telescopes_dict, rotate_angle=one_angle, show_tel_label=args_dict["show_tel_label"]
            )
            output_dir = io_handler_instance.get_output_directory(
                label, dir_type="application-plots"
            )
            plot_file = output_dir.joinpath(plot_file_name)

            for f in ["pdf", "png"]:
                logger.info(f"Saving figure to {plot_file}.{f}.")
                plt.savefig(str(plot_file) + "." + f, format=f, bbox_inches="tight")
            fig_out.clf()


if __name__ == "__main__":
    main()
