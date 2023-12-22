#!/usr/bin/python3

"""
    Summary
    -------
    This application plots the layout array and saves into disk.

    It accepts as input the telescope list file, the name of the layout, a sequence of arguments
    with the telescope files or a sequence of arguments with the layout names.

    A rotation angle in degrees can be passed in case the array should be rotated before plotting.
    A sequence of arguments for the rotation angle is also permitted, in which case all of them
    are plotted and saved separately.


    Command line arguments
    ----------------------
    figure_name (str, optional)
        File name for the pdf output.
    telescope_list (str, optional)
        The telescopes file (.ecsv) with the array information.
    array_layout_name (str, optional)
        Name of the layout array (e.g., North-TestLayout, South-TestLayout, North-4LST, etc.).
    rotate_angle (float, optional)
        Angle to rotate the array before plotting (in degrees).
    show_tel_label (bool, optional)
        Shows the telescope labels in the plot.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    .. code-block:: console

        simtools-plot-layout-array --figure_name northern_array_alpha \
        --array_layout_name North-TestLayout

"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from astropy import units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model import data_reader
from simtools.io_operations import io_handler
from simtools.layout.array_layout import ArrayLayout
from simtools.visualization.visualize import plot_array


def _parse(label, description, usage):
    """
    Parse command line configuration

    Parameters
    ----------
    label: str
        Label describing the application.
    description: str
        Description of the application.
    usage: str
        Example on how to use the application.

    Returns
    -------
    CommandLineParser
        Command line parser object

    """
    config = configurator.Configurator(label=label, description=description, usage=usage)

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
        "--array_layout_name",
        help="Name of the array layout.",
        nargs="+",
        type=str,
        required=False,
        default=None,
    )

    return config.initialize()


def main():
    label = Path(__file__).stem
    description = "Plots layout array."
    usage = "python applications/plot_array_layout.py --array_layout_name test_layout"
    args_dict, _ = _parse(label, description, usage)
    io_handler_instance = io_handler.IOHandler()

    if args_dict["rotate_angle"] is None:
        rotate_angles = [0 * u.deg]
    else:
        rotate_angles = []
        for one_angle in args_dict["rotate_angle"]:
            rotate_angles.append(float(one_angle) * u.deg)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if args_dict["telescope_list"] is not None:
        logger.info("Plotting array from telescope list file(s).")
        telescope_file = args_dict["telescope_list"]

    elif args_dict["array_layout_name"] is not None:
        logger.info("Plotting array from layout array name(s).")
        telescope_file = [
            io_handler_instance.get_input_data_file(
                "layout", f"telescope_positions-{one_array}.ecsv"
            )
            for one_array in args_dict["array_layout_name"]
        ]

    for one_file in telescope_file:
        logger.debug(f"Processing: {one_file}.")
        for one_angle in rotate_angles:
            logger.debug(f"Processing: {one_angle}.")
            if args_dict["figure_name"] is None:
                plot_file_name = (
                    f"plot_array_layout_{(Path(one_file).name).split('.')[0]}_"
                    f"{str((round((one_angle.to(u.deg).value))))}deg"
                )
            else:
                plot_file_name = args_dict["figure_name"]

            telescope_table = data_reader.read_table_from_file(one_file)
            telescopes_dict = ArrayLayout.include_radius_into_telescope_table(telescope_table)
            fig_out = plot_array(
                telescopes_dict, rotate_angle=one_angle, show_tel_label=args_dict["show_tel_label"]
            )
            output_dir = io_handler_instance.get_output_directory(
                label, sub_dir="application-plots"
            )
            plot_file = output_dir.joinpath(plot_file_name)

            allowed_extensions = ["jpeg", "jpg", "png", "tiff", "ps", "pdf", "bmp"]

            splitted_plot_file_name = plot_file_name.split(".")
            if len(splitted_plot_file_name) > 1:
                if splitted_plot_file_name[-1] in allowed_extensions:
                    logger.info(f"Saving figure as {plot_file}.")
                    plt.savefig(plot_file, bbox_inches="tight", dpi=400)
                else:
                    msg = (
                        f"Extension in {plot_file} is not valid. Valid extensions are:"
                        f" {allowed_extensions}."
                    )
                    raise NameError(msg)
            else:
                for ext in ["pdf", "png"]:
                    logger.info(f"Saving figure to {plot_file}.{ext}.")
                    plt.savefig(f"{str(plot_file)}.{ext}", bbox_inches="tight", dpi=400)
                fig_out.clf()


if __name__ == "__main__":
    main()
