#!/usr/bin/python3

"""
Plot array elements (array layout).

Plot an array layout and save it to file (e.g., pdf). Layouts are defined in the database,
or given as command line arguments (explicit listing or telescope list file). List of input
files are accepted.

A rotation angle in degrees allows to rotate the array before plotting.
A sequence of arguments for the rotation angle is also permitted, in which case all of them
are plotted and saved separately.

The typical image formats for the output figures are allowed (e.g., pdf, png, jpg). If no
``figure_name`` is given as output, layouts are plotted in pdf and png format.

Example of a layout plot:

.. _plot_array_layout_plot:
.. image:: images/plot_array_layout_example.png
    :width: 49 %

Command line arguments
----------------------
figure_name : str
    File name for the output figure.
telescope_list : str
    A telescopes file (.ecsv) with the list of telescopes.
array_layout_name : str
    Name of the layout array (e.g., North-TestLayout, South-TestLayout, North-4LST, etc.).
rotate_angle : float, optional
    Angle to rotate the array before plotting (in degrees).
show_tel_label : bool, optional
    Shows the telescope labels in the plot.

Examples
--------
.. code-block:: console

    simtools-plot-layout-array --figure_name northern_array_alpha
                               --array_layout_name North-TestLayout
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.array_model import ArrayModel
from simtools.utils import names
from simtools.visualization.visualize import plot_array


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

    return config.initialize(db_config=True, simulation_model="site")


def _get_site_from_telescope_list_name(telescope_list_file):
    """
    Get the site name from the telescope list file name.

    Parameters
    ----------
    telescope_list_file : str
        Telescope list file name.

    Returns
    -------
    str
        Site name.
    """
    for _site in names.site_names():
        if _site in str(telescope_list_file):
            return _site
    return None


def _get_list_of_plot_files(plot_file_name, output_dir):
    """
    Get list of output file names for plotting.

    Parameters
    ----------
    plot_file_name : str
        Name of the plot file.
    output_dir : str
        Output directory.

    Returns
    -------
    list
        List of output file names.

    Raises
    ------
    NameError
        If the file extension is not valid.
    """
    plot_file = output_dir.joinpath(plot_file_name)

    if len(plot_file.suffix) == 0:
        return [plot_file.with_suffix(f".{ext}") for ext in ["pdf", "png"]]

    allowed_extensions = [".jpeg", ".jpg", ".png", ".tiff", ".ps", ".pdf", ".bmp"]
    if plot_file.suffix in allowed_extensions:
        return [plot_file]
    msg = f"Extension in {plot_file} is not valid. Valid extensions are:" f" {allowed_extensions}."
    raise NameError(msg)


def main():
    """Plot array layout application."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label,
        "Plots array layout.",
        "python applications/plot_array_layout.py --array_layout_name test_layout",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    io_handler_instance = io_handler.IOHandler()

    rotate_angles = (
        [float(angle) * u.deg for angle in args_dict["rotate_angle"]]
        if args_dict["rotate_angle"] is not None
        else [0 * u.deg]
    )

    mpl.use("Agg")
    telescope_file = None
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
        site = (
            _get_site_from_telescope_list_name(one_file)
            if args_dict["site"] is None
            else args_dict["site"]
        )
        for one_angle in rotate_angles:
            logger.debug(f"Processing: {one_angle}.")
            if args_dict["figure_name"] is None:
                plot_file_name = (
                    f"plot_array_layout_{(Path(one_file).name).split('.')[0]}_"
                    f"{str(round(one_angle.to(u.deg).value))}deg"
                )
            else:
                plot_file_name = args_dict["figure_name"]

            array_model = ArrayModel(
                mongo_db_config=db_config,
                model_version=args_dict["model_version"],
                site=site,
                array_elements_file=one_file,
            )
            fig_out = plot_array(
                array_model.get_array_element_positions(),
                rotate_angle=one_angle,
                show_tel_label=args_dict["show_tel_label"],
            )

            _plot_files = _get_list_of_plot_files(
                plot_file_name,
                io_handler_instance.get_output_directory(label, sub_dir="application-plots"),
            )

            for file in _plot_files:
                logger.info(f"Saving figure as {file}")
                plt.savefig(file, bbox_inches="tight", dpi=400)
            fig_out.clf()
            plt.close()


if __name__ == "__main__":
    main()
