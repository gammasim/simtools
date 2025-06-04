#!/usr/bin/python3

"""
Plot array elements (array layout).

Plot an array layout and save it to file (e.g., pdf). Layouts are defined in the database,
or given as command line arguments (explicit listing or telescope list file). A list of input
files is also accepted.
Layouts can be plotted in ground or UTM coordinate systems.

Listing of array elements follows this logic:

* explicit listing: e.g., ``-array_element_list MSTN-01, MSTN05``
* listing of types: e.g, ``-array_element_list MSTN`` plots all telescopes of type MSTN.

A rotation angle in degrees allows to rotate the array before plotting.
The typical image formats (e.g., pdf, png, jpg) are allowed for the output figures.
If no ``figure_name`` is given as output, layouts are plotted in pdf and png format.

Example of a layout plot:

.. _plot_array_layout_plot:
.. image:: images/plot_array_layout_example.png
    :width: 49 %

Command line arguments
----------------------
figure_name : str
    File name for the output figure.
array_layout_file : str
    File (astropy table compatible) with a list of array elements.
array_layout_name : str
    Name of the layout array (e.g., test_layout, alpha, 4mst, etc.).
    Use 'plot_all' to plot all layouts from the database for the given site and model version.
array_layout_name_background: str, optional
    Name of the background layout array (e.g., test_layout, alpha, 4mst, etc.).
array_element_list : list
    List of array elements (e.g., telescopes) to plot (e.g., ``LSTN-01 LSTN-02 MSTN``).
coordinate_system : str, optional
    Coordinate system for the array layout (ground or utm).
rotate_angle : float, optional
    Angle to rotate the array before plotting (in degrees).
show_labels : bool, optional
    Shows the telescope labels in the plot.
axes_range : float, optional
    Range of the both axes in meters.
marker_scaling : float, optional.
    Scaling factor for plotting of array elements, optional.

Examples
--------
Plot layout with the name "test_layout":

.. code-block:: console

    simtools-plot-layout-array --figure_name northern_array_alpha
                               --array_layout_name test_layout


Plot layout with 2 LSTs and all northern MSTs in UTM coordinates:

.. code-block:: console

    simtools-plot-layout-array --array_element_list LSTN-01 LSTN-02 MSTN
                               --coordinate_system utm

Plot layout from a file with the list of telescopes:

.. code-block:: console

    simtools-plot-layout-array --array_element_list telescope_positions-test_layout.ecsv
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import units as u

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.layout.array_layout_utils import (
    get_array_layouts_from_db,
    get_array_layouts_from_file,
    get_array_layouts_using_telescope_lists_from_db,
)
from simtools.visualization.plot_array_layout import plot_array_layout


def _parse(label, description, usage=None):
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
        help="Angle to rotate the array (in degrees).",
        type=str,
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--show_labels",
        help="Plot array element labels.",
        action="store_true",
        required=False,
        default=False,
    )
    config.parser.add_argument(
        "--marker_scaling",
        help="Scaling factor for the markers.",
        type=float,
        required=False,
        default=1.0,
    )
    config.parser.add_argument(
        "--coordinate_system",
        help="Coordinate system for the array layout.",
        type=str,
        required=False,
        default="ground",
        choices=["ground", "utm"],
    )
    config.parser.add_argument(
        "--axes_range",
        help="Range of the both axes in meters.",
        type=float,
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--array_layout_name_background",
        help="Name of the background layout array (e.g., test_layout, alpha, 4mst, etc.).",
        type=str,
        required=False,
        default=None,
    )
    return config.initialize(
        db_config=True,
        simulation_model=["site", "model_version", "layout", "layout_file", "plot_all_layouts"],
    )


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
    msg = f"Extension in {plot_file} is not valid. Valid extensions are: {allowed_extensions}."
    raise NameError(msg)


def _get_plot_file_name(figure_name, layout_name, site, coordinate_system, rotate_angle):
    """
    Generate and return the file name for plots.

    Parameters
    ----------
    figure_name : str
        Figure name given through command line.
    layout_name : str
        Name of the layout.
    site : str
        Site name.
    coordinate_system : str
        Coordinate system for the array layout.
    rotate_angle : float
        Angle to rotate the array before plotting.

    Returns
    -------
    str
        Plot file name.
    """
    if figure_name is not None:
        return figure_name

    return (
        f"array_layout_{layout_name}_{site}_{coordinate_system}_"
        f"{round(rotate_angle.to(u.deg).value)!s}deg"
    )


def main():
    """Plot array layout application."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(
        label,
        (
            "Plots array layout."
            "Use '--array_layout_name plot_all' to plot all layouts for the given site "
            "and model version."
        ),
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))
    io_handler_instance = io_handler.IOHandler()

    layouts = []
    if args_dict["array_layout_name"] is not None or args_dict["plot_all_layouts"]:
        logger.info("Plotting array from DB using layout array name(s).")
        layouts = get_array_layouts_from_db(
            args_dict["array_layout_name"],
            args_dict["site"],
            args_dict["model_version"],
            db_config,
            args_dict["coordinate_system"],
        )
    elif args_dict["array_layout_file"] is not None:
        logger.info("Plotting array from telescope table file(s).")
        layouts = get_array_layouts_from_file(args_dict["array_layout_file"])
    elif args_dict["array_element_list"] is not None:
        logger.info("Plotting array from list of array elements.")
        layouts = get_array_layouts_using_telescope_lists_from_db(
            [args_dict["array_element_list"]],
            args_dict["site"],
            args_dict["model_version"],
            db_config,
            args_dict["coordinate_system"],
        )

    if args_dict.get("array_layout_name_background"):
        background_layout = get_array_layouts_from_db(
            args_dict["array_layout_name_background"],
            args_dict["site"],
            args_dict["model_version"],
            db_config,
            args_dict["coordinate_system"],
        )[0]["array_elements"]
    else:
        background_layout = None

    rotate_angle = (
        0.0 * u.deg
        if args_dict["rotate_angle"] is None
        else float(args_dict["rotate_angle"]) * u.deg
    )

    mpl.use("Agg")
    for layout in layouts:
        fig_out = plot_array_layout(
            telescopes=layout["array_elements"],
            rotate_angle=rotate_angle,
            show_tel_label=args_dict["show_labels"],
            axes_range=args_dict["axes_range"],
            marker_scaling=args_dict["marker_scaling"],
            background_telescopes=background_layout,
        )
        plot_file_name = _get_plot_file_name(
            args_dict["figure_name"],
            layout["name"],
            args_dict["site"],
            args_dict["coordinate_system"],
            rotate_angle,
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
