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
    return config.initialize(
        db_config=True, simulation_model=["site", "model_version", "layout", "layout_file"]
    )


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


def _layouts_from_array_layout_file(args_dict, db_config, rotate_angle):
    """
    Read array layout positions from file(s) and return a list of layouts.

    Parameters
    ----------
    args_dict : dict
        Dictionary with the command line arguments.
    db_config : dict
        Database configuration.
    rotate_angle : float
        Angle to rotate the array before plotting (in degrees).

    Returns
    -------
    list
        List of array layouts.
    """
    layouts = []
    telescope_files = args_dict["array_layout_file"]
    for one_file in telescope_files:
        site = (
            _get_site_from_telescope_list_name(one_file)
            if args_dict["site"] is None
            else args_dict["site"]
        )
        array_model = ArrayModel(
            mongo_db_config=db_config,
            model_version=args_dict["model_version"],
            site=site,
            array_elements=one_file,
        )
        layouts.append(
            {
                "array_elements": array_model.export_array_elements_as_table(),
                "plot_file_name": _get_plot_file_name(
                    args_dict["figure_name"],
                    (Path(one_file).name).split(".")[0],
                    site,
                    args_dict["coordinate_system"],
                    rotate_angle,
                ),
            }
        )
    return layouts


def _layouts_from_list(args_dict, db_config, rotate_angle):
    """
    Read positions for a list of array elements from the database and return a list of layouts.

    Parameters
    ----------
    args_dict : dict
        Dictionary with the command line arguments.
    db_config : dict
        Database configuration.
    rotate_angle : float
        Angle to rotate the array before plotting (in degrees).

    Returns
    -------
    list
        List of array layouts.
    """
    site = (
        names.get_site_from_array_element_name(args_dict["array_element_list"][0])
        if args_dict["site"] is None
        else args_dict["site"]
    )
    array_model = ArrayModel(
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        site=site,
        array_elements=args_dict["array_element_list"],
    )
    return [
        {
            "array_elements": array_model.export_array_elements_as_table(
                coordinate_system=args_dict["coordinate_system"]
            ),
            "plot_file_name": _get_plot_file_name(
                args_dict["figure_name"],
                "list",
                site,
                args_dict["coordinate_system"],
                rotate_angle,
            ),
        }
    ]


def _layouts_from_db(args_dict, db_config, rotate_angle):
    """
    Read array elements and their positions from data base using the layout name.

    Parameters
    ----------
    args_dict : dict
        Dictionary with the command line arguments.
    db_config : dict
        Database configuration.
    rotate_angle : float
        Angle to rotate the array before plotting (in degrees).

    Returns
    -------
    list
        List of array layouts.
    """
    layouts = []
    array_model = ArrayModel(
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        site=args_dict["site"],
        layout_name=args_dict["array_layout_name"],
    )
    layouts.append(
        {
            "array_elements": array_model.export_array_elements_as_table(
                coordinate_system=args_dict["coordinate_system"]
            ),
            "plot_file_name": _get_plot_file_name(
                figure_name=args_dict["figure_name"],
                layout_name=args_dict["array_layout_name"],
                site=args_dict["site"],
                coordinate_system=args_dict["coordinate_system"],
                rotate_angle=rotate_angle,
            ),
        }
    )
    return layouts


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

    rotate_angle = (
        0.0 * u.deg
        if args_dict["rotate_angle"] is None
        else float(args_dict["rotate_angle"]) * u.deg
    )

    layouts = []
    if args_dict["array_layout_name"] is not None:
        logger.info("Plotting array from layout array name.")
        layouts = _layouts_from_db(args_dict, db_config, rotate_angle)
    elif args_dict["array_layout_file"] is not None:
        logger.info("Plotting array from telescope list file.")
        layouts = _layouts_from_array_layout_file(args_dict, db_config, rotate_angle)
    elif args_dict["array_element_list"] is not None:
        logger.info("Plotting array from list of array elements.")
        layouts = _layouts_from_list(args_dict, db_config, rotate_angle)

    mpl.use("Agg")
    for layout in layouts:
        fig_out = plot_array(
            telescopes=layout["array_elements"],
            rotate_angle=rotate_angle,
            show_tel_label=args_dict["show_labels"],
            axes_range=args_dict["axes_range"],
            marker_scaling=args_dict["marker_scaling"],
        )
        _plot_files = _get_list_of_plot_files(
            layout["plot_file_name"],
            io_handler_instance.get_output_directory(label, sub_dir="application-plots"),
        )

        for file in _plot_files:
            logger.info(f"Saving figure as {file}")
            plt.savefig(file, bbox_inches="tight", dpi=400)
        fig_out.clf()
        plt.close()


if __name__ == "__main__":
    main()
