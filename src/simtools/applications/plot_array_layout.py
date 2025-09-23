#!/usr/bin/python3

"""
Plot array elements (array layouts).

Plot array layouts in ground or UTM coordinate systems from multiple sources.

For the following options, array element positions are retrieved from the model parameter database:

* from the model parameter database using the layout name (e.g., ``-array_layout_name alpha``)

* from the model parameter data, retrieving all layouts for the given site and model version
  (``--plot_all_layouts``)

* from a model parameter file
  (e.g., ``-array_layout_parameter_file tests/resources/model_parameters/array_layouts-2.0.2.json``)

* from a list of array elements (e.g., ``-array_element_list MSTN-01, MSTN-02``).
  Positions are retrieved from the database.
  * explicit listing: e.g., ``-array_element_list MSTN-01, MSTN05``
  * listing of types: e.g, ``-array_element_list MSTN`` plots all telescopes of type MSTN.

For this option, array element positions are retrieved from the input file:

* from a file containing an astropy table with a list of array elements and their positions
  (e.g., ``-array_layout_file tests/resources/telescope_positions-North-ground.ecsv``)

Plots are saved as pdf and png files in the output directory.

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
array_layout_parameter_file : str, optional
    File with array layouts similar in the model parameter file format (typically JSON).
array_layout_name_background: str, optional
    Name of the background layout array (e.g., test_layout, alpha, 4mst, etc.).
array_element_list : list
    List of array elements (e.g., telescopes) to plot (e.g., ``LSTN-01 LSTN-02 MSTN``).
coordinate_system : str, optional
    Coordinate system for the array layout (ground or utm).
show_labels : bool, optional
    Shows the telescope labels in the plot.
axes_range : float, optional
    Range of the both axes in meters.
marker_scaling : float, optional.
    Scaling factor for plotting of array elements, optional.

Examples
--------
Plot "alpha" layout for the North site with model version 6.0.0:

.. code-block:: console

    simtools-plot-array-layout --site North
                               --array_layout_name alpha
                               --model_version=6.0.0

Plot layout with 2 LSTs on top of north alpha layout:

.. code-block:: console

    simtools-plot-array-layout --site North
                               --array_element_list LSTN-01,LSTN-02
                               --model_version=6.0.0
                               --array_layout_name_background alpha

Plot layout from a file with a list of telescopes:

.. code-block:: console

    simtools-plot-array-layout
        --array_layout_file tests/resources/telescope_positions-North-ground.ecsv

Plot layout from a parameter file with a list of telescopes:

.. code-block:: console

    simtools-plot-array-layout
        --array_layout_parameter_file tests/resources/model_parameters/array_layouts-2.0.2.json
        --model_version 6.0.0


Plot all layouts for the North site and model version 6.0.0:

.. code-block:: console

    simtools-plot-array-layout --site North --plot_all_layouts --model_version=6.0.0
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

import simtools.layout.array_layout_utils as layout_utils
import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.io import io_handler
from simtools.visualization import visualize
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
        simulation_model=[
            "site",
            "model_version",
            "layout",
            "layout_file",
            "plot_all_layouts",
            "layout_parameter_file",
        ],
    )


def read_layouts(args_dict, db_config, logger):
    """
    Read array layouts from the database or parameter file.

    Parameters
    ----------
    args_dict : dict
        Dictionary with command line arguments.
    db_config : dict
        Database configuration.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    list
        List of array layouts.
    """
    if args_dict["array_layout_name"] is not None or args_dict["plot_all_layouts"]:
        logger.info("Plotting array from DB using layout array name(s).")
        layouts = layout_utils.get_array_layouts_from_db(
            args_dict["array_layout_name"],
            args_dict["site"],
            args_dict["model_version"],
            db_config,
            args_dict["coordinate_system"],
        )
        if isinstance(layouts, list):
            return layouts
        return [layouts]

    if args_dict["array_layout_parameter_file"] is not None:
        logger.info("Plotting array from parameter file(s).")
        return layout_utils.get_array_layouts_from_parameter_file(
            args_dict["array_layout_parameter_file"],
            args_dict["model_version"],
            db_config,
            args_dict["coordinate_system"],
        )

    if args_dict["array_layout_file"] is not None:
        logger.info("Plotting array from telescope table file(s).")
        return layout_utils.get_array_layouts_from_file(args_dict["array_layout_file"])

    if args_dict["array_element_list"] is not None:
        logger.info("Plotting array from list of array elements.")
        return layout_utils.get_array_layouts_using_telescope_lists_from_db(
            [args_dict["array_element_list"]],
            args_dict["site"],
            args_dict["model_version"],
            db_config,
            args_dict["coordinate_system"],
        )

    return []


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

    layouts = read_layouts(args_dict, db_config, logger)

    if args_dict.get("array_layout_name_background"):
        background_layout = layout_utils.get_array_layouts_from_db(
            args_dict["array_layout_name_background"],
            args_dict["site"],
            args_dict["model_version"],
            db_config,
            args_dict["coordinate_system"],
        )["array_elements"]
    else:
        background_layout = None

    mpl.use("Agg")
    for layout in layouts:
        fig_out = plot_array_layout(
            telescopes=layout["array_elements"],
            show_tel_label=args_dict["show_labels"],
            axes_range=args_dict["axes_range"],
            marker_scaling=args_dict["marker_scaling"],
            background_telescopes=background_layout,
        )
        site_string = ""
        if layout.get("site") is not None:
            site_string = f"_{layout['site']}"
        elif args_dict["site"] is not None:
            site_string = f"_{args_dict['site']}"
        coordinate_system_string = (
            f"_{args_dict['coordinate_system']}"
            if args_dict["coordinate_system"] not in layout["name"]
            else ""
        )
        plot_file_name = args_dict["figure_name"] or (
            f"array_layout_{layout['name']}{site_string}{coordinate_system_string}"
        )

        visualize.save_figure(
            fig_out,
            io_handler_instance.get_output_directory(sub_dir="application-plots") / plot_file_name,
            dpi=400,
        )
        plt.close()


if __name__ == "__main__":
    main()
