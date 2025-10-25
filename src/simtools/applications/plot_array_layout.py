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
grayed_out_array_elements : list, optional
    List of array elements to plot as gray circles.
highlighted_array_elements : list, optional
    List of array elements to plot with red circles around them.
legend_location : str, optional
    Location of the legend (default "best").
bounds : str, optional
    Axis bounds mode. Use "symmetric" for +-R with padding (default) or "exact" for
    per-axis min/max bounds.
padding : float, optional
    Fractional padding applied around computed extents in both modes (default 0.1).

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

Use exact bounds with default padding:

.. code-block:: console

    simtools-plot-array-layout --array_layout_name alpha \
        --site North --model_version 6.0.0 --bounds exact

Use symmetric bounds with custom padding:

.. code-block:: console

    simtools-plot-array-layout --array_layout_name alpha \
        --site North --model_version 6.0.0 --bounds symmetric --padding 0.15

Plot layout from a parameter file with a list of telescopes:

.. code-block:: console

    simtools-plot-array-layout
        --array_layout_parameter_file tests/resources/model_parameters/array_layouts-2.0.2.json
        --model_version 6.0.0


Plot all layouts for the North site and model version 6.0.0:

.. code-block:: console

    simtools-plot-array-layout --site North --plot_all_layouts --model_version=6.0.0

Plot layout with some telescopes grayed out and others highlighted:

.. code-block:: console

    simtools-plot-array-layout --site North
                               --array_layout_name alpha
                               --model_version=6.0.0
                               --grayed_out_array_elements LSTN-01 LSTN-02
                               --highlighted_array_elements MSTN-01 MSTN-02
                               --legend_location "upper right"
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

import simtools.layout.array_layout_utils as layout_utils
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.visualization import visualize
from simtools.visualization.plot_array_layout import plot_array_layout


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Plots array layout.",
        usage=(
            "Use '--array_layout_name plot_all' to plot all layouts for the given site "
            "and model version."
        ),
    )

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
    config.parser.add_argument(
        "--grayed_out_array_elements",
        help="List of array elements to plot as gray circles.",
        type=str,
        nargs="*",
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--highlighted_array_elements",
        help="List of array elements to plot with red circles around them.",
        type=str,
        nargs="*",
        required=False,
        default=None,
    )
    config.parser.add_argument(
        "--legend_location",
        help=(
            "Location of the legend (e.g., 'best', 'upper right', 'upper left', "
            "'lower left', 'lower right', 'right', 'center left', 'center right', "
            "'lower center', 'upper center', 'center', 'no_legend')."
        ),
        type=str,
        required=False,
        default="best",
    )
    config.parser.add_argument(
        "--bounds",
        help=("Axis bounds mode: 'symmetric' uses +-R with padding, 'exact' uses per-axis min/max"),
        type=str,
        choices=["symmetric", "exact"],
        required=False,
        default="symmetric",
    )
    config.parser.add_argument(
        "--padding",
        help=("Fractional padding applied around computed extents (used for both modes)."),
        type=float,
        required=False,
        default=0.1,
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
    logger : logging.app_context.logger
        app_context.logger instance.

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
    app_context = startup_application(_parse)

    layouts = read_layouts(app_context.args, app_context.db_config, app_context.logger)

    if app_context.args.get("array_layout_name_background"):
        background_layout = layout_utils.get_array_layouts_from_db(
            app_context.args["array_layout_name_background"],
            app_context.args["site"],
            app_context.args["model_version"],
            app_context.db_config,
            app_context.args["coordinate_system"],
        )["array_elements"]
    else:
        background_layout = None

    mpl.use("Agg")
    for layout in layouts:
        fig_out = plot_array_layout(
            telescopes=layout["array_elements"],
            show_tel_label=app_context.args["show_labels"],
            axes_range=app_context.args["axes_range"],
            marker_scaling=app_context.args["marker_scaling"],
            background_telescopes=background_layout,
            grayed_out_elements=app_context.args["grayed_out_array_elements"],
            highlighted_elements=app_context.args["highlighted_array_elements"],
            legend_location=app_context.args["legend_location"],
            bounds_mode=app_context.args["bounds"],
            padding=app_context.args["padding"],
        )
        site_string = ""
        if layout.get("site") is not None:
            site_string = f"_{layout['site']}"
        elif app_context.args["site"] is not None:
            site_string = f"_{app_context.args['site']}"
        coordinate_system_string = (
            f"_{app_context.args['coordinate_system']}"
            if app_context.args["coordinate_system"] not in layout["name"]
            else ""
        )
        plot_file_name = app_context.args["figure_name"] or (
            f"array_layout_{layout['name']}{site_string}{coordinate_system_string}"
        )

        visualize.save_figure(
            fig_out,
            app_context.io_handler.get_output_directory() / plot_file_name,
            dpi=400,
        )
        plt.close()


if __name__ == "__main__":
    main()
