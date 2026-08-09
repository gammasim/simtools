#!/usr/bin/python3

"""Plot array elements (array layouts)."""

import simtools.layout.array_layout_utils as layout_utils
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.visualization.plot_array_layout import (
    generate_plot_combinations,
    plot_array_layouts,
)

_ARGUMENTS = (
    cli.ALL_MODEL_VERSIONS,
    cli.ArgumentDefinition("all_sites", action="store_true", help="Plot layouts for all sites."),
    cli.ArgumentDefinition(
        "figure_name",
        help="Name of the output figure to be saved.",
        type=str,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "show_labels",
        help="Plot array element labels.",
        action="store_true",
        required=False,
        default=False,
    ),
    cli.ArgumentDefinition(
        "marker_scaling",
        help="Scaling factor for the markers.",
        type=float,
        required=False,
        default=1.0,
    ),
    cli.ArgumentDefinition(
        "coordinate_system",
        help="Coordinate system for the array layout.",
        type=str,
        required=False,
        default="ground",
        choices=["ground", "utm"],
    ),
    cli.ArgumentDefinition(
        "axes_range",
        help="Range of the both axes in meters.",
        type=float,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "x_lim",
        help="Explicit x-axis limits [xmin xmax] in meters.",
        type=float,
        nargs=2,
        required=False,
        default=None,
        metavar=("XMIN", "XMAX"),
    ),
    cli.ArgumentDefinition(
        "y_lim",
        help="Explicit y-axis limits [ymin ymax] in meters.",
        type=float,
        nargs=2,
        required=False,
        default=None,
        metavar=("YMIN", "YMAX"),
    ),
    cli.ArgumentDefinition(
        "array_layout_name_background",
        help="Name of the background layout array (e.g., test_layout, alpha, 4mst, etc.).",
        type=str,
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "grayed_out_array_elements",
        help="List of array elements to plot as gray circles.",
        type=str,
        nargs="*",
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "highlighted_array_elements",
        help="List of array elements to plot with red circles around them.",
        type=str,
        nargs="*",
        required=False,
        default=None,
    ),
    cli.ArgumentDefinition(
        "legend_location",
        help=(
            "Legend location: 'best', 'upper right', 'upper left', 'lower left', "
            "'lower right', 'right', 'center left', 'center right', 'lower center', "
            "'upper center', 'center', or 'no_legend'."
        ),
        type=str,
        required=False,
        default="best",
    ),
    cli.ArgumentDefinition(
        "bounds",
        help="Axis bounds mode: 'symmetric' uses +-R with padding, 'exact' uses per-axis min/max",
        type=str,
        choices=["symmetric", "exact"],
        required=False,
        default="symmetric",
    ),
    cli.ArgumentDefinition(
        "padding",
        help="Fractional padding applied around computed extents (used for both modes).",
        type=float,
        required=False,
        default=0.1,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.IGNORE_MISSING_DESIGN_MODEL,
        cli.SITE,
        *cli.layout_selection_arguments(
            include_file=True,
            include_parameter_file=True,
            include_plot_all=True,
        ),
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
    usage="Use '--plot_all_layouts' to plot all layouts for the given site and model version.",
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    if app_context.args.get("all_model_versions") or app_context.args.get("all_sites"):
        for model_version, site in generate_plot_combinations(app_context.args):
            run_args = app_context.args.copy()
            run_args.update(
                {
                    "model_version": model_version,
                    "site": site,
                    "ignore_software_version": True,
                }
            )
            layouts, background_layout = layout_utils.read_layouts(run_args)
            plot_array_layouts(
                run_args,
                app_context.io_handler.get_output_directory(),
                layouts,
                background_layout,
            )
    else:
        layouts, background_layout = layout_utils.read_layouts(app_context.args)
        plot_array_layouts(
            app_context.args,
            app_context.io_handler.get_output_directory(),
            layouts,
            background_layout,
        )


if __name__ == "__main__":
    main()
