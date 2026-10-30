#!/usr/bin/python3

"""Plot the camera of the requested telescope."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import bounded_int
from simtools.model.telescope_model import TelescopeModel
from simtools.simulation.configuration import export_model_files
from simtools.visualization.plot_camera import (
    PIXEL_IDS,
    parse_pixel_ids_to_print,
    plot_pixel_layout,
    plot_selected_trigger_patches,
    read_selected_trigger_patches,
    trigger_patches_for_pixel,
)
from simtools.visualization.visualize import save_figure

_TRIGGER_SELECTION = "trigger selection"
_TRIGGER_GROUPS = "camera_trigger_groups"
_TRIGGER_MEMBERS = "camera_trigger_members"
_MAX_PIXEL_ID = (1 << 63) - 1

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "trigger_pixel",
        type=bounded_int(0, _MAX_PIXEL_ID),
        default=None,
        exclusive_group=_TRIGGER_SELECTION,
        help="Highlight every trigger patch containing this model pixel ID.",
    ),
    cli.ArgumentDefinition(
        "trigger_patches",
        type=int,
        nargs="+",
        default=None,
        exclusive_group=_TRIGGER_SELECTION,
        help=(
            "Highlight distinct model trigger-group IDs, with a labelled close-up. "
            "The maximum is the model's largest number of patches sharing a pixel (six for LST)."
        ),
    ),
    cli.ArgumentDefinition(
        "camera_in_sky_coor",
        help=(
            "Plot the camera layout in sky coordinates "
            "(akin to looking at it from behind for single-mirror telescopes)"
        ),
        action="store_true",
        default=False,
    ),
    cli.ArgumentDefinition(
        "print_pixels_id",
        help=(
            "Highest pixel ID to print. Use zero (--print_pixels_id 0) to suppress pixel "
            "IDs, or 'All' to print every pixel."
        ),
        default=50,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("model", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    args = app_context.args
    label = "plot_camera_pixel_layout"
    tel_model = TelescopeModel(
        site=args["site"],
        telescope_name=args["telescope"],
        model_version=args["model_version"],
        label=label,
        model_reader=app_context.model_reader,
        context=app_context,
    )
    export_model_files(tel_model)
    app_context.logger.info(f"\nPlot camera for {tel_model.name}\n")

    focal_length = tel_model.get_telescope_effective_focal_length("cm")
    camera = tel_model.camera
    fov, r_edge_avg = camera.calc_fov()
    app_context.logger.info(f"\nEffective focal length = {focal_length:.3f} cm")
    app_context.logger.info(f"{tel_model.name} FoV = {fov:.3f} deg")
    app_context.logger.info(f"Avg. edge radius = {r_edge_avg:.3f} cm\n")

    camera_pixels = set(camera.pixels[PIXEL_IDS])
    patch_ids = args.get("trigger_patches")
    trigger_pixel = args.get("trigger_pixel")
    if trigger_pixel is not None:
        if patch_ids is not None:
            raise ValueError("Use either --trigger_pixel or --trigger_patches, not both")
        patch_ids = trigger_patches_for_pixel(
            tel_model.get_parameter_table(_TRIGGER_MEMBERS), trigger_pixel, camera_pixels
        )
    if patch_ids is None:
        fig = plot_pixel_layout(
            camera,
            args["camera_in_sky_coor"],
            parse_pixel_ids_to_print(args["print_pixels_id"], camera),
        )
    else:
        groups = read_selected_trigger_patches(
            tel_model.get_parameter_table(_TRIGGER_GROUPS),
            tel_model.get_parameter_table(_TRIGGER_MEMBERS),
            patch_ids,
            camera_pixels,
        )
        fig = plot_selected_trigger_patches(camera, groups, args["camera_in_sky_coor"])
    output_dir = app_context.paths.get_output_directory()
    plot_file_prefix = output_dir.joinpath(f"{label}_{tel_model.name}_pixel_layout")
    save_figure(fig, f"{plot_file_prefix!s}", log_title="camera")


if __name__ == "__main__":
    main()
