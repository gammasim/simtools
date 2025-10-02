#!/usr/bin/python3

r"""
    Calculate the camera FoV of the telescope requested and plot the camera.

    The orientation for the plotting is "as seen for an observer facing the camera".

    An example of the camera plot can be found below.

    .. _camera_fov_plot:
    .. image:: images/validate_camera_fov_North-LST-1_pixelLayout.png
      :width: 50 %


    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...)
    model_version (str, optional)
        Model version
    camera_in_sky_coor (bool, optional)
        Plot the camera layout in sky coordinates akin to looking at it from behind for single \
         mirror telescopes
    print_pixels_id (bool, optional)
        Up to which pixel ID to print. To suppress printing of pixel IDs, set to zero\
         (--print_pixels_id 0). To print all pixels, set to 'All'."

    Example
    -------
    LST - 5.0.0

    .. code-block:: console

        simtools-validate-camera-fov --site North \\
            --telescope LSTN-01 --model_version 5.0.0

    The output is saved in simtools-output/validate_camera_fov.

    Expected final print-out message:

    .. code-block:: console

        Saved camera plot in /workdir/external/simtools/simtools-output/validate_camera_fov\\
        /application-plots/validate_camera_fov_LST-1_pixel_layout.png

"""

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.model.telescope_model import TelescopeModel
from simtools.visualization import plot_camera, visualize


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Calculate the camera FoV of the telescope requested. "
            "Plot the camera, as seen for an observer facing the camera."
        ),
    )
    config.parser.add_argument(
        "--camera_in_sky_coor",
        help=(
            "Plot the camera layout in sky coordinates "
            "(akin to looking at it from behind for single mirror telescopes)"
        ),
        action="store_true",
        default=False,
    )
    config.parser.add_argument(
        "--print_pixels_id",
        help=(
            "Up to which pixel ID to print. "
            "To suppress printing of pixel IDs, set to zero (--print_pixels_id 0). "
            "To print all pixels, set to 'All'."
        ),
        default=50,
    )
    return config.initialize(db_config=True, simulation_model=["telescope", "model_version"])


def main():
    """Validate camera field of view."""
    app_context = startup_application(_parse)

    label = "validate_camera_fov"

    tel_model = TelescopeModel(
        site=app_context.args["site"],
        telescope_name=app_context.args["telescope"],
        mongo_db_config=app_context.db_config,
        model_version=app_context.args["model_version"],
        label=label,
    )
    tel_model.export_model_files()

    app_context.logger.info(f"\nValidating the camera FoV of {tel_model.name}\n")

    focal_length = tel_model.get_telescope_effective_focal_length("cm")
    camera = tel_model.camera

    fov, r_edge_avg = camera.calc_fov()

    app_context.logger.info(f"\nEffective focal length = {focal_length:.3f} cm")
    app_context.logger.info(f"{tel_model.name} FoV = {fov:.3f} deg")
    app_context.logger.info(f"Avg. edge radius = {r_edge_avg:.3f} cm\n")

    # Now plot the camera as well
    try:
        pixel_ids_to_print = int(app_context.args["print_pixels_id"])
        if pixel_ids_to_print == 0:
            pixel_ids_to_print = -1  # so not print the zero pixel
    except ValueError as exc:
        if app_context.args["print_pixels_id"].lower() == "all":
            pixel_ids_to_print = camera.get_number_of_pixels()
        else:
            raise ValueError(
                f"The value provided to --print_pixels_id ({app_context.args['print_pixels_id']}) "
                "should be an integer or All"
            ) from exc
    fig = plot_camera.plot_pixel_layout(
        camera, app_context.args["camera_in_sky_coor"], pixel_ids_to_print
    )
    output_dir = app_context.io_handler.get_output_directory()
    plot_file_prefix = output_dir.joinpath(f"{label}_{tel_model.name}_pixel_layout")
    visualize.save_figure(fig, f"{plot_file_prefix!s}", log_title="camera")


if __name__ == "__main__":
    main()
