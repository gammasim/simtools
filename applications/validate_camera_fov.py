#!/usr/bin/python3

"""
    Summary
    -------
    This application calculate the camera FoV of the telescope requested and plot the camera \
    as seen for an observer facing the camera.

    An example of the camera plot can be found below.

    .. _camera_fov_plot:
    .. image:: docs/source/images/validate_camera_fov_North-LST-1_pixelLayout.png
      :width: 50 %


    Command line arguments
    ----------------------
    site (str, required)
        North or South.
    telescope (str, required)
        Telescope model name (e.g. LST-1, SST-D, ...)
    model_version (str, optional)
        Model version (default='Current')
    verbosity (str, optional)
        Log level to print (default=INFO).

    Example
    -------
    LST - Prod5

    Runtime 1 min

    .. code-block:: console

        python applications/validate_camera_fov.py --site North \
            --telescope LST-1 --model_version prod5

    .. todo::

        * Change default model to default (after this feature is implemented in db_handler)
        * Fix the set_style. For some reason, sphinx cannot built docs with it on.
"""

import logging
from pathlib import Path

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools import io_handler
from simtools.model.telescope_model import TelescopeModel


def main():

    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=(
            "Calculate the camera FoV of the telescope requested. "
            "Plot the camera, as seen for an observer facing the camera."
        ),
    )
    config.parser.add_argument(
        "--camera_in_sky_coor",
        help=(
            "Plot the camera layout in sky coordinates "
            "(akin to looking at it from behind for single mirror telesecopes)"
        ),
        action="store_true",
        default=False,
    )
    config.parser.add_argument(
        "--print_pixels_id",
        help=(
            "Up to which pixel ID to print (default: 50). "
            "To suppress printing of pixel IDs, set to zero (--print_pixels_id 0). "
            "To print all pixels, set to 'All'."
        ),
        default=50,
    )

    args_dict, db_config = config.initialize(db_config=True, telescope_model=True)
    label = "validate_camera_fov"

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # Output directory to save files related directly to this app
    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, dir_type="application-plots")

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_model_name=args_dict["telescope"],
        mongo_db_config=db_config,
        model_version=args_dict["model_version"],
        label=label,
    )
    tel_model.export_model_files()

    print("\nValidating the camera FoV of {}\n".format(tel_model.name))

    focal_length = float(tel_model.get_parameter_value("effective_focal_length"))
    camera = tel_model.camera

    fov, r_edge_avg = camera.calc_fov()

    print("\nEffective focal length = " + "{0:.3f} cm".format(focal_length))
    print("{0} FoV = {1:.3f} deg".format(tel_model.name, fov))
    print("Avg. edge radius = {0:.3f} cm\n".format(r_edge_avg))

    # Now plot the camera as well
    try:
        pixel_ids_to_print = int(args_dict["print_pixels_id"])
        if pixel_ids_to_print == 0:
            pixel_ids_to_print = -1  # so not print the zero pixel
    except ValueError:
        if args_dict["print_pixels_id"].lower() == "all":
            pixel_ids_to_print = camera.get_number_of_pixels()
        else:
            raise ValueError(
                f"The value provided to --print_pixels_id ({args_dict['print_pixels_id']}) "
                "should be an integer or All"
            )
    fig = camera.plot_pixel_layout(args_dict["camera_in_sky_coor"], pixel_ids_to_print)
    plot_file_prefix = output_dir.joinpath(f"{label}_{tel_model.name}_pixel_layout")
    for suffix in ["pdf", "png"]:
        file_name = f"{str(plot_file_prefix)}.{suffix}"
        fig.savefig(file_name, format=suffix, bbox_inches="tight")
        print("\nSaved camera plot in {}\n".format(file_name))
    fig.clf()


if __name__ == "__main__":
    main()
