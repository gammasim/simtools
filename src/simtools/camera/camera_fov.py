"""Functions for validating camera field of view and plotting pixel layouts."""

import logging

from simtools.model.telescope_model import TelescopeModel
from simtools.visualization import plot_camera, visualize

__all__ = ["parse_pixel_ids_to_print", "run_camera_fov_validation"]

_logger = logging.getLogger(__name__)


def parse_pixel_ids_to_print(print_pixels_id_arg, camera):
    """
    Parse the print_pixels_id argument into an integer pixel count.

    Parameters
    ----------
    print_pixels_id_arg : str or int
        The raw argument value. Can be an integer string, 0 (suppress), or 'All'.
    camera : Camera
        Camera model instance used to determine total pixel count for 'All'.

    Returns
    -------
    int
        Number of pixel IDs to print. Returns -1 when printing should be suppressed.

    Raises
    ------
    ValueError
        If the argument is not an integer string or 'All'.
    """
    if str(print_pixels_id_arg).lower() == "all":
        return camera.get_number_of_pixels()

    try:
        n = int(print_pixels_id_arg)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--print_pixels_id must be integer or 'All', got: {print_pixels_id_arg}"
        ) from exc

    return -1 if n == 0 else n


def run_camera_fov_validation(args_dict, io_handler):
    """
    Build telescope model, compute FoV, and plot the pixel layout.

    Parameters
    ----------
    args_dict : dict
        Application arguments including site, telescope, model_version,
        camera_in_sky_coor, print_pixels_id.
    io_handler : IOHandler
        I/O handler for output file paths.
    """
    label = "validate_camera_fov"

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
        label=label,
    )
    tel_model.export_model_files()

    _logger.info(f"\nValidating the camera FoV of {tel_model.name}\n")

    focal_length = tel_model.get_telescope_effective_focal_length("cm")
    camera = tel_model.camera

    fov, r_edge_avg = camera.calc_fov()

    _logger.info(f"\nEffective focal length = {focal_length:.3f} cm")
    _logger.info(f"{tel_model.name} FoV = {fov:.3f} deg")
    _logger.info(f"Avg. edge radius = {r_edge_avg:.3f} cm\n")

    pixel_ids_to_print = parse_pixel_ids_to_print(args_dict["print_pixels_id"], camera)

    fig = plot_camera.plot_pixel_layout(camera, args_dict["camera_in_sky_coor"], pixel_ids_to_print)
    output_dir = io_handler.get_output_directory()
    plot_file_prefix = output_dir.joinpath(f"{label}_{tel_model.name}_pixel_layout")
    visualize.save_figure(fig, f"{plot_file_prefix!s}", log_title="camera")
