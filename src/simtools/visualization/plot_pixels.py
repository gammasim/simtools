#!/usr/bin/python3
"""Functions for plotting pixel layout information.

The camera (``simtools.model.camera.Camera``) and its raw pixel data are built by the caller,
so this module does not import the model package.
"""

import logging
from copy import deepcopy

import astropy.units as u
import numpy as np

from simtools.utils import names
from simtools.utils.names import is_two_mirror_telescope
from simtools.visualization import visualize
from simtools.visualization.camera_plot_utils import (
    add_pixel_legend,
    add_pixel_patch_collections,
    create_pixel_patches_by_type,
    setup_camera_axis_properties,
)
from simtools.visualization.matplotlib_backend import pyplot as plt

logger = logging.getLogger(__name__)


def plot(output_file, camera, raw_pixels, *, rotate_angle=None, pixels_id_to_print=80):
    """
    Plot a pixel layout and save it to file.

    Parameters
    ----------
    output_file : str or Path
        Path where to save the plot.
    camera : Camera
        Camera with pixel geometry (``pixels``, ``telescope_name``, ``get_edge_pixels``).
    raw_pixels : dict
        Unrotated pixel data with keys ``x``, ``y`` and ``rotate_angle`` (radians).
    rotate_angle : astropy.units.Quantity, optional
        Rotation applied to the telescope-specific pixel transform (default: from raw_pixels).
    pixels_id_to_print : int, optional
        Pixel IDs below this value are printed.
    """
    fig = plot_pixel_layout(
        camera, raw_pixels, rotate_angle=rotate_angle, pixels_id_to_print=pixels_id_to_print
    )
    visualize.save_figure(fig, output_file)
    plt.close(fig)


def raw_pixels_from_configuration(configuration):
    """
    Return unrotated pixel data from resolved camera components.

    Parameters
    ----------
    configuration : dict
        Output of :func:`resolve_camera_components`.

    Returns
    -------
    dict
        Keys ``x``, ``y`` (cm) and ``rotate_angle`` (radians).
    """
    return {
        "x": np.asarray([item["x_cm"] for item in configuration["pixels"]]),
        "y": np.asarray([item["y_cm"] for item in configuration["pixels"]]),
        "rotate_angle": np.deg2rad(float(configuration.get("rotate", 0.0))),
    }


def resolve_camera_components(model_reader, parameters):
    """Resolve camera component model parameters for plotting."""
    return {
        "rotate": parameters["camera_rotate"]["value"],
        "pixel_types": deepcopy(parameters["camera_pixel_types"]["value"]),
        "pixels": _table_records(model_reader, parameters["camera_pixel_layout"]),
        "triggers": _table_records(model_reader, parameters["camera_trigger_groups"]),
        "trigger_members": _table_records(model_reader, parameters["camera_trigger_members"]),
    }


def _table_records(model_reader, parameter):
    """Convert a model parameter table into scalar records."""
    table = model_reader.get_parameter_table(parameter)
    return [
        {column: getattr(row[column], "value", row[column]) for column in table.colnames}
        for row in table
    ]


def plot_pixel_layout(camera, raw_pixels, *, rotate_angle=None, pixels_id_to_print=50):
    """
    Create the pixel layout figure.

    Parameters
    ----------
    camera : Camera
        Camera with pixel geometry; its pixel positions are transformed in place.
    raw_pixels : dict
        Unrotated pixel data with keys ``x``, ``y`` and ``rotate_angle`` (radians).
    rotate_angle : astropy.units.Quantity, optional
        Rotation applied to the telescope-specific pixel transform.
    pixels_id_to_print : int, optional
        Pixel IDs below this value are printed.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    logger.info("Plotting pixel layout for %s", camera.telescope_name)
    _apply_telescope_specific_pixel_transform(camera, raw_pixels, rotate_angle)
    return _create_pixel_plot(camera, pixels_id_to_print)


def _create_pixel_plot(camera, pixels_id_to_print):
    """Create and configure the pixel layout plot."""
    fig, ax = plt.subplots(figsize=(8, 8))

    on_pixels, edge_pixels, off_pixels = create_pixel_patches_by_type(camera)
    add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels)

    array_element_type = names.get_array_element_type_from_name(camera.telescope_name)
    font_size = 2 if "SCT" in array_element_type else 4
    for x, y, pix_id in zip(camera.pixels["x"], camera.pixels["y"], camera.pixels["pix_id"]):
        if pix_id < pixels_id_to_print:
            plt.text(x, y, pix_id, ha="center", va="center", fontsize=font_size)

    _configure_plot(ax, camera)
    add_pixel_legend(ax, on_pixels, off_pixels)
    return fig


def _configure_plot(ax, camera):
    """Configure titles, labels, limits and the coordinate-axes overlay."""
    x_values, y_values = camera.pixels["x"], camera.pixels["y"]
    padding = 0.1 * max(max(x_values) - min(x_values), max(y_values) - min(y_values))
    setup_camera_axis_properties(ax, camera, grid=True, axis_below=True, padding=padding)

    plt.xlabel("Horizontal scale [cm]", fontsize=18, labelpad=0)
    plt.ylabel("Vertical scale [cm]", fontsize=18, labelpad=0)
    ax.set_title("Pixel layout", fontsize=15, y=1.02)
    plt.tick_params(axis="both", which="major", labelsize=15)

    rotation = camera.pixels.get("plot_rotate_angle", camera.pixels["rotate_angle"]) * u.rad
    _add_coordinate_axes(ax, rotation)
    x_text = min(x_values) - (max(x_values) - min(x_values)) * 0.05
    y_text = min(y_values) - (max(y_values) - min(y_values)) * 0.05
    ax.text(
        x_text, y_text, "For an observer facing the camera", fontsize=10, ha="left", va="bottom"
    )


def _add_coordinate_axes(ax, rotation=0 * u.deg):
    """Add the Az/Alt and pixel coordinate axis arrows to the plot."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    axis_length = min(x_max - x_min, y_max - y_min) * 0.08

    x_origin = x_max - axis_length * 1.0
    y_origin_az = y_min + axis_length * 2.5
    y_origin_pix = y_min + axis_length * 1.2
    arrow_style = {
        "head_width": axis_length * 0.15,
        "head_length": axis_length * 0.15,
        "width": axis_length * 0.02,
    }
    arrow_length = 0.6
    is_sst = abs(rotation - (90.0 * u.deg)).value < 1.0
    az_direction = 1 if is_sst else -1

    def add_arrow_label(ox, oy, dx, dy, label, offset, color="black", ha="center", va="center"):
        """Add an arrow with its label."""
        ax.arrow(ox, oy, dx, dy, fc=color, ec=color, **arrow_style)
        if np.sqrt(dx**2 + dy**2) > 0:
            dir_unit = np.sqrt(dx**2 + dy**2)
            ax.text(
                ox + dx + dx / dir_unit * axis_length * offset,
                oy + dy + dy / dir_unit * axis_length * offset,
                label,
                ha=ha,
                va=va,
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    # Az-Alt axes
    add_arrow_label(
        x_origin,
        y_origin_az,
        az_direction * axis_length * arrow_length,
        0,
        "Az",
        0.25,
        "red",
        ha="left" if az_direction > 0 else "right",
    )
    add_arrow_label(
        x_origin, y_origin_az, 0, -axis_length * arrow_length, "Alt", 0.25, "red", va="top"
    )

    # Pixel coordinate axes
    rot_angle = rotation.to(u.rad).value
    x_direction = -1 if is_sst else 1
    x_dir = x_direction * axis_length * arrow_length * np.cos(rot_angle)
    y_dir = x_direction * axis_length * arrow_length * np.sin(rot_angle)
    add_arrow_label(x_origin, y_origin_pix, x_dir, y_dir, "$\\mathrm{x}_\\mathrm{pix}$", 0.45)

    y_dx = axis_length * arrow_length * np.sin(rot_angle)
    y_dy = -axis_length * arrow_length * np.cos(rot_angle)
    add_arrow_label(x_origin, y_origin_pix, y_dx, y_dy, "$\\mathrm{y}_\\mathrm{pix}$", 0.45)


def _apply_telescope_specific_pixel_transform(camera, raw_pixels, rotate_angle=None):
    """Apply the telescope-specific pixel transform to ``camera.pixels`` in place."""
    x_pos = np.asarray(raw_pixels["x"], dtype=float)
    y_pos = np.asarray(raw_pixels["y"], dtype=float)
    two_mirror = is_two_mirror_telescope(camera.telescope_name)
    if not two_mirror:
        y_pos = -y_pos

    if rotate_angle is None:
        raw_rotate_angle = raw_pixels.get("rotate_angle")
        rotate_angle = 0.0 * u.deg if raw_rotate_angle is None else raw_rotate_angle * u.rad
    if isinstance(rotate_angle, (int, float, np.floating)):
        rotate_angle = float(rotate_angle) * u.deg
    total_rotation = (90 if two_mirror else -90) * u.deg - rotate_angle.to(u.deg)

    rot_angle = total_rotation.to(u.rad).value
    camera.pixels["x"] = x_pos * np.cos(rot_angle) - y_pos * np.sin(rot_angle)
    camera.pixels["y"] = y_pos * np.cos(rot_angle) + x_pos * np.sin(rot_angle)
    if camera.pixels.get("pixel_shape") in (1, 3):
        camera.pixels["orientation"] = 30 if camera.pixels.get("pixel_shape") == 3 else 0
    camera.pixels["plot_rotate_angle"] = float(rot_angle)
