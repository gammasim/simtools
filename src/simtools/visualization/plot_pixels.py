#!/usr/bin/python3
"""Functions for plotting pixel layout information."""

import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from simtools.db import db_handler
from simtools.io import io_handler
from simtools.model.camera import Camera
from simtools.utils import names
from simtools.visualization import visualize
from simtools.visualization.camera_plot_utils import (
    add_pixel_legend,
    add_pixel_patch_collections,
    create_pixel_patches_by_type,
    setup_camera_axis_properties,
)

logger = logging.getLogger(__name__)


def plot(config, output_file):
    """
    Plot pixel layout based on configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - file_name : str, name of camera config file
        - column_x : str, x-axis label
        - column_y : str, y-axis label
        - parameter_version: str, version of the parameter
        - telescope : str, name of the telescope
        - focal_length : float, optional focal length for Camera initialization
    output_file : str
        Path where to save the plot

    Returns
    -------
    None
        The function saves the plot to the specified output file.
    """
    db = db_handler.DatabaseHandler()
    db.export_model_file(
        parameter=config["parameter"],
        site=config["site"],
        array_element_name=config.get("telescope"),
        parameter_version=config.get("parameter_version"),
        model_version=config.get("model_version"),
        export_file_as_table=False,
    )
    data_file_path = Path(io_handler.IOHandler().get_output_directory() / f"{config['file_name']}")
    fig = plot_pixel_layout_from_file(
        data_file_path,
        config["telescope"],
        pixels_id_to_print=80,
        focal_length=config.get("focal_length", 1.0),
    )
    visualize.save_figure(fig, output_file)
    plt.close(fig)


def plot_pixel_layout_from_file(dat_file_path, telescope_model_name, **kwargs):
    """
    Plot the pixel layout from a camera config file.

    This function reads the pixel configuration from the specified camera config file and
    generates a plot of the pixel layout for the given telescope model.

    Parameters
    ----------
    dat_file_path : str or Path
        Path to the camera config file containing pixel configuration
    telescope_model_name : str
        Name/model of the telescope
    **kwargs
        pixels_id_to_print : int
            Number of pixel IDs to print in the plot
        title : str
            Plot title
        xtitle : str
            X-axis label
        ytitle : str
            Y-axis label
        focal_length : float
            Focal length to initialize the Camera (default 1.0)

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    logger.info(f"Plotting pixel layout for {telescope_model_name} from {dat_file_path}")

    camera = Camera(
        telescope_name=telescope_model_name,
        camera_config_file=dat_file_path,
        focal_length=kwargs.get("focal_length", 1.0),  # 1 used as placeholder, not relevant
    )

    return _create_pixel_plot(
        camera,
        pixels_id_to_print=kwargs.get("pixels_id_to_print", 50),
        title=kwargs.get("title"),
        xtitle=kwargs.get("xtitle"),
        ytitle=kwargs.get("ytitle"),
    )


def _create_pixel_plot(camera, pixels_id_to_print=50, title=None, xtitle=None, ytitle=None):
    """
    Create and configure the pixel layout plot.

    Parameters
    ----------
    camera : Camera
        Camera instance with pixel data
    pixels_id_to_print : int, optional
        Number of pixel IDs to print, default 50
    title : str, optional
        Plot title
    xtitle : str, optional
        X-axis label
    ytitle : str, optional
        Y-axis label

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    on_pixels, edge_pixels, off_pixels = create_pixel_patches_by_type(camera)
    add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels)

    array_element_type = names.get_array_element_type_from_name(camera.telescope_name)
    font_size = 2 if "SCT" in array_element_type else 4
    for i_pix, (x, y) in enumerate(zip(camera.pixels["x"], camera.pixels["y"])):
        if camera.pixels["pix_id"][i_pix] < pixels_id_to_print:
            plt.text(
                x,
                y,
                camera.pixels["pix_id"][i_pix],
                ha="center",
                va="center",
                fontsize=font_size,
            )

    _configure_plot(
        ax,
        camera,
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
    )
    add_pixel_legend(ax, on_pixels, off_pixels)

    return fig


def _configure_plot(ax, camera, title=None, xtitle=None, ytitle=None):
    """Configure the plot with titles, labels, and limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure
    camera : Camera
        Camera instance containing pixel data
    title : str, optional
        Plot title
    xtitle : str, optional
        X-axis label
    ytitle : str, optional
        Y-axis label


    Returns
    -------
    None
        The function modifies the plot axes in place.
    """
    x_min, x_max = min(camera.pixels["x"]), max(camera.pixels["x"])
    y_min, y_max = min(camera.pixels["y"]), max(camera.pixels["y"])
    padding = 0.1 * max(x_max - x_min, y_max - y_min)

    setup_camera_axis_properties(ax, camera, grid=True, axis_below=True, padding=padding)

    plt.xlabel(xtitle or "Horizontal scale [cm]", fontsize=18, labelpad=0)
    plt.ylabel(ytitle or "Vertical scale [cm]", fontsize=18, labelpad=0)
    ax.set_title(
        title or "Pixel layout",
        fontsize=15,
        y=1.02,
    )
    plt.tick_params(axis="both", which="major", labelsize=15)

    rotation = (camera.pixels["rotate_angle"] * u.rad) + (90 * u.deg)
    _add_coordinate_axes(ax, rotation)
    x_min = min(camera.pixels["x"]) - (max(camera.pixels["x"]) - min(camera.pixels["x"])) * 0.05
    y_min = min(camera.pixels["y"]) - (max(camera.pixels["y"]) - min(camera.pixels["y"])) * 0.05
    ax.text(x_min, y_min, "For an observer facing the camera", fontsize=10, ha="left", va="bottom")


def _add_coordinate_axes(ax, rotation=0 * u.deg):
    """Add coordinate system axes to the plot."""
    # Setup dimensions and positions
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    plot_size = min(x_max - x_min, y_max - y_min)
    axis_length = plot_size * 0.08

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
        """Adding arrows with label."""
        ax.arrow(ox, oy, dx, dy, fc=color, ec=color, **arrow_style)
        if np.sqrt(dx**2 + dy**2) > 0:  # If not zero vector
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
    az_dx = az_direction * axis_length * arrow_length
    add_arrow_label(
        x_origin,
        y_origin_az,
        az_dx,
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
