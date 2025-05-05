#!/usr/bin/python3
"""Functions for plotting pixel layout information."""

import logging

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from simtools.model.model_utils import is_two_mirror_telescope
from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h
from simtools.visualization import visualize

logger = logging.getLogger(__name__)


def plot(config, output_file):
    """
    Plot pixel layout based on configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - tables : list of dicts with table configurations
        - type : str, type of plot
        - title : str, plot title
        - xtitle : str, x-axis label
        - ytitle : str, y-axis label
    output_file : str
        Path where to save the plot
    """
    table_config = config["tables"][0]

    fig = plot_pixel_layout_from_file(
        table_config["file_name"],
        table_config["telescope"],
        pixels_id_to_print=80,
        title=config.get("title"),
        xtitle=config.get("xtitle"),
        ytitle=config.get("ytitle"),
    )
    visualize.save_figure(fig, output_file)


def plot_pixel_layout_from_file(dat_file_path, telescope_model_name, **kwargs):
    """
    Plot the pixel layout from a .dat file configuration.

    Parameters
    ----------
    dat_file_path : str or Path
        Path to the .dat file containing pixel configuration
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

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    logger.info(f"Plotting pixel layout for {telescope_model_name} from {dat_file_path}")

    # Read configuration and prepare data
    pixel_data = _prepare_pixel_data(
        dat_file_path,
        telescope_model_name,
    )

    return _create_pixel_plot(
        pixel_data,
        telescope_model_name,
        pixels_id_to_print=kwargs.get("pixels_id_to_print", 50),
        title=kwargs.get("title"),
        xtitle=kwargs.get("xtitle"),
        ytitle=kwargs.get("ytitle"),
    )


def _prepare_pixel_data(dat_file_path, telescope_model_name):
    """Prepare pixel data from configuration file."""
    config = _read_pixel_config(dat_file_path)
    x_pos = np.array(config["x"])
    y_pos = np.array(config["y"])

    # Don't flip y coordinates for two mirror telescopes (SCT and SST)
    if not is_two_mirror_telescope(telescope_model_name):
        y_pos = -y_pos

    # For SST, apply 90 degree rotation
    if "SST" in telescope_model_name:
        base_angle = 90
    else:
        base_angle = 248.2

    # Apply base rotation and any additional rotation from config
    total_rotation = base_angle
    if config["rotate_angle"] is not None:
        total_rotation += config["rotate_angle"]

    rot_angle = np.deg2rad(total_rotation)
    x_rot = x_pos * np.cos(rot_angle) - y_pos * np.sin(rot_angle)
    y_rot = y_pos * np.cos(rot_angle) + x_pos * np.sin(rot_angle)
    x_pos, y_pos = x_rot, y_rot

    return {
        "x": x_pos,
        "y": y_pos,
        "pixel_ids": config["pixel_ids"],
        "pixels_on": config["pixels_on"],
        "pixel_shape": config["pixel_shape"],
        "pixel_diameter": config["pixel_diameter"],
        "rotation": total_rotation,
    }


def _create_pixel_plot(
    pixel_data, telescope_model_name, pixels_id_to_print=50, title=None, xtitle=None, ytitle=None
):
    """
    Create and configure the pixel layout plot.

    Parameters
    ----------
    pixel_data : dict
        Dictionary containing pixel configuration data
    telescope_model_name : str
        Name of telescope model
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

    # Create patches
    on_pixels, edge_pixels, off_pixels = _create_pixel_patches(
        pixel_data["x"],
        pixel_data["y"],
        pixel_data["pixel_diameter"],
        pixel_data["pixel_shape"],
        pixel_data["pixels_on"],
        pixel_data["pixel_ids"],
        pixels_id_to_print,
        telescope_model_name,
    )

    # Add collections
    for patches, style in [
        (on_pixels, {"facecolor": "none", "edgecolor": "black", "linewidth": 0.2}),
        (
            edge_pixels,
            {
                "facecolor": (*mcolors.to_rgb("brown"), 0.5),
                "edgecolor": (*mcolors.to_rgb("black"), 1),
                "linewidth": 0.2,
            },
        ),
        (off_pixels, {"facecolor": "black", "edgecolor": "black", "linewidth": 0.2}),
    ]:
        if patches:
            ax.add_collection(PatchCollection(patches, **style))

    # Configure plot with titles
    _configure_plot(
        ax,
        telescope_model_name,
        pixel_data["x"],
        pixel_data["y"],
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
    )
    _add_legend(ax, on_pixels, off_pixels)

    plt.tight_layout()
    return fig


def _read_pixel_config(dat_file_path):
    """Read pixel configuration from .dat file."""
    config = {
        "x": [],
        "y": [],
        "pixel_ids": [],
        "pixels_on": [],
        "pixel_shape": None,
        "pixel_diameter": None,
        "trigger_groups": [],
        "rotate_angle": None,  # Add this field
    }

    with open(dat_file_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("Rotate"):
                # Parse rotation angle from line like "Rotate 10.893"
                config["rotate_angle"] = float(line.split()[1].strip())
            elif line.startswith("PixType"):
                parts = line.split()
                config["pixel_shape"] = int(parts[5].strip())  # funnel shape
                config["pixel_diameter"] = float(parts[6].strip())
            elif line.startswith("Pixel"):
                parts = line.split()
                config["x"].append(float(parts[3].strip()))
                config["y"].append(float(parts[4].strip()))
                config["pixel_ids"].append(int(parts[1].strip()))
                if len(parts) > 9:
                    config["pixels_on"].append(int(parts[9].strip()) != 0)
                else:
                    config["pixels_on"].append(True)

    return config


def _create_patch(x, y, diameter, shape):
    """Create a single matplotlib patch for a pixel.

    Parameters
    ----------
    x, y : float
        Center coordinates of the pixel
    diameter : float
        Diameter of the pixel
    shape : int
        Pixel shape type:
        0: circular
        1: hexagonal (flat x)
        2: square
        3: hexagonal (flat y)
    """
    if shape == 0:  # Circular
        return mpatches.Circle((x, y), radius=diameter / 2)
    if shape in (1, 3):  # Hexagonal
        return mpatches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=diameter / np.sqrt(3),
            orientation=np.deg2rad(30 if shape == 3 else 0),
        )
    # Square
    return mpatches.Rectangle((x - diameter / 2, y - diameter / 2), width=diameter, height=diameter)


def _is_edge_pixel(x, y, x_pos, y_pos, diameter, shape):
    """Determine if a pixel is on the edge based on neighbor count."""
    if shape == 0:  # Circular
        return _count_neighbors(x, y, x_pos, y_pos, diameter * 1.1) < 6
    if shape in (1, 3):  # Hexagonal
        return _count_neighbors(x, y, x_pos, y_pos, diameter * 1.1) < 6
    # Square
    return _count_neighbors(x, y, x_pos, y_pos, diameter * 1.4) < 4


def _add_pixel_id(x, y, pixel_id, telescope_model_name):
    """Add pixel ID text to the plot."""
    font_size = 2 if "SCT" in names.get_array_element_type_from_name(telescope_model_name) else 4
    plt.text(x, y, pixel_id, ha="center", va="center", fontsize=font_size)


def _create_pixel_patches(
    x_pos, y_pos, diameter, shape, pixels_on, pixel_ids, pixels_id_to_print, telescope_model_name
):
    """Create matplotlib patches for different pixel types."""
    on_pixels, edge_pixels, off_pixels = [], [], []

    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        patch = _create_patch(x, y, diameter, shape)

        if pixels_on[i]:
            if _is_edge_pixel(x, y, x_pos, y_pos, diameter, shape):
                edge_pixels.append(patch)
            else:
                on_pixels.append(patch)
        else:
            off_pixels.append(patch)

        if pixel_ids[i] < pixels_id_to_print:
            _add_pixel_id(x, y, pixel_ids[i], telescope_model_name)

    return on_pixels, edge_pixels, off_pixels


def _count_neighbors(x, y, x_pos, y_pos, max_dist):
    """Count number of neighboring pixels within max_dist."""
    count = 0
    for x2, y2 in zip(x_pos, y_pos):
        if (x != x2 or y != y2) and np.sqrt((x - x2) ** 2 + (y - y2) ** 2) <= max_dist:
            count += 1
    return count


def _configure_plot(ax, telescope_model_name, x_pos, y_pos, title=None, xtitle=None, ytitle=None):
    # First set the aspect ratio
    ax.set_aspect("equal")

    # Calculate the axis limits
    x_min, x_max = min(x_pos), max(x_pos)
    y_min, y_max = min(y_pos), max(y_pos)  # Fixed: Added max(y_pos)

    # Add some padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    # Set limits with padding
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Rest of your configuration code...
    plt.grid(True)
    ax.set_axisbelow(True)

    plt.xlabel(xtitle or "Horizontal scale [cm]", fontsize=18, labelpad=0)
    plt.ylabel(ytitle or "Vertical scale [cm]", fontsize=18, labelpad=0)
    ax.set_title(
        title or f"Pixels layout in {telescope_model_name} camera",
        fontsize=15,
        y=1.02,
    )
    plt.tick_params(axis="both", which="major", labelsize=15)


def _add_legend(ax, on_pixels, off_pixels):
    """Add legend to the plot."""
    legend_objects = [leg_h.PixelObject(), leg_h.EdgePixelObject()]
    legend_labels = ["Pixel", "Edge pixel"]

    # Choose handler based on pixel shape
    is_hex = isinstance(on_pixels[0], mpatches.RegularPolygon)
    legend_handler_map = {
        leg_h.PixelObject: leg_h.HexPixelHandler() if is_hex else leg_h.SquarePixelHandler(),
        leg_h.EdgePixelObject: leg_h.HexEdgePixelHandler()
        if is_hex
        else leg_h.SquareEdgePixelHandler(),
        leg_h.OffPixelObject: leg_h.HexOffPixelHandler()
        if is_hex
        else leg_h.SquareOffPixelHandler(),
    }

    if off_pixels:
        legend_objects.append(leg_h.OffPixelObject())
        legend_labels.append("Disabled pixel")

    ax.legend(
        legend_objects,
        legend_labels,
        handler_map=legend_handler_map,
        prop={"size": 11},
        loc="upper right",
    )
