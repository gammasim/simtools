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

    if not is_two_mirror_telescope(telescope_model_name):
        y_pos = -y_pos

    # Apply rotation
    base_angle = 0.0
    if "MSTS" in telescope_model_name:
        base_angle = 270.0
    elif "SST" in telescope_model_name or "SCT" in telescope_model_name:
        base_angle = 90.0
    else:
        base_angle = 248.2

    # Apply base rotation and any additional rotation from config
    total_rotation = base_angle
    if config.get("rotate_angle") is not None:
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
        # Add FOV info
        "fov_diameter": config["fov_diameter"],
        "focal_length": config["focal_length"],
        "edge_radius": config["edge_radius"],
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

    # Combine all patches into a single collection
    all_patches = on_pixels + edge_pixels + off_pixels
    facecolors = [
        "none"
        if i < len(on_pixels)
        else (*mcolors.to_rgb("brown"), 0.5)
        if i < len(on_pixels) + len(edge_pixels)
        else "black"
        for i in range(len(on_pixels) + len(edge_pixels) + len(off_pixels))
    ]
    edgecolors = (
        ["black"] * len(on_pixels)
        + [(*mcolors.to_rgb("black"), 1)] * len(edge_pixels)
        + ["black"] * len(off_pixels)
    )
    linewidths = [0.2] * len(all_patches)

    # Add the combined collection
    ax.add_collection(
        PatchCollection(
            all_patches,
            facecolor=facecolors,
            edgecolor=edgecolors,
            linewidth=linewidths,
            match_original=True,
        )
    )

    # Configure plot with titles
    _configure_plot(
        ax,
        telescope_model_name,
        pixel_data["x"],
        pixel_data["y"],
        rotation=pixel_data["rotation"],
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
        fov_info={
            "diameter": pixel_data.get("fov_diameter"),
            "focal_length": pixel_data.get("focal_length"),
            "edge_radius": pixel_data.get("edge_radius"),
        },
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
        "rotate_angle": None,
        "fov_diameter": None,
        "focal_length": None,
        "edge_radius": None,
    }

    with open(dat_file_path, encoding="utf-8") as f:
        for line in f:
            if "field-of-view diameter of" in line and "focal length of" in line:
                fov_part = line.split("field-of-view diameter of")[1]
                config["fov_diameter"] = float(fov_part.split("deg")[0].strip())
                focal_part = line.split("focal length of")[1]
                config["focal_length"] = float(focal_part.split("m")[0].strip())

            elif "Mean radius of camera edge" in line and "is" in line:
                radius_part = line.split("is")[1]
                config["edge_radius"] = float(radius_part.split("m")[0].strip())

            elif line.startswith("Rotate"):
                # Parse rotation angle from line like "Rotate 10.893"
                config["rotate_angle"] = float(line.split()[1].strip())
            elif line.startswith("PixType"):
                parts = line.split()
                config["pixel_shape"] = int(parts[5].strip())
                config["pixel_diameter"] = float(parts[6].strip())
            elif line.startswith("Pixel"):
                parts = line.split()
                config["x"].append(float(parts[3].strip()))
                config["y"].append(float(parts[4].strip()))
                config["pixel_ids"].append(int(parts[1].strip()))
                if len(parts) >= 9:
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
        return _count_neighbors(x, y, x_pos, y_pos, diameter * 2) < 6
    if shape in (1, 3):  # Hexagonal - increase threshold for neighbor detection
        return _count_neighbors(x, y, x_pos, y_pos, diameter * 1.4) < 6
    # Square
    return _count_neighbors(x, y, x_pos, y_pos, diameter * 2.2) < 6


def _add_pixel_id(x, y, pixel_id, font_size):
    """Add pixel ID text to the plot."""
    plt.text(x, y, pixel_id, ha="center", va="center", fontsize=font_size)


def _create_pixel_patches(
    x_pos, y_pos, diameter, shape, pixels_on, pixel_ids, pixels_id_to_print, telescope_model_name
):
    """Create matplotlib patches for different pixel types."""
    on_pixels, edge_pixels, off_pixels = [], [], []

    array_element_type = names.get_array_element_type_from_name(telescope_model_name)
    font_size = 2 if "SCT" in array_element_type else 4

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
            _add_pixel_id(x, y, pixel_ids[i], font_size)

    return on_pixels, edge_pixels, off_pixels


def _count_neighbors(x, y, x_pos, y_pos, max_dist):
    """Count number of neighboring pixels within max_dist."""
    count = 0
    for x2, y2 in zip(x_pos, y_pos):
        if (x != x2 or y != y2) and np.sqrt((x - x2) ** 2 + (y - y2) ** 2) <= max_dist:
            count += 1
    return count


def _configure_plot(
    ax,
    telescope_model_name,
    x_pos,
    y_pos,
    rotation=0,
    title=None,
    xtitle=None,
    ytitle=None,
    fov_info=None,
):
    # First set the aspect ratio
    ax.set_aspect("equal")

    # Calculate the axis limits
    x_min, x_max = min(x_pos), max(x_pos)
    y_min, y_max = min(y_pos), max(y_pos)

    # Add some padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    # Set limits with padding
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

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

    # Add coordinate system axes
    _add_coordinate_axes(ax, x_pos, y_pos, rotation)

    # Add observer note at bottom left
    x_min = min(x_pos) - (max(x_pos) - min(x_pos)) * 0.05  # Use same padding as plot limits
    y_min = min(y_pos) - (max(y_pos) - min(y_pos)) * 0.05
    ax.text(x_min, y_min, "For an observer facing the camera", fontsize=10, ha="left", va="bottom")

    # Add FOV info at top left
    if fov_info and fov_info["diameter"] and fov_info["focal_length"] and fov_info["edge_radius"]:
        info_text = (
            f"FoV diameter: {fov_info['diameter']:.2f}°\n"
            f"Focal length: {fov_info['focal_length']:.2f} m\n"
            f"Edge radius: {fov_info['edge_radius'] * 100:.1f} cm"
        )
        ax.text(
            x_min + x_padding * 0.2,  # Position near top left
            y_max - y_padding * 0.1,
            info_text,
            fontsize=10,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )


def _add_coordinate_axes(ax, x_pos, y_pos, rotation=0):
    """Add coordinate system axes to the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on
    x_pos, y_pos : array-like
        Arrays of x and y positions of pixels
    rotation : float
        Rotation angle in degrees
    """
    # Calculate the length of the axes
    x_range = max(x_pos) - min(x_pos)
    y_range = max(y_pos) - min(y_pos)
    axis_length = min(x_range, y_range) * 0.08

    # Find the rightmost and bottom-most pixels
    rightmost_x = max(x_pos)
    bottommost_y = min(y_pos)

    # Calculate typical pixel spacing
    pixel_distances = []
    for i, (x1, y1) in enumerate(zip(x_pos, y_pos)):
        for x2, y2 in zip(x_pos[i + 1 :], y_pos[i + 1 :]):
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist > 0:  # Avoid same pixel
                pixel_distances.append(dist)
    pixel_spacing = min(pixel_distances) if pixel_distances else axis_length

    # Position Az-Alt axes further away from pixels
    x_origin = rightmost_x + pixel_spacing
    y_origin = bottommost_y + pixel_spacing

    # Adjust position if too close to plot border
    x_padding = x_range * 0.1
    if x_origin + axis_length > rightmost_x + x_padding:
        x_origin = rightmost_x + x_padding - axis_length * 1.5

    # Make arrows thinner and slightly longer
    arrow_style = {
        "head_width": axis_length * 0.15,
        "head_length": axis_length * 0.15,
        "width": axis_length * 0.02,
    }

    arrow_length = 0.6

    # Check if this is an SST telescope by looking at the rotation angle
    # SST telescopes have a base rotation of 90 degrees
    is_sst = abs(rotation - 90.0) < 1.0

    az_direction = 1 if is_sst else -1
    ax.arrow(
        x_origin,
        y_origin,
        az_direction * axis_length * arrow_length,
        0,
        fc="red",
        ec="red",
        **arrow_style,
    )
    # Position Az text based on direction
    ax.text(
        x_origin + az_direction * axis_length * (arrow_length + 0.2),
        y_origin,
        "Az",
        ha="left" if az_direction > 0 else "right",
        va="center",
        color="red",
    )

    ax.arrow(x_origin, y_origin, 0, -axis_length * arrow_length, fc="red", ec="red", **arrow_style)
    # Position Alt text slightly further below arrow tip
    ax.text(
        x_origin,
        y_origin - axis_length * (arrow_length + 0.2),
        "Alt",
        ha="center",
        va="top",
        color="red",
    )

    # Calculate rotated axes for X_pix and Y_pix
    rot_angle = np.deg2rad(rotation)

    # Position x_pix-y_pix with increased separation from Az-Alt
    x_origin_pix = x_origin - axis_length * 2.0  # Increased separation
    y_origin_pix = y_origin

    # X_pix axis direction depends on telescope type
    x_direction = -1 if is_sst else 1  # Invert direction for SST
    x_dir = x_direction * axis_length * arrow_length * np.cos(rot_angle)
    y_dir = x_direction * axis_length * arrow_length * np.sin(rot_angle)
    ax.arrow(x_origin_pix, y_origin_pix, x_dir, y_dir, fc="black", ec="black", **arrow_style)
    # Position X_pix text with more spacing and LaTeX subscript
    ax.text(
        x_origin_pix + x_dir * 1.3,
        y_origin_pix + y_dir * 1.3,
        "$X_{pix}$",
        ha="right" if x_dir < 0 else "left",
        va="center",
    )

    # Y_pix axis
    y_x_dir = axis_length * arrow_length * np.sin(rot_angle)
    y_y_dir = -axis_length * arrow_length * np.cos(rot_angle)
    ax.arrow(x_origin_pix, y_origin_pix, y_x_dir, y_y_dir, fc="black", ec="black", **arrow_style)
    # Position Y_pix text with more spacing and LaTeX subscript
    ax.text(
        x_origin_pix + y_x_dir * 1.3,
        y_origin_pix + y_y_dir * 1.3,
        "$Y_{pix}$",
        ha="center",
        va="top" if y_y_dir < 0 else "bottom",
    )


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
