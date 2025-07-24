#!/usr/bin/python3
"""Functions for plotting pixel layout information."""

import logging
from pathlib import Path

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from simtools.db import db_handler
from simtools.io import io_handler
from simtools.model.model_utils import is_two_mirror_telescope
from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h
from simtools.visualization import visualize

logger = logging.getLogger(__name__)


def plot(config, output_file, db_config=None):
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
    output_file : str
        Path where to save the plot
    db_config : dict, optional
        Database configuration.

    Returns
    -------
    None
        The function saves the plot to the specified output file.
    """
    db = db_handler.DatabaseHandler(mongo_db_config=db_config)
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

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    logger.info(f"Plotting pixel layout for {telescope_model_name} from {dat_file_path}")

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
    """Prepare pixel data from sim_telarray camera configuration file.

    This function reads the pixel configuration from the specified camera config file and
    prepares the data for plotting, including applying any necessary rotations.

    Parameters
    ----------
    dat_file_path : str or Path
        Path to the camera config file containing pixel configuration
    telescope_model_name : str
        Name/model of the telescope

    Returns
    -------
    dict
        Dictionary containing pixel data
    """
    config = _read_pixel_config(dat_file_path)
    x_pos = np.array(config["x"])
    y_pos = np.array(config["y"])

    if not is_two_mirror_telescope(telescope_model_name):
        y_pos = -y_pos

    rotate_angle = (
        config.get("rotate_angle") if config.get("rotate_angle") is not None else (0.0 * u.deg)
    )

    # Apply telescope-specific adjustments
    if "SST" in telescope_model_name or "SCT" in telescope_model_name:
        total_rotation = (90 * u.deg) - (rotate_angle)
    else:
        total_rotation = (-90 * u.deg) - (rotate_angle)

    # Apply rotation
    rot_angle = total_rotation.to(u.rad).value
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
        "pixel_spacing": config["pixel_spacing"],
        "module_number": config["module_number"],
        "module_gap": config["module_gap"],
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
        pixel_data["module_number"],
        pixel_data["module_gap"],
        pixel_data["pixel_spacing"],
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
        pixel_data["x"],
        pixel_data["y"],
        rotation=pixel_data["rotation"],
        title=title,
        xtitle=xtitle,
        ytitle=ytitle,
    )
    _add_legend(ax, on_pixels, off_pixels)

    return fig


def _read_pixel_config(dat_file_path):
    """Read pixel configuration from a camera configuration file.

    This function reads the pixel configuration from the specified camera config file and
    returns it as a dictionary. It parses information such as pixel positions,
    module numbers, and other relevant parameters.

    Parameters
    ----------
    dat_file_path : str or Path
        Path to the camera config file containing pixel configuration

    Returns
    -------
    dict
        config containing pixel data
    """
    config = {
        "x": [],
        "y": [],
        "pixel_ids": [],
        "pixels_on": [],
        "pixel_shape": None,
        "pixel_diameter": None,
        "pixel_spacing": None,
        "module_gap": None,
        "trigger_groups": [],
        "rotate_angle": None,
        "module_number": [],
    }

    with open(dat_file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Parse specific information from the file
            if line.startswith("Rotate"):
                # Parse rotation angle from line like "Rotate 10.893" (u.deg)
                config["rotate_angle"] = float(line.split()[1].strip()) * u.deg

            elif line.startswith("PixType"):
                parts = line.split()
                config["pixel_shape"] = int(parts[5].strip())
                config["pixel_diameter"] = float(parts[6].strip())

            elif "Pixel spacing is" in line:
                config["pixel_spacing"] = float(line.split("spacing is")[1].strip().split()[0])

            elif "Between modules is an additional gap of" in line:
                config["module_gap"] = float(line.split("gap of")[1].strip().split()[0])

            elif line.startswith("Pixel"):
                parts = line.split()
                config["x"].append(float(parts[3].strip()))
                config["y"].append(float(parts[4].strip()))
                config["module_number"].append(float(parts[5].strip()))
                config["pixel_ids"].append(int(parts[1].strip()))
                config["pixels_on"].append(int(parts[9].strip()) != 0)

    config["pixel_spacing"] = (
        config["pixel_diameter"] if config["pixel_spacing"] is None else config["pixel_spacing"]
    )
    config["module_gap"] = 0.0 if config["module_gap"] is None else config["module_gap"]

    return config


def _create_patch(x, y, diameter, shape):
    """Create a single matplotlib patch for a pixel.

    This function creates a matplotlib patch (shape) for a single pixel based on
    its position, diameter, and shape type. Supported shapes are circles, squares,
    and hexagons.

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

    Returns
    -------
    matplotlib.patches.Patch
        The created patch object for the pixel
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


def _is_edge_pixel(
    x, y, x_pos, y_pos, module_ids, pixel_spacing, module_gap, shape, current_module_id
):
    """
    Determine if a pixel is on the edge based on neighbor count.

    Parameters
    ----------
    x, y : float
        Coordinates of the pixel being checked.
    x_pos, y_pos : array-like
        Arrays of x and y positions of all pixels.
    module_ids : array-like
        Array of module IDs corresponding to each pixel.
    pixel_spacing : float
        Center-to-center spacing between pixels.
    module_gap : float
        Additional gap between modules.
    shape : int
        Pixel shape type (0: circular, 1/3: hexagonal, 2: square).
    current_module_id : int
        Module ID of the current pixel.

    Returns
    -------
    bool
        True if the pixel is an edge pixel, False otherwise.
    """
    # Determine the maximum number of neighbors based on the pixel shape
    if shape == 0:  # Circular
        max_neighbors = 8
    elif shape in (1, 3):  # Hexagonal
        max_neighbors = 6
    elif shape == 2:  # Square
        max_neighbors = 4
    else:
        raise ValueError(f"Unsupported pixel shape: {shape}")

    neighbor_count = _count_neighbors(
        x, y, x_pos, y_pos, module_ids, pixel_spacing, module_gap, current_module_id
    )

    # A pixel is an edge pixel if it has fewer neighbors than the maximum
    return neighbor_count < max_neighbors


def _create_pixel_patches(
    x_pos,
    y_pos,
    diameter,
    module_number,
    module_gap,
    spacing,
    shape,
    pixels_on,
    pixel_ids,
    pixels_id_to_print,
    telescope_model_name,
):
    """Create matplotlib patches for different pixel types.

    This function creates the matplotlib patches (shapes) for all pixels in the
    layout, categorizing them into "on", "edge", and "off" pixels based on their
    status and position.

    Parameters
    ----------
    x_pos, y_pos : array-like
        X and Y coordinates of the pixel centers
    diameter : float
        Diameter of the pixels
    module_number : array-like
        Module numbers for each pixel
    module_gap : float
        Gap between modules
    spacing : float
        Pixel spacing
    shape : array-like
        Shape types for each pixel
    pixels_on : array-like
        Status indicating if each pixel is "on"
    pixel_ids : array-like
        Unique IDs for each pixel
    pixels_id_to_print : int
        Number of pixel IDs to print on the plot
    telescope_model_name : str
        Name of the telescope model

    Returns
    -------
    tuple
        Three lists of patches for "on", "edge", and "off" pixels
    """
    on_pixels, edge_pixels, off_pixels = [], [], []

    array_element_type = names.get_array_element_type_from_name(telescope_model_name)
    font_size = 2 if "SCT" in array_element_type else 4

    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        patch = _create_patch(x, y, diameter, shape)

        if pixels_on[i]:
            if _is_edge_pixel(
                x, y, x_pos, y_pos, module_number, spacing, module_gap, shape, module_number[i]
            ):
                edge_pixels.append(patch)
            else:
                on_pixels.append(patch)
        else:
            off_pixels.append(patch)

        if pixel_ids[i] < pixels_id_to_print:
            plt.text(x, y, pixel_ids[i], ha="center", va="center", fontsize=font_size)

    return on_pixels, edge_pixels, off_pixels


def _count_neighbors(x, y, x_pos, y_pos, module_ids, pixel_spacing, module_gap, current_module_id):
    """
    Count the number of neighboring pixels within the appropriate distance.

    Parameters
    ----------
    x, y : float
        Coordinates of the pixel being checked.
    x_pos, y_pos : array-like
        Arrays of x and y positions of all pixels.
    module_ids : array-like
        Array of module IDs corresponding to each pixel.
    pixel_spacing : float
        Center-to-center spacing between pixels.
    module_gap : float
        Additional gap between modules.
    current_module_id : int
        Module ID of the current pixel.

    Returns
    -------
    int
        Number of neighboring pixels.
    """
    count = 0
    tolerance = 1e-6

    for x2, y2, module_id2 in zip(x_pos, y_pos, module_ids):
        # Skip the pixel itself
        if x == x2 and y == y2:
            continue

        # Calculate the distance between the current pixel and the potential neighbor
        dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)

        # Determine max distance based on whether pixels are in same module
        max_distance = (
            pixel_spacing + (0 if current_module_id == module_id2 else module_gap) + tolerance
        ) * 1.2

        if dist <= max_distance:
            count += 1

    return count


def _configure_plot(
    ax,
    x_pos,
    y_pos,
    rotation=0 * u.deg,
    title=None,
    xtitle=None,
    ytitle=None,
):
    """Configure the plot with titles, labels, and limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure
    x_pos, y_pos : array-like
        Arrays of x and y positions of pixels
    rotation : Astropy quantity in degrees, optional
        Rotation angle in degrees, default 0
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
        title or "Pixel layout",
        fontsize=15,
        y=1.02,
    )
    plt.tick_params(axis="both", which="major", labelsize=15)

    _add_coordinate_axes(ax, rotation)
    x_min = min(x_pos) - (max(x_pos) - min(x_pos)) * 0.05
    y_min = min(y_pos) - (max(y_pos) - min(y_pos)) * 0.05
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
