"""Shared helpers for camera pixel plotting."""

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PatchCollection

from simtools.visualization import legend_handlers as leg_h


def pixel_shape(camera, x, y):
    """Return the shape of the pixel.

    Parameters
    ----------
    camera : Camera
        Camera object containing pixel configuration and properties.
    x : float
        X coordinate of the pixel center.
    y : float
        Y coordinate of the pixel center.

    Returns
    -------
    matplotlib.patches.Patch or None
        A patch object representing the pixel (hexagon, rectangle) or
        None if shape is not recognized.
    """
    if camera.pixels["pixel_shape"] in (1, 3):
        return mpatches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=camera.pixels["pixel_diameter"] / np.sqrt(3),
            orientation=np.deg2rad(camera.pixels["orientation"]),
        )
    if camera.pixels["pixel_shape"] == 2:
        return mpatches.Rectangle(
            (
                x - camera.pixels["pixel_diameter"] / 2.0,
                y - camera.pixels["pixel_diameter"] / 2.0,
            ),
            width=camera.pixels["pixel_diameter"],
            height=camera.pixels["pixel_diameter"],
        )
    return None


def create_pixel_patches_by_type(camera):
    """Create pixel patches categorized by type (on, edge, off).

    Parameters
    ----------
    camera : Camera
        Camera object containing pixel positions and status information.

    Returns
    -------
    tuple of (list, list, list)
        Three lists containing matplotlib patch objects:
        - on_pixels: patches for enabled pixels not on the edge
        - edge_pixels: patches for enabled pixels on the edge
        - off_pixels: patches for disabled pixels
    """
    on_pixels, edge_pixels, off_pixels = [], [], []
    edge_indices = set(camera.get_edge_pixels())

    for i_pix, (x, y) in enumerate(zip(camera.pixels["x"], camera.pixels["y"])):
        shape = pixel_shape(camera, x, y)
        if camera.pixels["pix_on"][i_pix]:
            if i_pix in edge_indices:
                edge_pixels.append(shape)
            else:
                on_pixels.append(shape)
        else:
            off_pixels.append(shape)

    return on_pixels, edge_pixels, off_pixels


def _get_legend_handler(handler_type, is_hex):
    """Get appropriate legend handler based on pixel shape.

    Parameters
    ----------
    handler_type : str
        Handler type prefix (e.g., "Pixel", "EdgePixel", "OffPixel").
    is_hex : bool
        True for hexagonal pixels, False for square pixels.

    Returns
    -------
    Handler instance
        Appropriate handler for the pixel shape and type.
    """
    shape = "Hex" if is_hex else "Square"
    handler_name = f"{shape}{handler_type}Handler"
    return getattr(leg_h, handler_name)()


def add_pixel_legend(ax, on_pixels, off_pixels):
    """Add pixel/edge/off legend to the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the legend to.
    on_pixels : list
        List of enabled pixel patches.
    off_pixels : list
        List of disabled pixel patches.

    Returns
    -------
    None
    """
    if not on_pixels:
        return

    legend_objects = [leg_h.PixelObject(), leg_h.EdgePixelObject()]
    legend_labels = ["Pixel", "Edge pixel"]

    is_hex = isinstance(on_pixels[0], mpatches.RegularPolygon)
    legend_handler_map = {
        leg_h.PixelObject: _get_legend_handler("Pixel", is_hex),
        leg_h.EdgePixelObject: _get_legend_handler("EdgePixel", is_hex),
        leg_h.OffPixelObject: _get_legend_handler("OffPixel", is_hex),
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


def add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels):
    """Add patch collections for on/edge/off pixels to the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the patch collections to.
    on_pixels : list
        List of enabled pixel patches not on the edge.
    edge_pixels : list
        List of enabled pixel patches on the edge.
    off_pixels : list
        List of disabled pixel patches.

    Returns
    -------
    None
    """
    ax.add_collection(
        PatchCollection(on_pixels, facecolor="none", edgecolor="black", linewidth=0.2)
    )
    ax.add_collection(
        PatchCollection(
            edge_pixels,
            facecolor=(*mcolors.to_rgb("brown"), 0.5),
            edgecolor=(*mcolors.to_rgb("black"), 1),
            linewidth=0.2,
        )
    )
    ax.add_collection(
        PatchCollection(off_pixels, facecolor="black", edgecolor="black", linewidth=0.2)
    )


def setup_camera_axis_properties(
    ax, camera, grid=False, axis_below=False, grid_alpha=None, y_scale_factor=1.0, padding=0
):
    """Set up common axis properties for camera plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to configure.
    camera : Camera
        Camera object containing pixel positions and extent information.
    grid : bool, optional
        Whether to display grid lines. Default is False.
    axis_below : bool, optional
        Whether to place axis below the plot elements. Default is False.
    grid_alpha : float, optional
        Transparency level for grid lines. If None and grid is True, uses default alpha.
    y_scale_factor : float, optional
        Scaling factor for y-axis limits. Default is 1.0.
    padding : float, optional
        Padding around the camera extent. Default is 0.

    Returns
    -------
    None
    """
    x_min, x_max = min(camera.pixels["x"]), max(camera.pixels["x"])
    y_min, y_max = min(camera.pixels["y"]), max(camera.pixels["y"])

    if y_scale_factor > 1.0:
        ax.axis([x_min, x_max, y_min * y_scale_factor, y_max * y_scale_factor])
    else:
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_aspect("equal", "datalim")

    if grid:
        if grid_alpha is not None:
            ax.grid(True, alpha=grid_alpha)
        else:
            ax.grid(True)

    if axis_below:
        ax.set_axisbelow(True)
