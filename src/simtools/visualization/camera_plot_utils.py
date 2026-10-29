"""Shared helpers for camera pixel plotting."""

import numpy as np

from simtools.visualization import legend_handlers as leg_h
from simtools.visualization.matplotlib_backend import lazy_module

mcolors = lazy_module("matplotlib.colors")
mpatches = lazy_module("matplotlib.patches")
collections = lazy_module("matplotlib.collections")


def pixel_shape(camera, x, y):
    """
    Return the matplotlib patch of one pixel.

    Parameters
    ----------
    camera : Camera
        Camera object containing pixel configuration.
    x, y : float
        Coordinates of the pixel center.

    Returns
    -------
    matplotlib.patches.Patch or None
        Hexagon or rectangle patch; None if the pixel shape is not recognized.
    """
    pixels = camera.pixels
    if pixels["pixel_shape"] in (1, 3):
        return mpatches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=pixels["pixel_diameter"] / np.sqrt(3),
            orientation=np.deg2rad(pixels["orientation"]),
        )
    if pixels["pixel_shape"] == 2:
        half = pixels["pixel_diameter"] / 2.0
        return mpatches.Rectangle(
            (x - half, y - half), width=pixels["pixel_diameter"], height=pixels["pixel_diameter"]
        )
    return None


def create_pixel_patches_by_type(camera):
    """
    Create pixel patches categorized by type.

    Parameters
    ----------
    camera : Camera
        Camera object containing pixel positions and status.

    Returns
    -------
    tuple of (list, list, list)
        Patches for enabled inner pixels, enabled edge pixels and disabled pixels.
    """
    on_pixels, edge_pixels, off_pixels = [], [], []
    edge_indices = set(camera.get_edge_pixels())
    for i_pix, (x, y) in enumerate(zip(camera.pixels["x"], camera.pixels["y"])):
        shape = pixel_shape(camera, x, y)
        if not camera.pixels["pix_on"][i_pix]:
            off_pixels.append(shape)
        elif i_pix in edge_indices:
            edge_pixels.append(shape)
        else:
            on_pixels.append(shape)
    return on_pixels, edge_pixels, off_pixels


def add_pixel_legend(ax, on_pixels, off_pixels):
    """
    Add the pixel, edge pixel and disabled pixel legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the legend to.
    on_pixels : list
        Enabled pixel patches (selects hexagonal or square legend handlers).
    off_pixels : list
        Disabled pixel patches.
    """
    if not on_pixels:
        return
    shape = "Hex" if isinstance(on_pixels[0], mpatches.RegularPolygon) else "Square"
    handler_map = {
        leg_h.PixelObject: getattr(leg_h, f"{shape}PixelHandler")(),
        leg_h.EdgePixelObject: getattr(leg_h, f"{shape}EdgePixelHandler")(),
        leg_h.OffPixelObject: getattr(leg_h, f"{shape}OffPixelHandler")(),
    }
    legend_objects = [leg_h.PixelObject(), leg_h.EdgePixelObject()]
    legend_labels = ["Pixel", "Edge pixel"]
    if off_pixels:
        legend_objects.append(leg_h.OffPixelObject())
        legend_labels.append("Disabled pixel")
    ax.legend(
        legend_objects,
        legend_labels,
        handler_map=handler_map,
        prop={"size": 11},
        loc="upper right",
    )


def add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels):
    """
    Add patch collections for on, edge and off pixels to the axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the patch collections to.
    on_pixels, edge_pixels, off_pixels : list
        Pixel patches (see :func:`create_pixel_patches_by_type`).
    """
    brown = (*mcolors.to_rgb("brown"), 0.5)
    black = (*mcolors.to_rgb("black"), 1)
    for pixels, face, edge in (
        (on_pixels, "none", "black"),
        (edge_pixels, brown, black),
        (off_pixels, "black", "black"),
    ):
        ax.add_collection(
            collections.PatchCollection(pixels, facecolor=face, edgecolor=edge, linewidth=0.2)
        )


def setup_camera_axis_properties(
    ax, camera, grid=False, axis_below=False, grid_alpha=None, y_scale_factor=1.0, padding=0
):
    """
    Set up common axis properties for camera plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to configure.
    camera : Camera
        Camera object containing pixel positions.
    grid, axis_below : bool, optional
        Show grid lines; place axis below the plot elements.
    grid_alpha : float, optional
        Transparency of grid lines (default grid alpha if None).
    y_scale_factor : float, optional
        Scaling factor for y-axis limits.
    padding : float, optional
        Padding around the camera extent.
    """
    x_min, x_max = min(camera.pixels["x"]), max(camera.pixels["x"])
    y_min, y_max = min(camera.pixels["y"]), max(camera.pixels["y"])

    if y_scale_factor > 1.0:
        ax.axis([x_min, x_max, y_min * y_scale_factor, y_max * y_scale_factor])
    else:
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_aspect("equal", adjustable="box")
    if grid and grid_alpha is None:
        ax.grid(True)
    elif grid:
        ax.grid(True, alpha=grid_alpha)
    if axis_below:
        ax.set_axisbelow(True)
