"""Plot camera pixel layout."""

import logging

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from simtools.model.model_utils import is_two_mirror_telescope
from simtools.utils import names
from simtools.visualization.camera_plot_utils import (
    add_pixel_legend,
    add_pixel_patch_collections,
    create_pixel_patches_by_type,
    pixel_shape,
    setup_camera_axis_properties,
)

logger = logging.getLogger(__name__)


def plot_pixel_layout(camera, camera_in_sky_coor=False, pixels_id_to_print=50):
    """
    Plot the pixel layout for an observer facing the camera.

    Including in the plot edge pixels, off pixels, pixel ID for the first 50 pixels,
    coordinate systems, FOV, focal length and the average edge radius.

    Parameters
    ----------
    camera : Camera
        Camera object to plot.
    camera_in_sky_coor : bool, optional
        Flag to plot the camera in the sky coordinate system.
    pixels_id_to_print : int, optional
        Number of pixel IDs to print in the plot.

    Returns
    -------
    fig : plt.Figure
        Figure with the pixel layout.
    """
    logger.info(f"Plotting the {camera.telescope_name} camera")

    fig, ax = plt.subplots(figsize=(8, 8))

    if not is_two_mirror_telescope(camera.telescope_name) and not camera_in_sky_coor:
        camera.pixels["y"] = [(-1) * y_val for y_val in camera.pixels["y"]]

    on_pixels, edge_pixels, off_pixels = create_pixel_patches_by_type(camera)
    for i_pix, (x, y) in enumerate(zip(camera.pixels["x"], camera.pixels["y"])):
        if camera.pixels["pix_id"][i_pix] < pixels_id_to_print + 1:
            font_size = (
                4 if "SCT" in names.get_array_element_type_from_name(camera.telescope_name) else 2
            )
            plt.text(
                x, y, camera.pixels["pix_id"][i_pix], ha="center", va="center", fontsize=font_size
            )
    add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels)
    setup_camera_axis_properties(ax, camera, grid=True, axis_below=True, y_scale_factor=1.42)
    plt.xlabel("Horizontal scale [cm]", fontsize=18, labelpad=0)
    plt.ylabel("Vertical scale [cm]", fontsize=18, labelpad=0)
    ax.set_title(
        f"Pixels layout in {camera.telescope_name} camera",
        fontsize=15,
        y=1.02,
    )
    plt.tick_params(axis="both", which="major", labelsize=15)

    _plot_axes_def(camera, plt, camera.pixels["rotate_angle"])

    description = {
        False: "For an observer facing the camera",
        True: "For an observer behind the camera looking through",
        None: "For an observer looking from secondary to camera",
    }[camera_in_sky_coor and not is_two_mirror_telescope(camera.telescope_name)]
    ax.text(0.02, 0.02, description, transform=ax.transAxes, color="black", fontsize=12)

    fov, r_edge_avg = camera.calc_fov()
    ax.text(
        0.02,
        0.96,
        rf"$f_{{\mathrm{{eff}}}} = {camera.focal_length:.3f}\,\mathrm{{cm}}$",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
    )
    ax.text(
        0.02,
        0.92,
        rf"Avg. edge radius = {r_edge_avg:.3f}$\,\mathrm{{cm}}$",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
    )
    ax.text(
        0.02,
        0.88,
        rf"FoV = {fov:.3f}$\,\mathrm{{deg}}$",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
    )
    add_pixel_legend(ax, on_pixels, off_pixels)
    plt.tight_layout()

    return fig


def plot_pixel_layout_with_image(
    camera,
    image=None,
    colormap="viridis",
    norm="lin",
    ax=None,
    vmin=None,
    vmax=None,
    add_color_bar=True,
    color_bar_label="Pixel Value",
    **kwargs,
):
    """
    Plot pixel layout with optional per-pixel values as colors.

    Used to display DL1 images (e.g., integrated signal, peak timing) on the camera.

    Parameters
    ----------
    camera : Camera
        Camera object with pixel geometry.
    image : np.ndarray, optional
        Per-pixel values to display as colors (shape: n_pix,).
        If None, plot layout without colored values.
    colormap : str, optional
        Colormap name (default "viridis").
    norm : str, optional
        Normalization type: "lin" (linear), "log", or "symlog" (default "lin").
    ax : plt.Axes, optional
        Existing axes to plot on. If None, create new figure.
    vmin, vmax : float, optional
        Value range for normalization. If None, use data range.
    add_color_bar : bool, optional
        Whether to add a color bar (default True).
    color_bar_label : str, optional
        Label for the color bar (default "Pixel Value").
    **kwargs
        Additional arguments passed to plt.subplots() if ax is None.

    Returns
    -------
    plt.Figure
        Figure with the pixel layout and optional image overlay.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 8)))
    else:
        fig = ax.figure

    colors, cmap, norm_obj = _color_normalization(image, colormap, norm, vmin, vmax)

    for i_pix, (x, y) in enumerate(zip(camera.pixels["x"], camera.pixels["y"])):
        shape = pixel_shape(camera, x, y)

        if colors is not None and image is not None:
            facecolor = colors[i_pix]
            edgecolor = "black"
            linewidth = 0.1
        elif camera.pixels["pix_on"][i_pix]:
            facecolor = "none"
            edgecolor = "black"
            linewidth = 0.2
        else:
            facecolor = "black"
            edgecolor = "black"
            linewidth = 0.2

        ax.add_patch(shape)
        shape.set_facecolor(facecolor)
        shape.set_edgecolor(edgecolor)
        shape.set_linewidth(linewidth)

    # Add colorbar if image provided
    if colors is not None and image is not None and add_color_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.05)
        cbar.set_label(color_bar_label, fontsize=10)
    setup_camera_axis_properties(ax, camera, grid=True, grid_alpha=0.3, padding=0.1)
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)

    return fig


def _color_normalization(image, color_map, norm_type="lin", vmin=None, vmax=None):
    """
    Color normalize an image array for plotting.

    Parameters
    ----------
    image : np.ndarray
        Array of pixel values.
    color_map : str
        Colormap name.
    norm_type : str, optional
        Normalization type: "lin", "log", or "symlog" (default "lin").
    vmin, vmax : float, optional
        Value range for normalization.

    Returns
    -------
    np.ndarray, plt.Colormap, mcolors.Normalize
        Array of RGBA colors for each pixel.
        Colormap instance.
        Normalization instance.
        Returns None for colors if image is None.
    """
    if image is None:
        return None

    if norm_type == "log":
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm_type == "symlog":
        norm = mcolors.SymLogNorm(vmin=vmin, vmax=vmax)
    else:  # "lin"
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    normalized_image = norm(image)
    cmap = plt.get_cmap(color_map)
    colors = cmap(normalized_image)

    return colors, cmap, norm


def _plot_axes_def(camera, plot, rotate_angle):
    """
    Plot three axes definitions on the pyplot.plt instance provided.

    The three axes are Alt/Az, the camera coordinate system and the original coordinate
    system the pixel list was provided.

    Parameters
    ----------
    camera : Camera
        Camera object.
    plot: pyplot.plt instance
        A pyplot.plt instance where to add the axes definitions.
    rotate_angle: float
        The rotation angle applied
    """
    invert_yaxis = False
    x_left = 0.7  # Position of the left most axis
    if not is_two_mirror_telescope(camera.telescope_name):
        invert_yaxis = True
        x_left = 0.8

    x_title = r"$x_{\!pix}$"
    y_title = r"$y_{\!pix}$"
    x_pos, y_pos = (x_left, 0.12)
    # The rotation of LST (above 100 degrees) raises the axes.
    # In this case, lower the starting point.
    if np.rad2deg(rotate_angle) > 100:
        y_pos -= 0.09
        x_pos -= 0.05
    kwargs = {
        "x_title": x_title,
        "y_title": y_title,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "rotate_angle": rotate_angle - (1 / 2.0) * np.pi,
        "fc": "black",
        "ec": "black",
        "invert_yaxis": invert_yaxis,
    }
    _plot_one_axis_def(plot, **kwargs)

    x_title = r"$x_{\!cam}$"
    y_title = r"$y_{\!cam}$"
    x_pos, y_pos = (x_left + 0.15, 0.12)
    kwargs = {
        "x_title": x_title,
        "y_title": y_title,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "rotate_angle": (3 / 2.0) * np.pi,
        "fc": "blue",
        "ec": "blue",
        "invert_yaxis": invert_yaxis,
    }
    _plot_one_axis_def(plot, **kwargs)

    x_title = "Alt"
    y_title = "Az"
    x_pos, y_pos = (x_left + 0.15, 0.25)
    kwargs = {
        "x_title": x_title,
        "y_title": y_title,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "rotate_angle": (3 / 2.0) * np.pi,
        "fc": "red",
        "ec": "red",
        "invert_yaxis": invert_yaxis,
    }
    _plot_one_axis_def(plot, **kwargs)


def _plot_one_axis_def(plot, **kwargs):
    """
    Plot an axis on the pyplot.plt instance provided.

    Parameters
    ----------
    plot: pyplot.plt instance
        A pyplot.plt instance where to add the axes definitions.
    **kwargs: dict
            x_title: str
            x-axis title
            y_title: str
            y-axis title,
            x_pos: float
            x position of the axis to draw
            y_pos: float
            y position of the axis to draw
            rotate_angle: float
            rotation angle of the axis in radians
            fc: str
            face colour of the axis
            ec: str
            edge colour of the axis
            invert_yaxis: bool
            Flag to invert the y-axis (for dual mirror telescopes).
    """
    x_title = kwargs["x_title"]
    y_title = kwargs["y_title"]
    x_pos, y_pos = (kwargs["x_pos"], kwargs["y_pos"])

    r = 0.1  # size of arrow
    sign = 1.0
    if kwargs["invert_yaxis"]:
        sign *= -1.0
    x_text1 = x_pos + sign * r * np.cos(kwargs["rotate_angle"])
    y_text1 = y_pos + r * np.sin(0 + kwargs["rotate_angle"])
    x_text2 = x_pos + sign * r * np.cos(np.pi / 2.0 + kwargs["rotate_angle"])
    y_text2 = y_pos + r * np.sin(np.pi / 2.0 + kwargs["rotate_angle"])

    plot.gca().annotate(
        x_title,
        xy=(x_pos, y_pos),
        xytext=(x_text1, y_text1),
        xycoords="axes fraction",
        ha="center",
        va="center",
        size="xx-large",
        arrowprops={
            "arrowstyle": "<|-",
            "shrinkA": 0,
            "shrinkB": 0,
            "fc": kwargs["fc"],
            "ec": kwargs["ec"],
        },
    )

    plot.gca().annotate(
        y_title,
        xy=(x_pos, y_pos),
        xytext=(x_text2, y_text2),
        xycoords="axes fraction",
        ha="center",
        va="center",
        size="xx-large",
        arrowprops={
            "arrowstyle": "<|-",
            "shrinkA": 0,
            "shrinkB": 0,
            "fc": kwargs["fc"],
            "ec": kwargs["ec"],
        },
    )
