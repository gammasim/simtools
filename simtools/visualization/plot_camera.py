import logging

import matplotlib as mlp
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

import simtools.visualization.legend_handlers as leg_h
from simtools.model.model_utils import is_two_mirror_telescope
from simtools.utils import names

logger = logging.getLogger(__name__)


def plot_pixel_layout(camera, camera_in_sky_coor=False, pixels_id_to_print=50):
    """
    Plot the pixel layout for an observer facing the camera. Including in the plot edge pixels,\
    off pixels, pixel ID for the first 50 pixels, coordinate systems, FOV, focal length and the\
    average edge radius.

    Parameters
    ----------
    camera : Camera
        Camera object to plot.
    camera_in_sky_coor : bool
        Flag to plot the camera in the sky coordinate system.
    pixels_id_to_print : int
        Number of pixel IDs to print in the plot.

    Returns
    -------
    fig: plt.figure instance
        Figure with the pixel layout.
    """

    logger.info(f"Plotting the {camera.telescope_model_name} camera")

    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(8, 8)

    if not is_two_mirror_telescope(camera.telescope_model_name):
        if not camera_in_sky_coor:
            camera.pixels["y"] = [(-1) * y_val for y_val in camera.pixels["y"]]

    on_pixels, edge_pixels, off_pixels = [], [], []

    for i_pix, xy_pix_pos in enumerate(zip(camera.pixels["x"], camera.pixels["y"])):
        if camera.pixels["pixel_shape"] == 1 or camera.pixels["pixel_shape"] == 3:
            hexagon = mpatches.RegularPolygon(
                (xy_pix_pos[0], xy_pix_pos[1]),
                numVertices=6,
                radius=camera.pixels["pixel_diameter"] / np.sqrt(3),
                orientation=np.deg2rad(camera.pixels["orientation"]),
            )
            if camera.pixels["pix_on"][i_pix]:
                if len(camera.get_neighbour_pixels()[i_pix]) < 6:
                    edge_pixels.append(hexagon)
                else:
                    on_pixels.append(hexagon)
            else:
                off_pixels.append(hexagon)
        elif camera.pixels["pixel_shape"] == 2:
            square = mpatches.Rectangle(
                (
                    xy_pix_pos[0] - camera.pixels["pixel_diameter"] / 2.0,
                    xy_pix_pos[1] - camera.pixels["pixel_diameter"] / 2.0,
                ),
                width=camera.pixels["pixel_diameter"],
                height=camera.pixels["pixel_diameter"],
            )
            if camera.pixels["pix_on"][i_pix]:
                if len(camera.get_neighbour_pixels()[i_pix]) < 4:
                    edge_pixels.append(square)
                else:
                    on_pixels.append(square)
            else:
                off_pixels.append(square)

        if camera.pixels["pix_id"][i_pix] < pixels_id_to_print + 1:
            font_size = 4
            if "SCT" in names.get_telescope_type_from_telescope_name(camera.telescope_model_name):
                font_size = 2
            plt.text(
                xy_pix_pos[0],
                xy_pix_pos[1],
                camera.pixels["pix_id"][i_pix],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_size,
            )

    ax.add_collection(
        PatchCollection(on_pixels, facecolor="none", edgecolor="black", linewidth=0.2)
    )
    ax.add_collection(
        PatchCollection(
            edge_pixels,
            facecolor=mcolors.to_rgb("brown") + (0.5,),
            edgecolor=mcolors.to_rgb("black") + (1,),
            linewidth=0.2,
        )
    )
    ax.add_collection(
        PatchCollection(off_pixels, facecolor="black", edgecolor="black", linewidth=0.2)
    )

    legend_objects = [leg_h.PixelObject(), leg_h.EdgePixelObject()]
    legend_labels = ["Pixel", "Edge pixel"]
    if isinstance(on_pixels[0], mlp.patches.RegularPolygon):
        legend_handler_map = {
            leg_h.PixelObject: leg_h.HexPixelHandler(),
            leg_h.EdgePixelObject: leg_h.HexEdgePixelHandler(),
            leg_h.OffPixelObject: leg_h.HexOffPixelHandler(),
        }
    elif isinstance(on_pixels[0], mlp.patches.Rectangle):
        legend_handler_map = {
            leg_h.PixelObject: leg_h.SquarePixelHandler(),
            leg_h.EdgePixelObject: leg_h.SquareEdgePixelHandler(),
            leg_h.OffPixelObject: leg_h.SquareOffPixelHandler(),
        }

    if len(off_pixels) > 0:
        legend_objects.append(leg_h.OffPixelObject())
        legend_labels.append("Disabled pixel")

    plt.axis("equal")
    plt.grid(True)
    ax.set_axisbelow(True)
    plt.axis(
        [
            min(camera.pixels["x"]),
            max(camera.pixels["x"]),
            min(camera.pixels["y"]) * 1.42,
            max(camera.pixels["y"]) * 1.42,
        ]
    )
    plt.xlabel("Horizontal scale [cm]", fontsize=18, labelpad=0)
    plt.ylabel("Vertical scale [cm]", fontsize=18, labelpad=0)
    ax.set_title(
        f"Pixels layout in {camera.telescope_model_name:s} camera",
        fontsize=15,
        y=1.02,
    )
    plt.tick_params(axis="both", which="major", labelsize=15)

    _plot_axes_def(camera, plt, camera.pixels["rotate_angle"])
    description = "For an observer facing the camera"
    if camera_in_sky_coor and not is_two_mirror_telescope(camera.telescope_model_name):
        description = "For an observer behind the camera looking through"
    if is_two_mirror_telescope(camera.telescope_model_name):
        description = "For an observer looking from secondary to camera"
    ax.text(
        0.02,
        0.02,
        description,
        transform=ax.transAxes,
        color="black",
        fontsize=12,
    )

    fov, r_edge_avg = camera.calc_fov()
    ax.text(
        0.02,
        0.96,
        r"$f_{\mathrm{eff}}$ = " + f"{camera.focal_length:.3f} cm",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
    )
    ax.text(
        0.02,
        0.92,
        f"Avg. edge radius = {r_edge_avg:.3f} cm",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
    )
    ax.text(
        0.02,
        0.88,
        f"FoV = {fov:.3f} deg",
        transform=ax.transAxes,
        color="black",
        fontsize=12,
    )

    plt.legend(
        legend_objects,
        legend_labels,
        handler_map=legend_handler_map,
        prop={"size": 11},
        loc="upper right",
    )

    ax.set_aspect("equal", "datalim")
    plt.tight_layout()

    return fig


def _plot_axes_def(camera, plot, rotate_angle):
    """
    Plot three axes definitions on the pyplot.plt instance provided. The three axes are Alt/Az,\
    the camera coordinate system and the original coordinate system the pixel list was provided.

    Parameters
    ----------
    plot: pyplot.plt instance
        A pyplot.plt instance where to add the axes definitions.
    rotate_angle: float
        The rotation angle applied
    """

    invert_yaxis = False
    x_left = 0.7  # Position of the left most axis
    if not is_two_mirror_telescope(camera.telescope_model_name):
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
