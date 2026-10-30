"""Plot camera pixel layout and trigger patches.

Model data (camera, trigger tables) are passed in by the caller so that this module does not
import the model package.
"""

import copy
import logging
from itertools import pairwise

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, PathPatch
from matplotlib.transforms import Bbox

from simtools.utils import names
from simtools.utils.names import is_two_mirror_telescope
from simtools.visualization.camera_plot_utils import (
    add_pixel_legend,
    add_pixel_patch_collections,
    create_pixel_patches_by_type,
    pixel_shape,
    setup_camera_axis_properties,
)
from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt

logger = logging.getLogger(__name__)
mcolors = lazy_module("matplotlib.colors")
GROUP_ID = "group_id"
MEMBER_PIXEL_ID = "pixel_id"
PIXEL_IDS = "pix_id"
TRIGGER_PATCH = "Trigger patch"
_CENTER = "center"


def plot_pixel_layout(camera, camera_in_sky_coor=False, pixels_id_to_print=50):
    """
    Plot the pixel layout for an observer facing the camera.

    Includes edge pixels, off pixels, pixel IDs (up to ``pixels_id_to_print``),
    coordinate axes, the focal length, the average edge radius and the FoV.

    Parameters
    ----------
    camera : Camera
        Camera object to plot.
    camera_in_sky_coor : bool, optional
        Plot the camera in the sky coordinate system.
    pixels_id_to_print : int, optional
        Number of pixel IDs to print in the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the pixel layout.
    """
    logger.info(f"Plotting the {camera.telescope_name} camera")
    fig, ax = plt.subplots(figsize=(8, 8))
    two_mirror = is_two_mirror_telescope(camera.telescope_name)

    plot_camera = camera
    if not two_mirror and not camera_in_sky_coor:
        plot_camera = copy.copy(camera)
        plot_camera.pixels = {**camera.pixels, "y": [-y for y in camera.pixels["y"]]}

    on_pixels, edge_pixels, off_pixels = create_pixel_patches_by_type(plot_camera)
    font_size = 4 if "SCT" in names.get_array_element_type_from_name(camera.telescope_name) else 2
    for x, y, pix_id in zip(
        plot_camera.pixels["x"], plot_camera.pixels["y"], plot_camera.pixels[PIXEL_IDS]
    ):
        if pix_id < pixels_id_to_print + 1:
            plt.text(x, y, pix_id, ha=_CENTER, va=_CENTER, fontsize=font_size)
    add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels)
    setup_camera_axis_properties(ax, plot_camera, grid=True, axis_below=True, y_scale_factor=1.42)
    plt.xlabel("Horizontal scale [cm]", fontsize=18, labelpad=0)
    plt.ylabel("Vertical scale [cm]", fontsize=18, labelpad=0)
    ax.set_title(f"Pixels layout in {plot_camera.telescope_name} camera", fontsize=15, y=1.02)
    plt.tick_params(axis="both", which="major", labelsize=15)

    _plot_axes_def(plot_camera, plt, plot_camera.pixels["rotate_angle"])

    if camera_in_sky_coor is None:
        description = "For an observer looking from secondary to camera"
    elif camera_in_sky_coor and not two_mirror:
        description = "For an observer behind the camera looking through"
    else:
        description = "For an observer facing the camera"
    ax.text(0.02, 0.02, description, transform=ax.transAxes, color="black", fontsize=12)

    fov, r_edge_avg = plot_camera.calc_fov()
    for y_position, text in (
        (0.96, rf"$f_{{\mathrm{{eff}}}} = {plot_camera.focal_length:.3f}\,\mathrm{{cm}}$"),
        (0.92, rf"Avg. edge radius = {r_edge_avg:.3f}$\,\mathrm{{cm}}$"),
        (0.88, rf"FoV = {fov:.3f}$\,\mathrm{{deg}}$"),
    ):
        ax.text(0.02, y_position, text, transform=ax.transAxes, color="black", fontsize=12)
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
    has_values = colors is not None and image is not None

    patches, facecolors, edgecolors, linewidths = [], [], [], []
    for i_pix, (x, y) in enumerate(zip(camera.pixels["x"], camera.pixels["y"])):
        patches.append(pixel_shape(camera, x, y))
        edgecolors.append("black")
        if has_values:
            facecolors.append(colors[i_pix])
            linewidths.append(0.1)
        else:
            facecolors.append("none" if camera.pixels["pix_on"][i_pix] else "black")
            linewidths.append(0.2)
    ax.add_collection(
        PatchCollection(patches, facecolor=facecolors, edgecolor=edgecolors, linewidth=linewidths)
    )

    if has_values and add_color_bar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.05)
        cbar.set_label(color_bar_label, fontsize=10)
    setup_camera_axis_properties(ax, camera, grid=True, grid_alpha=0.3, padding=0.1)
    ax.set_xlabel("x [m]", fontsize=12)
    ax.set_ylabel("y [m]", fontsize=12)
    return fig


def _color_normalization(image, color_map, norm_type="lin", vmin=None, vmax=None):
    """Return RGBA colors, colormap and norm for an image; ``(None, None, None)`` if no image."""
    if image is None:
        return None, None, None
    norm_classes = {"log": mcolors.LogNorm, "symlog": mcolors.SymLogNorm}
    norm = norm_classes.get(norm_type, mcolors.Normalize)(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(color_map)
    return cmap(norm(image)), cmap, norm


def _plot_axes_def(camera, plot, rotate_angle):
    """Plot the Alt/Az, camera and pixel coordinate axis arrows (rotate_angle in radians)."""
    invert_yaxis = not is_two_mirror_telescope(camera.telescope_name)
    x_left = 0.8 if invert_yaxis else 0.7
    x_pix, y_pix = x_left, 0.12
    # The rotation of LST (above 100 degrees) raises the axes; lower the starting point.
    if np.rad2deg(rotate_angle) > 100:
        y_pix -= 0.09
        x_pix -= 0.05
    definitions = (
        (r"$x_{\!pix}$", r"$y_{\!pix}$", x_pix, y_pix, rotate_angle - (1 / 2.0) * np.pi, "black"),
        (r"$x_{\!cam}$", r"$y_{\!cam}$", x_left + 0.15, 0.12, (3 / 2.0) * np.pi, "blue"),
        ("Alt", "Az", x_left + 0.15, 0.25, (3 / 2.0) * np.pi, "red"),
    )
    for x_title, y_title, x_pos, y_pos, angle, colour in definitions:
        _plot_one_axis_def(plot, x_title, y_title, x_pos, y_pos, angle, colour, invert_yaxis)


def _plot_one_axis_def(plot, x_title, y_title, x_pos, y_pos, rotate_angle, colour, invert_yaxis):
    """Plot one pair of labelled axis arrows (rotate_angle in radians) on ``plot``."""
    r = 0.1  # size of arrow
    sign = -1.0 if invert_yaxis else 1.0
    tips = (
        (x_title, sign * r * np.cos(rotate_angle), r * np.sin(rotate_angle)),
        (
            y_title,
            sign * r * np.cos(np.pi / 2.0 + rotate_angle),
            r * np.sin(np.pi / 2.0 + rotate_angle),
        ),
    )
    for title, dx, dy in tips:
        plot.gca().annotate(
            title,
            xy=(x_pos, y_pos),
            xytext=(x_pos + dx, y_pos + dy),
            xycoords="axes fraction",
            ha=_CENTER,
            va=_CENTER,
            size="xx-large",
            arrowprops={
                "arrowstyle": "<|-",
                "shrinkA": 0,
                "shrinkB": 0,
                "fc": colour,
                "ec": colour,
            },
        )


def trigger_patches_for_pixel(members, pixel_id, camera_pixels):
    """
    Find the distinct trigger-group IDs containing a camera pixel.

    Parameters
    ----------
    members : astropy.table.Table
        Trigger memberships with ``group_id`` and ``pixel_id`` columns.
    pixel_id : int
        Camera pixel ID.
    camera_pixels : set of int
        Pixel IDs present in the camera.

    Returns
    -------
    list of int
        Sorted trigger-group IDs.
    """
    if pixel_id not in camera_pixels:
        raise ValueError(f"Pixel {pixel_id} is not in the camera; choose a model pixel ID")
    patch_ids = sorted(set(map(int, members[GROUP_ID][members[MEMBER_PIXEL_ID] == pixel_id])))
    if not patch_ids:
        raise ValueError(f"Pixel {pixel_id} belongs to no trigger patches in this model")
    logger.info("Trigger patches containing pixel %s: %s", pixel_id, patch_ids)
    return patch_ids


def read_selected_trigger_patches(definitions, members, patch_ids, camera_pixels):
    """
    Resolve the selected trigger groups into their pixel sets, with validation.

    Parameters
    ----------
    definitions : astropy.table.Table
        Trigger group definitions with a ``group_id`` column.
    members : astropy.table.Table
        Trigger memberships with ``group_id`` and ``pixel_id`` columns.
    patch_ids : list of int
        Selected trigger-group IDs.
    camera_pixels : set of int
        Pixel IDs present in the camera.

    Returns
    -------
    dict
        Mapping of group ID to its set of pixel IDs.
    """
    if not patch_ids:
        raise ValueError("Select at least one trigger patch with --trigger_patches")
    if len(set(patch_ids)) != len(patch_ids):
        raise ValueError("Trigger patch IDs must be distinct")
    maximum = max(1, _maximum_patch_membership(members))
    if len(patch_ids) > maximum:
        raise ValueError(
            f"Select at most {maximum} trigger patches: this model's maximum patches per pixel"
        )
    available = set(definitions[GROUP_ID])
    groups = {}
    for group_id in patch_ids:
        if group_id not in available:
            raise ValueError(f"Unknown trigger patch {group_id}; choose a group_id from the model")
        pixels = set(map(int, members[MEMBER_PIXEL_ID][members[GROUP_ID] == group_id]))
        if not pixels:
            raise ValueError(f"{TRIGGER_PATCH} {group_id} has no members in the model")
        missing = pixels - camera_pixels
        if missing:
            raise ValueError(
                f"{TRIGGER_PATCH} {group_id} references pixels absent from the camera: "
                f"{sorted(missing)}"
            )
        groups[group_id] = pixels
    return groups


def _maximum_patch_membership(members):
    """Count distinct groups per pixel, including each group only once per pixel."""
    memberships = {}
    for row in members:
        memberships.setdefault(int(row[MEMBER_PIXEL_ID]), set()).add(int(row[GROUP_ID]))
    return max(map(len, memberships.values()), default=0)


def plot_selected_trigger_patches(camera, groups, sky_coordinates):
    """
    Draw selected trigger groups on a camera overview and a labelled close-up of their union.

    Parameters
    ----------
    camera : Camera
        Camera object.
    groups : dict
        Mapping of group ID to its set of pixel IDs (see :func:`read_selected_trigger_patches`).
    sky_coordinates : bool
        Plot in sky coordinates (no vertical flip for single-mirror telescopes).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    plot_camera = copy.copy(camera)
    plot_camera.pixels = dict(camera.pixels)
    if not is_two_mirror_telescope(camera.telescope_name) and not sky_coordinates:
        plot_camera.pixels["y"] = -np.asarray(camera.pixels["y"])
    memberships, selected_positions = _selected_patch_positions(plot_camera, groups)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    _draw_patch_memberships(axes, plot_camera, groups, memberships)
    for ax in axes:
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Camera x [cm]")
        ax.set_ylabel("Camera y [cm]")
    axes[0].set_title(f"{camera.telescope_name}: full camera")
    axes[1].set_title("Selected trigger patches (pixel IDs)")
    for pixel_id, x, y in selected_positions:
        axes[1].text(x, y, str(pixel_id), ha=_CENTER, va=_CENTER, fontsize=7)
    coordinates = np.asarray([(x, y) for _, x, y in selected_positions])
    margin = 2 * camera.pixels["pixel_diameter"]
    axes[1].set_xlim(coordinates[:, 0].min() - margin, coordinates[:, 0].max() + margin)
    axes[1].set_ylim(coordinates[:, 1].min() - margin, coordinates[:, 1].max() + margin)
    return fig


def _selected_patch_positions(camera, groups):
    """Partition camera pixels by their exact membership in the selected groups."""
    memberships = {}
    selected_positions = []
    for pixel_id, x, y in zip(camera.pixels[PIXEL_IDS], camera.pixels["x"], camera.pixels["y"]):
        membership = tuple(group_id for group_id, pixels in groups.items() if pixel_id in pixels)
        memberships.setdefault(membership, []).append((pixel_id, x, y))
        if membership:
            selected_positions.append((pixel_id, x, y))
    return memberships, selected_positions


def _draw_patch_memberships(axes, camera, groups, memberships):
    """Show intersections as stripes retaining each contributing patch's colour."""
    palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    if len(groups) > len(palette):
        palette = plt.get_cmap("turbo")(np.linspace(0, 1, len(groups)))
    colours = dict(zip(groups, palette))
    handles = []
    for membership, positions in memberships.items():
        member_colours = ["0.92"]
        if membership:
            member_colours = [colours[group_id] for group_id in membership]
            label = "Patch " + str(membership[0])
            if len(membership) > 1:
                label = "Patches " + " + ".join(map(str, membership))
            handles.append(tuple(Patch(facecolor=colour) for colour in member_colours))
            handles[-1][0].set_label(label)
        for ax in axes:
            patches, facecolours = _striped_pixel_shapes(camera, positions, member_colours)
            ax.add_collection(PatchCollection(patches, facecolor=facecolours, edgecolor="none"))
    outlines = [pixel_shape(camera, x, y) for x, y in zip(camera.pixels["x"], camera.pixels["y"])]
    for ax in axes:
        ax.add_collection(
            PatchCollection(outlines, facecolor="none", edgecolor="0.45", linewidth=0.5)
        )
    axes[0].legend(
        handles=handles,
        labels=[handle[0].get_label() for handle in handles],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        loc="upper left",
        fontsize=9,
    )


def _striped_pixel_shapes(camera, positions, colours):
    """Clip repeating vertical colour bands to each pixel's actual outline."""
    shapes, facecolours = [], []
    for _, x, y in positions:
        pixel = pixel_shape(camera, x, y)
        path = pixel.get_path().transformed(pixel.get_transform())
        if len(colours) == 1:
            shapes.append(PathPatch(path))
            facecolours.append(colours[0])
            continue
        bounds = path.get_extents()
        edges = np.linspace(bounds.xmin, bounds.xmax, 3 * len(colours) + 1)
        for index, (left, right) in enumerate(pairwise(edges)):
            stripe = path.clip_to_bbox(Bbox.from_extents(left, bounds.ymin, right, bounds.ymax))
            shapes.append(PathPatch(stripe))
            facecolours.append(colours[index % len(colours)])
    return shapes, facecolours


def parse_pixel_ids_to_print(print_pixels_id_arg, camera):
    """
    Parse the print_pixels_id argument into an integer pixel count.

    Parameters
    ----------
    print_pixels_id_arg : str or int
        Integer, 0 (suppress) or 'All'.
    camera : Camera
        Camera used to determine the total pixel count for 'All'.

    Returns
    -------
    int
        Number of pixel IDs to print; -1 when printing should be suppressed.

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
