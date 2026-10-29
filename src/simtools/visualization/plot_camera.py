"""Plot camera pixel layout."""

import copy
import logging
from itertools import pairwise

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, PathPatch
from matplotlib.transforms import Bbox

from simtools.model.model_utils import is_two_mirror_telescope
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import names
from simtools.visualization.camera_plot_utils import (
    add_pixel_legend,
    add_pixel_patch_collections,
    create_pixel_patches_by_type,
    pixel_shape,
    setup_camera_axis_properties,
)
from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

logger = logging.getLogger(__name__)
mcolors = lazy_module("matplotlib.colors")
_GROUP_ID = "group_id"
_CENTER = "center"
_PIXEL_IDS = "pix_id"
_SKY_COORDINATES = "camera_in_sky_coor"
_TRIGGER_PATCH = "Trigger patch"
_MEMBER_PIXEL_ID = "pixel_id"
_TRIGGER_MEMBERS = "camera_trigger_members"
_PIXEL_LABEL = "Pixel"


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

    plot_camera = camera
    if not is_two_mirror_telescope(camera.telescope_name) and not camera_in_sky_coor:
        plot_camera = copy.copy(camera)
        plot_camera.pixels = {
            **camera.pixels,
            "y": [-y_value for y_value in camera.pixels["y"]],
        }

    on_pixels, edge_pixels, off_pixels = create_pixel_patches_by_type(plot_camera)
    for i_pix, (x, y) in enumerate(zip(plot_camera.pixels["x"], plot_camera.pixels["y"])):
        if plot_camera.pixels[_PIXEL_IDS][i_pix] < pixels_id_to_print + 1:
            font_size = (
                4
                if "SCT" in names.get_array_element_type_from_name(plot_camera.telescope_name)
                else 2
            )
            plt.text(
                x,
                y,
                plot_camera.pixels[_PIXEL_IDS][i_pix],
                ha=_CENTER,
                va=_CENTER,
                fontsize=font_size,
            )
    add_pixel_patch_collections(ax, on_pixels, edge_pixels, off_pixels)
    setup_camera_axis_properties(ax, plot_camera, grid=True, axis_below=True, y_scale_factor=1.42)
    plt.xlabel("Horizontal scale [cm]", fontsize=18, labelpad=0)
    plt.ylabel("Vertical scale [cm]", fontsize=18, labelpad=0)
    ax.set_title(
        f"Pixels layout in {plot_camera.telescope_name} camera",
        fontsize=15,
        y=1.02,
    )
    plt.tick_params(axis="both", which="major", labelsize=15)

    _plot_axes_def(plot_camera, plt, plot_camera.pixels["rotate_angle"])

    description = {
        False: "For an observer facing the camera",
        True: "For an observer behind the camera looking through",
        None: "For an observer looking from secondary to camera",
    }[camera_in_sky_coor and not is_two_mirror_telescope(plot_camera.telescope_name)]
    ax.text(0.02, 0.02, description, transform=ax.transAxes, color="black", fontsize=12)

    fov, r_edge_avg = plot_camera.calc_fov()
    ax.text(
        0.02,
        0.96,
        rf"$f_{{\mathrm{{eff}}}} = {plot_camera.focal_length:.3f}\,\mathrm{{cm}}$",
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

    patches = []
    facecolors = []
    edgecolors = []
    linewidths = []
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

        patches.append(shape)
        facecolors.append(facecolor)
        edgecolors.append(edgecolor)
        linewidths.append(linewidth)

    ax.add_collection(
        PatchCollection(
            patches,
            facecolor=facecolors,
            edgecolor=edgecolors,
            linewidth=linewidths,
        )
    )

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
    tuple[np.ndarray | None, mcolors.Colormap | None, mcolors.Normalize | None]
        Array of RGBA colors for each pixel.
        Colormap instance.
        Normalization instance.
        Returns ``(None, None, None)`` if image is None.
    """
    if image is None:
        return None, None, None

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
            Flag to invert the horizontal direction for single-mirror camera views.
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
        ha=_CENTER,
        va=_CENTER,
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
        ha=_CENTER,
        va=_CENTER,
        size="xx-large",
        arrowprops={
            "arrowstyle": "<|-",
            "shrinkA": 0,
            "shrinkB": 0,
            "fc": kwargs["fc"],
            "ec": kwargs["ec"],
        },
    )


def plot_camera_pixel_layout_from_args(app_context, *, context=None):
    """
    Build telescope model, compute FoV, and plot the pixel layout.

    Parameters
    ----------
    app_context : object
        Application context with ``args`` and ``io_handler`` attributes.
    """
    context = context or app_context.context
    args_dict = app_context.args

    label = "plot_camera_pixel_layout"

    tel_model = TelescopeModel(
        site=args_dict["site"],
        telescope_name=args_dict["telescope"],
        model_version=args_dict["model_version"],
        label=label,
        model_reader=app_context.model_reader,
        context=context,
    )
    tel_model.export_model_files()

    logger.info(f"\nPlot camera for {tel_model.name}\n")

    focal_length = tel_model.get_telescope_effective_focal_length("cm")
    camera = tel_model.camera

    fov, r_edge_avg = camera.calc_fov()

    logger.info(f"\nEffective focal length = {focal_length:.3f} cm")
    logger.info(f"{tel_model.name} FoV = {fov:.3f} deg")
    logger.info(f"Avg. edge radius = {r_edge_avg:.3f} cm\n")

    pixel_ids_to_print = _parse_pixel_ids_to_print(args_dict["print_pixels_id"], camera)

    patch_ids = args_dict.get("trigger_patches")
    trigger_pixel = args_dict.get("trigger_pixel")
    if trigger_pixel is not None:
        if patch_ids is not None:
            raise ValueError("Use either --trigger_pixel or --trigger_patches, not both")
        patch_ids = _trigger_patches_for_pixel(
            tel_model, trigger_pixel, set(camera.pixels[_PIXEL_IDS])
        )
    if patch_ids is None:
        fig = plot_pixel_layout(camera, args_dict[_SKY_COORDINATES], pixel_ids_to_print)
    else:
        groups = _read_selected_trigger_patches(
            tel_model, patch_ids, set(camera.pixels[_PIXEL_IDS])
        )
        fig = _plot_selected_trigger_patches(camera, groups, args_dict[_SKY_COORDINATES])
    output_dir = context.paths.get_output_directory()
    plot_file_prefix = output_dir.joinpath(f"{label}_{tel_model.name}_pixel_layout")
    save_figure(fig, f"{plot_file_prefix!s}", log_title="camera")


def _trigger_patches_for_pixel(telescope, pixel_id, camera_pixels):
    """Find distinct model group IDs containing the requested camera pixel."""
    if pixel_id not in camera_pixels:
        raise ValueError(f"{_PIXEL_LABEL} {pixel_id} is not in the camera; choose a model pixel ID")
    members = telescope.get_parameter_table(_TRIGGER_MEMBERS)
    patch_ids = sorted(set(map(int, members[_GROUP_ID][members[_MEMBER_PIXEL_ID] == pixel_id])))
    if not patch_ids:
        raise ValueError(f"{_PIXEL_LABEL} {pixel_id} belongs to no trigger patches in this model")
    logger.info("Trigger patches containing pixel %s: %s", pixel_id, patch_ids)
    return patch_ids


def _read_selected_trigger_patches(telescope, patch_ids, camera_pixels):
    """Resolve groups, limiting selection to the model's maximum patches per pixel."""
    if not patch_ids:
        raise ValueError("Select at least one trigger patch with --trigger_patches")
    if len(set(patch_ids)) != len(patch_ids):
        raise ValueError("Trigger patch IDs must be distinct")
    definitions = telescope.get_parameter_table("camera_trigger_groups")
    members = telescope.get_parameter_table(_TRIGGER_MEMBERS)
    maximum = max(1, _maximum_patch_membership(members))
    if len(patch_ids) > maximum:
        raise ValueError(
            f"Select at most {maximum} trigger patches: this model's maximum patches per pixel"
        )
    available = set(definitions[_GROUP_ID])
    groups = {}
    for group_id in patch_ids:
        if group_id not in available:
            raise ValueError(f"Unknown trigger patch {group_id}; choose a group_id from the model")
        pixels = set(map(int, members[_MEMBER_PIXEL_ID][members[_GROUP_ID] == group_id]))
        if not pixels:
            raise ValueError(f"{_TRIGGER_PATCH} {group_id} has no members in the model")
        missing = pixels - camera_pixels
        if missing:
            raise ValueError(
                f"{_TRIGGER_PATCH} {group_id} references pixels absent from the camera: "
                f"{sorted(missing)}"
            )
        groups[group_id] = pixels
    return groups


def _maximum_patch_membership(members):
    """Count distinct groups per pixel, including each group only once per pixel."""
    memberships = {}
    for row in members:
        memberships.setdefault(int(row[_MEMBER_PIXEL_ID]), set()).add(int(row[_GROUP_ID]))
    return max(map(len, memberships.values()), default=0)


def _plot_selected_trigger_patches(camera, groups, sky_coordinates):
    """Draw selected groups on a camera overview and a labelled close-up of their union."""
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
    for pixel_id, x, y in zip(camera.pixels[_PIXEL_IDS], camera.pixels["x"], camera.pixels["y"]):
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
    for ax in axes:
        outlines = [
            pixel_shape(camera, x, y) for x, y in zip(camera.pixels["x"], camera.pixels["y"])
        ]
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


def _parse_pixel_ids_to_print(print_pixels_id_arg, camera):
    """
    Parse the print_pixels_id argument into an integer pixel count.

    Parameters
    ----------
    print_pixels_id_arg : str or int
        The raw argument value. Can be an integer string, 0 (suppress), or 'All'.
    camera : Camera
        Camera model instance used to determine total pixel count for 'All'.

    Returns
    -------
    int
        Number of pixel IDs to print. Returns -1 when printing should be suppressed.

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
