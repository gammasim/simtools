#!/usr/bin/python3
"""Plot array elements for a layout."""

import logging
from collections import Counter
from typing import NamedTuple

import astropy.units as u
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from astropy.table import Column
from matplotlib.collections import PatchCollection

from simtools.utils import geometry as transf
from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h
from simtools.visualization import visualize

logging.getLogger("adjustText").setLevel(logging.CRITICAL)


class PlotBounds(NamedTuple):
    """Axis-aligned bounds for the layout in meters.

    Attributes
    ----------
    x_lim : tuple[float, float]
        Min/max for x (meters).
    y_lim : tuple[float, float]
        Min/max for y (meters).
    """

    x_lim: tuple[float, float]
    y_lim: tuple[float, float]


def plot_array_layouts(args_dict, output_path, layouts, background_layout=None):
    """
    Plot multiple array layouts.

    Parameters
    ----------
    args_dict : dict
        Application arguments.
    output_path : Path
        Output path for figures.
    layouts : dict
        Dictionary of layout name to telescope table.
    background_layout : Table or None
        Optional background telescope table.

    Returns
    -------
    figs : dict
        Dictionary of layout name to matplotlib figure object.

    """
    mpl.use("Agg")
    for layout in layouts:
        fig_out = plot_array_layout(
            telescopes=layout["array_elements"],
            show_tel_label=args_dict["show_labels"],
            axes_range=args_dict["axes_range"],
            marker_scaling=args_dict["marker_scaling"],
            background_telescopes=background_layout,
            grayed_out_elements=args_dict["grayed_out_array_elements"],
            highlighted_elements=args_dict["highlighted_array_elements"],
            legend_location=args_dict["legend_location"],
            bounds_mode=args_dict["bounds"],
            padding=args_dict["padding"],
            x_lim=tuple(args_dict["x_lim"]) if args_dict["x_lim"] else None,
            y_lim=tuple(args_dict["y_lim"]) if args_dict["y_lim"] else None,
        )
        site_string = ""
        if layout.get("site") is not None:
            site_string = f"_{layout['site']}"
        elif args_dict["site"] is not None:
            site_string = f"_{args_dict['site']}"
        coordinate_system_string = (
            f"_{args_dict['coordinate_system']}"
            if args_dict["coordinate_system"] not in layout["name"]
            else ""
        )
        plot_file_name = args_dict["figure_name"] or (
            f"array_layout_{layout['name']}{site_string}{coordinate_system_string}"
        )

        visualize.save_figure(fig_out, output_path / plot_file_name, dpi=400)
        plt.close()


def plot_array_layout(
    telescopes,
    show_tel_label=False,
    axes_range=None,
    marker_scaling=1.0,
    background_telescopes=None,
    grayed_out_elements=None,
    highlighted_elements=None,
    legend_location="best",
    bounds_mode="exact",
    padding=0.1,
    x_lim=None,
    y_lim=None,
):
    """
    Plot telescope array layout.

    Parameters
    ----------
    telescopes : Table
        Telescope data table.
    show_tel_label : bool
        Show telescope labels (default False).
    axes_range : float or None
        Axis range, auto if None.
    marker_scaling : float
        Marker size scale factor.
    background_telescopes : Table or None
        Optional background telescope table.
    grayed_out_elements : list or None
        List of telescope names to plot as gray circles.
    highlighted_elements : list or None
        List of telescope names to plot with red circles around them.
    legend_location : str
        Location of the legend (default "best").

    Returns
    -------
    fig : Figure
        Matplotlib figure object.

    Other Parameters
    ----------------
    bounds_mode : {"symmetric", "exact"}
        Controls axis limits calculation. "symmetric" uses +-R where R is the padded
        maximum extent (default), while "exact" uses individual x/y min/max bounds.
    padding : float
        Fractional padding applied around computed extents (used for both modes).
    x_lim, y_lim : tuple(float, float), optional
        Explicit axis limits in meters. If provided, these override axes_range and bounds_mode
        for the respective axis. If only one is provided, the other axis is derived per mode.
    """
    fig, ax = plt.subplots(1)

    # If explicit limits are provided (one or both), filter patches accordingly
    filter_x = x_lim
    filter_y = y_lim

    patches, plot_range, highlighted_patches, bounds, text_objects = get_patches(
        ax,
        telescopes,
        show_tel_label,
        axes_range,
        marker_scaling,
        grayed_out_elements,
        highlighted_elements,
        filter_x_lim=filter_x,
        filter_y_lim=filter_y,
    )

    plot_range, bounds = _get_patches_for_background_telescopes(
        ax,
        background_telescopes,
        axes_range,
        marker_scaling,
        bounds_mode,
        plot_range,
        bounds,
        filter_x_lim=filter_x,
        filter_y_lim=filter_y,
    )

    if legend_location != "no_legend":
        update_legend(ax, telescopes, grayed_out_elements, legend_location)

    x_lim, y_lim = _get_axis_limits(
        axes_range, bounds_mode, padding, plot_range, bounds, x_lim, y_lim
    )

    finalize_plot(ax, patches, "Easting [m]", "Northing [m]", x_lim, y_lim, highlighted_patches)

    if text_objects:
        adjust_text(
            text_objects,
            ax=ax,
            arrowprops={"arrowstyle": "->", "color": "grey", "alpha": 0.8, "lw": 0.8, "ls": "--"},
            expand=(2.0, 2.0),
            prevent_crossings=True,
            min_arrow_len=8,
            ensure_inside_axes=True,
        )

    return fig


def _get_axis_limits(
    axes_range,
    bounds_mode,
    padding,
    plot_range,
    bounds,
    x_lim_override=None,
    y_lim_override=None,
):
    """Get axis limits based on mode and padding."""

    def _derive_axis(axis: str) -> tuple[float, float]:
        if bounds_mode == "exact":
            if axis == "x":
                span = bounds.x_lim[1] - bounds.x_lim[0]
                pad = padding * span
                return (bounds.x_lim[0] - pad, bounds.x_lim[1] + pad)
            span = bounds.y_lim[1] - bounds.y_lim[0]
            pad = padding * span
            return (bounds.y_lim[0] - pad, bounds.y_lim[1] + pad)
        # symmetric
        sym = plot_range
        padf = max(0.0, min(1.0, float(padding))) if padding is not None else 0.0
        sym *= 1.0 + padf
        return (-sym, sym)

    # Highest priority: explicit overrides (per axis)
    if x_lim_override is not None or y_lim_override is not None:
        x_lim = x_lim_override if x_lim_override is not None else _derive_axis("x")
        y_lim = y_lim_override if y_lim_override is not None else _derive_axis("y")
        return x_lim, y_lim

    if axes_range is not None:
        return (-axes_range, axes_range), (-axes_range, axes_range)
    # Derive both axes using selected mode
    return _derive_axis("x"), _derive_axis("y")


def _get_patches_for_background_telescopes(
    ax,
    background_telescopes,
    axes_range,
    marker_scaling,
    bounds_mode,
    plot_range,
    bounds,
    filter_x_lim=None,
    filter_y_lim=None,
):
    """Get background telescope patches and update plot range/bounds."""
    if background_telescopes is None:
        return plot_range, bounds

    bg_patches, bg_range, _, bg_bounds, _ = get_patches(
        ax,
        background_telescopes,
        False,
        axes_range,
        marker_scaling,
        None,
        None,
        filter_x_lim=filter_x_lim,
        filter_y_lim=filter_y_lim,
    )
    ax.add_collection(PatchCollection(bg_patches, match_original=True, alpha=0.1))
    if axes_range is None:
        if bounds_mode == "symmetric":
            plot_range = max(plot_range, bg_range)
        else:
            bounds = PlotBounds(
                x_lim=(
                    min(bounds.x_lim[0], bg_bounds.x_lim[0]),
                    max(bounds.x_lim[1], bg_bounds.x_lim[1]),
                ),
                y_lim=(
                    min(bounds.y_lim[0], bg_bounds.y_lim[0]),
                    max(bounds.y_lim[1], bg_bounds.y_lim[1]),
                ),
            )
    return plot_range, bounds


def _apply_limits_filter(telescopes, pos_x, pos_y, filter_x_lim, filter_y_lim):
    """Filter telescope table and positions by optional axis limits."""
    if filter_x_lim is None and filter_y_lim is None:
        return telescopes, pos_x, pos_y

    px = np.asarray(pos_x.to_value(u.m))
    py = np.asarray(pos_y.to_value(u.m))
    mask = np.ones(px.shape, dtype=bool)
    if filter_x_lim is not None:
        mask &= (px >= float(filter_x_lim[0])) & (px <= float(filter_x_lim[1]))
    if filter_y_lim is not None:
        mask &= (py >= float(filter_y_lim[0])) & (py <= float(filter_y_lim[1]))

    if mask.size and mask.any():
        return telescopes[mask], pos_x[mask], pos_y[mask]

    # No telescopes within limits
    return telescopes[:0], pos_x[:0], pos_y[:0]


def get_patches(
    ax,
    telescopes,
    show_tel_label,
    axes_range,
    marker_scaling,
    grayed_out_elements=None,
    highlighted_elements=None,
    filter_x_lim=None,
    filter_y_lim=None,
):
    """
    Get plot patches and axis range.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    telescopes : Table
        Telescope data table.
    show_tel_label : bool
        Show telescope labels.
    axes_range : float or None
        Axis range, auto if None.
    marker_scaling : float
        Marker size scale factor.
    grayed_out_elements : list or None
        List of telescope names to plot as gray circles.
    highlighted_elements : list or None
        List of telescope names to plot with red circles around them.

    Returns
    -------
    patches : list
        List of telescope patches.
    axes_range : float
        Calculated or input symmetric axis range (meters).
    highlighted_patches : list
        List of highlighted telescope patches.
    bounds : PlotBounds
        Min/max for x and y in meters.
    text_objects : list
        List of text objects for labels.
    """
    pos_x, pos_y = get_positions(telescopes)
    tel_table, pos_x, pos_y = _apply_limits_filter(
        telescopes, pos_x, pos_y, filter_x_lim, filter_y_lim
    )

    tel_table["pos_x_rotated"] = Column(pos_x)
    tel_table["pos_y_rotated"] = Column(pos_y)

    patches, radii, highlighted_patches, text_objects = create_patches(
        tel_table, marker_scaling, show_tel_label, ax, grayed_out_elements, highlighted_elements
    )

    if len(radii) == 0:
        r = 0.0
    else:
        radii_q = u.Quantity(radii)
        r = float(np.nanmax(radii_q).to_value(u.m))

    if len(pos_x) == 0:
        bounds = PlotBounds(x_lim=(0.0, 0.0), y_lim=(0.0, 0.0))
        if axes_range:
            return patches, axes_range, highlighted_patches, bounds, text_objects
        return patches, 0.0, highlighted_patches, bounds, text_objects

    x_min = float(np.nanmin(pos_x).to_value(u.m)) - r
    x_max = float(np.nanmax(pos_x).to_value(u.m)) + r
    y_min = float(np.nanmin(pos_y).to_value(u.m)) - r
    y_max = float(np.nanmax(pos_y).to_value(u.m)) + r
    bounds = PlotBounds(x_lim=(x_min, x_max), y_lim=(y_min, y_max))

    if axes_range:
        return patches, axes_range, highlighted_patches, bounds, text_objects

    max_x = max(abs(x_min), abs(x_max))
    max_y = max(abs(y_min), abs(y_max))
    updated_axes_range = max(max_x, max_y) * 1.1

    return patches, updated_axes_range, highlighted_patches, bounds, text_objects


@u.quantity_input(x=u.m, y=u.m, radius=u.m)
def get_telescope_patch(tel_type, x, y, radius, is_grayed_out=False):
    """
    Create patch for a telescope.

    Parameters
    ----------
    tel_type: str
        Telescope type.
    x : Quantity
        X position.
    y : Quantity
        Y position.
    radius : Quantity
        Telescope radius.
    is_grayed_out : bool
        Whether to plot telescope in gray.

    Returns
    -------
    patch : Patch
        Circle or rectangle patch.
    """
    config = leg_h.get_telescope_config(tel_type)
    x, y, r = x.to(u.m), y.to(u.m), radius.to(u.m)

    color = "gray" if is_grayed_out else config["color"]
    fill_flag = True if is_grayed_out else bool(config.get("filled", True))

    if config.get("shape", "circle") == "square":
        return mpatches.Rectangle(
            ((x - r / 2).value, (y - r / 2).value),
            width=r.value,
            height=r.value,
            fill=fill_flag,
            color=color,
        )
    if config.get("shape") == "hexagon":
        return mpatches.RegularPolygon(
            (x.value, y.value),
            numVertices=6,
            radius=r.value * np.sqrt(3) / 2,
            orientation=np.pi / 6,
            fill=fill_flag,
            color=color,
        )

    return mpatches.Circle(
        (x.value, y.value),
        radius=r.value,
        fill=fill_flag,
        alpha=0.5 if is_grayed_out else 1.0,
        color=color,
    )


def get_positions(telescopes):
    """
    Get X/Y positions depending on coordinate system.

    For ground coordinates, rotates the positions by 90 degrees.

    Returns
    -------
    x_rot, y_rot : Quantity
        Position coordinates.
    """
    if "position_x" in telescopes.colnames:
        x, y = telescopes["position_x"], telescopes["position_y"]
        locale_rotate_angle = 90 * u.deg
    elif "utm_east" in telescopes.colnames:
        x, y = telescopes["utm_east"], telescopes["utm_north"]
        locale_rotate_angle = 0 * u.deg
    else:
        raise ValueError("Missing required position columns.")

    return transf.rotate(x, y, locale_rotate_angle) if locale_rotate_angle != 0 else (x, y)


def create_patches(
    telescopes, scale, show_label, ax, grayed_out_elements=None, highlighted_elements=None
):
    """
    Create telescope patches and labels.

    Parameters
    ----------
    telescopes : Table
        Telescope data table.
    scale : float
        Marker size scale factor.
    show_label : bool
        Show telescope labels.
    ax : Axes
        Matplotlib axes object.
    grayed_out_elements : list or None
        List of telescope names to plot as gray circles.
    highlighted_elements : list or None
        List of telescope names to plot with red circles around them.

    Returns
    -------
    patches : list
        Shape patches.
    radii : list
        Telescope radii.
    highlighted_patches : list
        List of highlighted telescope patches.
    text_objects : list
        List of text objects for labels.
    """
    patches, radii, highlighted_patches, text_objects = [], [], [], []
    fontsize, scale_factor = (4, 2) if len(telescopes) > 30 else (8, 1)

    grayed_out_set = set(grayed_out_elements) if grayed_out_elements else set()
    highlighted_set = set(highlighted_elements) if highlighted_elements else set()

    for tel in telescopes:
        name = get_telescope_name(tel)
        radius = get_sphere_radius(tel)
        radii.append(radius)
        try:
            tel_type = names.get_array_element_type_from_name(name)
        except ValueError:
            tel_type = None

        is_grayed_out = name in grayed_out_set
        is_highlighted = name in highlighted_set

        patches.append(
            get_telescope_patch(
                tel_type,
                tel["pos_x_rotated"],
                tel["pos_y_rotated"],
                scale_factor * radius * scale,
                is_grayed_out=is_grayed_out,
            )
        )

        if is_highlighted:
            highlight_patch = mpatches.Circle(
                (tel["pos_x_rotated"].value, tel["pos_y_rotated"].value),
                radius=(scale_factor * radius * scale * 4).value,
                fill=False,
                color="red",
                linewidth=1,
            )
            highlighted_patches.append(highlight_patch)

        if show_label:
            text_objects.append(
                ax.text(
                    tel["pos_x_rotated"].value,
                    tel["pos_y_rotated"].value + scale_factor * radius.value,
                    name,
                    ha="center",
                    va="center",
                    fontsize=fontsize * 0.8,
                )
            )

    return patches, radii, highlighted_patches, text_objects


def get_telescope_name(tel):
    """
    Get telescope name.

    Returns
    -------
    name : str
        Telescope name or fallback identifier.
    """
    if "telescope_name" in tel.colnames:
        return tel["telescope_name"]
    if "asset_code" in tel.colnames and "sequence_number" in tel.colnames:
        return f"{tel['asset_code']}-{tel['sequence_number']}"
    return f"tel_{tel.index}"


def get_sphere_radius(tel):
    """
    Get telescope sphere radius.

    Returns
    -------
    radius : Quantity
        Radius with units.
    """
    return tel["sphere_radius"] if "sphere_radius" in tel.colnames else 10.0 * u.m


def update_legend(ax, telescopes, grayed_out_elements=None, legend_location="best"):
    """Add legend for telescope types and counts."""
    grayed_out_set = set(grayed_out_elements) if grayed_out_elements else set()

    types = []
    for tel in telescopes:
        tel_name = get_telescope_name(tel)
        if tel_name not in grayed_out_set:
            types.append(names.get_array_element_type_from_name(tel_name))

    counts = Counter(types)

    objs, labels = [], []
    handler_map = {}

    for telescope_type in names.get_list_of_array_element_types():
        if counts[telescope_type]:
            objs.append(telescope_type)
            labels.append(f"{telescope_type} ({counts[telescope_type]})")

            class BaseLegendHandlerWrapper:  # pylint: disable=R0903
                """Wrapper for BaseLegendHandler to use in legend."""

                def __init__(self, tel_type):
                    self.tel_type = tel_type

                def legend_artist(self, legend, orig_handle, fontsize, handlebox):
                    handler = leg_h.BaseLegendHandler(self.tel_type)
                    return handler.legend_artist(legend, orig_handle, fontsize, handlebox)

            handler_map[telescope_type] = BaseLegendHandlerWrapper(telescope_type)

    ax.legend(objs, labels, handler_map=handler_map, prop={"size": 11}, loc=legend_location)


def finalize_plot(
    ax,
    patches,
    x_title,
    y_title,
    x_lim=None,
    y_lim=None,
    highlighted_patches=None,
):
    """Finalize plot appearance and limits."""
    ax.add_collection(PatchCollection(patches, match_original=True))

    if highlighted_patches:
        ax.add_collection(PatchCollection(highlighted_patches, match_original=True))

    ax.set(xlabel=x_title, ylabel=y_title)
    ax.tick_params(labelsize=8)
    ax.axis("square")
    if x_lim is not None and y_lim is not None:
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
    plt.tight_layout()
