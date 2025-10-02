#!/usr/bin/python3
"""Plot array elements for a layout."""

from collections import Counter

import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.table import Column
from matplotlib.collections import PatchCollection

from simtools.utils import geometry as transf
from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h


def plot_array_layout(
    telescopes,
    show_tel_label=False,
    axes_range=None,
    marker_scaling=1.0,
    background_telescopes=None,
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

    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(1)

    patches, plot_range = get_patches(ax, telescopes, show_tel_label, axes_range, marker_scaling)

    if background_telescopes is not None:
        bg_patches, bg_range = get_patches(
            ax, background_telescopes, False, axes_range, marker_scaling
        )
        ax.add_collection(PatchCollection(bg_patches, match_original=True, alpha=0.1))
        if axes_range is None:
            plot_range = max(plot_range, bg_range)

    update_legend(ax, telescopes)
    finalize_plot(ax, patches, "Easting [m]", "Northing [m]", plot_range)

    return fig


def get_patches(ax, telescopes, show_tel_label, axes_range, marker_scaling):
    """
    Get plot patches and axis range.

    Returns
    -------
    patches : list
        List of telescope patches.
    axes_range : float
        Calculated or input axis range.
    """
    pos_x, pos_y = get_positions(telescopes)
    telescopes["pos_x_rotated"] = Column(pos_x)
    telescopes["pos_y_rotated"] = Column(pos_y)

    patches, radii = create_patches(telescopes, marker_scaling, show_tel_label, ax)

    if axes_range:
        return patches, axes_range

    r = max(radii).value
    max_x = max(abs(pos_x.min().value), abs(pos_x.max().value)) + r
    max_y = max(abs(pos_y.min().value), abs(pos_y.max().value)) + r
    updated_axes_range = max(max_x, max_y) * 1.1

    return patches, updated_axes_range


@u.quantity_input(x=u.m, y=u.m, radius=u.m)
def get_telescope_patch(name, x, y, radius):
    """
    Create patch for a telescope.

    Returns
    -------
    patch : Patch
        Circle or rectangle patch.
    """
    tel_type = names.get_array_element_type_from_name(name)
    x, y, r = x.to(u.m), y.to(u.m), radius.to(u.m)

    if tel_type == "SCTS":
        return mpatches.Rectangle(
            ((x - r / 2).value, (y - r / 2).value),
            width=r.value,
            height=r.value,
            fill=False,
            color=leg_h.get_telescope_config(tel_type)["color"],
        )

    return mpatches.Circle(
        (x.value, y.value),
        radius=r.value,
        fill=tel_type.startswith("MST"),
        color=leg_h.get_telescope_config(tel_type)["color"],
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


def create_patches(telescopes, scale, show_label, ax):
    """
    Create telescope patches and labels.

    Returns
    -------
    patches : list
        Shape patches.
    radii : list
        Telescope radii.
    """
    patches, radii = [], []
    fontsize, scale_factor = (4, 2) if len(telescopes) > 30 else (8, 1)

    for tel in telescopes:
        name = get_telescope_name(tel)
        radius = get_sphere_radius(tel)
        radii.append(radius)
        tel_type = names.get_array_element_type_from_name(name)

        patches.append(
            get_telescope_patch(
                tel_type,
                tel["pos_x_rotated"],
                tel["pos_y_rotated"],
                scale_factor * radius * scale,
            )
        )

        if show_label:
            ax.text(
                tel["pos_x_rotated"].value,
                tel["pos_y_rotated"].value + scale_factor * radius.value,
                name,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    return patches, radii


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


def update_legend(ax, telescopes):
    """Add legend for telescope types and counts."""
    names_list = [get_telescope_name(tel) for tel in telescopes]
    types = [names.get_array_element_type_from_name(n) for n in names_list]
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

    ax.legend(objs, labels, handler_map=handler_map, prop={"size": 11}, loc="best")


def finalize_plot(ax, patches, x_title, y_title, axes_range):
    """Finalize plot appearance and limits."""
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set(xlabel=x_title, ylabel=y_title)
    ax.tick_params(labelsize=8)
    ax.axis("square")
    if axes_range:
        ax.set_xlim(-axes_range, axes_range)
        ax.set_ylim(-axes_range, axes_range)
    plt.tight_layout()
