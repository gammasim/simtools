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
    grayed_out_elements=None,
    highlighted_elements=None,
    legend_location="best",
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
    """
    fig, ax = plt.subplots(1)

    patches, plot_range, highlighted_patches = get_patches(
        ax,
        telescopes,
        show_tel_label,
        axes_range,
        marker_scaling,
        grayed_out_elements,
        highlighted_elements,
    )

    if background_telescopes is not None:
        bg_patches, bg_range, _ = get_patches(
            ax, background_telescopes, False, axes_range, marker_scaling
        )
        ax.add_collection(PatchCollection(bg_patches, match_original=True, alpha=0.1))
        if axes_range is None:
            plot_range = max(plot_range, bg_range)

    update_legend(ax, telescopes, grayed_out_elements, legend_location)
    finalize_plot(ax, patches, "Easting [m]", "Northing [m]", plot_range, highlighted_patches)

    return fig


def get_patches(
    ax,
    telescopes,
    show_tel_label,
    axes_range,
    marker_scaling,
    grayed_out_elements=None,
    highlighted_elements=None,
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
        Calculated or input axis range.
    highlighted_patches : list
        List of highlighted telescope patches.
    """
    pos_x, pos_y = get_positions(telescopes)
    telescopes["pos_x_rotated"] = Column(pos_x)
    telescopes["pos_y_rotated"] = Column(pos_y)

    patches, radii, highlighted_patches = create_patches(
        telescopes, marker_scaling, show_tel_label, ax, grayed_out_elements, highlighted_elements
    )

    if axes_range:
        return patches, axes_range, highlighted_patches

    r = max(radii).value
    max_x = max(abs(pos_x.min().value), abs(pos_x.max().value)) + r
    max_y = max(abs(pos_y.min().value), abs(pos_y.max().value)) + r
    updated_axes_range = max(max_x, max_y) * 1.1

    return patches, updated_axes_range, highlighted_patches


@u.quantity_input(x=u.m, y=u.m, radius=u.m)
def get_telescope_patch(name, x, y, radius, is_grayed_out=False):
    """
    Create patch for a telescope.

    Parameters
    ----------
    name : str
        Telescope name.
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
    tel_type = names.get_array_element_type_from_name(name)
    x, y, r = x.to(u.m), y.to(u.m), radius.to(u.m)

    color = "gray" if is_grayed_out else leg_h.get_telescope_config(tel_type)["color"]

    if tel_type == "SCTS":
        return mpatches.Rectangle(
            ((x - r / 2).value, (y - r / 2).value),
            width=r.value,
            height=r.value,
            fill=is_grayed_out,  # Fill if grayed out
            color=color,
        )

    return mpatches.Circle(
        (x.value, y.value),
        radius=r.value,
        fill=is_grayed_out or tel_type.startswith("MST"),  # Always fill if grayed out
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
    """
    patches, radii, highlighted_patches = [], [], []
    fontsize, scale_factor = (4, 2) if len(telescopes) > 30 else (8, 1)

    # Convert lists to sets for faster lookup
    grayed_out_set = set(grayed_out_elements) if grayed_out_elements else set()
    highlighted_set = set(highlighted_elements) if highlighted_elements else set()

    for tel in telescopes:
        name = get_telescope_name(tel)
        radius = get_sphere_radius(tel)
        radii.append(radius)
        tel_type = names.get_array_element_type_from_name(name)

        # Check if telescope should be grayed out
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

        # Add red circle for highlighted telescopes
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
            ax.text(
                tel["pos_x_rotated"].value,
                tel["pos_y_rotated"].value + scale_factor * radius.value,
                name,
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    return patches, radii, highlighted_patches


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

    # Filter out grayed out telescopes from legend counts
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


def finalize_plot(ax, patches, x_title, y_title, axes_range, highlighted_patches=None):
    """Finalize plot appearance and limits."""
    ax.add_collection(PatchCollection(patches, match_original=True))

    # Add highlighted patches if any
    if highlighted_patches:
        ax.add_collection(PatchCollection(highlighted_patches, match_original=True))

    ax.set(xlabel=x_title, ylabel=y_title)
    ax.tick_params(labelsize=8)
    ax.axis("square")
    if axes_range:
        ax.set_xlim(-axes_range, axes_range)
        ax.set_ylim(-axes_range, axes_range)
    plt.tight_layout()
