#!/usr/bin/python3
"""Array element plotting."""

import logging
from collections import Counter

import astropy.units as u
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from astropy.table import Column
from matplotlib.collections import PatchCollection

from simtools.utils import geometry as transf
from simtools.utils import names
from simtools.visualization import legend_handlers as leg_h

__all__ = [
    "get_telescope_patch",
    "plot_array_layout",
]


_logger = logging.getLogger(__name__)


@u.quantity_input(rotate_angle=u.deg)
def plot_array_layout(
    telescopes,
    rotate_angle=0,
    show_tel_label=False,
    axes_range=None,
    marker_scaling=1.0,
    background_telescopes=None,
):
    """Plot array elements."""
    fig, ax = plt.subplots(1)

    patches, plot_range = get_patches(
        ax, telescopes, rotate_angle, show_tel_label, axes_range, marker_scaling
    )

    if background_telescopes is not None:
        patches_background, background_plot_range = get_patches(
            ax, background_telescopes, rotate_angle, False, axes_range, marker_scaling
        )
        background_collection = PatchCollection(patches_background, match_original=True, alpha=0.1)
        ax.add_collection(background_collection)

        if axes_range is None:
            plot_range = max(plot_range, background_plot_range)

    update_legend(ax, telescopes)
    finalize_plot(ax, patches, x_title="Easting [m]", y_title="Northing [m]", axes_range=plot_range)

    return fig


def get_patches(ax, telescopes, rotate_angle, show_tel_label, axes_range, marker_scaling):
    """Return patches and axes range for the telescopes."""
    pos_x_rotated, pos_y_rotated = get_rotated_positions(telescopes, rotate_angle)
    telescopes.add_column(Column(pos_x_rotated, name="pos_x_rotated"))
    telescopes.add_column(Column(pos_y_rotated, name="pos_y_rotated"))

    patches, sphere_radius = create_patches(telescopes, marker_scaling, show_tel_label, ax)

    if axes_range is None:
        r = max(sphere_radius).value
        max_x = max(abs(pos_x_rotated.min().value), abs(pos_x_rotated.max().value)) + r
        max_y = max(abs(pos_y_rotated.min().value), abs(pos_y_rotated.max().value)) + r
        axes_range = max(max_x, max_y) * 1.1

    return patches, axes_range


@u.quantity_input(x=u.m, y=u.m, radius=u.m)
def get_telescope_patch(name, x, y, radius):
    """
    Collect the patch of one telescope to be plotted by plot_array.

    Parameters
    ----------
    name: str
        Name of the telescope (type).
    x: astropy.units.Quantity
        x position of the telescope usually in meters.
    y: astropy.units.Quantity
        y position of the telescope usually in meters.
    radius: astropy.units.Quantity
        Radius of the telescope sphere usually in meters.

    Returns
    -------
    patch
        Instance of mpatches.Circle.
    """
    tel_obj = leg_h.TelescopeHandler()
    valid_name = names.get_array_element_type_from_name(name)
    fill_flag = False

    x = x.to(u.m)
    y = y.to(u.m)
    radius = radius.to(u.m)

    if valid_name.startswith("MST"):
        fill_flag = True
    if valid_name == "SCTS":
        patch = mpatches.Rectangle(
            ((x - radius / 2).value, (y - radius / 2).value),
            width=radius.value,
            height=radius.value,
            fill=False,
            color=tel_obj.colors_dict["SCTS"],
        )
    else:
        patch = mpatches.Circle(
            (x.value, y.value),
            radius=radius.value,
            fill=fill_flag,
            color=tel_obj.colors_dict[valid_name],
        )
    return patch


def get_rotated_positions(telescopes, rotate_angle):
    """
    Rotate the positions of the telescopes based on the given angle.

    Parameters
    ----------
    telescopes: astropy.table
        Table with the telescope positions.
    rotate_angle: astropy.units.Quantity
        Angle to rotate the positions, in degrees or radians.

    Returns
    -------
    pos_x_rotated: astropy.units.Quantity
        Rotated x positions of the telescopes.
    pos_y_rotated: astropy.units.Quantity
        Rotated y positions of the telescopes.
    """
    pos_x_rotated = pos_y_rotated = None
    if "position_x" in telescopes.colnames and "position_y" in telescopes.colnames:
        pos_x_rotated, pos_y_rotated = telescopes["position_x"], telescopes["position_y"]
        rotate_angle = rotate_angle + 90.0 * u.deg
    elif "utm_east" in telescopes.colnames and "utm_north" in telescopes.colnames:
        pos_x_rotated, pos_y_rotated = telescopes["utm_east"], telescopes["utm_north"]
    else:
        raise ValueError(
            "Telescopes table must contain either 'position_x'/'position_y'"
            "or 'utm_east'/'utm_north' columns"
        )
    if rotate_angle != 0:
        pos_x_rotated, pos_y_rotated = transf.rotate(pos_x_rotated, pos_y_rotated, rotate_angle)
    return pos_x_rotated, pos_y_rotated


def get_plot_params(position_length):
    if position_length > 30:
        return 4, 2
    return 8, 1


def create_patches(telescopes, marker_scaling, show_tel_label, ax):
    """Create patches for telescopes."""
    patches = []
    sphere_radius = []
    fontsize, scale = get_plot_params(len(telescopes))

    for tel_now in telescopes:
        telescope_name = get_telescope_name(tel_now)
        sphere_radius.append(get_sphere_radius(tel_now))
        i_tel_name = names.get_array_element_type_from_name(telescope_name)

        patches.append(
            get_telescope_patch(
                i_tel_name,
                tel_now["pos_x_rotated"],
                tel_now["pos_y_rotated"],
                scale * sphere_radius[-1] * marker_scaling,
            )
        )

        if show_tel_label:
            ax.text(
                tel_now["pos_x_rotated"].value,
                tel_now["pos_y_rotated"].value + scale * sphere_radius[-1].value,
                telescope_name,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=fontsize,
            )

    return patches, sphere_radius


def get_telescope_name(tel_now):
    """Get the telescope name from the table row."""
    try:
        return tel_now["telescope_name"]
    except KeyError:
        return tel_now["asset_code"] + "-" + tel_now["sequence_number"]


def update_tel_counters(tel_counters, telescope_name):
    """Update the counter for the given telescope type."""
    for tel_type in tel_counters:
        if tel_type in telescope_name:
            tel_counters[tel_type] += 1


def get_sphere_radius(tel_now):
    """Get the sphere radius of the telescope."""
    return 1.0 * u.m if "sphere_radius" not in tel_now.colnames else tel_now["sphere_radius"]


def update_legend(ax, telescopes):
    """Update the legend with the telescope counts."""
    # Count telescopes directly from the data
    telescope_names = [get_telescope_name(tel) for tel in telescopes]
    tel_types = [names.get_array_element_type_from_name(name) for name in telescope_names]
    tel_counts = Counter(tel_types)

    legend_objects = []
    legend_labels = []

    for tel_type in names.get_list_of_array_element_types():
        if tel_counts[tel_type] > 0:
            legend_objects.append(leg_h.all_telescope_objects[tel_type]())
            legend_labels.append(f"{tel_type} ({tel_counts[tel_type]})")

    legend_handler_map = {k: v() for k, v in leg_h.legend_handler_map.items()}
    ax.legend(
        legend_objects,
        legend_labels,
        handler_map=legend_handler_map,
        prop={"size": 11},
        loc="best",
    )


def finalize_plot(ax, patches, x_title, y_title, axes_range):
    """Finalize the plot by adding titles, setting limits, and adding patches."""
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_xlabel(x_title, fontsize=12, labelpad=0)
    ax.set_ylabel(y_title, fontsize=12, labelpad=0)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_axisbelow(True)
    ax.axis("square")
    if axes_range is not None:
        ax.set_xlim(-axes_range, axes_range)
        ax.set_ylim(-axes_range, axes_range)
    plt.tight_layout()
