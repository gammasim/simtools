"""Plot the optical structure of a telescope: mirrors, camera housing and telescope structure.

The geometry is read from the model parameters with the nominal positions, that is
without the random alignment errors of a particular instrument. Only simtools is
needed. With ``shadowing``, the structure is coloured by the light each element stops
on its own; that needs the optional obdeect package.
"""

import numpy as np

from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import save_figure

mpatches = lazy_module("matplotlib.patches")

MIRROR_FACE, MIRROR_EDGE, HOUSING, NEUTRAL = "#e6eef6", "#a9bdd0", "#52514e", "#c3c2b7"


def _convex_hull(points):
    """Convex hull (closed ring) of 2-D points, monotone chain; a polyline if degenerate."""
    unique = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(unique) < 3:
        return np.array(unique)

    def half(sequence):
        out = []
        for q in sequence:
            while len(out) >= 2 and _cross(out[-2], out[-1], q) <= 0:
                out.pop()
            out.append(q)
        return out

    ring = half(unique)[:-1] + half(reversed(unique))[:-1]
    return np.array([*ring, ring[0]])


def _cross(o, a, b):
    """Cross product of the vectors o->a and o->b."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _draw(ax, element, i, j, face, edge, *, lw, zorder, linestyle="-"):
    """Draw an element projected on axes (i, j): a line for wires and cylinders, else a polygon."""
    if element.side_only and (i, j) != (0, 2):
        return
    projected = element.points[:, [i, j]]
    if element.kind == "wire" or element.is_cylinder:
        x, y = projected[:, 0], projected[:, 1]
        ax.plot(x, y, color=face, lw=lw, solid_capstyle="round", zorder=zorder, ls=linestyle)
        return
    ring = _convex_hull(projected)
    if len(ring) < 3:
        ax.plot(ring[:, 0], ring[:, 1], color=edge, lw=lw, zorder=zorder)
    else:
        ax.fill(ring[:, 0], ring[:, 1], fc=face, ec=edge, lw=lw, zorder=zorder, ls=linestyle)


def _structure_style(element, shares, top, cmap):
    """Colour, z-order and line style of a structure element."""
    if shares is None or element.name not in shares:
        return NEUTRAL, 1.0, "-"
    share, crossed = shares[element.name]
    colour = cmap(0.25 + 0.75 * min(share / top, 1.0)) if share > 0 and top > 0 else NEUTRAL
    return colour, 2.0 + share, ":" if crossed == 0 else "-"


def _draw_mirrors(ax, mirrors, i, j):
    """Mirror facets or dual-mirror surfaces, holes in white, profiles as lines."""
    for element in mirrors:
        if element.group == "mirror_hole":
            _draw(ax, element, i, j, "white", "white", lw=0.0, zorder=1.5)
        elif element.kind == "wire":
            _draw(ax, element, i, j, MIRROR_EDGE, MIRROR_EDGE, lw=1.2, zorder=1.2)
        else:
            _draw(ax, element, i, j, MIRROR_FACE, MIRROR_EDGE, lw=0.4, zorder=1.0)


def _draw_view(ax, geometry, shares, top, cmap, view):
    """Draw the mirrors, camera housing and structure in one view and label the axes."""
    (i, j), title, ylabel = view
    _draw_mirrors(ax, geometry.mirrors, i, j)
    for element in geometry.camera:
        _draw(ax, element, i, j, HOUSING, HOUSING, lw=1.2, zorder=3.0)
    for element in geometry.structure:
        colour, zorder, linestyle = _structure_style(element, shares, top, cmap)
        _draw(ax, element, i, j, colour, colour, lw=1.6, zorder=zorder, linestyle=linestyle)
    ax.set(aspect="equal", xlabel="x [m]", ylabel=ylabel, title=title)
    ax.grid(color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def _add_legend(ax, geometry):
    """Legend for the mirrors and the camera housing, if the geometry has them."""
    handles = []
    if geometry.mirrors:
        handles.append(mpatches.Patch(fc=MIRROR_FACE, ec=MIRROR_EDGE, label="mirrors"))
    if geometry.camera:
        handles.append(mpatches.Patch(fc=HOUSING, ec=HOUSING, label="camera housing"))
    if handles:
        ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)


def _add_orientation_arrows(ax):
    """Show telescope top (-x) and right (+y) in the sky view."""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    length = min(x_max - x_min, y_max - y_min) * 0.12
    x_start = x_max - 1.6 * length
    y_start = y_max - 1.6 * length
    dx, dy = 0.04 * (x_max - x_min), 0.04 * (y_max - y_min)
    arrow = {"arrowstyle": "->", "color": "0.25", "lw": 1.0}
    text = {"fontsize": 8, "color": "0.25"}
    for xy in ((x_start - length, y_start), (x_start, y_start + length)):
        ax.annotate("", xy=xy, xytext=(x_start, y_start), arrowprops=arrow)
    ax.text(x_start - length, y_start - dy, "top", ha="center", va="top", **text)
    ax.text(x_start + dx, y_start + length, "right", ha="left", va="center", **text)


def plot_telescope_structure(geometry, output_file, shares=None):
    """Plot the telescope seen from the sky (x, y) and from the side (x, z).

    Parameters
    ----------
    geometry: TelescopeGeometry
        Elements to draw (see ``telescope_geometry``).
    output_file: Path or str
        Output file name without suffix (see ``save_figure``).
    shares: dict, optional
        Element name -> (share of light stopped alone in per cent, photons crossing it),
        as from ``shadowing_shares``. Without it, all structure elements are grey.
    """
    cmap = plt.get_cmap("Blues")
    top = max((share for share, _ in shares.values()), default=0.0) if shares else 0.0
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    views = (
        ((0, 1), "seen from the sky (x, y)", "y [m]"),
        ((0, 2), "side view (x, z)", "z [m]"),
    )
    for ax, view in zip(axes, views, strict=True):
        _draw_view(ax, geometry, shares, top, cmap, view)
    _add_orientation_arrows(axes[0])
    _add_legend(axes[0], geometry)
    if shares is not None and top > 0:
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, top))
        colourbar = figure.colorbar(mappable, ax=axes, shrink=0.7, pad=0.02)
        colourbar.set_label("light stopped by this element alone [% of the unobscured light]")
    figure.suptitle(f"{geometry.name}: telescope structure", color="0.3")
    save_figure(figure, output_file, log_title="telescope structure")
    plt.close(figure)
