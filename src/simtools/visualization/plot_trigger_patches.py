"""Plot camera trigger groups and their pre-summed pixel patches."""

from collections import defaultdict

from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from simtools.visualization.matplotlib_backend import pyplot as plt


def plot_trigger_patches(configuration, telescope_model_name=None, max_groups=None):
    """Plot pixel positions, trigger members, and required master pixels.

    Bracketed trigger members are drawn with a heavy outline around their
    convex hull. A required master pixel is marked with ``+``. The plot uses
    the same normalized camera component records as the sim_telarray writer.

    Parameters
    ----------
    configuration : dict
        Mapping with ``pixels``, ``triggers``, and ``trigger_members`` entries.
    telescope_model_name : str, optional
        Telescope name used in the figure title.
    max_groups : int, optional
        Limit the number of trigger groups shown, useful for large cameras.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    pixels = configuration.get("pixels", [])
    triggers = configuration.get("triggers", configuration.get("trigger_groups", []))
    members = configuration.get("trigger_members", [])
    pixel_map = {pixel["pixel_id"]: pixel for pixel in pixels}
    members_by_group = _members_by_group(members)
    shown_triggers = triggers if max_groups is None else triggers[:max_groups]

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_pixels(ax, pixels)
    colors = plt.get_cmap("tab20")
    has_patches, has_masters = _draw_triggers(
        ax, shown_triggers, members_by_group, pixel_map, colors
    )
    ax.legend(handles=_legend_handles(has_patches, has_masters))
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal scale [cm]")
    ax.set_ylabel("Vertical scale [cm]")
    title = (
        "Trigger patches"
        if telescope_model_name is None
        else f"{telescope_model_name} trigger patches"
    )
    ax.set_title(title)
    return fig


def _members_by_group(members):
    """Group normalized trigger-member rows by trigger group."""
    grouped = defaultdict(list)
    for member in members:
        grouped[member["group_id"]].append(member)
    return grouped


def _draw_pixels(ax, pixels):
    """Draw the camera pixel positions."""
    ax.scatter(
        [pixel["x_cm"] for pixel in pixels],
        [pixel["y_cm"] for pixel in pixels],
        s=5,
        color="0.75",
        label="pixels",
    )


def _draw_triggers(ax, triggers, members_by_group, pixel_map, colors):
    """Draw trigger members and return whether patches or masters were found."""
    has_patches = has_masters = False
    for index, trigger in enumerate(triggers):
        patch, master = _draw_trigger(
            ax,
            members_by_group.get(trigger["group_id"], []),
            pixel_map,
            colors(index % 20),
        )
        has_patches |= patch
        has_masters |= master
    return has_patches, has_masters


def _draw_trigger(ax, members, pixel_map, color):
    """Draw all members in one trigger group."""
    grouped = defaultdict(list)
    for member in members:
        grouped[member["member_order"]].append(member)
    has_patches = has_masters = False
    for rows in grouped.values():
        rows.sort(key=lambda row: row["pixel_order"])
        has_patches |= _draw_member(ax, rows, pixel_map, color)
        has_masters |= _draw_required_masters(ax, rows, pixel_map, color)
    return has_patches, has_masters


def _draw_member(ax, rows, pixel_map, color):
    """Draw one pre-summed trigger member."""
    points = [
        (pixel_map[row["pixel_id"]]["x_cm"], pixel_map[row["pixel_id"]]["y_cm"]) for row in rows
    ]
    if len(points) <= 1:
        return False
    hull = _convex_hull(points)
    if len(hull) > 2:
        ax.add_patch(Polygon(hull, closed=True, fill=False, linewidth=2.5, color=color))
    else:
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color=color,
            linewidth=2.5,
        )
    return True


def _draw_required_masters(ax, rows, pixel_map, color):
    """Draw required master-pixel markers."""
    has_masters = False
    for row in rows:
        if row.get("required"):
            pixel = pixel_map[row["pixel_id"]]
            ax.plot(pixel["x_cm"], pixel["y_cm"], marker="+", markersize=10, color=color)
            has_masters = True
    return has_masters


def _legend_handles(has_patches, has_masters):
    """Build legend handles for the elements present in the plot."""
    handles = [Line2D([], [], color="0.75", marker="o", linestyle="None", label="pixels")]
    if has_patches:
        handles.append(Line2D([], [], color="black", linewidth=2.5, label="pre-summed patch"))
    if has_masters:
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                marker="+",
                linestyle="None",
                markersize=10,
                label="required master",
            )
        )
    return handles


def _convex_hull(points):
    """Return the convex hull of two-dimensional points."""
    unique = sorted(set(points))
    if len(unique) <= 1:
        return unique

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]
