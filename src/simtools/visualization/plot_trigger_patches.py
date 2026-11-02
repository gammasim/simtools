"""Analyze and plot camera trigger-patch configuration."""

import textwrap
from pathlib import Path

import numpy as np
from astropy.table import Table
from matplotlib.patches import Rectangle

from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_camera import plot_pixel_layout_with_image
from simtools.visualization.visualize import pdf_report

GROUP_ID = "group_id"
GROUP_KIND = "kind"
PIXEL_IDS = "pix_id"
PATCH_COUNT = "patch_count"
PATCH_IDS = "patch_ids"
TYPICAL_DISTANCE = "typical_distance_cm"
DISTANCE_STATUS = "distance_status"
MEDIAN_PAIR_DISTANCE = "median_pair_distance_cm"
MAX_PAIR_DISTANCE = "max_pair_distance_cm"
CONNECTED_COMPONENTS = "connected_components"
REPORT_INK = "#17324d"
REPORT_MUTED = "#5c7186"


def analyze_trigger_patch_configuration(camera, trigger_groups, trigger_members):
    """
    Calculate per-pixel patch distances and per-group spatial summaries.

    Parameters
    ----------
    camera : Camera
        Camera geometry with pixel IDs and neighbor lookup.
    trigger_groups : astropy.table.Table
        Group definitions with ``group_id`` and ``kind`` columns.
    trigger_members : astropy.table.Table
        Memberships with ``group_id`` and ``pixel_id`` columns.

    Returns
    -------
    tuple of astropy.table.Table
        Per-pixel and per-group diagnostics. Distances are in camera-plane centimeters;
        undefined distances are NaN.

    Raises
    ------
    ValueError
        If the camera or trigger tables contain inconsistent identifiers.
    """
    pixel_ids, x_coordinates, y_coordinates = _camera_geometry(camera)
    pixel_indices = {int(pixel_id): index for index, pixel_id in enumerate(pixel_ids)}
    group_ids, group_kinds, group_members = _validated_group_membership(
        trigger_groups, trigger_members, pixel_indices
    )
    neighbors = camera.get_neighbor_pixels()
    if len(neighbors) != len(pixel_ids):
        raise ValueError("Camera neighbor indices must match the camera pixel count")

    pixel_distances, pixel_groups, group_rows = _group_memberships(
        group_ids,
        group_kinds,
        group_members,
        pixel_indices,
        x_coordinates,
        y_coordinates,
        neighbors,
    )
    pixel_rows = [
        _pixel_row(
            pixel_id,
            x_coordinates[index],
            y_coordinates[index],
            pixel_groups[index],
            pixel_distances[index],
        )
        for index, pixel_id in enumerate(pixel_ids)
    ]

    pixel_table = Table(
        rows=pixel_rows,
        names=[
            PIXEL_IDS,
            "x_cm",
            "y_cm",
            PATCH_COUNT,
            TYPICAL_DISTANCE,
            PATCH_IDS,
            "trigger_status",
            DISTANCE_STATUS,
        ],
    )
    group_table = Table(
        rows=group_rows,
        names=[
            GROUP_ID,
            GROUP_KIND,
            "pixel_count",
            MEDIAN_PAIR_DISTANCE,
            MAX_PAIR_DISTANCE,
            CONNECTED_COMPONENTS,
        ],
    )
    return pixel_table, group_table


def _group_memberships(
    group_ids, group_kinds, group_members, pixel_indices, x_coordinates, y_coordinates, neighbors
):
    """Return per-pixel distances and group IDs, and the per-group summary rows."""
    pixel_distances = [[] for _ in x_coordinates]
    pixel_groups = [[] for _ in x_coordinates]
    group_rows = []
    for group_id, group_kind in zip(group_ids, group_kinds):
        group_id = int(group_id)
        indices = sorted(pixel_indices[pixel_id] for pixel_id in group_members[group_id])
        distances, pairwise = _group_distances(x_coordinates, y_coordinates, indices)
        for index in indices:
            pixel_groups[index].append(group_id)
            if index in distances:
                pixel_distances[index].append(distances[index])
        group_rows.append(
            (
                group_id,
                str(group_kind),
                len(indices),
                float(np.median(pairwise)) if pairwise else np.nan,
                float(np.max(pairwise)) if pairwise else np.nan,
                _count_connected_components(indices, neighbors),
            )
        )
    return pixel_distances, pixel_groups, group_rows


def _pixel_distance_status(selected, distances):
    """Return the human-readable distance status of one pixel."""
    if not selected:
        return "no patch membership"
    if not distances:
        return "no other members in patch"
    return "available"


def _pixel_row(pixel_id, x, y, groups, distances):
    """Return the pixel-table row for one camera pixel."""
    selected = sorted(groups)
    return (
        int(pixel_id),
        float(x),
        float(y),
        len(selected),
        float(np.median(distances)) if distances else np.nan,
        ",".join(map(str, selected)),
        "configured" if selected else "unclassified",
        _pixel_distance_status(selected, distances),
    )


def _camera_geometry(camera):
    """Validate and return camera pixel IDs and physical coordinates."""
    pixel_ids = np.asarray(camera.pixels[PIXEL_IDS], dtype=int)
    x_coordinates = np.asarray(camera.pixels["x"], dtype=float)
    y_coordinates = np.asarray(camera.pixels["y"], dtype=float)
    if len(set(pixel_ids.tolist())) != len(pixel_ids):
        raise ValueError("Camera pixel IDs must be unique")
    if len(x_coordinates) != len(pixel_ids) or len(y_coordinates) != len(pixel_ids):
        raise ValueError("Camera pixel IDs and coordinates must have matching lengths")
    if not (np.all(np.isfinite(x_coordinates)) and np.all(np.isfinite(y_coordinates))):
        raise ValueError("Camera pixel coordinates must be finite")
    return pixel_ids, x_coordinates, y_coordinates


def _validated_group_membership(trigger_groups, trigger_members, pixel_indices):
    """Validate trigger identifiers and return distinct pixel membership sets."""
    group_ids = _required_column(trigger_groups, GROUP_ID)
    group_kinds = _required_column(trigger_groups, GROUP_KIND)
    member_groups = _required_column(trigger_members, GROUP_ID)
    member_pixels = _required_column(trigger_members, "pixel_id")
    if len(set(map(int, group_ids))) != len(group_ids):
        raise ValueError("Trigger group IDs must be unique")
    if not group_ids:
        raise ValueError("Camera configuration contains no trigger groups")

    group_members = {int(group_id): set() for group_id in group_ids}
    for group_id, pixel_id in zip(member_groups, member_pixels):
        group_id, pixel_id = int(group_id), int(pixel_id)
        if group_id not in group_members:
            raise ValueError(f"Trigger member references unknown trigger group {group_id}")
        if pixel_id not in pixel_indices:
            raise ValueError(f"Trigger member pixel {pixel_id} is not in the camera")
        group_members[group_id].add(pixel_id)

    empty_groups = [group_id for group_id, pixels in group_members.items() if not pixels]
    if empty_groups:
        raise ValueError(f"Trigger groups have no members: {empty_groups}")
    return group_ids, group_kinds, group_members


def _required_column(table, column_name):
    """Return a table column or raise an actionable error for missing data."""
    try:
        return list(table[column_name])
    except KeyError as exc:
        raise ValueError(f"Trigger configuration is missing required column {column_name}") from exc


def _group_distances(x_coordinates, y_coordinates, indices):
    """Return per-pixel medians and unique distances within one group."""
    if len(indices) < 2:
        return {}, []
    points = np.column_stack((x_coordinates[indices], y_coordinates[indices]))
    matrix = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    medians = {
        indices[index]: float(np.median(np.delete(matrix[index], index)))
        for index in range(len(indices))
    }
    return medians, matrix[np.triu_indices(len(indices), k=1)].tolist()


def _count_connected_components(indices, neighbors):
    """Count components in a trigger group using camera neighbor indices."""
    remaining, components = set(indices), 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            connected = remaining.intersection(neighbors[pending.pop()])
            remaining.difference_update(connected)
            pending.extend(connected)
    return components


def write_trigger_patch_report(camera, pixel_table, group_table, pdf_path):
    """
    Write a two-page PDF for trigger-patch diagnostics.

    Parameters
    ----------
    camera : Camera
        Camera geometry used for both pages.
    pixel_table : astropy.table.Table
        Per-pixel output from :func:`analyze_trigger_patch_configuration`.
    group_table : astropy.table.Table
        Per-group output from :func:`analyze_trigger_patch_configuration`.
    pdf_path : pathlib.Path or str
        Destination PDF path.

    Returns
    -------
    str
        The page-two summary text for application logging.
    """
    page_summary = format_trigger_patch_summary(pixel_table, group_table)
    distance_map = np.ma.masked_invalid(np.asarray(pixel_table[TYPICAL_DISTANCE], dtype=float))
    colormap = plt.get_cmap("viridis").with_extremes(bad="#aeb9c5")
    pages = (
        (
            "patch configuration",
            "Color shows how many configured trigger patches contain each pixel.",
            np.asarray(pixel_table[PATCH_COUNT]),
            "Configured patches per pixel",
            None,
            None,
        ),
        (
            "typical distance to pixels in the same patch",
            "Median camera-plane distance to the other pixels in each patch.",
            distance_map,
            "Typical distance to pixels in the same patch [cm]",
            page_summary,
            colormap,
        ),
    )
    with pdf_report(Path(pdf_path)) as pdf:
        for title, subtitle, image, label, summary, cmap in pages:
            _write_report_page(pdf, camera, title, subtitle, image, label, summary, cmap)
    return page_summary


def _write_report_page(
    pdf, camera, title, subtitle, image, color_bar_label, summary=None, colormap=None
):
    """Write one camera map and, when provided, its summary panel."""
    figure, (axis, details_axis) = plt.subplots(
        ncols=2, figsize=(15, 9), gridspec_kw={"width_ratios": [3.5, 1.5]}
    )
    figure.subplots_adjust(top=0.84, bottom=0.08, left=0.055, right=0.97, wspace=0.10)
    try:
        figure.patch.set_facecolor("white")
        figure.suptitle(
            f"{camera.telescope_name}: {title}",
            x=0.055,
            y=0.98,
            ha="left",
            va="top",
            fontsize=19,
            fontweight="bold",
            color=REPORT_INK,
        )
        figure.text(0.057, 0.935, subtitle, ha="left", va="top", fontsize=10.5, color=REPORT_MUTED)
        plot_kwargs = {"ax": axis, "color_bar_label": color_bar_label}
        if colormap is not None:
            plot_kwargs["colormap"] = colormap
        plot_pixel_layout_with_image(camera, image=image, **plot_kwargs)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Camera x [cm]", fontsize=10, color=REPORT_MUTED)
        axis.set_ylabel("Camera y [cm]", fontsize=10, color=REPORT_MUTED)
        axis.tick_params(axis="both", labelsize=9, colors=REPORT_MUTED)
        details_axis.axis("off")
        if summary is not None:
            details_axis.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=details_axis.transAxes,
                    facecolor="#f1f5f9",
                    edgecolor="#d7e1eb",
                    linewidth=1,
                    zorder=-1,
                )
            )
            details_axis.text(
                0.06,
                0.96,
                summary,
                va="top",
                transform=details_axis.transAxes,
                wrap=True,
                fontsize=9.5,
                linespacing=1.4,
                color=REPORT_INK,
                family="DejaVu Sans Mono",
            )
        pdf.add(figure)
    finally:
        plt.close(figure)


def format_trigger_patch_summary(pixel_table, group_table):
    """
    Format the page-two summary for both the report and application log.

    Parameters
    ----------
    pixel_table : astropy.table.Table
        Per-pixel diagnostics.
    group_table : astropy.table.Table
        Per-group diagnostics.

    Returns
    -------
    str
        Multi-line summary text.
    """
    uncovered_ids = pixel_table[PIXEL_IDS][pixel_table[PATCH_COUNT] == 0].tolist()
    typical = np.asarray(pixel_table[TYPICAL_DISTANCE], dtype=float)
    undefined_ids = pixel_table[PIXEL_IDS][~np.isfinite(typical)].tolist()
    outliers = _distance_outlier_group_ids(group_table)
    alert_groups = [
        row
        for row in group_table
        if int(row[CONNECTED_COMPONENTS]) > 1 or str(row[GROUP_ID]) in outliers
    ]
    sorted_groups = sorted(
        alert_groups,
        key=lambda row: (int(row[CONNECTED_COMPONENTS]) > 1, float(row[MAX_PAIR_DISTANCE])),
        reverse=True,
    )
    lines = [
        "PATCH SUMMARY",
        f"Groups: {len(group_table)}",
        f"Camera coverage: {len(pixel_table) - len(uncovered_ids)} / {len(pixel_table)} pixels",
        f"Distances available: {np.count_nonzero(np.isfinite(typical))} / {len(pixel_table)}",
        "",
        "REVIEW",
        f"Uncovered pixels: {len(uncovered_ids)}",
        "  IDs: " + _format_pixel_ids(uncovered_ids),
        f"Undefined distances: {len(undefined_ids)}",
        "  IDs: " + _format_pixel_ids(undefined_ids),
        "  Gray pixels: see distance_status table",
        "",
        "DISTANCE OUTLIERS / DISCONNECTED GROUPS",
    ]
    if not sorted_groups:
        lines.append("No outlying spans or disconnected groups.")
    for row in sorted_groups[:5]:
        maximum = row[MAX_PAIR_DISTANCE]
        count = int(row[CONNECTED_COMPONENTS])
        span = f"{maximum:.2f} cm" if np.isfinite(maximum) else "undefined"
        label = "component" if count == 1 else "components"
        lines.append(f"{row[GROUP_ID]} ({row[GROUP_KIND]})\n  max span {span} | {count} {label}")
        lines.append(
            textwrap.fill(
                ", ".join(map(str, _pixels_in_group(pixel_table, row[GROUP_ID]))),
                width=39,
                initial_indent="  pixels: ",
                subsequent_indent="    ",
            )
        )
    if len(sorted_groups) > 5:
        lines.append(f"... and {len(sorted_groups) - 5} more; see groups table")
    return "\n".join(lines)


def _distance_outlier_group_ids(group_table):
    """Find per-kind max-span outliers using Tukey's upper fence (values rounded to cm)."""
    spans = [
        (str(row[GROUP_KIND]), str(row[GROUP_ID]), round(float(row[MAX_PAIR_DISTANCE]), 2))
        for row in group_table
        if np.isfinite(float(row[MAX_PAIR_DISTANCE]))
    ]
    outliers = set()
    for kind in {kind for kind, _, _ in spans}:
        values = [span for k, _, span in spans if k == kind]
        first_quartile, third_quartile = np.percentile(values, (25, 75))
        fence = third_quartile + 1.5 * (third_quartile - first_quartile)
        outliers.update(group for k, group, span in spans if k == kind and span > fence)
    return outliers


def _pixels_in_group(pixel_table, group_id):
    """Return all camera pixel IDs listed for a trigger group."""
    return sorted(
        int(row[PIXEL_IDS])
        for row in pixel_table
        if str(group_id) in str(row[PATCH_IDS]).split(",")
    )


def _format_pixel_ids(pixel_ids):
    """Format a bounded set of pixel IDs for the PDF summary."""
    shown_ids = ", ".join(map(str, pixel_ids[:20])) or "none"
    return f"{shown_ids}, ..." if len(pixel_ids) > 20 else shown_ids
