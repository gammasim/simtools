"""Analyze and plot camera trigger-patch configuration."""

import textwrap
from pathlib import Path

import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_camera import plot_pixel_layout_with_image

GROUP_ID = "group_id"
GROUP_KIND = "kind"
MEMBER_PIXEL_ID = "pixel_id"
PIXEL_IDS = "pix_id"
PATCH_COUNT = "patch_count"
PATCH_IDS = "patch_ids"
PATCH_ID_SEPARATOR = ","
TYPICAL_DISTANCE = "typical_distance_cm"
TRIGGER_STATUS = "trigger_status"
DISTANCE_STATUS = "distance_status"
MEDIAN_PAIR_DISTANCE = "median_pair_distance_cm"
MAX_PAIR_DISTANCE = "max_pair_distance_cm"
CONNECTED_COMPONENTS = "connected_components"
PATCH_CONFIGURATION_TITLE = "patch configuration"
PATCH_DISTANCE_TITLE = "typical distance to pixels in the same patch"
PATCH_COVERAGE_LABEL = "Configured patches per pixel"
PATCH_DISTANCE_LABEL = "Typical distance to pixels in the same patch [cm]"
PATCH_CONFIGURATION_SUBTITLE = "Color shows how many configured trigger patches contain each pixel."
PATCH_DISTANCE_SUBTITLE = "Median camera-plane distance to the other pixels in each patch."
UNCLASSIFIED = "unclassified"
CONFIGURED = "configured"
DISTANCE_AVAILABLE = "available"
NO_PATCH_MEMBERSHIP = "no patch membership"
NO_PATCH_PEERS = "no other members in patch"
EQUAL_ASPECT = "equal"
REPORT_INK = "#17324d"
REPORT_MUTED = "#5c7186"
REPORT_PANEL = "#f1f5f9"
REPORT_BORDER = "#d7e1eb"
REPORT_UNDEFINED = "#aeb9c5"
REPORT_BACKGROUND = "white"
SUMMARY_ID_INDENT = "  IDs: "
GROUP_PIXEL_INDENT = "  pixels: "
GROUP_PIXEL_CONTINUATION_INDENT = "    "
GROUP_PIXEL_WRAP_WIDTH = 39
DISTANCE_REPORT_PRECISION = 2
OUTLIER_QUARTILES = (25, 75)
OUTLIER_IQR_MULTIPLIER = 1.5
REPORT_TOP = 0.84
REPORT_BOTTOM = 0.08
REPORT_LEFT = 0.055
REPORT_RIGHT = 0.97
REPORT_WSPACE = 0.10
CAMERA_X_LABEL = "Camera x [cm]"
CAMERA_Y_LABEL = "Camera y [cm]"


def analyze_trigger_patch_configuration(camera, trigger_groups, trigger_members):
    """Calculate per-pixel patch distances and per-group spatial summaries.

    Parameters
    ----------
    camera : Camera
        Camera geometry with pixel IDs and neighbor lookup.
    trigger_groups : astropy.table.Table
        Resolved group definitions with ``group_id`` and ``kind`` columns.
    trigger_members : astropy.table.Table
        Resolved memberships with ``group_id`` and ``pixel_id`` columns.

    Returns
    -------
    tuple of astropy.table.Table
        Per-pixel diagnostics followed by per-group diagnostics. Distances are
        in camera-plane centimeters; undefined distances are represented by NaN.

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

    group_rows, pixel_group_distances, pixel_group_ids = _analyze_groups(
        pixel_ids,
        x_coordinates,
        y_coordinates,
        pixel_indices,
        group_ids,
        group_kinds,
        group_members,
        neighbors,
    )
    pixel_rows = _build_pixel_rows(
        pixel_ids,
        x_coordinates,
        y_coordinates,
        pixel_group_distances,
        pixel_group_ids,
    )
    pixel_table = _build_pixel_table(pixel_rows)
    group_table = _build_group_table(group_rows)
    return pixel_table, group_table


def _analyze_groups(
    pixel_ids,
    x_coordinates,
    y_coordinates,
    pixel_indices,
    group_ids,
    group_kinds,
    group_members,
    neighbors,
):
    """Collect group metrics and per-pixel distance memberships."""
    pixel_group_distances = [[] for _ in pixel_ids]
    pixel_group_ids = [[] for _ in pixel_ids]
    group_rows = []
    for group_id, group_kind in zip(group_ids, group_kinds):
        group_id = int(group_id)
        indices = sorted(pixel_indices[pixel_id] for pixel_id in group_members[group_id])
        distances = _pairwise_distances(x_coordinates, y_coordinates, indices)
        for index in indices:
            pixel_group_ids[index].append(group_id)
            if index in distances:
                pixel_group_distances[index].append(distances[index])
        pairwise_values = _all_pairwise_distances(x_coordinates, y_coordinates, indices)
        group_rows.append(
            (
                group_id,
                str(group_kind),
                len(indices),
                float(np.median(pairwise_values)) if pairwise_values else np.nan,
                float(np.max(pairwise_values)) if pairwise_values else np.nan,
                _count_connected_components(indices, neighbors),
            )
        )
    return group_rows, pixel_group_distances, pixel_group_ids


def _build_pixel_rows(pixel_ids, x_coordinates, y_coordinates, group_distances, group_ids):
    """Build diagnostic rows for all camera pixels."""
    pixel_rows = []
    for index, pixel_id in enumerate(pixel_ids):
        distances = group_distances[index]
        selected_groups = sorted(group_ids[index])
        pixel_rows.append(
            (
                int(pixel_id),
                float(x_coordinates[index]),
                float(y_coordinates[index]),
                len(selected_groups),
                float(np.median(distances)) if distances else np.nan,
                ",".join(map(str, selected_groups)),
                CONFIGURED if selected_groups else UNCLASSIFIED,
                _distance_status(selected_groups, distances),
            )
        )
    return pixel_rows


def _build_pixel_table(pixel_rows):
    """Build the output table containing per-pixel diagnostics."""
    return Table(
        rows=pixel_rows,
        names=(
            PIXEL_IDS,
            "x_cm",
            "y_cm",
            PATCH_COUNT,
            TYPICAL_DISTANCE,
            PATCH_IDS,
            TRIGGER_STATUS,
            DISTANCE_STATUS,
        ),
    )


def _build_group_table(group_rows):
    """Build the output table containing per-group diagnostics."""
    return Table(
        rows=group_rows,
        names=(
            GROUP_ID,
            GROUP_KIND,
            "pixel_count",
            MEDIAN_PAIR_DISTANCE,
            MAX_PAIR_DISTANCE,
            CONNECTED_COMPONENTS,
        ),
    )


def _distance_status(selected_groups, distances):
    """Describe why a per-pixel patch distance is or is not available."""
    if not selected_groups:
        return NO_PATCH_MEMBERSHIP
    if not distances:
        return NO_PATCH_PEERS
    return DISTANCE_AVAILABLE


def _camera_geometry(camera):
    """Validate and return camera pixel IDs and physical coordinates."""
    pixel_ids = np.asarray(camera.pixels[PIXEL_IDS], dtype=int)
    x_coordinates = np.asarray(camera.pixels["x"], dtype=float)
    y_coordinates = np.asarray(camera.pixels["y"], dtype=float)
    if len(set(pixel_ids.tolist())) != len(pixel_ids):
        raise ValueError("Camera pixel IDs must be unique")
    if len(x_coordinates) != len(pixel_ids) or len(y_coordinates) != len(pixel_ids):
        raise ValueError("Camera pixel IDs and coordinates must have matching lengths")
    if not np.all(np.isfinite(x_coordinates)) or not np.all(np.isfinite(y_coordinates)):
        raise ValueError("Camera pixel coordinates must be finite")
    return pixel_ids, x_coordinates, y_coordinates


def _validated_group_membership(trigger_groups, trigger_members, pixel_indices):
    """Validate trigger identifiers and return distinct pixel membership sets."""
    group_ids = _required_column(trigger_groups, GROUP_ID)
    group_kinds = _required_column(trigger_groups, GROUP_KIND)
    member_groups = _required_column(trigger_members, GROUP_ID)
    member_pixels = _required_column(trigger_members, MEMBER_PIXEL_ID)
    if len(set(map(int, group_ids))) != len(group_ids):
        raise ValueError("Trigger group IDs must be unique")
    if not group_ids:
        raise ValueError("Camera configuration contains no trigger groups")

    group_members = {int(group_id): set() for group_id in group_ids}
    for group_id, pixel_id in zip(member_groups, member_pixels):
        group_id = int(group_id)
        pixel_id = int(pixel_id)
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


def _pairwise_distances(x_coordinates, y_coordinates, indices):
    """Return each group's per-pixel median distance to its other members."""
    if len(indices) < 2:
        return {}
    points = np.column_stack((x_coordinates[indices], y_coordinates[indices]))
    distance_matrix = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    return {
        indices[index]: float(np.median(np.delete(distance_matrix[index], index)))
        for index in range(len(indices))
    }


def _all_pairwise_distances(x_coordinates, y_coordinates, indices):
    """Return unique pairwise distances for one trigger group."""
    if len(indices) < 2:
        return []
    points = np.column_stack((x_coordinates[indices], y_coordinates[indices]))
    distance_matrix = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
    return distance_matrix[np.triu_indices(len(indices), k=1)].tolist()


def _count_connected_components(indices, neighbors):
    """Count components in a trigger group using camera neighbor indices."""
    remaining = set(indices)
    components = 0
    while remaining:
        components += 1
        pending = [remaining.pop()]
        while pending:
            current = pending.pop()
            connected = remaining.intersection(neighbors[current])
            remaining.difference_update(connected)
            pending.extend(connected)
    return components


def write_trigger_patch_report(camera, pixel_table, group_table, pdf_path):
    """Write a two-page PDF for trigger-patch diagnostics.

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
    pdf_path = Path(pdf_path)
    page_summary = format_trigger_patch_summary(pixel_table, group_table)
    with PdfPages(pdf_path) as pdf:
        _write_coverage_page(pdf, camera, pixel_table)
        _write_distance_page(pdf, camera, pixel_table, page_summary)
    return page_summary


def _write_coverage_page(pdf, camera, pixel_table):
    """Write the existing patch-coverage map as the first report page."""
    figure, (axis, details_axis) = plt.subplots(
        ncols=2,
        figsize=(15, 9),
        gridspec_kw={"width_ratios": [3.5, 1.5]},
    )
    figure.subplots_adjust(
        top=REPORT_TOP,
        bottom=REPORT_BOTTOM,
        left=REPORT_LEFT,
        right=REPORT_RIGHT,
        wspace=REPORT_WSPACE,
    )
    try:
        _style_report_figure(
            figure,
            f"{camera.telescope_name}: {PATCH_CONFIGURATION_TITLE}",
            PATCH_CONFIGURATION_SUBTITLE,
        )
        plot_pixel_layout_with_image(
            camera,
            image=np.asarray(pixel_table[PATCH_COUNT]),
            ax=axis,
            color_bar_label=PATCH_COVERAGE_LABEL,
        )
        axis.set_aspect(EQUAL_ASPECT, adjustable="box")
        axis.set_xlabel(CAMERA_X_LABEL)
        axis.set_ylabel(CAMERA_Y_LABEL)
        _style_camera_axis(axis)
        details_axis.axis("off")
        pdf.savefig(figure, facecolor=REPORT_BACKGROUND)
    finally:
        plt.close(figure)


def _write_distance_page(pdf, camera, pixel_table, page_summary):
    """Write per-pixel physical distances and a compact group summary."""
    distance_map = np.ma.masked_invalid(np.asarray(pixel_table[TYPICAL_DISTANCE], dtype=float))
    colormap = plt.get_cmap("viridis").with_extremes(bad=REPORT_UNDEFINED)
    figure, (axis, details_axis) = plt.subplots(
        ncols=2,
        figsize=(15, 9),
        gridspec_kw={"width_ratios": [3.5, 1.5]},
    )
    figure.subplots_adjust(
        top=REPORT_TOP,
        bottom=REPORT_BOTTOM,
        left=REPORT_LEFT,
        right=REPORT_RIGHT,
        wspace=REPORT_WSPACE,
    )
    try:
        _style_report_figure(
            figure,
            f"{camera.telescope_name}: {PATCH_DISTANCE_TITLE}",
            PATCH_DISTANCE_SUBTITLE,
        )
        plot_pixel_layout_with_image(
            camera,
            image=distance_map,
            colormap=colormap,
            ax=axis,
            color_bar_label=PATCH_DISTANCE_LABEL,
        )
        axis.set_aspect(EQUAL_ASPECT, adjustable="box")
        axis.set_xlabel(CAMERA_X_LABEL)
        axis.set_ylabel(CAMERA_Y_LABEL)
        _style_camera_axis(axis)
        details_axis.axis("off")
        details_axis.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=details_axis.transAxes,
                facecolor=REPORT_PANEL,
                edgecolor=REPORT_BORDER,
                linewidth=1,
                zorder=-1,
            )
        )
        details_axis.text(
            0.06,
            0.96,
            page_summary,
            va="top",
            transform=details_axis.transAxes,
            wrap=True,
            fontsize=9.5,
            linespacing=1.4,
            color=REPORT_INK,
            family="DejaVu Sans Mono",
        )
        pdf.savefig(figure, facecolor=REPORT_BACKGROUND)
    finally:
        plt.close(figure)


def _style_report_figure(figure, title, subtitle):
    """Apply consistent title and subtitle styling to report pages."""
    figure.patch.set_facecolor(REPORT_BACKGROUND)
    figure.suptitle(
        title,
        x=0.055,
        y=0.98,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=REPORT_INK,
    )
    figure.text(
        0.057,
        0.935,
        subtitle,
        ha="left",
        va="top",
        fontsize=10.5,
        color=REPORT_MUTED,
    )


def _style_camera_axis(axis):
    """Polish camera-map labels without changing the plotted values."""
    axis.tick_params(axis="both", labelsize=9, colors=REPORT_MUTED)
    axis.xaxis.label.set_color(REPORT_MUTED)
    axis.yaxis.label.set_color(REPORT_MUTED)
    axis.xaxis.label.set_size(10)
    axis.yaxis.label.set_size(10)


def format_trigger_patch_summary(pixel_table, group_table):
    """Format the page-two summary for both the report and application log."""
    uncovered = pixel_table[PATCH_COUNT] == 0
    uncovered_ids = pixel_table[PIXEL_IDS][uncovered].tolist()
    undefined_distances = pixel_table[PIXEL_IDS][
        ~np.isfinite(np.asarray(pixel_table[TYPICAL_DISTANCE], dtype=float))
    ].tolist()
    outlier_group_ids = _distance_outlier_group_ids(group_table)
    alert_groups = [
        row
        for row in group_table
        if int(row[CONNECTED_COMPONENTS]) > 1 or str(row[GROUP_ID]) in outlier_group_ids
    ]
    sorted_groups = sorted(
        alert_groups,
        key=lambda row: (int(row[CONNECTED_COMPONENTS]) > 1, float(row[MAX_PAIR_DISTANCE])),
        reverse=True,
    )
    covered_count = int(np.count_nonzero(~uncovered))
    distance_count = int(np.count_nonzero(np.isfinite(pixel_table[TYPICAL_DISTANCE])))
    lines = [
        "PATCH SUMMARY",
        f"Groups: {len(group_table)}",
        f"Camera coverage: {covered_count} / {len(pixel_table)} pixels",
        f"Distances available: {distance_count} / {len(pixel_table)}",
        "",
        "REVIEW",
        f"Uncovered pixels: {len(uncovered_ids)}",
        SUMMARY_ID_INDENT + _format_pixel_ids(uncovered_ids),
        f"Undefined distances: {len(undefined_distances)}",
        SUMMARY_ID_INDENT + _format_pixel_ids(undefined_distances),
        "  Gray pixels: see distance_status table",
        "",
        "DISTANCE OUTLIERS / DISCONNECTED GROUPS",
    ]
    if not sorted_groups:
        lines.append("No outlying spans or disconnected groups.")
    for row in sorted_groups[:5]:
        maximum_distance = row[MAX_PAIR_DISTANCE]
        component_count = int(row[CONNECTED_COMPONENTS])
        component_label = "component" if component_count == 1 else "components"
        distance_text = (
            f"{maximum_distance:.2f} cm" if np.isfinite(maximum_distance) else "undefined"
        )
        pixel_ids = _pixels_in_group(pixel_table, row[GROUP_ID])
        lines.append(
            f"{row[GROUP_ID]} ({row[GROUP_KIND]})\n"
            f"  max span {distance_text} | {component_count} {component_label}"
        )
        lines.append(
            textwrap.fill(
                ", ".join(map(str, pixel_ids)),
                width=GROUP_PIXEL_WRAP_WIDTH,
                initial_indent=GROUP_PIXEL_INDENT,
                subsequent_indent=GROUP_PIXEL_CONTINUATION_INDENT,
            )
        )
    if len(sorted_groups) > 5:
        lines.append(f"... and {len(sorted_groups) - 5} more; see groups table")
    return "\n".join(lines)


def _distance_outlier_group_ids(group_table):
    """Find per-kind max-span outliers using Tukey's upper fence."""
    distances_by_kind = {}
    for row in group_table:
        maximum_distance = float(row[MAX_PAIR_DISTANCE])
        if np.isfinite(maximum_distance):
            kind = str(row[GROUP_KIND])
            distances_by_kind.setdefault(kind, []).append(
                round(maximum_distance, DISTANCE_REPORT_PRECISION)
            )

    outlier_group_ids = set()
    for kind, distances in distances_by_kind.items():
        first_quartile, third_quartile = np.percentile(distances, OUTLIER_QUARTILES)
        upper_fence = third_quartile + OUTLIER_IQR_MULTIPLIER * (third_quartile - first_quartile)
        for row in group_table:
            maximum_distance = float(row[MAX_PAIR_DISTANCE])
            if (
                str(row[GROUP_KIND]) == kind
                and np.isfinite(maximum_distance)
                and round(maximum_distance, DISTANCE_REPORT_PRECISION) > upper_fence
            ):
                outlier_group_ids.add(str(row[GROUP_ID]))
    return outlier_group_ids


def _pixels_in_group(pixel_table, group_id):
    """Return all camera pixel IDs listed for a trigger group."""
    selected_group_id = str(group_id)
    return sorted(
        int(row[PIXEL_IDS])
        for row in pixel_table
        if selected_group_id in str(row[PATCH_IDS]).split(PATCH_ID_SEPARATOR)
    )


def _format_pixel_ids(pixel_ids):
    """Format a bounded set of pixel IDs for the PDF summary."""
    shown_ids = ", ".join(map(str, pixel_ids[:20])) or "none"
    if len(pixel_ids) > 20:
        shown_ids = f"{shown_ids}, ..."
    return shown_ids
