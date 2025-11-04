#!/usr/bin/python3
"""Functions for plotting mirror panel layout information."""

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from simtools.db import db_handler
from simtools.io import io_handler
from simtools.model.mirrors import Mirrors
from simtools.visualization import visualize

logger = logging.getLogger(__name__)


def _detect_segmentation_type(data_file_path):
    """
    Detect the type of segmentation file (ring, petal/polygon, or standard).

    Parameters
    ----------
    data_file_path : Path
        Path to the segmentation data file

    Returns
    -------
    str
        One of "ring", "petal", or "standard"
    """
    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            line_lower = line.strip().lower()
            if line_lower.startswith("#") or not line_lower:
                continue
            if line_lower.startswith("ring"):
                return "ring"
            if line_lower.startswith(("poly", "hex", "square", "circular", "yhex")):
                return "petal"
    return "standard"


def plot(config, output_file, db_config=None):
    """
    Plot mirror panel layout based on configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - parameter : str, should be "mirror_list", "primary_mirror_segmentation",
          or "secondary_mirror_segmentation"
        - site : str, site name
        - telescope : str, name of the telescope
        - parameter_version: str, version of the parameter
        - model_version: str, version of the model
    output_file : str or Path
        Path where to save the plot
    db_config : dict, optional
        Database configuration.

    Returns
    -------
    None
        The function saves the plot to the specified output file.
    """
    db = db_handler.DatabaseHandler(db_config=db_config)

    parameters = db.get_model_parameter(
        config["parameter"],
        config["site"],
        config.get("telescope"),
        parameter_version=config.get("parameter_version"),
        model_version=config.get("model_version"),
    )

    db.export_model_files(parameters=parameters, dest=io_handler.IOHandler().get_output_directory())

    mirror_file = parameters[config["parameter"]]["value"]
    data_file_path = Path(io_handler.IOHandler().get_output_directory() / mirror_file)

    parameter_type = config["parameter"]

    if parameter_type in ("primary_mirror_segmentation", "secondary_mirror_segmentation"):
        segmentation_type = _detect_segmentation_type(data_file_path)

        if segmentation_type == "ring":
            fig = plot_mirror_ring_segmentation(
                data_file_path=data_file_path,
                telescope_model_name=config["telescope"],
                parameter_type=parameter_type,
                title=config.get("title"),
            )
        elif segmentation_type == "petal":
            fig = plot_mirror_petal_segmentation(
                data_file_path=data_file_path,
                telescope_model_name=config["telescope"],
                parameter_type=parameter_type,
                title=config.get("title"),
            )
        else:
            fig = plot_mirror_segmentation(
                data_file_path=data_file_path,
                telescope_model_name=config["telescope"],
                parameter_type=parameter_type,
                title=config.get("title"),
            )
    else:
        mirrors = Mirrors(mirror_list_file=data_file_path)
        fig = plot_mirror_layout(
            mirrors=mirrors,
            telescope_model_name=config["telescope"],
            title=config.get("title"),
        )

    visualize.save_figure(fig, output_file)
    plt.close(fig)


def plot_mirror_layout(mirrors, telescope_model_name, title=None):
    """
    Plot the mirror panel layout from a Mirrors object.

    This function creates a visualization of mirror panel positions,
    showing their spatial arrangement on the telescope dish structure.

    Parameters
    ----------
    mirrors : Mirrors
        Mirrors object containing mirror panel information
    telescope_model_name : str
        Name/model of the telescope
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    logger.info(f"Plotting mirror layout for {telescope_model_name}")

    fig, ax = plt.subplots(figsize=(10, 10))

    x_pos = mirrors.mirror_table["mirror_x"].to("cm").value
    y_pos = mirrors.mirror_table["mirror_y"].to("cm").value
    diameter = mirrors.mirror_diameter.to("cm").value
    shape_type = mirrors.shape_type
    focal_lengths = mirrors.mirror_table["focal_length"].to("cm").value

    mirror_ids = (
        mirrors.mirror_table["mirror_panel_id"]
        if "mirror_panel_id" in mirrors.mirror_table.colnames
        else list(range(len(x_pos)))
    )

    patches, colors = _create_mirror_patches(x_pos, y_pos, diameter, shape_type, focal_lengths)

    collection = PatchCollection(
        patches,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
    )
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    mean_outer_edge_radius = _calculate_mean_outer_edge_radius(x_pos, y_pos, diameter, shape_type)
    outer_edge_circle = mpatches.Circle(
        (0, 0), mean_outer_edge_radius, fill=False, edgecolor="darkorange", linewidth=2.0
    )
    ax.add_patch(outer_edge_circle)

    _add_mirror_labels(ax, x_pos, y_pos, mirror_ids, max_labels=20)

    _configure_mirror_plot(
        ax,
        x_pos,
        y_pos,
        title=title,
        telescope_model_name=telescope_model_name,
    )

    cbar = plt.colorbar(collection, ax=ax, pad=0.02)
    cbar.set_label("Focal length [cm]", fontsize=14)

    _add_mirror_statistics(ax, mirrors, x_pos, y_pos, diameter)

    return fig


def plot_mirror_segmentation(data_file_path, telescope_model_name, parameter_type, title=None):
    """
    Plot mirror segmentation layout from a segmentation file.

    This function creates a visualization of mirror segments,
    showing their spatial arrangement and grouping.

    Parameters
    ----------
    data_file_path : Path
        Path to the segmentation data file
    telescope_model_name : str
        Name/model of the telescope
    parameter_type : str
        Type of segmentation ("primary_mirror_segmentation" or "secondary_mirror_segmentation")
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    logger.info(f"Plotting {parameter_type} for {telescope_model_name}")

    segmentation_data = _read_segmentation_file(data_file_path)

    fig, ax = plt.subplots(figsize=(10, 10))

    x_pos = segmentation_data["x"]
    y_pos = segmentation_data["y"]
    diameter = segmentation_data["diameter"]
    shape_type = segmentation_data["shape_type"]
    segment_ids = segmentation_data["segment_ids"]

    patches, colors = _create_segmentation_patches(x_pos, y_pos, diameter, shape_type, segment_ids)

    collection = PatchCollection(
        patches,
        cmap="tab20",
        edgecolor="black",
        linewidth=0.8,
    )
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    _add_mirror_labels(ax, x_pos, y_pos, segment_ids, max_labels=30)

    _configure_mirror_plot(
        ax,
        x_pos,
        y_pos,
        title=title,
        telescope_model_name=telescope_model_name,
    )

    cbar = plt.colorbar(collection, ax=ax, pad=0.02)
    cbar.set_label("Segment ID", fontsize=14)

    n_segments = len(set(segment_ids))
    stats_text = (
        f"Number of segments: {len(x_pos)}\n"
        f"Number of segment groups: {n_segments}\n"
        f"Segment diameter: {diameter:.1f} cm"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    return fig


def _create_mirror_patches(x_pos, y_pos, diameter, shape_type, focal_lengths):
    """
    Create matplotlib patches for mirror panels.

    Parameters
    ----------
    x_pos, y_pos : array-like
        X and Y coordinates of mirror panel centers
    diameter : float
        Diameter of mirror panels
    shape_type : int
        Shape type (0: circular, 1/3: hexagonal, 2: square)
    focal_lengths : array-like
        Focal length values for each mirror panel (used for coloring)

    Returns
    -------
    tuple
        (patches, colors) - list of matplotlib patches and corresponding color values
    """
    patches = []
    colors = []

    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        patch = _create_single_mirror_patch(x, y, diameter, shape_type)
        patches.append(patch)
        colors.append(focal_lengths[i])

    return patches, colors


def _read_segmentation_file(data_file_path):
    """
    Read mirror segmentation file and extract segment information.

    Parameters
    ----------
    data_file_path : Path
        Path to the segmentation data file

    Returns
    -------
    dict
        Dictionary containing x, y, diameter, shape_type, and segment_ids
    """
    x_pos = []
    y_pos = []
    diameter = None
    shape_type = None
    segment_ids = []

    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                x_pos.append(float(parts[0]))
                y_pos.append(float(parts[1]))
            except ValueError:
                continue

            diameter = _extract_diameter(parts, diameter)
            shape_type = _extract_shape_type(parts, shape_type)
            segment_ids.append(_extract_segment_id(parts, len(segment_ids)))

    if len(x_pos) == 0:
        logger.warning(f"No valid numeric data found in segmentation file: {data_file_path}")

    return {
        "x": np.array(x_pos),
        "y": np.array(y_pos),
        "diameter": diameter if diameter is not None else 150.0,
        "shape_type": shape_type if shape_type is not None else 3,
        "segment_ids": segment_ids,
    }


def _extract_diameter(parts, current_diameter):
    """Extract diameter from parts or return current value."""
    return float(parts[2]) if current_diameter is None else current_diameter


def _extract_shape_type(parts, current_shape_type):
    """Extract shape type from parts or return current value."""
    return int(parts[4]) if current_shape_type is None else current_shape_type


def _extract_segment_id(parts, default_id):
    """Extract segment ID from parts or return default."""
    if len(parts) >= 8:
        seg_id_str = parts[7].split("=")[-1] if "=" in parts[7] else parts[7]
        return int("".join(filter(str.isdigit, seg_id_str)))
    return default_id


def _create_segmentation_patches(x_pos, y_pos, diameter, shape_type, segment_ids):
    """
    Create matplotlib patches for mirror segments.

    Parameters
    ----------
    x_pos, y_pos : array-like
        X and Y coordinates of segment centers
    diameter : float
        Diameter of segments
    shape_type : int
        Shape type (0: circular, 1/3: hexagonal, 2: square)
    segment_ids : array-like
        Segment ID values for each segment (used for coloring)

    Returns
    -------
    tuple
        (patches, colors) - list of matplotlib patches and corresponding color values
    """
    patches = []
    colors = []

    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        patch = _create_single_mirror_patch(x, y, diameter, shape_type)
        patches.append(patch)
        colors.append(segment_ids[i])

    return patches, colors


def _create_single_mirror_patch(x, y, diameter, shape_type):
    """
    Create a single matplotlib patch for a mirror panel.

    Parameters
    ----------
    x, y : float
        Center coordinates of the mirror panel
    diameter : float
        Diameter of the mirror panel
    shape_type : int
        Shape type:
        0: circular
        1: hexagonal (flat x)
        2: square
        3: hexagonal (flat y)

    Returns
    -------
    matplotlib.patches.Patch
        The created patch object for the mirror panel
    """
    if shape_type == 0:
        return mpatches.Circle((x, y), radius=diameter / 2)
    if shape_type in (1, 3):
        orientation = 0 if shape_type == 1 else np.pi / 2
        return mpatches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=diameter / np.sqrt(3),
            orientation=orientation,
        )
    return mpatches.Rectangle(
        (x - diameter / 2, y - diameter / 2),
        width=diameter,
        height=diameter,
    )


def _add_mirror_labels(ax, x_pos, y_pos, mirror_ids, max_labels=20):
    """
    Add mirror panel ID labels to the plot.

    Labels the mirrors with the lowest IDs (first few mirrors by ID number).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add labels to
    x_pos, y_pos : array-like
        X and Y coordinates of mirror panel centers
    mirror_ids : array-like
        Mirror panel IDs
    max_labels : int, optional
        Maximum number of labels to display
    """
    mirror_data = sorted(zip(mirror_ids, x_pos, y_pos), key=lambda item: item[0])

    for i, (mid, x, y) in enumerate(mirror_data):
        if i < max_labels:
            ax.text(
                x,
                y,
                str(mid),
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                weight="bold",
            )


def _configure_mirror_plot(ax, x_pos, y_pos, title, telescope_model_name):
    """
    Configure the mirror plot with titles, labels, and limits.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure
    x_pos, y_pos : array-like
        Arrays of x and y positions of mirror panels
    title : str
        Plot title
    telescope_model_name : str
        Name of the telescope model

    Returns
    -------
    None
        The function modifies the plot axes in place.
    """
    ax.set_aspect("equal")

    if len(x_pos) == 0 or len(y_pos) == 0:
        logger.warning("No valid mirror data found for plotting")
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.text(0, 0, "No valid mirror data", ha="center", va="center", fontsize=14, color="red")
        return

    x_min, x_max = np.min(x_pos), np.max(x_pos)
    y_min, y_max = np.min(y_pos), np.max(y_pos)

    x_padding = (x_max - x_min) * 0.15
    y_padding = (y_max - y_min) * 0.15

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.xlabel("X position [cm]", fontsize=16)
    plt.ylabel("Y position [cm]", fontsize=16)
    if title:
        ax.set_title(title, fontsize=18, pad=20)
    plt.tick_params(axis="both", which="major", labelsize=14)

    _add_camera_frame_indicator(ax, telescope_model_name)


def _add_camera_frame_indicator(ax, telescope_model_name=None):
    """
    Add camera frame coordinate system indicator to the plot (bottom-right corner).

    Shows the camera coordinate system convention based on telescope type:
    - LST/SST: X_cam (down), Y_cam (right) - camera frame coordinates
    - MST: North (down), West (right) - cardinal directions

    This matches sim_telarray convention where mirrors are positioned
    in the camera reference frame or cardinal directions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the indicator to
    telescope_model_name : str, optional
        Name of the telescope model; used to determine coordinate labels.
    """
    if telescope_model_name and "MST" in telescope_model_name.upper():
        x_label = "$North$"
        y_label = "$West$"
    else:
        x_label = "$X_{cam}$"
        y_label = "$Y_{cam}$"

    arrow_props = {
        "arrowstyle": "->",
        "lw": 2.0,
        "color": "darkblue",
    }

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    arrow_length = min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) * 0.08

    x_origin = x_lim[1] - (x_lim[1] - x_lim[0]) * 0.15
    y_origin = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.12

    dx_first = 0.0
    dy_first = -arrow_length

    dx_second = arrow_length
    dy_second = 0.0

    ax.annotate(
        "",
        xy=(x_origin + dx_first, y_origin + dy_first),
        xytext=(x_origin, y_origin),
        arrowprops=arrow_props,
    )
    ax.text(
        x_origin + dx_first * 1.3,
        y_origin + dy_first * 1.3,
        x_label,
        fontsize=12,
        color="darkblue",
        weight="bold",
        ha="center",
        va="center",
    )

    ax.annotate(
        "",
        xy=(x_origin + dx_second, y_origin + dy_second),
        xytext=(x_origin, y_origin),
        arrowprops=arrow_props,
    )
    ax.text(
        x_origin + dx_second * 1.5,
        y_origin + dy_second * 1.3,
        y_label,
        fontsize=12,
        color="darkblue",
        weight="bold",
        ha="center",
        va="center",
    )


def _add_mirror_statistics(ax, mirrors, x_pos, y_pos, diameter):
    """
    Add mirror statistics text to the plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add statistics to
    mirrors : Mirrors
        Mirrors object containing mirror information
    x_pos, y_pos : array-like
        Arrays of x and y positions
    diameter : float
        Mirror panel diameter
    """
    n_mirrors = mirrors.number_of_mirrors

    max_radius = np.sqrt(np.max(x_pos**2 + y_pos**2)) / 100.0
    mean_outer_edge_radius = (
        _calculate_mean_outer_edge_radius(x_pos, y_pos, diameter, mirrors.shape_type) / 100.0
    )

    panel_area = np.pi * (diameter / 200.0) ** 2
    if mirrors.shape_type in (1, 3):
        panel_area = 3 * np.sqrt(3) / 2 * (diameter / 200.0) ** 2
    elif mirrors.shape_type == 2:
        panel_area = (diameter / 100.0) ** 2

    total_area = n_mirrors * panel_area

    stats_text = (
        f"Number of mirrors: {n_mirrors}\n"
        f"Mirror diameter: {diameter:.1f} cm\n"
        f"Max radius: {max_radius:.2f} m\n"
        f"Mean outer edge radius: {mean_outer_edge_radius:.2f} m\n"
        f"Total surface area: {total_area:.2f} m²"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )


def _calculate_mean_outer_edge_radius(x_pos, y_pos, diameter, shape_type):
    """
    Calculate the mean radius of the outer edge of the mirror array.

    Parameters
    ----------
    x_pos, y_pos : array-like
        X and Y coordinates of mirror panel centers
    diameter : float
        Diameter of mirror panels
    shape_type : int
        Shape type (0: circular, 1/3: hexagonal, 2: square)

    Returns
    -------
    float
        Mean outer edge radius in cm
    """
    if shape_type == 0:
        radius_offset = diameter / 2
    elif shape_type in (1, 3):
        radius_offset = diameter / np.sqrt(3)
    else:
        radius_offset = diameter / np.sqrt(2)

    radii = np.sqrt(x_pos**2 + y_pos**2) + radius_offset
    return np.mean(radii)


def _read_ring_segmentation_data(data_file_path):
    """
    Read ring segmentation data from file.

    Parameters
    ----------
    data_file_path : Path
        Path to the segmentation data file

    Returns
    -------
    tuple
        (radii, phi0, nseg) - inner/outer radii, rotation angle, number of segments
    """
    radii = []
    phi0 = 0
    nseg = 6

    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#") and not line.startswith("%"):
                if line.lower().startswith("ring"):
                    parts = line.split()
                    nseg = int(parts[1].strip())
                    radii.append(float(parts[2].strip()))
                    radii.append(float(parts[3].strip()))
                    phi0 = float(parts[5].strip())

    return radii, phi0, nseg


def _plot_ring_structure(ax, theta, r, nseg, phi0, cmap, norm):
    """
    Plot the ring structure including boundaries and segments.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes to plot on
    theta : array-like
        Angular coordinates
    r : array-like
        Radial coordinates [inner_hole, inner_ring, outer_ring]
    nseg : int
        Number of segments
    phi0 : float
        Rotation angle in degrees
    cmap : matplotlib colormap
        Colormap for segments
    norm : matplotlib normalization
        Color normalization
    """
    linewidth = 2

    for i in range(r.shape[0]):
        ax.plot(theta, np.repeat(r[i], theta.shape), "-k", lw=linewidth)

    if nseg > 1:
        for i in range(nseg):
            theta_i = (i - 1) * 60 * np.pi / 180 + phi0 * np.pi / 180
            ax.plot([theta_i, theta_i], [r[1], r[2]], "-k", lw=linewidth)

    if phi0 == 0:
        ax.plot([30 * np.pi / 180, 30 * np.pi / 180], [r[1], r[2]], "--k", lw=1)

    r0 = r[1:3]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(nseg):
        theta0 = theta[i * 128 : i * 128 + 128] + phi0 * np.pi / 180
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2)) * 11
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)


def plot_mirror_ring_segmentation(data_file_path, telescope_model_name, parameter_type, title=None):
    """
    Plot mirror ring segmentation layout (for SCT-type telescopes).

    This function creates a polar visualization of mirror segments arranged in rings.

    Parameters
    ----------
    data_file_path : Path
        Path to the segmentation data file
    telescope_model_name : str
        Name/model of the telescope
    parameter_type : str
        Type of segmentation ("primary_mirror_segmentation" or "secondary_mirror_segmentation")
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    logger.info(f"Plotting ring {parameter_type} for {telescope_model_name}")

    radii, phi0, nseg = _read_ring_segmentation_data(data_file_path)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))

    cmap = mcolors.LinearSegmentedColormap.from_list("mirror_blue", ["#deebf7", "#3182bd"])
    norm = mcolors.Normalize(vmin=1, vmax=20)

    theta = np.linspace(0, 2 * np.pi, 128 * nseg)
    r = np.array([0, radii[0], radii[1]])

    _plot_ring_structure(ax, theta, r, nseg, phi0, cmap, norm)

    ax.set_ylim([r[0], r[2]])
    ax.set_yticklabels([])

    if r[1] > 20:
        plt.text(30 * np.pi / 180, (5 / 9) * r[1], f"{r[1]:.0f}")
    else:
        plt.text(225 * np.pi / 180, 3, f"{r[1]:.0f}")
    plt.text(30 * np.pi / 180, r[2] + 5, f"{r[2]:.0f} [cm]")

    if title:
        plt.subplots_adjust(top=0.85)
        ax.set_title(title, fontsize=18, y=1.10)
    plt.tight_layout()

    return fig


def _read_petal_segmentation_file(data_file_path):
    """
    Read petal/shape segmentation file.

    Parameters
    ----------
    data_file_path : Path
        Path to segmentation file

    Returns
    -------
    tuple
        (segments, shape_segments, segment_ids)
    """
    segments = []
    shape_segments = []
    segment_ids = []

    with open(data_file_path, encoding="utf-8") as f:
        current_segment_id = 0
        for line in f:
            line_stripped = line.strip()

            if "segment id" in line_stripped.lower():
                try:
                    current_segment_id = int(line_stripped.split()[-1])
                except (ValueError, IndexError):
                    pass
                continue

            if not line_stripped or line_stripped.startswith("#") or line_stripped.startswith("%"):
                continue

            if line_stripped.startswith("poly"):
                entries = line.split()
                angle = int(entries[2].strip())
                corners_x = [float(entries[i].strip()) for i in range(3, len(entries) - 1, 2)]
                corners_y = [float(entries[i].strip()) for i in range(4, len(entries), 2)]
                segments.append({"angle": angle, "x": corners_x, "y": corners_y})
                segment_ids.append(current_segment_id if current_segment_id > 0 else len(segments))
            elif any(
                line_stripped.lower().startswith(shape)
                for shape in ["hex", "square", "circular", "yhex"]
            ):
                entries = line.split()
                if len(entries) >= 5:
                    shape_type = entries[0].lower()
                    x = float(entries[2])
                    y = float(entries[3])
                    diam = float(entries[4])
                    rot = float(entries[5]) if len(entries) > 5 else 0.0

                    shape_segments.append(
                        {"shape": shape_type, "x": x, "y": y, "diameter": diam, "rotation": rot}
                    )
                    segment_ids.append(
                        current_segment_id if current_segment_id > 0 else len(shape_segments)
                    )

    return segments, shape_segments, segment_ids


def _create_polygon_patches(segments, segment_ids, ax):
    """
    Create patches for polygon segments.

    Parameters
    ----------
    segments : list
        List of polygon segment dictionaries
    segment_ids : list
        List of segment IDs
    ax : matplotlib.axes.Axes
        Axes to add text labels to

    Returns
    -------
    tuple
        (patches, maximum_radius)
    """
    patches = []
    maximum_radius = 0

    for i_seg, segment in enumerate(segments):
        angle_rad = np.deg2rad(segment["angle"])
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        rotated_x = []
        rotated_y = []
        for x, y in zip(segment["x"], segment["y"]):
            rot_x = x * cos_a - y * sin_a
            rot_y = x * sin_a + y * cos_a
            rotated_x.append(rot_x)
            rotated_y.append(rot_y)
            maximum_radius = max(maximum_radius, abs(rot_x), abs(rot_y))

        polygon = mpatches.Polygon(
            list(zip(rotated_x, rotated_y)),
            alpha=0.8,
            edgecolor="black",
            facecolor="dodgerblue",
        )
        patches.append(polygon)

        center_x = np.mean(rotated_x)
        center_y = np.mean(rotated_y)
        ax.text(
            center_x,
            center_y,
            str(segment_ids[i_seg]) if i_seg < len(segment_ids) else str(i_seg + 1),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    return patches, maximum_radius


def _create_shape_patches(shape_segments, segment_ids, segments, ax):
    """
    Create patches for shape segments (hex, square, circular).

    Parameters
    ----------
    shape_segments : list
        List of shape segment dictionaries
    segment_ids : list
        List of segment IDs
    segments : list
        List of polygon segments (for offset calculation)
    ax : matplotlib.axes.Axes
        Axes to add text labels to

    Returns
    -------
    tuple
        (patches, maximum_radius)
    """
    patches = []
    maximum_radius = 0

    for i_seg, segment in enumerate(shape_segments):
        x, y = segment["x"], segment["y"]
        diam = segment["diameter"]
        rot = segment["rotation"]
        shape = segment["shape"]

        maximum_radius = max(maximum_radius, abs(x) + diam / 2, abs(y) + diam / 2)

        if "hex" in shape:
            orientation = np.deg2rad(rot)
            patch = mpatches.RegularPolygon(
                (x, y),
                numVertices=6,
                radius=diam / np.sqrt(3),
                orientation=orientation,
                alpha=0.8,
                edgecolor="black",
                facecolor="dodgerblue",
            )
        elif "square" in shape:
            patch = mpatches.Rectangle(
                (x - diam / 2, y - diam / 2),
                width=diam,
                height=diam,
                angle=rot,
                alpha=0.8,
                edgecolor="black",
                facecolor="dodgerblue",
            )
        else:
            patch = mpatches.Circle(
                (x, y),
                radius=diam / 2,
                alpha=0.8,
                edgecolor="black",
                facecolor="dodgerblue",
            )

        patches.append(patch)

        idx = len(segments) + i_seg
        label = str(segment_ids[idx]) if idx < len(segment_ids) else str(i_seg + 1)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    return patches, maximum_radius


def plot_mirror_petal_segmentation(
    data_file_path, telescope_model_name, parameter_type, title=None
):
    """
    Plot mirror petal/polygon segmentation layout.

    This function creates a visualization of mirror segments defined as polygons.

    Parameters
    ----------
    data_file_path : Path
        Path to the segmentation data file
    telescope_model_name : str
        Name/model of the telescope
    parameter_type : str
        Type of segmentation ("primary_mirror_segmentation" or "secondary_mirror_segmentation")
    title : str, optional
        Plot title

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    logger.info(f"Plotting petal {parameter_type} for {telescope_model_name}")

    segments, shape_segments, segment_ids = _read_petal_segmentation_file(data_file_path)

    fig, ax = plt.subplots(figsize=(10, 10))

    all_patches = []
    maximum_radius = 0

    # Handle polygon segments
    patches, max_r = _create_polygon_patches(segments, segment_ids, ax)
    all_patches.extend(patches)
    maximum_radius = max(maximum_radius, max_r)

    # Handle shape segments
    patches, max_r = _create_shape_patches(shape_segments, segment_ids, segments, ax)
    all_patches.extend(patches)
    maximum_radius = max(maximum_radius, max_r)

    collection = PatchCollection(all_patches, match_original=True)
    ax.add_collection(collection)

    ax.set_aspect("equal")
    padding = maximum_radius * 0.1
    ax.set_xlim(-maximum_radius - padding, maximum_radius + padding)
    ax.set_ylim(-maximum_radius - padding, maximum_radius + padding)

    plt.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.xlabel("X position [cm]", fontsize=16)
    plt.ylabel("Y position [cm]", fontsize=16)

    if title:
        ax.set_title(title, fontsize=18, pad=20)
    plt.tick_params(axis="both", which="major", labelsize=14)

    # Count unique segment IDs if available
    total_segments = len(segments) + len(shape_segments)
    if segment_ids and total_segments > 0:
        unique_segments = len(set(segment_ids))
        stats_text = f"Number of segments: {unique_segments}"
    elif total_segments > 0:
        stats_text = f"Number of segments: {total_segments}"
    else:
        stats_text = "No segment data"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    return fig
