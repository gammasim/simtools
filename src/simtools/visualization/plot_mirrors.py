#!/usr/bin/python3
"""Functions for plotting mirror panel layout information."""

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from simtools.io import io_handler
from simtools.model.mirrors import Mirrors
from simtools.model.telescope_model import TelescopeModel
from simtools.visualization import visualize

logger = logging.getLogger(__name__)

PATCH_STYLE = {"alpha": 0.8, "edgecolor": "black", "facecolor": "dodgerblue"}
LABEL_STYLE = {"ha": "center", "va": "center", "fontsize": 10, "color": "white", "weight": "bold"}
STATS_BOX_STYLE = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
OBSERVER_TEXT = "for an observer facing the mirrors"


def _rotate_coordinates_clockwise_90(x_pos, y_pos):
    """
    Rotate coordinates 90 degrees clockwise for observer-facing view.

    Plots are rotated by 90 degrees clockwise to present the point of view of a
    person standing on the camera platform.

    Important: Any transformation applied to one mirror list must be applied consistently to
    all (primary) mirror lists. Inconsistent transformations would result in incorrect mirror
    configurations when running sim_telarray simulations.
    """
    return y_pos, -x_pos


def _detect_segmentation_type(data_file_path):
    """
    Detect the type of segmentation file (ring, shape, or standard).

    Parameters
    ----------
    data_file_path : Path
        Path to the segmentation data file

    Returns
    -------
    str
        One of "ring", "shape", or "standard"
    """
    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            line_lower = line.strip().lower()
            if line_lower.startswith("#") or not line_lower:
                continue
            if line_lower.startswith("ring"):
                return "ring"
            if line_lower.startswith(("hex", "yhex")):
                return "shape"
    return "standard"


def plot(config, output_file):
    """
    Plot mirror panel layout based on configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - parameter: str, parameter name (e.g., "mirror_list", "primary_mirror_segmentation")
        - site: str, site name (e.g., "North", "South")
        - telescope: str, telescope name (e.g., "LSTN-01")
        - parameter_version: str, optional, parameter version
        - model_version: str, optional, model version
        - title: str, optional, plot title
    output_file : str or Path
        Path where to save the plot (without extension)

    Returns
    -------
    None
        The function saves the plot to the specified output file.
    """
    tel_model = TelescopeModel(
        site=config["site"],
        telescope_name=config["telescope"],
        model_version=config.get("model_version"),
        ignore_software_version=True,
    )

    output_path = io_handler.IOHandler().get_output_directory()

    parameter_name = config["parameter"]
    parameter_value = tel_model.get_parameter_value(parameter_name)
    tel_model.export_model_files(destination_path=output_path)

    mirror_file = parameter_value
    data_file_path = Path(output_path / mirror_file)

    parameter_type = config["parameter"]

    if parameter_type in ("primary_mirror_segmentation", "secondary_mirror_segmentation"):
        segmentation_type = _detect_segmentation_type(data_file_path)

        if segmentation_type == "ring":
            fig = plot_mirror_ring_segmentation(
                data_file_path=data_file_path,
                telescope_model_name=config["telescope"],
                parameter_type=parameter_type,
            )
        elif segmentation_type == "shape":
            fig = plot_mirror_shape_segmentation(
                data_file_path=data_file_path,
                telescope_model_name=config["telescope"],
                parameter_type=parameter_type,
            )
        else:
            fig = plot_mirror_segmentation(
                data_file_path=data_file_path,
                telescope_model_name=config["telescope"],
                parameter_type=parameter_type,
            )
    else:
        mirrors = Mirrors(mirror_list_file=data_file_path)
        fig = plot_mirror_layout(
            mirrors=mirrors,
            mirror_file_path=data_file_path,
            telescope_model_name=config["telescope"],
        )

    visualize.save_figure(fig, output_file)
    plt.close(fig)


def plot_mirror_layout(mirrors, mirror_file_path, telescope_model_name):
    """
    Plot the mirror panel layout from a Mirrors object.

    Parameters
    ----------
    mirrors : Mirrors
        Mirrors object containing mirror panel data including positions,
        diameters, focal lengths, and shape types
    mirror_file_path : Path or str
        Path to the mirror list file
    telescope_model_name : str
        Name of the telescope model (e.g., "LSTN-01", "MSTN-01")

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    logger.info(f"Plotting mirror layout for {telescope_model_name}")

    fig, ax = plt.subplots(figsize=(10, 10))

    x_pos = mirrors.mirror_table["mirror_x"].to("cm").value
    y_pos = mirrors.mirror_table["mirror_y"].to("cm").value
    x_pos, y_pos = _rotate_coordinates_clockwise_90(x_pos, y_pos)
    diameter = mirrors.mirror_diameter.to("cm").value
    shape_type = mirrors.shape_type
    focal_lengths = mirrors.mirror_table["focal_length"].to("cm").value

    mirror_ids = (
        mirrors.mirror_table["mirror_panel_id"]
        if "mirror_panel_id" in mirrors.mirror_table.colnames
        else list(range(len(x_pos)))
    )

    # MST mirrors are numbered from 1 at the bottom to N at the top
    if telescope_model_name and "MST" in telescope_model_name.upper():
        n_mirrors = len(mirror_ids)
        mirror_ids = [n_mirrors - mid for mid in mirror_ids]

    patches, colors = _create_mirror_patches(x_pos, y_pos, diameter, shape_type, focal_lengths)

    collection = PatchCollection(
        patches,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.5,
    )
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    _add_mirror_labels(ax, x_pos, y_pos, mirror_ids, max_labels=20)

    _configure_mirror_plot(ax, x_pos, y_pos)

    cbar = plt.colorbar(collection, ax=ax, pad=0.02)
    cbar.set_label("Focal length [cm]", fontsize=14)

    _add_mirror_statistics(ax, mirrors, mirror_file_path, x_pos, y_pos, diameter)

    return fig


def plot_mirror_segmentation(data_file_path, telescope_model_name, parameter_type):
    """
    Plot mirror segmentation layout from a segmentation file.

    Parameters
    ----------
    data_file_path : Path or str
        Path to the segmentation data file containing mirror segment positions,
        diameters, and shape types in standard numeric format
    telescope_model_name : str
        Name of the telescope model (e.g., "LSTN-01", "MSTN-01")
    parameter_type : str
        Type of segmentation parameter (e.g., "primary_mirror_segmentation",
        "secondary_mirror_segmentation")

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    logger.info(f"Plotting {parameter_type} for {telescope_model_name}")

    segmentation_data = _read_segmentation_file(data_file_path)

    fig, ax = plt.subplots(figsize=(10, 10))

    x_pos = segmentation_data["x"]
    y_pos = segmentation_data["y"]
    x_pos, y_pos = _rotate_coordinates_clockwise_90(x_pos, y_pos)
    diameter = segmentation_data["diameter"]
    shape_type = segmentation_data["shape_type"]
    segment_ids = segmentation_data["segment_ids"]

    patches, colors = _create_mirror_patches(x_pos, y_pos, diameter, shape_type, segment_ids)

    collection = PatchCollection(
        patches,
        cmap="tab20",
        edgecolor="black",
        linewidth=0.8,
    )
    collection.set_array(np.array(colors))
    ax.add_collection(collection)

    _add_mirror_labels(ax, x_pos, y_pos, segment_ids, max_labels=30)

    _configure_mirror_plot(ax, x_pos, y_pos)

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
        bbox=STATS_BOX_STYLE,
    )

    return fig


def _create_mirror_patches(x_pos, y_pos, diameter, shape_type, color_values):
    """Create matplotlib patches for mirror panels or segments."""
    patches = [
        _create_single_mirror_patch(x, y, diameter, shape_type) for x, y in zip(x_pos, y_pos)
    ]
    return patches, list(color_values)


def _read_segmentation_file(data_file_path):
    """Read mirror segmentation file and extract segment information."""
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


def _create_single_mirror_patch(x, y, diameter, shape_type):
    """Create a single matplotlib patch for a mirror panel (hexagonal only)."""
    base_orientation = 0 if shape_type == 1 else np.pi / 2
    # Rotate hexagon counter-clockwise to compensate for clockwise coordinate rotation
    orientation = base_orientation - np.pi / 2

    return mpatches.RegularPolygon(
        (x, y),
        numVertices=6,
        radius=diameter / np.sqrt(3),
        orientation=orientation,
    )


def _add_mirror_labels(ax, x_pos, y_pos, mirror_ids, max_labels=20):
    """Add mirror panel ID labels to the plot."""
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


def _configure_mirror_plot(ax, x_pos, y_pos):
    """Add titles, labels, and limits."""
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

    plt.xlabel("X position [cm]", fontsize=14)
    plt.ylabel("Y position [cm]", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", which="major", labelsize=12)

    ax.text(
        0.02,
        0.02,
        OBSERVER_TEXT,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        style="italic",
        color="black",
    )


def _extract_float_after_keyword(line, keyword):
    """Extract first float value after keyword in line."""
    if keyword not in line:
        return None
    try:
        part = line.split(keyword, 1)[1] if keyword == "=" else line.split(keyword)[-1]
        return float(part.strip().split()[0])
    except (ValueError, IndexError):
        return None


def _read_mirror_file_metadata(mirror_file_path):
    """Read metadata from mirror .dat file header (Rmax and total surface area)."""
    metadata = {}
    patterns = [
        (("Total surface area:", "Total mirror surface area:"), ":", "total_surface_area"),
        (("Rmax =", "Rmax="), "=", "rmax"),
        (("mirrors are inside a radius of",), "of", "rmax"),
    ]

    try:
        with open(mirror_file_path, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                for triggers, keyword, key in patterns:
                    if key not in metadata and any(t in line for t in triggers):
                        if (val := _extract_float_after_keyword(line, keyword)) is not None:
                            metadata[key] = val
                            break
    except OSError as e:
        logger.warning(f"Could not read mirror file metadata: {e}")

    return metadata


def _add_mirror_statistics(ax, mirrors, mirror_file_path, x_pos, y_pos, diameter):
    """Add mirror statistics text to the plot."""
    n_mirrors = mirrors.number_of_mirrors

    metadata = _read_mirror_file_metadata(mirror_file_path)
    max_radius = metadata.get("rmax")
    total_area = metadata.get("total_surface_area")

    # Calculate values if not available from file
    if max_radius is None:
        max_radius = np.sqrt(np.max(x_pos**2 + y_pos**2)) / 100.0

    if total_area is None:
        panel_area = 3 * np.sqrt(3) / 2 * (diameter / 200.0) ** 2
        total_area = n_mirrors * panel_area

    stats_text = (
        f"Number of mirrors: {n_mirrors}\n"
        f"Mirror diameter: {diameter:.1f} cm\n"
        f"Max radius: {max_radius:.2f} m\n"
        f"Total surface area: {total_area:.2f} $m^{2}$"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=STATS_BOX_STYLE,
    )


def _read_ring_segmentation_data(data_file_path):
    """Read ring segmentation data from file."""
    rings = []

    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#") and not line.startswith("%"):
                if line.lower().startswith("ring"):
                    parts = line.split()
                    rings.append(
                        {
                            "nseg": int(parts[1].strip()),
                            "rmin": float(parts[2].strip()),
                            "rmax": float(parts[3].strip()),
                            "dphi": float(parts[4].strip()),
                            "phi0": float(parts[5].strip()),
                        }
                    )

    return rings


def _plot_single_ring(ax, ring, cmap, color_index):
    """Plot a single ring with its segments."""
    rmin, rmax = ring["rmin"], ring["rmax"]
    nseg, phi0 = ring["nseg"], ring["phi0"]
    dphi = ring["dphi"]

    # Angular gap between segments (in degrees) - represents the physical gaps
    angular_gap = 0.3  # degrees

    if nseg > 1:
        dphi_rad = dphi * np.pi / 180
        phi0_rad = phi0 * np.pi / 180
        gap_rad = angular_gap * np.pi / 180

        for i in range(nseg):
            theta_i = i * dphi_rad + phi0_rad

            # Fill segment with small gap on each side
            n_theta = 100
            theta_seg = np.linspace(theta_i + gap_rad, theta_i + dphi_rad - gap_rad, n_theta)

            color_value = (color_index + i % 10) / 20.0  # Normalize to 0-1
            color = cmap(color_value)
            ax.fill_between(theta_seg, rmin, rmax, color=color, alpha=0.8)


def _add_ring_radius_label(ax, angle, radius, label_text):
    """Add a radius label at the specified angle and radius."""
    ax.text(
        angle,
        radius,
        label_text,
        ha="center",
        va="center",
        fontsize=9,
        color="red",
        weight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )


def plot_mirror_ring_segmentation(data_file_path, telescope_model_name, parameter_type):
    """
    Plot mirror ring segmentation layout.

    Parameters
    ----------
    data_file_path : Path or str
        Path to the segmentation data file containing ring definitions with format:
        ring <nseg> <rmin> <rmax> <dphi> <phi0>
    telescope_model_name : str
        Name of the telescope model (e.g., "LSTN-01", "MSTN-01")
    parameter_type : str
        Type of segmentation parameter (e.g., "primary_mirror_segmentation",
        "secondary_mirror_segmentation")

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure object, or None if no ring data found
    """
    logger.info(f"Plotting ring {parameter_type} for {telescope_model_name}")

    rings = _read_ring_segmentation_data(data_file_path)

    if not rings:
        logger.warning(f"No ring data found in {data_file_path}")
        return None

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))

    cmap = mcolors.LinearSegmentedColormap.from_list("mirror_blue", ["#deebf7", "#3182bd"])

    for i, ring in enumerate(rings):
        _plot_single_ring(ax, ring, cmap, color_index=i * 10)

    max_radius = max(ring["rmax"] for ring in rings)
    label_padding = max_radius * 0.04
    ax.set_ylim([0, max_radius + label_padding])
    ax.set_yticklabels([])
    ax.set_rgrids([])
    ax.spines["polar"].set_visible(False)

    label_angle = 30 * np.pi / 180

    for ring in rings:
        theta_full = np.linspace(0, 2 * np.pi, 360)
        ax.plot(
            theta_full,
            np.repeat(ring["rmin"], len(theta_full)),
            ":",
            color="gray",
            lw=0.8,
            alpha=0.5,
        )
        ax.plot(
            theta_full,
            np.repeat(ring["rmax"], len(theta_full)),
            ":",
            color="gray",
            lw=0.8,
            alpha=0.5,
        )

        _add_ring_radius_label(ax, label_angle, ring["rmin"], f"{ring['rmin']:.3f}")
        _add_ring_radius_label(ax, label_angle, ring["rmax"], f"{ring['rmax']:.3f}")

    ax.text(
        label_angle,
        max_radius + label_padding * 2.5,
        "[cm]",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color="red",
    )

    if len(rings) == 2:
        stats_text = (
            f"Inner ring segments: {rings[0]['nseg']}\nOuter ring segments: {rings[1]['nseg']}"
        )
    else:
        stats_lines = [f"Ring {i + 1} segments: {ring['nseg']}" for i, ring in enumerate(rings)]
        stats_text = "\n".join(stats_lines)

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=STATS_BOX_STYLE,
    )

    ax.text(
        0.02,
        0.02,
        OBSERVER_TEXT,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        style="italic",
        color="black",
    )

    plt.tight_layout()

    return fig


def _parse_segment_id_line(line_stripped):
    """Extract segment ID from a line if it contains segment ID information."""
    try:
        return int(line_stripped.split()[-1])
    except (ValueError, IndexError):
        return 0


def _is_skippable_line(line_stripped):
    """Check if line should be skipped (empty or comment)."""
    return not line_stripped or line_stripped.startswith(("#", "%"))


def _parse_shape_line(line_stripped, shape_segments, segment_ids, current_segment_id):
    """Parse and append a single shape segmentation line."""
    entries = line_stripped.split()

    if any(line_stripped.lower().startswith(s) for s in ["hex", "yhex"]) and len(entries) >= 5:
        shape_segments.append(
            {
                "shape": entries[0].lower(),
                "x": float(entries[2]),
                "y": float(entries[3]),
                "diameter": float(entries[4]),
                "rotation": float(entries[5]) if len(entries) > 5 else 0.0,
            }
        )
        segment_ids.append(current_segment_id if current_segment_id > 0 else len(shape_segments))


def _read_shape_segmentation_file(data_file_path):
    """
    Read shape segmentation file.

    Parameters
    ----------
    data_file_path : Path
        Path to segmentation file

    Returns
    -------
    tuple
        (shape_segments, segment_ids)
    """
    shape_segments, segment_ids = [], []
    current_segment_id = 0

    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()

            if "segment id" in line_stripped.lower():
                current_segment_id = _parse_segment_id_line(line_stripped)
            elif not _is_skippable_line(line_stripped):
                _parse_shape_line(line_stripped, shape_segments, segment_ids, current_segment_id)

    return shape_segments, segment_ids


def _add_segment_label(ax, x, y, label):
    """Add a label at the specified position."""
    ax.text(x, y, str(label), **LABEL_STYLE)


def _create_shape_patches(ax, shape_segments, segment_ids):
    """
    Create patches for shape segments (hexagons).

    Parameters
    ----------
    shape_segments : list
        List of shape segment dictionaries
    segment_ids : list
        List of segment IDs
    ax : matplotlib.axes.Axes
        Axes to add text labels to

    Returns
    -------
    tuple
        (patches, maximum_radius)
    """
    patches, maximum_radius = [], 0

    for i_seg, seg in enumerate(shape_segments):
        x, y, diam, rot = seg["x"], seg["y"], seg["diameter"], seg["rotation"]
        maximum_radius = max(maximum_radius, abs(x) + diam / 2, abs(y) + diam / 2)

        patch = mpatches.RegularPolygon(
            (x, y),
            numVertices=6,
            radius=diam / np.sqrt(3),
            orientation=np.deg2rad(rot),
            **PATCH_STYLE,
        )

        patches.append(patch)
        label = segment_ids[i_seg] if i_seg < len(segment_ids) else i_seg + 1
        _add_segment_label(ax, x, y, label)

    return patches, maximum_radius


def plot_mirror_shape_segmentation(data_file_path, telescope_model_name, parameter_type):
    """
    Plot mirror shape segmentation layout.

    Parameters
    ----------
    data_file_path : Path or str
        Path to the segmentation data file containing explicit shape definitions
        (hex, yhex) with positions, diameters, and rotations
    telescope_model_name : str
        Name of the telescope model (e.g., "LSTN-01", "MSTN-design")
    parameter_type : str
        Type of segmentation parameter (e.g., "primary_mirror_segmentation",
        "secondary_mirror_segmentation")

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    logger.info(f"Plotting shape {parameter_type} for {telescope_model_name}")

    shape_segments, segment_ids = _read_shape_segmentation_file(data_file_path)

    for seg in shape_segments:
        seg["x"], seg["y"] = _rotate_coordinates_clockwise_90(seg["x"], seg["y"])
        seg["rotation"] = seg["rotation"] - 90

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create patches for shape segments
    all_patches, maximum_radius = _create_shape_patches(ax, shape_segments, segment_ids)

    collection = PatchCollection(all_patches, match_original=True)
    ax.add_collection(collection)

    ax.set_aspect("equal")
    padding = maximum_radius * 0.1 if maximum_radius > 0 else 100
    ax.set_xlim(-maximum_radius - padding, maximum_radius + padding)
    ax.set_ylim(-maximum_radius - padding, maximum_radius + padding)

    plt.xlabel("X position [cm]", fontsize=14)
    plt.ylabel("Y position [cm]", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", which="major", labelsize=12)

    ax.text(
        0.02,
        0.02,
        OBSERVER_TEXT,
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        style="italic",
        color="black",
    )

    total_segments = len(shape_segments)
    if segment_ids and total_segments > 0:
        stats_text = f"Number of segments: {len(set(segment_ids))}"
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
        bbox=STATS_BOX_STYLE,
    )

    return fig
