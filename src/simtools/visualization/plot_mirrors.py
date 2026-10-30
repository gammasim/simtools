#!/usr/bin/python3
"""Functions for plotting mirror panel layout information."""

import logging
from pathlib import Path

import numpy as np

from simtools.visualization import visualize
from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt

logger = logging.getLogger(__name__)
mcolors = lazy_module("matplotlib.colors")
mpatches = lazy_module("matplotlib.patches")
collections = lazy_module("matplotlib.collections")

PATCH_STYLE = {"alpha": 0.8, "edgecolor": "black", "facecolor": "dodgerblue"}
LABEL_STYLE = {"ha": "center", "va": "center", "fontsize": 10, "color": "white", "weight": "bold"}
STATS_BOX_STYLE = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
OBSERVER_TEXT = "for an observer facing the mirrors"


def plot(config, output_file, data_file_path, mirrors=None):
    """
    Plot mirror panel layout based on configuration.

    Parameters
    ----------
    config : dict
        Keys ``parameter`` (e.g., "mirror_list", "primary_mirror_segmentation") and
        ``telescope`` (e.g., "LSTN-01").
    output_file : str or Path
        Path where to save the plot (without extension).
    data_file_path : str or Path
        Exported mirror or segmentation file.
    mirrors : Mirrors, optional
        Parsed mirror data; required for mirror lists.
    """
    data_file_path = Path(data_file_path)
    parameter_type = config["parameter"]
    telescope = config["telescope"]

    if parameter_type in ("primary_mirror_segmentation", "secondary_mirror_segmentation"):
        plotters = {
            "ring": plot_mirror_ring_segmentation,
            "shape": plot_mirror_shape_segmentation,
            "standard": plot_mirror_segmentation,
        }
        fig = plotters[_detect_segmentation_type(data_file_path)](
            data_file_path=data_file_path,
            telescope_model_name=telescope,
            parameter_type=parameter_type,
        )
    else:
        if mirrors is None:
            raise ValueError("Mirror layout plotting requires parsed mirror data")
        fig = plot_mirror_layout(
            mirrors=mirrors, mirror_file_path=data_file_path, telescope_model_name=telescope
        )

    visualize.save_figure(fig, output_file)
    plt.close(fig)


def _rotate_coordinates_clockwise_90(x_pos, y_pos):
    """
    Rotate coordinates 90 degrees clockwise for the observer-facing view.

    Any transformation applied to one mirror list must be applied consistently to all
    primary mirror lists, otherwise sim_telarray mirror configurations are wrong.
    """
    return y_pos, -x_pos


def _detect_segmentation_type(data_file_path):
    """Return "ring", "shape" or "standard" from the first data line of the file."""
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


def plot_mirror_layout(mirrors, mirror_file_path, telescope_model_name):
    """
    Plot the mirror panel layout from a Mirrors object.

    Parameters
    ----------
    mirrors : Mirrors
        Mirror panel positions, diameters, focal lengths and shape types.
    mirror_file_path : Path or str
        Mirror list file (its header provides the statistics).
    telescope_model_name : str
        Telescope model name (e.g., "LSTN-01", "MSTN-01").

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    logger.info(f"Plotting mirror layout for {telescope_model_name}")
    fig, ax = plt.subplots(figsize=(10, 10))

    x_pos, y_pos = _rotate_coordinates_clockwise_90(
        mirrors.mirror_table["mirror_x"].to("cm").value,
        mirrors.mirror_table["mirror_y"].to("cm").value,
    )
    diameter = mirrors.mirror_diameter.to("cm").value
    focal_lengths = mirrors.mirror_table["focal_length"].to("cm").value
    mirror_ids = (
        mirrors.mirror_table["mirror_panel_id"]
        if "mirror_panel_id" in mirrors.mirror_table.colnames
        else list(range(len(x_pos)))
    )
    # MST mirrors are numbered from 1 at the bottom to N at the top
    if telescope_model_name and "MST" in telescope_model_name.upper():
        mirror_ids = [len(mirror_ids) - mid for mid in mirror_ids]

    _draw_panels(
        ax,
        x_pos,
        y_pos,
        diameter,
        mirrors.shape_type,
        focal_lengths,
        mirror_ids,
        cmap="viridis",
        linewidth=0.5,
        max_labels=20,
        colorbar_label="Focal length [cm]",
    )
    _add_mirror_statistics(ax, mirrors, mirror_file_path, x_pos, y_pos, diameter)
    return fig


def plot_mirror_segmentation(data_file_path, telescope_model_name, parameter_type):
    """
    Plot mirror segmentation layout from a standard segmentation file.

    Parameters
    ----------
    data_file_path : Path or str
        Segmentation file with positions, diameters, shape types and segment IDs.
    telescope_model_name : str
        Telescope model name (e.g., "LSTN-01").
    parameter_type : str
        Segmentation parameter (e.g., "primary_mirror_segmentation").

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    logger.info(f"Plotting {parameter_type} for {telescope_model_name}")
    data = _read_segmentation_file(data_file_path)
    fig, ax = plt.subplots(figsize=(10, 10))

    x_pos, y_pos = _rotate_coordinates_clockwise_90(data["x"], data["y"])
    diameter = data["diameter"]
    segment_ids = data["segment_ids"]
    _draw_panels(
        ax,
        x_pos,
        y_pos,
        diameter,
        data["shape_type"],
        segment_ids,
        segment_ids,
        cmap="tab20",
        linewidth=0.8,
        max_labels=30,
        colorbar_label="Segment ID",
    )
    _add_stats_box(
        ax,
        f"Number of segments: {len(x_pos)}\n"
        f"Number of segment groups: {len(set(segment_ids))}\n"
        f"Segment diameter: {diameter:.1f} cm",
    )
    return fig


def _draw_panels(
    ax,
    x_pos,
    y_pos,
    diameter,
    shape_type,
    colors,
    label_ids,
    *,
    cmap,
    linewidth,
    max_labels,
    colorbar_label,
):
    """Draw hexagonal panels coloured by ``colors``, with labels, axes and a colour bar."""
    patches = [
        _create_single_mirror_patch(x, y, diameter, shape_type) for x, y in zip(x_pos, y_pos)
    ]
    collection = collections.PatchCollection(
        patches, cmap=cmap, edgecolor="black", linewidth=linewidth
    )
    collection.set_array(np.array(list(colors)))
    ax.add_collection(collection)
    _add_mirror_labels(ax, x_pos, y_pos, label_ids, max_labels=max_labels)
    _configure_mirror_plot(ax, x_pos, y_pos)
    plt.colorbar(collection, ax=ax, pad=0.02).set_label(colorbar_label, fontsize=14)


def _split_segmentation_line(line):
    """Return the whitespace-separated fields of a data line, or None for comments and blanks."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    return parts if len(parts) >= 5 else None


def _parse_segment_id(parts):
    """Return the integer segment ID from the eighth column, or None if the column is absent."""
    if len(parts) < 8:
        return None
    seg_id_str = parts[7].split("=")[-1] if "=" in parts[7] else parts[7]
    return int("".join(filter(str.isdigit, seg_id_str)))


def _read_segmentation_file(data_file_path):
    """Read a standard segmentation file into positions, diameter, shape type and segment IDs."""
    x_pos, y_pos, segment_ids = [], [], []
    diameter = shape_type = None

    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            parts = _split_segmentation_line(line)
            if parts is None:
                continue
            try:
                x_pos.append(float(parts[0]))
                y_pos.append(float(parts[1]))
            except ValueError:
                continue
            diameter = float(parts[2]) if diameter is None else diameter
            shape_type = int(parts[4]) if shape_type is None else shape_type
            segment_id = _parse_segment_id(parts)
            segment_ids.append(len(segment_ids) if segment_id is None else segment_id)

    if len(x_pos) == 0:
        logger.warning(f"No valid numeric data found in segmentation file: {data_file_path}")

    return {
        "x": np.array(x_pos),
        "y": np.array(y_pos),
        "diameter": diameter if diameter is not None else 150.0,
        "shape_type": shape_type if shape_type is not None else 3,
        "segment_ids": segment_ids,
    }


def _create_single_mirror_patch(x, y, diameter, shape_type):
    """Create a hexagonal patch for a mirror panel, rotated to the observer-facing view."""
    base_orientation = 0 if shape_type == 1 else np.pi / 2
    # Rotate hexagon counter-clockwise to compensate for clockwise coordinate rotation
    orientation = base_orientation - np.pi / 2
    return mpatches.RegularPolygon(
        (x, y), numVertices=6, radius=diameter / np.sqrt(3), orientation=orientation
    )


def _add_mirror_labels(ax, x_pos, y_pos, mirror_ids, max_labels=20):
    """Add panel ID labels (at most ``max_labels``, sorted by ID)."""
    mirror_data = sorted(zip(mirror_ids, x_pos, y_pos), key=lambda item: item[0])
    for mid, x, y in mirror_data[:max_labels]:
        ax.text(x, y, str(mid), ha="center", va="center", fontsize=6, color="white", weight="bold")


def _configure_mirror_plot(ax, x_pos, y_pos):
    """Set aspect, limits, labels and observer note of a mirror panel plot."""
    ax.set_aspect("equal")

    if len(x_pos) == 0 or len(y_pos) == 0:
        logger.warning("No valid mirror data found for plotting")
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.text(0, 0, "No valid mirror data", ha="center", va="center", fontsize=14, color="red")
        return

    x_min, x_max = np.min(x_pos), np.max(x_pos)
    y_min, y_max = np.min(y_pos), np.max(y_pos)
    x_padding = max((x_max - x_min) * 0.15, abs(x_min) * 0.15, 1.0)
    y_padding = max((y_max - y_min) * 0.15, abs(y_min) * 0.15, 1.0)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    _decorate_panel_axes(ax)


def _decorate_panel_axes(ax):
    """Add axis labels, grid, tick sizes and the observer note."""
    plt.xlabel("X position [cm]", fontsize=14)
    plt.ylabel("Y position [cm]", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis="both", which="major", labelsize=12)
    _add_observer_text(ax)


def _add_observer_text(ax):
    """Add the observer note in the lower left corner."""
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


def _add_stats_box(ax, text):
    """Add a statistics box in the upper left corner."""
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=STATS_BOX_STYLE,
    )


def _extract_float_after_keyword(line, keyword):
    """Extract the first float after ``keyword`` in ``line``; None if not found."""
    if keyword not in line:
        return None
    try:
        part = line.split(keyword, 1)[1] if keyword == "=" else line.split(keyword)[-1]
        return float(part.strip().split()[0])
    except ValueError, IndexError:
        return None


def _read_mirror_file_metadata(mirror_file_path):
    """Read Rmax and total surface area from the mirror file header."""
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
    """Add the mirror statistics box (header values, else computed from the layout)."""
    n_mirrors = mirrors.number_of_mirrors
    metadata = _read_mirror_file_metadata(mirror_file_path)
    max_radius = metadata.get("rmax")
    total_area = metadata.get("total_surface_area")
    if max_radius is None:
        max_radius = np.sqrt(np.max(x_pos**2 + y_pos**2)) / 100.0
    if total_area is None:
        panel_area = 3 * np.sqrt(3) / 2 * (diameter / 200.0) ** 2
        total_area = n_mirrors * panel_area
    _add_stats_box(
        ax,
        f"Number of mirrors: {n_mirrors}\n"
        f"Mirror diameter: {diameter:.1f} cm\n"
        f"Max radius: {max_radius:.2f} m\n"
        f"Total surface area: {total_area:.2f} $m^{2}$",
    )


def _read_ring_segmentation_data(data_file_path):
    """Read ``ring <nseg> <rmin> <rmax> <dphi> <phi0>`` lines."""
    rings = []
    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith(("#", "%")) and line.lower().startswith("ring"):
                parts = [part.strip() for part in line.split()]
                rings.append(
                    {
                        "nseg": int(parts[1]),
                        "rmin": float(parts[2]),
                        "rmax": float(parts[3]),
                        "dphi": float(parts[4]),
                        "phi0": float(parts[5]),
                    }
                )
    return rings


def _plot_single_ring(ax, ring, cmap, color_index):
    """Fill the segments of one ring; 0.3 degree gaps separate neighbouring segments."""
    if ring["nseg"] <= 1:
        return
    dphi_rad = ring["dphi"] * np.pi / 180
    phi0_rad = ring["phi0"] * np.pi / 180
    gap_rad = 0.3 * np.pi / 180
    for i in range(ring["nseg"]):
        theta_i = i * dphi_rad + phi0_rad
        theta_seg = np.linspace(theta_i + gap_rad, theta_i + dphi_rad - gap_rad, 100)
        color = cmap((color_index + i % 10) / 20.0)
        ax.fill_between(theta_seg, ring["rmin"], ring["rmax"], color=color, alpha=0.8)


def _add_ring_radius_label(ax, angle, radius, label_text):
    """Add a radius label at the given angle and radius."""
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
    Plot mirror ring segmentation layout (polar view).

    Parameters
    ----------
    data_file_path : Path or str
        Segmentation file with ``ring <nseg> <rmin> <rmax> <dphi> <phi0>`` lines.
    telescope_model_name : str
        Telescope model name (e.g., "LSTN-01").
    parameter_type : str
        Segmentation parameter (e.g., "primary_mirror_segmentation").

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure object, or None if no ring data is found.
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
    theta_full = np.linspace(0, 2 * np.pi, 360)
    for ring in rings:
        for radius in (ring["rmin"], ring["rmax"]):
            ax.plot(theta_full, np.repeat(radius, 360), ":", color="gray", lw=0.8, alpha=0.5)
        for radius in (ring["rmin"], ring["rmax"]):
            _add_ring_radius_label(ax, label_angle, radius, f"{radius:.3f}")

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
        stats_text = "\n".join(
            f"Ring {i + 1} segments: {ring['nseg']}" for i, ring in enumerate(rings)
        )
    _add_stats_box(ax, stats_text)
    _add_observer_text(ax)
    plt.tight_layout()
    return fig


def _parse_segment_id_header(line):
    """Return the segment ID given on a ``segment id`` header line (0 if unreadable)."""
    try:
        return int(line.split()[-1])
    except ValueError, IndexError:
        return 0


def _read_shape_segmentation_file(data_file_path):
    """Read hex and yhex shape entries; returns ``(shape_segments, segment_ids)``."""
    shape_segments, segment_ids = [], []
    current_segment_id = 0

    with open(data_file_path, encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if "segment id" in line_stripped.lower():
                current_segment_id = _parse_segment_id_header(line_stripped)
                continue
            if not line_stripped or line_stripped.startswith(("#", "%")):
                continue
            entries = line_stripped.split()
            if line_stripped.lower().startswith(("hex", "yhex")) and len(entries) >= 5:
                shape_segments.append(
                    {
                        "x": float(entries[2]),
                        "y": float(entries[3]),
                        "diameter": float(entries[4]),
                        "rotation": float(entries[5]) if len(entries) > 5 else 0.0,
                    }
                )
                segment_ids.append(
                    current_segment_id if current_segment_id > 0 else len(shape_segments)
                )

    return shape_segments, segment_ids


def _create_shape_patches(ax, shape_segments, segment_ids):
    """Create hexagon patches and their labels; returns ``(patches, maximum_radius)``."""
    patches, maximum_radius = [], 0
    for i_seg, seg in enumerate(shape_segments):
        x, y, diam, rot = seg["x"], seg["y"], seg["diameter"], seg["rotation"]
        maximum_radius = max(maximum_radius, abs(x) + diam / 2, abs(y) + diam / 2)
        patches.append(
            mpatches.RegularPolygon(
                (x, y),
                numVertices=6,
                radius=diam / np.sqrt(3),
                orientation=np.deg2rad(rot),
                **PATCH_STYLE,
            )
        )
        label = segment_ids[i_seg] if i_seg < len(segment_ids) else i_seg + 1
        ax.text(x, y, str(label), **LABEL_STYLE)
    return patches, maximum_radius


def plot_mirror_shape_segmentation(data_file_path, telescope_model_name, parameter_type):
    """
    Plot mirror shape segmentation layout (explicit hex and yhex shapes).

    Parameters
    ----------
    data_file_path : Path or str
        Segmentation file with explicit shape definitions (hex, yhex).
    telescope_model_name : str
        Telescope model name (e.g., "LSTN-01", "MSTN-design").
    parameter_type : str
        Segmentation parameter (e.g., "primary_mirror_segmentation").

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    logger.info(f"Plotting shape {parameter_type} for {telescope_model_name}")
    shape_segments, segment_ids = _read_shape_segmentation_file(data_file_path)
    for seg in shape_segments:
        seg["x"], seg["y"] = _rotate_coordinates_clockwise_90(seg["x"], seg["y"])
        seg["rotation"] = seg["rotation"] - 90

    fig, ax = plt.subplots(figsize=(10, 10))
    all_patches, maximum_radius = _create_shape_patches(ax, shape_segments, segment_ids)
    ax.add_collection(collections.PatchCollection(all_patches, match_original=True))

    ax.set_aspect("equal")
    padding = maximum_radius * 0.1 if maximum_radius > 0 else 100
    ax.set_xlim(-maximum_radius - padding, maximum_radius + padding)
    ax.set_ylim(-maximum_radius - padding, maximum_radius + padding)
    _decorate_panel_axes(ax)

    if segment_ids and shape_segments:
        stats_text = f"Number of segments: {len(set(segment_ids))}"
    elif shape_segments:
        stats_text = f"Number of segments: {len(shape_segments)}"
    else:
        stats_text = "No segment data"
    _add_stats_box(ax, stats_text)
    return fig
