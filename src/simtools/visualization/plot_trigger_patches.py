"""Plot sim_telarray trigger-patch configuration and simulation results."""

import json
import textwrap

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator

from simtools.camera.trace_analysis import get_trace_data
from simtools.simtel import trigger_patch_validation as validation
from simtools.visualization.camera_plot_utils import pixel_shape
from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_camera import plot_pixel_layout_with_image

spatial = lazy_module("scipy.spatial")

PIXEL_IDS = "pix_id"
ADC_SAMPLES = "adc_samples"
MC_PHOTOELECTRONS = "photoelectrons"
PHOTOELECTRON_PIXELS = "pe_pixels"
NONE_RECORDED = "none recorded"
NO_MATCHING_GROUPS = "none"
EQUAL_ASPECT = "equal"
CONSTRAINED_LAYOUT = "constrained"
GROUP_COUNT = "groups"
GROUPS_SAMPLED = "groups_sampled"
MAPPING_MISMATCHES = "illumination_mapping_mismatches"
PATCH_COLORS = ("#b2182b", "#ef3b2c", "#fb8d7a")
PATCH_LINESTYLES = ("solid", "dashed", "dotted", "dashdot")
PATCH_ID_LABEL = "Patch ID"
CONFIGURED_PATCHES_LABEL = "Configured patches"
TELESCOPE_TRIGGERED_LABEL = "Telescope triggered"


def write_report(
    output_file, camera, groups, rows, summary, events, telescope_id, trigger_groups, input_patterns
):
    """Render trigger-patch summaries and sampled event pages into a PDF.

    Parameters
    ----------
    output_file : str or pathlib.Path
        Destination PDF path.
    camera : simtools.model.camera.Camera
        Camera geometry used for the pixel maps.
    groups : sequence of dict
        Per-patch definitions and trial counts from trigger-patch validation.
    rows : sequence of dict
        Accounted event rows, including JSON-encoded photoelectron pixel IDs.
    summary : dict
        Aggregate trial counts and illumination mapping results.
    events : iterable of dict
        Sampled sim_telarray events to display after the three summary pages.
    telescope_id : int
        Telescope identifier in the event stream.
    trigger_groups : sequence of set
        Pixel membership indexed by native trigger-sector ID.
    input_patterns : sequence of sequence of int
        Illuminated pixel IDs indexed by shower number minus one.
    """
    with PdfPages(output_file) as pdf:
        _plot_configuration_summary(pdf, camera, groups)
        _plot_inspection_summary(pdf, camera, rows, summary)
        _plot_patch_efficiency(pdf, groups)
        _plot_trigger_events(events, telescope_id, camera, pdf, trigger_groups, input_patterns)


def _plot_configuration_summary(pdf, camera, groups):
    """Show configured patch coverage independently of illumination."""
    coverage = _pixel_coverage(camera, (json.loads(group[validation.MEMBERS]) for group in groups))
    figure, axis = plt.subplots(figsize=(11, 8), layout=CONSTRAINED_LAYOUT)
    try:
        plot_pixel_layout_with_image(
            camera, image=coverage, ax=axis, color_bar_label="Configured patches per pixel"
        )
        axis.set_aspect(EQUAL_ASPECT, adjustable="box")
        axis.set_title(f"{camera.telescope_name}: patch configuration")
        pdf.savefig(figure)
    finally:
        plt.close(figure)


def _plot_patch_efficiency(pdf, groups):
    """Show observed trigger fractions and the underlying per-patch trial counts."""
    ids = [group[validation.GROUP_ID] for group in groups]
    figure, axes = plt.subplots(nrows=2, figsize=(11, 8), layout=CONSTRAINED_LAYOUT)
    try:
        axes[0].plot(ids, _observed_patch_efficiency(groups), "o-", markersize=2.5)
        axes[0].set(
            xlabel=PATCH_ID_LABEL,
            ylabel="Intended patch fired / recorded trials",
            ylim=(-0.05, 1.05),
            title="Observed patch trigger fraction (untested patches omitted)",
        )
        for key, label in (
            (validation.PLANNED, "Planned trials"),
            (validation.OBSERVED, "Recorded trials"),
            (validation.TRIGGERED, TELESCOPE_TRIGGERED_LABEL),
            (validation.INTENDED_SECTOR_FIRED, "Intended patch fired"),
        ):
            axes[1].plot(ids, [group[key] for group in groups], label=label)
        axes[1].set(xlabel=PATCH_ID_LABEL, ylabel="Trials", title="Per-patch trial outcomes")
        axes[1].set_ylim(bottom=0)
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].legend()
        pdf.savefig(figure)
    finally:
        plt.close(figure)


def _plot_inspection_summary(pdf, camera, rows, summary):
    """Summarize simulated photoelectron arrivals and trial outcomes on one page."""
    coverage = _pixel_coverage(camera, (json.loads(row[PHOTOELECTRON_PIXELS]) for row in rows))
    figure, (axis, text_axis) = plt.subplots(ncols=2, figsize=(11, 8))
    try:
        plot_pixel_layout_with_image(
            camera, image=coverage, ax=axis, color_bar_label="Events with detected photoelectrons"
        )
        axis.set_aspect(EQUAL_ASPECT, adjustable="box")
        axis.set_title(f"{camera.telescope_name}: photoelectron arrivals")
        text_axis.axis("off")
        displayed = (
            ("Trials planned", validation.PLANNED),
            ("Trials recorded", validation.OBSERVED),
            (TELESCOPE_TRIGGERED_LABEL, validation.TRIGGERED),
            ("Telescope did not trigger", validation.NON_TRIGGERED),
            ("Missing trials", validation.MISSING),
            (CONFIGURED_PATCHES_LABEL, GROUP_COUNT),
            ("Patches tested", GROUPS_SAMPLED),
            ("Off-target illumination trials", MAPPING_MISMATCHES),
        )
        lines = [f"{label}: {summary[key]}" for label, key in displayed]
        lines.append("\nMap: photoelectron arrivals, not all pixels in fired patches.")
        text_axis.text(
            0,
            1,
            "\n".join(lines),
            va="top",
            wrap=True,
            transform=text_axis.transAxes,
            fontsize=9,
        )
        pdf.savefig(figure)
    finally:
        plt.close(figure)


def _plot_trigger_events(events, telescope_id, camera, pdf, trigger_groups, input_patterns):
    """Plot one page per recorded telescope event without accumulating images."""
    page_count = 0
    for event in events:
        telescope_event = event.get(validation.TELESCOPE_EVENTS, {}).get(telescope_id, {})
        image, image_label = _event_image(event, telescope_event, camera)
        input_pixels, pe_pixels, fired_groups, matching_groups = _trigger_event_context(
            event, telescope_event, telescope_id, trigger_groups, input_patterns
        )
        figure, (axis, details_axis) = plt.subplots(
            ncols=2, figsize=(11, 8), gridspec_kw={"width_ratios": [4, 1.6]}
        )
        try:
            plot_pixel_layout_with_image(camera, image=image, ax=axis, color_bar_label=image_label)
            axis.set_aspect(EQUAL_ASPECT, adjustable="box")
            _overlay_trigger_groups(axis, camera, trigger_groups, fired_groups, input_pixels)
            details_axis.axis("off")
            details_axis.text(
                0,
                1,
                "Illuminated pixels\n"
                f"{textwrap.fill(_format_ids(input_pixels), width=30)}\n\n"
                "Patches containing all illuminated pixels\n"
                f"{_format_ids(matching_groups, empty_label=NO_MATCHING_GROUPS)}\n\n"
                "Pixels with simulated photoelectrons\n"
                f"{textwrap.fill(_format_ids(pe_pixels), width=30)}\n\n"
                "Fired patches (sim_telarray)\n"
                f"{_format_ids(fired_groups)}\n\n"
                "Cyan rings: illuminated pixels\nRed shades: fired patches (see legend)",
                va="top",
                transform=details_axis.transAxes,
                wrap=True,
            )
            axis.set_title(
                f"{camera.telescope_name}: event {event[validation.EVENT_ID]}\n"
                f"{textwrap.fill(f'Illuminated pixels: {_format_ids(input_pixels)}', width=65)}"
            )
            pdf.savefig(figure)
        finally:
            plt.close(figure)
        page_count += 1
    return page_count


def _pixel_coverage(camera, pixel_lists):
    """Count occurrences of each camera pixel across membership or arrival lists."""
    coverage = np.zeros(len(camera.pixels[PIXEL_IDS]))
    pixel_indices = {int(pixel): index for index, pixel in enumerate(camera.pixels[PIXEL_IDS])}
    for pixels in pixel_lists:
        for pixel in pixels:
            coverage[pixel_indices[pixel]] += 1
    return coverage


def _observed_patch_efficiency(groups):
    """Return intended-patch firing fractions, leaving unobserved groups undefined."""
    observed = np.array([group[validation.OBSERVED] for group in groups])
    fired = np.array([group[validation.INTENDED_SECTOR_FIRED] for group in groups])
    return np.divide(fired, observed, out=np.full(len(groups), np.nan), where=observed > 0)


def _trigger_event_context(event, telescope_event, telescope_id, trigger_groups, input_patterns):
    """Return input pixels, photoelectron pixels, and fired sector IDs for an event."""
    shower_number = event.get(validation.MC_EVENT, {}).get(validation.SHOWER_NUMBER)
    pattern_index = shower_number - 1 if shower_number is not None else -1
    input_pixels = input_patterns[pattern_index] if 0 <= pattern_index < len(input_patterns) else []
    photoelectrons = event.get(MC_PHOTOELECTRONS, {})
    pixel_data = photoelectrons.get(telescope_id - 1)
    if pixel_data is None and len(photoelectrons) == 1:
        pixel_data = next(iter(photoelectrons.values()))
    source_pixels = (
        np.flatnonzero(pixel_data[MC_PHOTOELECTRONS] > 0).tolist()
        if pixel_data and MC_PHOTOELECTRONS in pixel_data
        else []
    )
    header = telescope_event.get(validation.HEADER, {})
    fired_groups = [int(group) for group in header.get(validation.LIST_TRGSECT, [])]
    input_set = set(input_pixels)
    matching_groups = (
        [group_id for group_id, members in enumerate(trigger_groups) if input_set <= members]
        if input_set
        else []
    )
    return input_pixels, source_pixels, fired_groups, matching_groups


def _event_image(event, telescope_event, camera):
    """Use integrated readout when available, otherwise show photoelectron counts."""
    if ADC_SAMPLES in telescope_event:
        _samples, _pedestals, image = get_trace_data(telescope_event[ADC_SAMPLES])
        return image, "Integrated signal [d.c.]"
    image = np.zeros(len(camera.pixels[PIXEL_IDS]))
    photoelectrons = next(iter(event.get(MC_PHOTOELECTRONS, {}).values()), {})
    if MC_PHOTOELECTRONS in photoelectrons:
        image = photoelectrons[MC_PHOTOELECTRONS]
    return image, "MC photoelectrons (no recorded waveform)"


def _format_ids(values, empty_label=NONE_RECORDED):
    """Format a short list of integer IDs for a PDF annotation."""
    return ", ".join(str(value) for value in values) if values else empty_label


def _overlay_trigger_groups(axis, camera, trigger_groups, fired_groups, input_pixels):
    """Outline each fired patch and mark the illuminated pixel centers."""
    for index, group_id in enumerate(sorted(set(fired_groups))):
        if 0 <= group_id < len(trigger_groups):
            _outline_patch(
                axis,
                camera,
                trigger_groups[group_id],
                color=PATCH_COLORS[index % len(PATCH_COLORS)],
                linestyle=PATCH_LINESTYLES[(index // len(PATCH_COLORS)) % len(PATCH_LINESTYLES)],
                label=f"Patch {group_id}",
            )
    if axis.patches:
        axis.legend(loc="upper right", fontsize=8, title="Fired patches")
    pixel_ids = camera.pixels[PIXEL_IDS]
    pixel_x = camera.pixels["x"]
    pixel_y = camera.pixels["y"]
    pixel_indices = {int(pixel_id): index for index, pixel_id in enumerate(pixel_ids)}
    input_indices = [pixel_indices[pixel] for pixel in input_pixels if pixel in pixel_indices]
    if input_indices:
        axis.scatter(
            [pixel_x[index] for index in input_indices],
            [pixel_y[index] for index in input_indices],
            marker="o",
            s=110,
            facecolors="none",
            edgecolors="#00e5ff",
            linewidths=1.8,
            zorder=6,
        )


def _outline_patch(
    axis, camera, selected_pixels, color=PATCH_COLORS[0], linestyle=PATCH_LINESTYLES[0], label=None
):
    """Draw the convex outer boundary of a patch using its full pixel footprints."""
    patches = [
        pixel_shape(camera, x, y)
        for pixel_id, x, y in zip(camera.pixels[PIXEL_IDS], camera.pixels["x"], camera.pixels["y"])
        if int(pixel_id) in selected_pixels
    ]
    patches = [patch for patch in patches if patch is not None]
    if patches:
        vertices = np.concatenate(
            [patch.get_patch_transform().transform(patch.get_path().vertices) for patch in patches]
        )
        hull = spatial.ConvexHull(vertices)
        axis.add_patch(
            Polygon(
                vertices[hull.vertices],
                facecolor="none",
                edgecolor=color,
                linestyle=linestyle,
                label=label,
                linewidth=1.4,
                zorder=5,
            )
        )
