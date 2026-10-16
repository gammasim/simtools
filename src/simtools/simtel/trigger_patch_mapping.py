"""Generate sim_telarray trigger-patch displays with ``pixled``."""

import json
import logging
import os
import re
import shlex
import shutil
import textwrap
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from eventio import SimTelFile
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator

from simtools import settings
from simtools.camera.trace_analysis import get_trace_data
from simtools.job_execution import job_manager
from simtools.model.camera import Camera
from simtools.runners.simtel_runner import SIM_TELARRAY_ENV
from simtools.simtel import trigger_patch_validation as validation
from simtools.simulation.configuration import get_model_writer
from simtools.visualization.camera_plot_utils import pixel_shape
from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.plot_camera import plot_pixel_layout_with_image

MIRROR_CLASS = "mirror_class"
TRIGGER_SETTINGS_KEY = "trigger_settings"
MAPPING_MISMATCHES = "illumination_mapping_mismatches"
GROUP_COUNT = "groups"
GROUPS_SAMPLED = "groups_sampled"
HEADER_COMPARISON = "header_membership_and_kind"
MODEL_COMPARISON = "model_membership_and_roles"
STREAM_COMPLETE = "stream_complete"


FAILED = "failed"
STATUS = "status"
PIXEL_IDS = "pix_id"
EQUAL_ASPECT = "equal"
CONSTRAINED_LAYOUT = "constrained"
ADC_SAMPLES = "adc_samples"
CAMERA_ORGANIZATION = "camera_organization"
TRIGGER_SETTINGS = frozenset(
    {
        "default_trigger",
        "trigger_pixels",
        "discriminator_threshold",
        "discriminator_amplitude",
        "asum_threshold",
        "asum_clipping",
        "dsum_threshold",
        "nsb_scaling_factor",
    }
)

PIXLED = "pixled"
PDF_SUFFIX = ".pdf"
MC_PHOTOELECTRONS = "photoelectrons"
NONE_RECORDED = "none recorded"
NO_MATCHING_GROUPS = "none"
PATCH_COLORS = ("#b2182b", "#ef3b2c", "#fb8d7a")
PATCH_LINESTYLES = ("solid", "dashed", "dotted", "dashdot")
PATCH_ID_LABEL = "Patch ID"
CONFIGURED_PATCHES_LABEL = "Configured patches"
TELESCOPE_TRIGGERED_LABEL = "Telescope triggered"
PIXLED_FIRING_PREFIX = "Firing"
PIXLED_PIXEL_ID_PATTERN = re.compile(r"\(pixel (\d+)\)")
PIXLED_SELECTION_PATTERN = re.compile(r"Using \d+ pixels:\s*(.*)")
PIXLED_SLAVES_MARKER = "Slaves pixels also firing"
UTF_8 = "utf-8"
CAMERA_CONTEXT = ". Camera: "
PDF_CONTEXT = "; PDF: "
_logger = logging.getLogger(__name__)
spatial = lazy_module("scipy.spatial")


@dataclass(frozen=True)
class TriggerPatchMappingFiles:
    """Files produced while mapping trigger patches.

    Attributes
    ----------
    camera_file : pathlib.Path
        Original exported model camera used for structural comparisons.
    iact_file : pathlib.Path
        LED event stream produced by ``pixled``.
    simtel_file : pathlib.Path
        sim_telarray event file.
    pdf_file : pathlib.Path
        Compact PDF inspection report.
    log_files : tuple[pathlib.Path, ...]
        Standard-output and standard-error logs for each program.
    """

    camera_file: Path
    iact_file: Path
    simtel_file: Path
    pdf_file: Path
    log_files: tuple[Path, ...]


def run_trigger_patch_mapping(
    telescope_model,
    site_model,
    output_directory,
    *,
    required_pixels=None,
    include_presum_slaves=False,
    photons_per_pixel=500,
    events_per_combination=1,
    run_number=1,
    output_name=None,
    max_events=None,
    max_pixel_radius=None,
    exhaustive=False,
    max_plotted_events=4,
):
    """Inspect complete trigger structure with bounded, explicit illumination trials.

    The model writer exports the camera first. A deterministic manifest selects
    whole groups by default. pixled generates those exact patterns, sim_telarray
    processes them, and simtools accounts for every trial in tables and a compact PDF.

    Parameters
    ----------
    telescope_model : simtools.model.telescope_model.TelescopeModel
        Telescope model whose camera trigger tables are mapped.
    site_model : simtools.model.site_model.SiteModel
        Site model supplying sim_telarray atmosphere settings.
    output_directory : str or pathlib.Path
        Directory for generated event files, display, and logs.
    required_pixels : int, optional
        Illuminated master count; omit to illuminate every master in each group.
    include_presum_slaves : bool, optional
        Include pre-sum slave pixels when a master pixel fires.
    photons_per_pixel : int, optional
        Number of photons emitted by every selected LED.
    events_per_combination : int, optional
        Number of repeated events per selected LED combination.
    run_number : int, optional
        Run number recorded by ``pixled``.
    output_name : str, optional
        PDF output filename. A deterministic name is used when omitted.
    max_events : int, optional
        Maximum generated trials, including repetitions.
    max_pixel_radius : float, optional
        Maximum radius of illuminated pixels in camera-plane metres.
    exhaustive : bool, optional
        Enumerate all eligible combinations instead of one pattern per group.
    max_plotted_events : int, optional
        Maximum event pages in simulation order; zero writes only the summary.

    Returns
    -------
    TriggerPatchMappingFiles
        Paths of all reviewable output files.

    Raises
    ------
    ValueError
        If a numeric option is not positive, the PDF name is unsafe,
        or a camera has no trigger groups or fewer masters than required.
    FileNotFoundError
        If a required sim_telarray executable or generated camera file is absent.
    PermissionError
        If a required program cannot be executed.
    simtools.job_execution.job_manager.JobExecutionError
        If a program fails, its output is missing/empty, or PDF generation fails.
    """
    _validate_options(photons_per_pixel, events_per_combination, run_number)
    if (
        isinstance(max_plotted_events, bool)
        or not isinstance(max_plotted_events, Integral)
        or not 0 <= max_plotted_events <= 2147483647
    ):
        raise ValueError("max_plotted_events must be a nonnegative 32-bit integer")
    if required_pixels is not None:
        _validate_options(required_pixels)
    elif exhaustive:
        raise ValueError("exhaustive mode requires an explicit required_pixels stimulus size")
    if max_events is not None:
        _validate_options(max_events)
    if max_pixel_radius is not None and (
        isinstance(max_pixel_radius, bool)
        or not isinstance(max_pixel_radius, Real)
        or not np.isfinite(max_pixel_radius)
        or max_pixel_radius <= 0
    ):
        raise ValueError("max_pixel_radius must be finite and positive, in metres")
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = f"trigger-patches-{telescope_model.name}-run{run_number}"
    pdf_file = _output_file(output_directory, output_name or f"{prefix}{PDF_SUFFIX}")
    pixled, simtel = _executables()
    camera_file = _export_camera_file(telescope_model, site_model)
    if camera_file.parent != output_directory:
        retained_camera = output_directory / f"{prefix}.camera.dat"
        shutil.copyfile(camera_file, retained_camera)
        camera_file = retained_camera
    if required_pixels is not None:
        _validate_trigger_group_sizes(camera_file, required_pixels)
    atmospheric_transmission = _configuration_value(
        telescope_model.config_file_path, "atmospheric_transmission"
    )
    altitude = site_model.get_parameter_value_with_unit("corsika_observation_level").to_value(u.m)
    files = TriggerPatchMappingFiles(
        camera_file=camera_file,
        iact_file=output_directory / f"{prefix}.iact.gz",
        simtel_file=output_directory / f"{prefix}.simtel.gz",
        pdf_file=pdf_file,
        log_files=tuple(
            output_directory / f"{prefix}.{step}.{stream}.log"
            for step in (PIXLED, "simtel")
            for stream in ("stdout", "stderr")
        ),
    )
    _prepare_manifest(
        files,
        telescope_model,
        required_pixels,
        photons_per_pixel,
        events_per_combination,
        include_presum_slaves,
        exhaustive,
        max_events,
        max_pixel_radius,
    )
    _prepare_pixled_atmosphere(site_model, output_directory)
    _run_pixled(
        pixled,
        files,
        photons_per_pixel,
        run_number,
        altitude,
    )
    _run_simtel(simtel, telescope_model, files, altitude, atmospheric_transmission)
    _write_pdf(files, telescope_model.name, max_plotted_events)
    return files


def _validate_options(*values):
    """Reject values outside pixled's positive integer range."""
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 2147483647
        for value in values
    ):
        raise ValueError("Trigger-patch mapping numeric options must be positive 32-bit integers")


def _output_file(directory, name):
    """Return a safe output file in ``directory``."""
    path = Path(name)
    if path.name != str(path) or path.suffix != PDF_SUFFIX:
        raise ValueError("output_name must be a PDF filename without a directory")
    return directory / path


def _export_camera_file(telescope_model, site_model):
    """Export the telescope configuration and return its camera file."""
    telescope_model.write_sim_telarray_config_file(additional_models=site_model)
    config_file = Path(telescope_model.config_file_path).resolve()
    camera_file = config_file.parent / _configuration_value(config_file, "camera_config_file")
    if not camera_file.is_file():
        raise FileNotFoundError(f"Generated camera file not found: {camera_file}")
    return camera_file


def _validate_trigger_group_sizes(camera_file, required_pixels):
    """Reject trigger groups that pixled would silently skip."""
    groups, _pixels, _rotation = validation.read_camera(camera_file)
    for group_id, group in enumerate(groups):
        member_count = len(group[validation.MASTERS])
        if member_count < required_pixels:
            raise ValueError(
                f"Trigger group {group_id} in {camera_file} has {member_count} master pixels, "
                f"but required_pixels={required_pixels}. Reduce required_pixels to include "
                "every trigger group."
            )


def _configuration_value(config_file, parameter):
    """Read a filename from the exported configuration without guessing its name."""
    config_file = Path(config_file)
    for line in config_file.read_text(encoding=UTF_8).splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == parameter:
            tokens = shlex.split(value, comments=True)
            if len(tokens) == 1:
                return tokens[0]
            raise ValueError(f"Invalid {parameter} in generated config: {config_file}")
    raise ValueError(f"No {parameter} defined in generated config: {config_file}")


def _executables():
    """Return the required sim_telarray executable paths."""
    simtel_path = settings.config.sim_telarray_path
    paths = (
        simtel_path / "LightEmission" / PIXLED,
        Path(settings.config.sim_telarray_exe),
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required sim_telarray executable not found: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Required sim_telarray program is not executable: {path}")
    return paths


def _prepare_pixled_atmosphere(site_model, output_directory):
    """Provide the selected site's native profile under LightEmission's default alias."""
    writer = get_model_writer(site_model)
    writer.export_atmospheric_transmission_file(output_directory)
    profile_file = output_directory / writer.get_atmospheric_profile_file_name()
    if not profile_file.is_file():
        raise FileNotFoundError(f"Exported site atmospheric profile not found: {profile_file}")
    alias = output_directory / "atmprof1.dat"
    if profile_file != alias:
        alias.unlink(missing_ok=True)
        shutil.copyfile(profile_file, alias)


def _run_pixled(
    executable,
    files,
    photons_per_pixel,
    run_number,
    altitude,
):
    """Generate LED events from all trigger lines."""
    command = [
        str(executable),
        f"-I{files.iact_file.parent}",
        "--atmosphere",
        "1",
        "--camera-file",
        str(files.pdf_file.with_suffix(validation.ILLUMINATION_SUFFIX)),
        "--use-trg-all",
        "--require",
        "1",
        "--photons",
        str(photons_per_pixel),
        "--events",
        "1",
        "--run",
        str(run_number),
        "--altitude",
        str(altitude),
        "-o",
        str(files.iact_file),
    ]
    _submit(command, files.log_files[0:2], files.iact_file, files)


def _run_simtel(executable, telescope_model, files, altitude, atmospheric_transmission):
    """Process the LED event stream through sim_telarray."""
    histogram_file = files.pdf_file.with_suffix(".hdata")
    command = [
        str(executable),
        "-c",
        str(Path(telescope_model.config_file_path).resolve()),
        f"-I{Path(telescope_model.config_file_directory).resolve()}",
        "-DNUM_TELESCOPES=1",
        "-C",
        "Bypass_Optics=2",
        "-C",
        "maximum_telescopes=1",
        "-C",
        "trigger_telescopes=1",
        "-C",
        "random_mono_prob=0",
        "-C",
        "iobuf_maximum=1000000000",
        "-C",
        f"Altitude={altitude}",
        "-C",
        f"atmospheric_transmission={atmospheric_transmission}",
        "-C",
        "telescope_theta=0",
        "-C",
        "telescope_phi=0",
        "-C",
        f"histogram_file={histogram_file}",
        "-r",
        str(files.simtel_file),
        str(files.iact_file),
    ]
    _submit(command, files.log_files[2:4], files.simtel_file, files)


def _write_pdf(files, telescope_name, max_plotted_events=4):
    """Stream simulated events to a PDF using simtools camera and trace analysis."""
    files.pdf_file.unlink(missing_ok=True)
    try:
        with (
            SimTelFile(
                files.simtel_file, skip_calibration=True, skip_non_triggered=False
            ) as events,
            PdfPages(files.pdf_file) as pdf,
        ):
            if len(events.telescope_descriptions) != 1:
                raise ValueError("Trigger-patch display requires exactly one telescope")
            telescope_id, description = next(iter(events.telescope_descriptions.items()))
            camera_settings = description["camera_settings"]
            camera = Camera(
                telescope_name=telescope_name,
                camera_config_file=None,
                focal_length=camera_settings["focal_length"],
                camera_config_dict=camera_settings,
            )
            groups, _pixels, _rotation = validation.read_camera(files.camera_file)
            patterns, metadata = _inspection_manifest(files, groups)
            group_rows = validation.build_group_rows(
                groups,
                description[CAMERA_ORGANIZATION],
                metadata.get(validation.MODEL_MATCHES, [None] * len(groups)),
            )
            metadata[validation.HEADER_SECTOR_COUNT] = int(
                description[CAMERA_ORGANIZATION]["n_sectors"]
            )
            rows, unexpected, duplicates, samples = validation.account_events(
                events, telescope_id, patterns, group_rows, sample_limit=max_plotted_events
            )
            metadata[STREAM_COMPLETE] = events.histograms is not None
            if not metadata[STREAM_COMPLETE]:
                for row in rows:
                    if row[validation.STATE] == validation.NON_TRIGGERED:
                        row[validation.STATE] = validation.UNRESOLVED
            summary = _inspection_summary(rows, group_rows, unexpected, duplicates, metadata)
            validation.save_results(
                files.pdf_file, rows, group_rows, summary, files.camera_file, metadata
            )
            if summary[validation.OBSERVED] == 0:
                raise ValueError("No MC events available for trigger-patch display")
            _plot_configuration_summary(pdf, camera, group_rows)
            _plot_inspection_summary(pdf, camera, rows, summary)
            _plot_patch_efficiency(pdf, group_rows)
            _plot_trigger_events(
                samples,
                telescope_id,
                camera,
                pdf,
                [set(json.loads(group[validation.MEMBERS])) for group in group_rows],
                [json.loads(row[validation.INPUT_PIXELS]) for row in patterns],
            )
    except (OSError, EOFError, ValueError, KeyError) as exc:
        files.pdf_file.unlink(missing_ok=True)
        raise job_manager.JobExecutionError(
            f"Trigger-patch PDF failed for {files.simtel_file}: {exc}"
            f"{CAMERA_CONTEXT}{files.camera_file}{PDF_CONTEXT}{files.pdf_file}"
        ) from exc
    if summary[STATUS] == FAILED:
        raise job_manager.JobExecutionError(
            f"Trigger-patch inspection found inconsistencies; retained report: {files.pdf_file}"
        )


def _inspection_manifest(files, groups):
    """Read the authoritative manifest, with recovery for earlier branch artifacts."""
    path = files.pdf_file.with_suffix(validation.EVENTS_SUFFIX)
    if path.is_file():
        table = Table.read(path)
        table.sort(validation.EVENT_NUMBER)
        if list(table[validation.EVENT_NUMBER]) != list(range(1, len(table) + 1)):
            raise ValueError("Stimulus manifest event numbers must be unique and contiguous from 1")
        return [dict(row) for row in table], table.meta
    return _patterns_from_log(files, groups), {
        "manifest_origin": "recovered pixled log; original model comparison unavailable"
    }


def _prepare_manifest(
    files,
    telescope_model,
    required_pixels,
    photons_per_pixel,
    repetitions,
    include_slaves,
    exhaustive,
    max_events,
    max_radius,
):
    """Plan a bounded batch and retain its identities before either executable runs."""
    groups, pixels, _rotation = validation.read_camera(files.camera_file)
    matches, count_matches = validation.compare_model(
        groups,
        telescope_model.get_parameter_table("camera_trigger_groups"),
        telescope_model.get_parameter_table("camera_trigger_members"),
    )
    rows = validation.plan_patterns(
        groups,
        pixels,
        required_pixels,
        repetitions,
        include_slaves,
        exhaustive=exhaustive,
        max_events=max_events,
        max_radius=max_radius,
    )
    _logger.info(
        "Planned %d LED trials for %d configured groups (exhaustive=%s)",
        len(rows),
        len(groups),
        exhaustive,
    )
    validation.write_stimulus_camera(
        files.camera_file,
        files.pdf_file.with_suffix(validation.ILLUMINATION_SUFFIX),
        rows,
        int(telescope_model.get_parameter_value(MIRROR_CLASS)),
    )
    metadata = {
        validation.MODEL_MATCHES: matches,
        validation.MODEL_GROUP_COUNT_MATCH: count_matches,
        "telescope": telescope_model.name,
        "site": telescope_model.site,
        "model_version": telescope_model.model_version,
        "photons_per_pixel": int(photons_per_pixel),
        "configuration_sha256": validation.file_digest(telescope_model.config_file_path),
        "illumination_sha256": validation.file_digest(
            files.pdf_file.with_suffix(validation.ILLUMINATION_SUFFIX)
        ),
        MIRROR_CLASS: int(telescope_model.get_parameter_value(MIRROR_CLASS)),
        TRIGGER_SETTINGS_KEY: _trigger_settings(telescope_model.config_file_path),
        "max_events": int(max_events) if max_events is not None else None,
        "max_pixel_radius": float(max_radius) if max_radius is not None else None,
        "exhaustive": exhaustive,
        "required_pixels": int(required_pixels) if required_pixels is not None else None,
        "events_per_combination": int(repetitions),
    }
    Table(rows=rows, meta=metadata).write(
        files.pdf_file.with_suffix(validation.EVENTS_SUFFIX), overwrite=True
    )


def _trigger_settings(config_file):
    """Retain the effective exported trigger parameters without inventing conversions."""
    values = {}
    for line in Path(config_file).read_text(encoding=UTF_8).splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() in TRIGGER_SETTINGS:
            values[key.strip().lower()] = value.strip()
    return values


def _patterns_from_log(files, groups):
    """Recover original branch stimuli without treating recorded events as a denominator."""
    log = files.log_files[0] if files.log_files else None
    patterns = _read_pixled_input_patterns(log)
    by_pixel = {}
    for group_id, group in enumerate(groups):
        for pixel in validation.member_pixels(group):
            by_pixel.setdefault(pixel, set()).add(group_id)
    rows = []
    for number, pixels in enumerate(patterns, 1):
        group_ids = (
            sorted(set.intersection(*(by_pixel.get(pixel, set()) for pixel in pixels)))
            if pixels
            else []
        )
        rows.append(
            {
                validation.EVENT_NUMBER: number,
                validation.INPUT_PIXELS: json.dumps(pixels),
                validation.SOURCE_GROUPS: json.dumps(group_ids),
            }
        )
    return rows


def _inspection_summary(rows, groups, unexpected, duplicates, metadata):
    """Reconcile observed, non-triggering, and missing trials without physics assumptions."""
    missing = sum(row[validation.STATE] == validation.MISSING for row in rows)
    structural = all(row[validation.HEADER_MATCH] for row in groups) and metadata.get(
        validation.HEADER_SECTOR_COUNT
    ) == len(groups)
    model_mismatch = any(row[validation.MODEL_MATCH] == validation.MISMATCH for row in groups)
    model_mismatch = model_mismatch or metadata.get(validation.MODEL_GROUP_COUNT_MATCH) is False
    model_available = metadata.get(validation.MODEL_GROUP_COUNT_MATCH) is not None and all(
        row[validation.MODEL_MATCH] != "unavailable" for row in groups
    )
    mapping_mismatches = sum(row[validation.MAPPING_STATUS] == validation.MISMATCH for row in rows)
    unresolved = sum(row[validation.STATE] == validation.UNRESOLVED for row in rows)
    status = (
        FAILED
        if missing
        or unexpected
        or duplicates
        or not structural
        or model_mismatch
        or mapping_mismatches
        or unresolved
        or not metadata.get(STREAM_COMPLETE, True)
        else validation.OBSERVED
    )
    return {
        **metadata,
        STATUS: status,
        validation.PLANNED: len(rows),
        validation.OBSERVED: len(rows) - missing,
        "missing": missing,
        validation.UNRESOLVED: unresolved,
        "non_triggered": sum(row[validation.STATE] == validation.NON_TRIGGERED for row in rows),
        "triggered": sum(row[validation.STATE] == validation.TRIGGERED for row in rows),
        MAPPING_MISMATCHES: mapping_mismatches,
        "unexpected_event_ids": unexpected,
        "duplicate_event_numbers": duplicates,
        GROUP_COUNT: len(groups),
        GROUPS_SAMPLED: sum(row[validation.PLANNED] > 0 for row in groups),
        HEADER_COMPARISON: structural,
        MODEL_COMPARISON: validation.model_status(not model_mismatch if model_available else None),
    }


def _pixel_coverage(camera, pixel_lists):
    """Count occurrences of each camera pixel across membership or arrival lists."""
    coverage = np.zeros(len(camera.pixels[PIXEL_IDS]))
    pixel_indices = {int(pixel): index for index, pixel in enumerate(camera.pixels[PIXEL_IDS])}
    for pixels in pixel_lists:
        for pixel in pixels:
            coverage[pixel_indices[pixel]] += 1
    return coverage


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


def _observed_patch_efficiency(groups):
    """Return intended-patch firing fractions, leaving unobserved groups undefined."""
    observed = np.array([group[validation.OBSERVED] for group in groups])
    fired = np.array([group[validation.INTENDED_SECTOR_FIRED] for group in groups])
    return np.divide(fired, observed, out=np.full(len(groups), np.nan), where=observed > 0)


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
    coverage = _pixel_coverage(camera, (json.loads(row["pe_pixels"]) for row in rows))
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


def _read_pixled_input_patterns(log_file):
    """Read the emitted pixel combinations from pixled's event log, in event order."""
    if log_file is None or not Path(log_file).is_file():
        return []
    patterns = []
    selected = []
    for line in Path(log_file).read_text(encoding=UTF_8).splitlines():
        selection = PIXLED_SELECTION_PATTERN.search(line)
        if selection:
            selected = [int(pixel.lstrip("+")) for pixel in selection[1].split()]
        elif line.lstrip().startswith(PIXLED_FIRING_PREFIX):
            emitted = [int(pixel) for pixel in PIXLED_PIXEL_ID_PATTERN.findall(line)]
            patterns.append(emitted or selected.copy())
        elif PIXLED_SLAVES_MARKER in line and patterns:
            patterns[-1].extend(int(pixel) for pixel in line.partition(":")[2].split())
    return patterns


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
    """Use the integrated readout when available, otherwise show photoelectron counts."""
    if ADC_SAMPLES in telescope_event:
        _samples, _pedestals, image = get_trace_data(telescope_event[ADC_SAMPLES])
        return image, "Integrated signal [d.c.]"
    image = np.zeros(len(camera.pixels[PIXEL_IDS]))
    photoelectrons = next(iter(event.get(MC_PHOTOELECTRONS, {}).values()), {})
    if MC_PHOTOELECTRONS in photoelectrons:
        image = photoelectrons[MC_PHOTOELECTRONS]
    return image, "MC photoelectrons (no recorded waveform)"


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


def _submit(command, log_files, output_file, files):
    """Submit one command with explicit standard-output and standard-error logs."""
    try:
        output_file.unlink(missing_ok=True)
        job_manager.submit(
            command, out_file=log_files[0], err_file=log_files[1], env=SIM_TELARRAY_ENV
        )
        if not output_file.is_file() or output_file.stat().st_size == 0:
            raise job_manager.JobExecutionError(f"Missing or empty output: {output_file}")
    except (job_manager.JobExecutionError, OSError) as exc:
        raise job_manager.JobExecutionError(
            f"Trigger-patch step {Path(command[0]).name} failed: {exc}"
            f"{CAMERA_CONTEXT}{files.camera_file}; IACT: {files.iact_file}; "
            f"simtel: {files.simtel_file}{PDF_CONTEXT}{files.pdf_file}; "
            f"logs: {log_files[0]}, {log_files[1]}"
        ) from exc
