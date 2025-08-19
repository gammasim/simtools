#!/usr/bin/python3
"""Plots for light emission (flasher/calibration) sim_telarray events."""

from __future__ import annotations

import logging

import astropy.units as u
import matplotlib.pyplot as plt

__all__ = [
    "plot_simtel_event_image",
    "plot_simtel_integrated_pedestal_image",
    "plot_simtel_integrated_signal_image",
    "plot_simtel_peak_timing",
    "plot_simtel_step_traces",
    "plot_simtel_time_traces",
    "plot_simtel_waveform_pcolormesh",
]

_logger = logging.getLogger(__name__)

# Reusable literal constants (duplicated from visualize to avoid circular deps)
AXES_FRACTION = "axes fraction"
NO_R1_WAVEFORMS_MSG = "No R1 waveforms available in event"
TIME_NS_LABEL = "time [ns]"
R1_SAMPLES_LABEL = "R1 samples [a.u.]"


def _select_event_by_type(source):
    """Return the first event from the source."""
    for ev in source:
        return ev
    _logger.warning("No events available from source")
    return None


def plot_simtel_event_image(filename, distance=None):
    """
    Read in a sim_telarray file and plot DL1 image via ctapipe.

    Returns matplotlib.figure.Figure or None.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.calib import CameraCalibrator
    from ctapipe.io import EventSource
    from ctapipe.visualization import CameraDisplay

    source = EventSource(filename, max_events=1)
    event = next(iter(source), None)
    if not event:
        _logger.warning("No event found in the file.")
        return None

    tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if not tel_ids:
        _logger.warning("First event has no R1 telescope data")
        return None
    tel_id = tel_ids[0]

    calib = CameraCalibrator(subarray=source.subarray)
    calib(event)

    geometry = source.subarray.tel[tel_id].camera.geometry
    image = event.dl1.tel[tel_id].image

    fig, ax = plt.subplots(dpi=300)
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    title = f"{tel_label}, run {event.index.obs_id} event {event.index.event_id}"
    disp = CameraDisplay(geometry, image=image, norm="symlog", ax=ax)
    disp.cmap = "RdBu_r"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)
    ax.set_title(title, pad=20)

    try:
        d_str = f"{distance.to(u.m)}"
    except (AttributeError, TypeError, ValueError):
        d_str = str(distance)

    ax.annotate(
        f"tel type: {source.subarray.tel[tel_id].type.name}\n"
        f"optics: {source.subarray.tel[tel_id].optics.name}\n"
        f"camera: {source.subarray.tel[tel_id].camera_name}\n"
        f"distance: {d_str}",
        xy=(0, 0),
        xytext=(0.1, 1),
        xycoords=AXES_FRACTION,
        va="top",
        size=7,
    )
    ax.annotate(
        f"dl1 image,\ntotal ADC counts: {np.round(np.sum(image))}\n",
        xy=(0, 0),
        xytext=(0.75, 1),
        xycoords=AXES_FRACTION,
        va="top",
        ha="left",
        size=7,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_simtel_time_traces(
    filename,
    tel_id: int | None = None,
    n_pixels: int = 3,
):
    """Plot time traces (R1 waveforms) for a few camera pixels of a selected event."""
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.calib import CameraCalibrator
    from ctapipe.io import EventSource

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source)

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        dl1_tel_ids = sorted(getattr(event.dl1, "tel", {}).keys())
        tel_id = tel_id or dl1_tel_ids[0]

    calib = CameraCalibrator(subarray=source.subarray)
    try:
        calib(event)
        image = event.dl1.tel[tel_id].image
    except (RuntimeError, ValueError, KeyError, AttributeError):
        image = None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning(NO_R1_WAVEFORMS_MSG)
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    _, n_samp = w.shape

    if image is not None:
        pix_ids = np.argsort(image)[-n_pixels:][::-1]
    else:
        integrals = w.sum(axis=1)
        pix_ids = np.argsort(integrals)[-n_pixels:][::-1]

    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0
    t = np.arange(n_samp) * dt

    fig, ax = plt.subplots(dpi=300)
    for pid in pix_ids:
        ax.plot(t, w[pid], label=f"pix {int(pid)}", drawstyle="steps-mid")
    ax.set_xlabel(TIME_NS_LABEL)
    ax.set_ylabel(R1_SAMPLES_LABEL)
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    ax.set_title(f"{tel_label} waveforms ({et_name})")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return fig


def plot_simtel_waveform_pcolormesh(
    filename,
    tel_id: int | None = None,
    pixel_step: int | None = None,
    vmax: float | None = None,
):
    """Pseudocolor image of waveforms (samples vs pixel id) for one event."""
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source)

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for waveform plot")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning(NO_R1_WAVEFORMS_MSG)
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    if pixel_step and pixel_step > 1:
        pix_idx = np.arange(0, n_pix, pixel_step)
        w_sel = w[pix_idx]
    else:
        pix_idx = np.arange(n_pix)
        w_sel = w

    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0
    t = np.arange(n_samp) * dt

    fig, ax = plt.subplots(dpi=300)
    mesh = ax.pcolormesh(t, pix_idx, w_sel, shading="auto", vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(R1_SAMPLES_LABEL)
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    ax.set_title(f"{tel_label} waveform matrix ({et_name})")
    ax.set_xlabel(TIME_NS_LABEL)
    ax.set_ylabel("pixel id")
    fig.tight_layout()
    return fig


def plot_simtel_step_traces(
    filename,
    tel_id: int | None = None,
    pixel_step: int = 100,
    max_pixels: int | None = None,
):
    """Plot step-style traces for regularly sampled pixels: pix 0, N, 2N, ..."""
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source)

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for traces plot")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning(NO_R1_WAVEFORMS_MSG)
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0
    t = np.arange(n_samp) * dt

    pix_ids = np.arange(0, n_pix, max(1, pixel_step))
    if max_pixels is not None:
        pix_ids = pix_ids[:max_pixels]

    fig, ax = plt.subplots(dpi=300)
    for pid in pix_ids:
        ax.plot(t, w[int(pid)], label=f"pix {int(pid)}", drawstyle="steps-mid")
    ax.set_xlabel(TIME_NS_LABEL)
    ax.set_ylabel(R1_SAMPLES_LABEL)
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    ax.set_title(f"{tel_label} step traces ({et_name})")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def _detect_peaks(trace, peak_width, signal_mod):
    """Return indices of peaks using CWT if available, else find_peaks fallback."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    peaks = []
    try:
        if hasattr(signal_mod, "find_peaks_cwt"):
            peaks = signal_mod.find_peaks_cwt(trace, widths=np.array([peak_width]))
        if not np.any(peaks):
            peaks, _ = signal_mod.find_peaks(trace, prominence=np.max(trace) * 0.1)
    except (ValueError, RuntimeError, TypeError):
        peaks = []
    return np.asarray(peaks, dtype=int) if np.size(peaks) else np.array([], dtype=int)


def _collect_peak_samples(w, sum_threshold, peak_width, signal_mod):
    """Compute peak sample per pixel, return samples, considered pixel ids and count with peaks."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    n_pix, _ = w.shape
    sums = w.sum(axis=1)
    has_signal = sums > float(sum_threshold)
    pix_ids = np.arange(n_pix)[has_signal]
    if pix_ids.size == 0:
        return None, None, 0

    peak_samples = []
    found_count = 0
    for pid in pix_ids:
        trace = w[int(pid)]
        pks = _detect_peaks(trace, peak_width, signal_mod)
        if pks.size:
            found_count += 1
            peak_idx = int(pks[np.argmax(trace[pks])])
        else:
            peak_idx = int(np.argmax(trace))
        peak_samples.append(peak_idx)

    return np.asarray(peak_samples), pix_ids, found_count


def _histogram_edges(n_samp, timing_bins):
    """Return contiguous histogram bin edges for sample indices."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    if timing_bins and timing_bins > 0:
        return np.linspace(-0.5, n_samp - 0.5, int(timing_bins) + 1)
    return np.arange(-0.5, n_samp + 0.5, 1.0)


def _draw_peak_hist(
    ax,
    peak_samples,
    edges,
    mean_sample,
    std_sample,
    tel_label,
    et_name,
    considered,
    found_count,
):
    """Draw contiguous-bar histogram, stats overlays, and annotations."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    counts, edges = np.histogram(peak_samples, bins=edges)
    ax.bar(edges[:-1], counts, width=np.diff(edges), color="#5B90DC", align="edge")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("peak sample")
    ax.set_ylabel("N pixels")
    ax.axvline(
        mean_sample,
        color="#D8153C",
        linestyle="--",
        label=f"mean={mean_sample:.2f}",
    )
    ax.axvspan(
        mean_sample - std_sample,
        mean_sample + std_sample,
        color="#D8153C",
        alpha=0.2,
        label=f"std={std_sample:.2f}",
    )
    ax.set_title(f"{tel_label} peak timing ({et_name})")
    ax.text(
        0.98,
        0.95,
        f"considered: {considered}\npeaks found: {found_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.6,
            "linewidth": 0.0,
        },
    )
    ax.legend(fontsize=7)


def plot_simtel_peak_timing(
    filename,
    tel_id: int | None = None,
    sum_threshold: float = 10.0,
    peak_width: int = 8,
    examples: int = 3,
    timing_bins: int | None = None,
    return_stats: bool = False,
):
    """
    Peak finding per pixel; report mean/std of peak sample and plot a histogram.

    Returns matplotlib.figure.Figure or (fig, stats_dict) if return_stats.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource
    from scipy import signal as _signal

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source)

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning("Event has no R1 data for peak timing plot")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning(NO_R1_WAVEFORMS_MSG)
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    _, n_samp = w.shape

    peak_samples, pix_ids, found_count = _collect_peak_samples(
        w, sum_threshold, peak_width, _signal
    )
    if peak_samples is None or pix_ids is None:
        _logger.warning("No pixels exceeded sum_threshold for peak timing")
        return None

    mean_sample = float(np.mean(peak_samples))
    std_sample = float(np.std(peak_samples))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    edges = _histogram_edges(n_samp, timing_bins)
    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    _draw_peak_hist(
        ax1,
        peak_samples,
        edges,
        mean_sample,
        std_sample,
        tel_label,
        et_name,
        pix_ids.size,
        found_count,
    )

    readout = source.subarray.tel[tel_id].camera.readout
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, ZeroDivisionError, TypeError):
        dt = 1.0
    t = np.arange(n_samp) * dt

    ex_ids = pix_ids[: max(1, int(examples))]
    for pid in ex_ids:
        trace = w[int(pid)]
        pks = _detect_peaks(trace, peak_width, _signal)
        ax2.plot(t, trace, drawstyle="steps-mid", label=f"pix {int(pid)}")
        if pks.size:
            ax2.scatter(t[pks], trace[pks], s=10)
    ax2.set_xlabel(TIME_NS_LABEL)
    ax2.set_ylabel(R1_SAMPLES_LABEL)
    ax2.legend(fontsize=7)

    fig.tight_layout()

    if return_stats:
        stats = {
            "considered": int(pix_ids.size),
            "found": int(found_count),
            "mean": float(mean_sample),
            "std": float(std_sample),
        }
        return fig, stats
    return fig


def _prepare_waveforms_for_image(filename, tel_id, context_no_r1):
    """Fetch R1 waveforms for one event/telescope and return prepared arrays.

    Returns (w, n_pix, n_samp, source, event, tel_id) or None on failure.
    """
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.io import EventSource

    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source)

    r1_tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if r1_tel_ids:
        tel_id = tel_id or r1_tel_ids[0]
    else:
        _logger.warning(f"Event has no R1 data for {context_no_r1}")
        return None

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning(NO_R1_WAVEFORMS_MSG)
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape
    return w, n_pix, n_samp, source, event, tel_id


def plot_simtel_integrated_signal_image(
    filename,
    tel_id: int | None = None,
    half_width: int = 8,
):
    """Plot camera image of integrated signal per pixel around the flasher peak."""
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.visualization import CameraDisplay

    prepared = _prepare_waveforms_for_image(filename, tel_id, "integrated-signal image")
    if prepared is None:
        return None
    w, n_pix, n_samp, source, event, tel_id = prepared

    win_len = 2 * int(half_width) + 1
    img = np.zeros(n_pix, dtype=float)

    for pid in range(n_pix):
        trace = w[pid]
        peak_idx = int(np.argmax(trace))
        a = max(0, peak_idx - half_width)
        b = min(n_samp, peak_idx + half_width + 1)
        img[pid] = float(np.sum(trace[a:b]))

    geometry = source.subarray.tel[tel_id].camera.geometry
    fig, ax = plt.subplots(dpi=300)
    disp = CameraDisplay(geometry, image=img, norm="lin", ax=ax)
    disp.cmap = "viridis"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)

    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    ax.set_title(
        f"{tel_label} integrated signal (win {win_len}) ({et_name})",
        pad=20,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def plot_simtel_integrated_pedestal_image(
    filename,
    tel_id: int | None = None,
    half_width: int = 8,
    gap: int = 16,
):
    """Plot camera image of integrated pedestal per pixel away from the flasher peak."""
    # pylint:disable=import-outside-toplevel
    import numpy as np
    from ctapipe.visualization import CameraDisplay

    prepared = _prepare_waveforms_for_image(filename, tel_id, "integrated-pedestal image")
    if prepared is None:
        return None
    w, n_pix, n_samp, source, event, tel_id = prepared

    win_len = 2 * int(half_width) + 1
    img = np.zeros(n_pix, dtype=float)
    for pid in range(n_pix):
        trace = w[pid]
        peak_idx = int(np.argmax(trace))
        start = peak_idx + int(gap)
        if start + win_len <= n_samp:
            a = start
            b = start + win_len
        else:
            start = max(0, peak_idx - int(gap) - win_len)
            a = start
            b = min(n_samp, start + win_len)
        if a >= b:
            a = 0
            b = min(n_samp, win_len)
        img[pid] = float(np.sum(trace[a:b]))

    geometry = source.subarray.tel[tel_id].camera.geometry
    fig, ax = plt.subplots(dpi=300)
    disp = CameraDisplay(geometry, image=img, norm="lin", ax=ax)
    disp.cmap = "cividis"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)

    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    ax.set_title(
        f"{tel_label} integrated pedestal (win {win_len}, gap {gap}) ({et_name})",
        pad=20,
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig
