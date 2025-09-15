#!/usr/bin/python3
"""Plots for light emission (flasher/calibration) sim_telarray events."""

import logging
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.io import EventSource
from ctapipe.visualization import CameraDisplay
from scipy import signal as _signal

from simtools.corsika.corsika_histograms_visualize import save_figs_to_pdf
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.visualization.visualize import save_figure

__all__ = [
    "generate_and_save_plots",
    "plot_simtel_event_image",
    "plot_simtel_integrated_pedestal_image",
    "plot_simtel_integrated_signal_image",
    "plot_simtel_peak_timing",
    "plot_simtel_step_traces",
    "plot_simtel_time_traces",
    "plot_simtel_waveform_matrix",
]

_logger = logging.getLogger(__name__)

# Reusable literal constants (duplicated from visualize to avoid circular deps)
AXES_FRACTION = "axes fraction"
NO_R1_WAVEFORMS_MSG = "No R1 waveforms available in event"
TIME_NS_LABEL = "time [ns]"
R1_SAMPLES_LABEL = "R1 samples [d.c.]"

# Choices understood by the dispatcher used below
PLOT_CHOICES = {
    "event_image": "event_image",
    "time_traces": "time_traces",
    "waveform_matrix": "waveform_matrix",
    "step_traces": "step_traces",
    "integrated_signal_image": "integrated_signal_image",
    "integrated_pedestal_image": "integrated_pedestal_image",
    "peak_timing": "peak_timing",
    "all": "all",
}


def _get_event_source_and_r1_tel(filename, event_index=None, warn_context=None):
    """Return (source, event, first_r1_tel_id) or None if unavailable.

    Centralizes creation of EventSource, event selection, and first R1 tel-id lookup.

    When no event exists, logs a standard warning. When the event has no R1 tel data,
    logs either a contextual message ("Event has no R1 data for <context>") if
    warn_context is provided, or the generic "First event has no R1 telescope data".
    """
    source = EventSource(filename, max_events=None)
    event = _select_event_by_type(source)(event_index=event_index)
    if not event:
        _logger.warning("No event found in the file.")
        return None

    tel_ids = sorted(getattr(event.r1, "tel", {}).keys())
    if not tel_ids:
        if warn_context:
            _logger.warning("Event has no R1 data for %s", warn_context)
        else:
            _logger.warning("First event has no R1 telescope data")
        return None
    return source, event, int(tel_ids[0])


def _compute_integration_window(peak_idx, n_samp, half_width, mode, offset):
    """Return [a, b) window bounds for integration for signal/pedestal modes."""
    hw = int(half_width)
    win_len = 2 * hw + 1
    if mode == "signal":
        a = max(0, peak_idx - hw)
        b = min(n_samp, peak_idx + hw + 1)
        return a, b

    g = int(offset) if offset is not None else 16
    start = peak_idx + g
    if start + win_len <= n_samp:
        return start, start + win_len
    start = max(0, peak_idx - g - win_len)
    a = start
    b = min(n_samp, start + win_len)
    if a >= b:
        return 0, min(n_samp, win_len)
    return a, b


def _format_integrated_title(tel_label, et_name, half_width, mode, offset):
    win_len = 2 * int(half_width) + 1
    if mode == "signal":
        return f"{tel_label} integrated signal (win {win_len}) ({et_name})"
    g = int(offset) if offset is not None else 16
    return f"{tel_label} integrated pedestal (win {win_len}, offset {g}) ({et_name})"


def _select_event_by_type(source):
    """
    Build an event selector for a ctapipe EventSource.

    Parameters
    ----------
    source : ctapipe.io.EventSource
        Iterable event source.

    Returns
    -------
    callable
        A function ``select_event(event_index: int | None) -> Any`` that returns
        the first event (when ``event_index`` is None) or the event at the given
        index. Returns ``None`` if no event is available or the index is out of range.
    """

    def select_event(event_index=None):
        if event_index is None:
            for ev in source:
                return ev
        else:
            for idx, ev in enumerate(source):
                if idx == event_index:
                    return ev
        _logger.warning("No events available from source or event_index out of range")
        return None

    return select_event


def _time_axis_from_readout(readout, n_samp):
    """
    Compute time axis in nanoseconds from a camera readout.

    Parameters
    ----------
    readout : Any
        Camera readout providing ``sampling_rate`` as an astropy Quantity.
    n_samp : int
        Number of samples per trace.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_samp,)`` with time in nanoseconds.
    """
    try:
        dt = (1 / readout.sampling_rate).to(u.ns).value
    except (AttributeError, TypeError):
        dt = 1.0
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0
    return np.arange(int(n_samp)) * float(dt)


def plot_simtel_event_image(filename, distance=None, event_index=None):
    """
    Read a sim_telarray file and plot the DL1 image for one event via ctapipe.

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the ``.simtel`` file.
    distance : astropy.units.Quantity | float | None, optional
        Distance to annotate in the plot. If a Quantity, interpreted in meters.
        If not provided, no unit conversion is attempted.
    event_index : int | None, optional
        Zero-based index of the event to plot. If None, the first event is used.

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure, or ``None`` if no suitable event/image is available.
    """
    prepared = _get_event_source_and_r1_tel(filename, event_index=event_index, warn_context=None)
    if prepared is None:
        return None
    source, event, tel_id = prepared

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
    tel_id=None,
    n_pixels=3,
    event_index=None,
):
    """
    Plot R1 time traces for a few pixels of one event.

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the ``.simtel`` file.
    tel_id : int | None, optional
        Telescope ID to use. If None, the first telescope with R1 data is chosen.
    n_pixels : int, optional
        Number of pixels with highest signal to plot. Default is 3.
    event_index : int | None, optional
        Zero-based index of the event to plot. If None, the first event is used.

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure, or ``None`` if R1 waveforms are unavailable.
    """
    prepared = _get_event_source_and_r1_tel(
        filename, event_index=event_index, warn_context="time traces plot"
    )
    if prepared is None:
        return None
    source, event, tel_id_default = prepared
    tel_id = tel_id or tel_id_default

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
    t = _time_axis_from_readout(readout, n_samp)

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


def plot_simtel_waveform_matrix(
    filename,
    tel_id=None,
    vmax=None,
    event_index=None,
    pixel_step=None,
):
    """
    Create a pseudocolor image of R1 waveforms (sample index vs. pixel id).

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the ``.simtel`` file.
    tel_id : int | None, optional
        Telescope ID to use. If None, the first telescope with R1 data is chosen.
    vmax : float | None, optional
        Upper limit for color normalization. If None, determined automatically.
    event_index : int | None, optional
        Zero-based index of the event to plot. If None, the first event is used.
    pixel_step : int | None, optional
        Step between plotted pixel ids (e.g., 1 plots all, 2 plots every second pixel).

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure, or ``None`` if R1 waveforms are unavailable.
    """
    prepared = _get_event_source_and_r1_tel(
        filename, event_index=event_index, warn_context="waveform plot"
    )
    if prepared is None:
        return None
    source, event, tel_id_default = prepared
    tel_id = tel_id or tel_id_default

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning(NO_R1_WAVEFORMS_MSG)
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    step = max(1, int(pixel_step)) if pixel_step is not None else 1
    pix_idx = np.arange(n_pix)[::step]
    w_sel = w[pix_idx]

    readout = source.subarray.tel[tel_id].camera.readout
    t = _time_axis_from_readout(readout, n_samp)

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
    tel_id=None,
    pixel_step=100,
    max_pixels=None,
    event_index=None,
):
    """
    Plot step-style R1 traces for regularly sampled pixels (0, N, 2N, ...).

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the ``.simtel`` file.
    tel_id : int | None, optional
        Telescope ID to use. If None, the first telescope with R1 data is chosen.
    pixel_step : int, optional
        Interval between pixel indices to plot. Default is 100.
    max_pixels : int | None, optional
        Maximum number of pixels to plot. If None, plot all selected by ``pixel_step``.
    event_index : int | None, optional
        Zero-based index of the event to plot. If None, the first event is used.

    Returns
    -------
    matplotlib.figure.Figure | None
        The created figure, or ``None`` if R1 waveforms are unavailable.
    """
    prepared = _get_event_source_and_r1_tel(
        filename, event_index=event_index, warn_context="traces plot"
    )
    if prepared is None:
        return None
    source, event, tel_id_default = prepared
    tel_id = tel_id or tel_id_default

    waveforms = getattr(event.r1.tel.get(tel_id, None), "waveform", None)
    if waveforms is None:
        _logger.warning(NO_R1_WAVEFORMS_MSG)
        return None

    w = np.asarray(waveforms)
    if w.ndim == 3:
        w = w[0]
    n_pix, n_samp = w.shape

    readout = source.subarray.tel[tel_id].camera.readout
    t = _time_axis_from_readout(readout, n_samp)

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
    """
    Detect peak indices using CWT if available, otherwise ``find_peaks``.

    Parameters
    ----------
    trace : numpy.ndarray
        One-dimensional waveform samples for a single pixel.
    peak_width : int
        Characteristic peak width (samples) for CWT.
    signal_mod : module
        SciPy ``signal``-like module providing ``find_peaks_cwt``/``find_peaks``.

    Returns
    -------
    numpy.ndarray
        Array of integer indices of detected peaks (possibly empty).
    """
    peaks = []
    try:
        if hasattr(signal_mod, "find_peaks_cwt"):
            peaks = signal_mod.find_peaks_cwt(trace, widths=np.array([peak_width]))
        if not np.any(peaks):
            peaks, _ = signal_mod.find_peaks(trace, prominence=np.max(trace) * 0.1)
    except (ValueError, TypeError):
        peaks = []
    return np.asarray(peaks, dtype=int) if np.size(peaks) else np.array([], dtype=int)


def _collect_peak_samples(w, sum_threshold, peak_width, signal_mod):
    """
    Compute peak-sample indices per pixel from waveform matrix.

    Parameters
    ----------
    w : numpy.ndarray
        Waveform array of shape ``(n_pix, n_samples)`` (or ``(1, n_pix, n_samples)``).
    sum_threshold : float
        Minimum sum over samples for a pixel to be considered.
    peak_width : int
        Characteristic peak width (samples) for CWT.
    signal_mod : module
        SciPy ``signal``-like module providing peak finding routines.

    Returns
    -------
    tuple[numpy.ndarray | None, numpy.ndarray | None, int]
        ``(peak_samples, pix_ids, found_count)`` where ``peak_samples`` are the
        selected peak indices per considered pixel, ``pix_ids`` are the pixel
        indices that passed ``sum_threshold``, and ``found_count`` is the number
        of pixels with at least one detected peak. Returns ``(None, None, 0)`` if
        no pixels passed the threshold.
    """
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
    """
    Compute contiguous histogram bin edges for sample indices.

    Parameters
    ----------
    n_samp : int
        Number of samples per trace.
    timing_bins : int | None
        Number of histogram bins. If None, use unit-width bins.

    Returns
    -------
    numpy.ndarray
        Array of bin edges spanning the sample index range.
    """
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
    """
    Draw a histogram of peak samples with overlays and annotations.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw into.

        Peak sample indices per pixel.
    edges : numpy.ndarray
        Histogram bin edges.
    mean_sample : float
        Mean of peak sample indices.
    std_sample : float
        Standard deviation of peak sample indices.
    tel_label : str
        Telescope label used in the title.
    et_name : str
        Event type name used in the title.
    considered : int
        Number of pixels considered (passed threshold).
    found_count : int
        Number of pixels with at least one detected peak.

    Returns
    -------
    None
    """
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
    tel_id=None,
    sum_threshold=10.0,
    peak_width=8,
    examples=3,
    timing_bins=None,
    return_stats=False,
    event_index=None,
):
    """
    Peak finding per pixel; report mean/std of peak sample and plot a histogram.

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the ``.simtel`` file.
    tel_id : int | None, optional
        Telescope ID to use. If None, the first telescope with R1 data is chosen.
    sum_threshold : float, optional
        Minimum sum over samples for a pixel to be considered. Default is 10.0.
    peak_width : int, optional
        Characteristic peak width (samples) for CWT. Default is 8.
    examples : int, optional
        Number of example pixel traces to overlay. Default is 3.
    timing_bins : int | None, optional
        Number of histogram bins. If None, use unit-width bins.
    return_stats : bool, optional
        If True, also return a statistics dictionary. Default is False.
    event_index : int | None, optional
        Zero-based index of the event to plot. If None, the first event is used.

    Returns
    -------
    matplotlib.figure.Figure | tuple[matplotlib.figure.Figure, dict] | None
        The created figure, or ``None`` if R1 waveforms are unavailable. If
        ``return_stats`` is True, a tuple ``(fig, stats)`` is returned, where
        ``stats`` has keys ``{"considered", "found", "mean", "std"}``.
    """
    prepared = _get_event_source_and_r1_tel(
        filename, event_index=event_index, warn_context="peak timing plot"
    )
    if prepared is None:
        return None
    source, event, tel_id_default = prepared
    tel_id = tel_id or tel_id_default

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
    t = _time_axis_from_readout(readout, n_samp)

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


def _prepare_waveforms_for_image(filename, tel_id, context_no_r1, event_index=None):
    """
    Fetch R1 waveforms for one event/telescope and return prepared arrays.

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the ``.simtel`` file.
    tel_id : int | None
        Telescope ID to use. If None, the first telescope with R1 data is chosen.
    context_no_r1 : str
        Short description used in warnings when no R1 data is available.
    event_index : int | None, optional
        Zero-based index of the event to use. If None, the first event is used.

    Returns
    -------
    tuple | None
        ``(w, n_pix, n_samp, source, event, tel_id)`` where ``w`` is a
        ``numpy.ndarray`` of shape ``(n_pix, n_samples)``, ``n_pix`` and
        ``n_samp`` are integers, and ``source``, ``event`` and ``tel_id`` are
        the ctapipe objects used. Returns ``None`` on failure.
    """
    prepared = _get_event_source_and_r1_tel(
        filename, event_index=event_index, warn_context=context_no_r1
    )
    if prepared is None:
        return None
    source, event, tel_id_default = prepared
    tel_id = tel_id or tel_id_default

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
    tel_id=None,
    half_width=8,
    event_index=None,
):
    """Plot camera image of integrated signal per pixel around the flasher peak."""
    return _plot_simtel_integrated_image(
        filename=filename,
        tel_id=tel_id,
        half_width=half_width,
        event_index=event_index,
        mode="signal",
    )


def plot_simtel_integrated_pedestal_image(
    filename,
    tel_id=None,
    half_width=8,
    offset=16,
    event_index=None,
):
    """Plot camera image of integrated pedestal per pixel away from the flasher peak."""
    return _plot_simtel_integrated_image(
        filename=filename,
        tel_id=tel_id,
        half_width=half_width,
        event_index=event_index,
        mode="pedestal",
        offset=offset,
    )


def _plot_simtel_integrated_image(
    filename,
    tel_id,
    half_width,
    event_index,
    mode,
    offset=None,
):
    """Shared implementation for integrated signal/pedestal images.

    mode: "signal" or "pedestal". For "pedestal", ``offset`` is used.
    """
    context = "integrated-signal image" if mode == "signal" else "integrated-pedestal image"
    prepared = _prepare_waveforms_for_image(filename, tel_id, context, event_index=event_index)
    if prepared is None:
        return None

    w, n_pix, n_samp, source, event, tel_id = prepared
    img = np.zeros(n_pix, dtype=float)

    for pid in range(n_pix):
        trace = w[pid]
        peak_idx = int(np.argmax(trace))
        a, b = _compute_integration_window(peak_idx, n_samp, half_width, mode, offset)
        img[pid] = float(np.sum(trace[a:b]))

    geometry = source.subarray.tel[tel_id].camera.geometry
    fig, ax = plt.subplots(dpi=300)
    disp = CameraDisplay(geometry, image=img, norm="lin", ax=ax)
    disp.cmap = "viridis" if mode == "signal" else "cividis"
    disp.add_colorbar(fraction=0.02, pad=-0.1)
    disp.set_limits_percent(100)

    et_name = getattr(getattr(event.trigger, "event_type", None), "name", "?")
    tel = source.subarray.tel[tel_id]
    tel_label = getattr(tel, "name", f"CT{tel_id}")
    ax.set_title(_format_integrated_title(tel_label, et_name, half_width, mode, offset), pad=20)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _make_output_paths(ioh, base, input_file):
    """Return (out_dir, pdf_path) based on base name and input file."""
    out_dir = ioh.get_output_directory(label=Path(__file__).stem)
    pdf_path = ioh.get_output_file(f"{base}_{input_file.stem}" if base else input_file.stem)
    pdf_path = Path(f"{pdf_path}.pdf") if Path(pdf_path).suffix != ".pdf" else Path(pdf_path)
    return out_dir, pdf_path


def _call_peak_timing(
    filename,
    tel_id=None,
    sum_threshold=10.0,
    peak_width=8,
    examples=3,
    timing_bins=None,
    event_index=None,
):
    """Call plot_simtel_peak_timing while tolerating older signature.

    Returns a matplotlib Figure or None.
    """
    try:
        fig_stats = plot_simtel_peak_timing(
            filename,
            tel_id=tel_id,
            sum_threshold=sum_threshold,
            peak_width=peak_width,
            examples=examples,
            timing_bins=timing_bins,
            return_stats=True,
            event_index=event_index,
        )
        return fig_stats[0] if isinstance(fig_stats, tuple) else fig_stats
    except TypeError:
        return plot_simtel_peak_timing(
            filename,
            tel_id=tel_id,
            sum_threshold=sum_threshold,
            peak_width=peak_width,
            examples=examples,
            timing_bins=timing_bins,
            event_index=event_index,
        )


def _collect_figures_for_file(
    filename,
    plots,
    args,
    out_dir,
    base_stem,
    save_pngs,
    dpi,
):
    """Generate selected plots for a single sim_telarray file.

    Returns a list of figures. If ``save_pngs`` is True, also writes PNGs to ``out_dir``.
    """
    figures = []

    def add(fig, tag):
        if fig is not None:
            figures.append(fig)
            if save_pngs:
                base_path = out_dir / f"{base_stem}_{tag}"
                try:
                    save_figure(fig, base_path, figure_format=["png"], dpi=int(dpi))
                except Exception as ex:  # pylint:disable=broad-except
                    _logger.warning("Failed to save PNG %s: %s", base_path.with_suffix(".png"), ex)
        else:
            _logger.warning("Plot '%s' returned no figure for %s", tag, filename)

    plots_to_run = (
        [
            "event_image",
            "time_traces",
            "waveform_matrix",
            "step_traces",
            "integrated_signal_image",
            "integrated_pedestal_image",
            "peak_timing",
        ]
        if "all" in plots
        else list(plots)
    )

    dispatch = {
        "event_image": (
            plot_simtel_event_image,
            {"distance": None, "event_index": None},
        ),
        "time_traces": (
            plot_simtel_time_traces,
            {"tel_id": None, "n_pixels": 3, "event_index": None},
        ),
        "waveform_matrix": (
            plot_simtel_waveform_matrix,
            {"tel_id": None, "vmax": None, "event_index": None},
        ),
        "step_traces": (
            plot_simtel_step_traces,
            {"tel_id": None, "pixel_step": None, "max_pixels": None, "event_index": None},
        ),
        "integrated_signal_image": (
            plot_simtel_integrated_signal_image,
            {"tel_id": None, "half_width": 8, "event_index": None},
        ),
        "integrated_pedestal_image": (
            plot_simtel_integrated_pedestal_image,
            {"tel_id": None, "half_width": 8, "offset": 16, "event_index": None},
        ),
        "peak_timing": (
            _call_peak_timing,
            {
                "tel_id": None,
                "sum_threshold": 10.0,
                "peak_width": 8,
                "examples": 3,
                "timing_bins": None,
                "event_index": None,
            },
        ),
    }

    for plot_name in plots_to_run:
        entry = dispatch.get(plot_name)
        if entry is None:
            _logger.warning("Unknown plot selection '%s'", plot_name)
            continue
        func, defaults = entry
        kwargs = {k: args.get(k, v) for k, v in defaults.items()}
        fig = func(filename, **kwargs)  # type: ignore[misc]
        add(fig, plot_name)

    return figures


def generate_and_save_plots(
    simtel_files,
    plots,
    args,
    ioh,
):
    """Generate plots for files and save a multi-page PDF per input.

    Also writes metadata JSON next to the PDF.
    """
    for simtel in simtel_files:
        out_dir, pdf_path = _make_output_paths(ioh, args.get("output_file"), simtel)
        figures = _collect_figures_for_file(
            filename=simtel,
            plots=plots,
            args=args,
            out_dir=out_dir,
            base_stem=simtel.stem,
            save_pngs=bool(args.get("save_pngs", False)),
            dpi=int(args.get("dpi", 300)),
        )

        if not figures:
            _logger.warning("No figures produced for %s", simtel)
            continue

        try:
            save_figs_to_pdf(figures, pdf_path)
            _logger.info("Saved PDF: %s", pdf_path)
        except Exception as ex:  # pylint:disable=broad-except
            _logger.error("Failed to save PDF %s: %s", pdf_path, ex)

        try:
            MetadataCollector.dump(args, pdf_path, add_activity_name=True)
        except Exception as ex:  # pylint:disable=broad-except
            _logger.warning("Failed to write metadata for %s: %s", pdf_path, ex)
