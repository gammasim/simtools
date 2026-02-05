"""Trace analysis."""

import numpy as np
from scipy import signal


def compute_integration_window(peak_idx, n_samp, half_width, mode, offset=None):
    """
    Compute integration window bounds for signal or pedestal modes.

    Parameters
    ----------
    peak_idx : int
        Index of the peak in the waveform.
    n_samp : int
        Total number of samples in the waveform.
    half_width : int
        Half-width of the integration window.
    mode : str
        Either "signal" or "pedestal".
    offset : int, optional
        Offset for pedestal window from peak. Default is 16.

    Returns
    -------
    tuple of int
        Window bounds (start, end) for integration [start:end).
    """
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


def calculate_pedestals(trace, start=None, end=None, half_width=2, offset=16):
    """
    Calculate pedestal per pixel.

    If start/end are provided, uses fixed window. Otherwise, computes pedestal
    window dynamically based on detected peak position.

    Parameters
    ----------
    trace : array-like
        Trace, shape (n_pixels, n_samples) or (1, n_pixels, n_samples).
    start : int, optional
        Start index for fixed window. If None, uses dynamic windowing.
    end : int, optional
        End index for fixed window.
    half_width : int, optional
        Half-width for dynamic pedestal window. Default is 2.
    offset : int, optional
        Offset from peak for dynamic pedestal window. Default is 16.

    Returns
    -------
    np.ndarray
        Mean pedestal value per pixel.
    """
    w = np.atleast_2d(trace)
    if w.ndim == 3:
        w = w[0]

    if start is not None and end is not None:
        return np.mean(w[:, start:end], axis=-1)

    # Dynamic windowing based on peak detection
    n_pix, n_samp = w.shape
    pedestals = np.zeros(n_pix)
    for i in range(n_pix):
        peak_idx = int(np.argmax(w[i]))
        a, b = compute_integration_window(peak_idx, n_samp, half_width, "pedestal", offset)
        pedestals[i] = np.mean(w[i, a:b])

    return pedestals


def trace_integration(trace, pedestals, window):
    """
    Integrate traces over specific window.

    Parameters
    ----------
    trace : array-like
        Trace, shape (n_pixels, n_samples) or (1, n_pixels, n_samples).
    pedestals : array-like or None
        Pedestal values per pixel. If None, pedestals are calculated.
    window : tuple of int
        Integration window (start, end) indices.

    Returns
    -------
    np.ndarray
        Integrated trace values per pixel.
    """
    if pedestals is None:
        pedestals = calculate_pedestals(trace)

    w = np.squeeze(trace)
    if w.ndim == 1:
        w = w[np.newaxis, :]

    calibrated = w - np.atleast_1d(pedestals)[:, None]
    return np.sum(calibrated[:, int(window[0]) : int(window[1])], axis=1)


def find_signal_peaks(trace, prominence=None):
    """Find peaks using scipy.signal with automatic prominence."""
    trace = np.asarray(trace)
    prom = prominence or (np.max(np.abs(trace)) * 0.1 + 1e-6)
    peaks, _ = signal.find_peaks(trace, prominence=prom)

    return peaks if peaks.size > 0 else np.array([np.argmax(np.abs(trace))])


def get_time_axis(sampling_rate, n_samples):
    """
    Generate time axis using sampling frequency logic.

    Parameters
    ----------
    sampling_rate : astropy.units.Quantity
        Sampling rate with time units (e.g., ns).
    n_samples : int
        Number of samples.

    Returns
    -------
    np.ndarray
        Time axis array.
    """
    dt = 1.0 / sampling_rate.to("ns").value if sampling_rate.to("ns").value > 0 else 1.0
    return np.linspace(0, (n_samples - 1) * dt, n_samples)


def get_adc_samples_per_gain(adc_samples, low_gain=False):
    """
    Get ADC samples for low gain channel if specified.

    Parameters
    ----------
    adc_samples : array-like
        ADC samples array, shape (n_pixels, n_samples) or (1, n_pixels, n_samples).
    low_gain : bool
        If True, return low gain channel samples.

    Returns
    -------
    np.ndarray
        ADC samples for the specified gain channel.
    """
    if adc_samples.ndim == 3 and low_gain:
        adc_samples = np.asarray(adc_samples[1])
    return np.asarray(adc_samples[0])


def trace_maxima(trace, sum_threshold):
    """
    Compute trace maxima sample per pixel from waveform matrix.

    Parameters
    ----------
    trace : numpy.ndarray
        Trace array of shape ``(n_pix, n_samples)`` (or ``(1, n_pix, n_samples)``).
    sum_threshold : float
        Minimum sum over samples for a pixel to be considered.

    Returns
    -------
    tuple[numpy.ndarray | None, numpy.ndarray | None, int]
        ``(peak_samples, pix_ids, found_count)`` where ``peak_samples`` are the
        selected peak indices per considered pixel, ``pix_ids`` are the pixel
        indices that passed ``sum_threshold``, and ``found_count`` is the number
        of pixels with at least one detected peak. Returns ``(None, None, 0)`` if
        no pixels passed the threshold.
    """
    sums = trace.sum(axis=1)
    pix_ids = np.flatnonzero(sums > float(sum_threshold))
    if pix_ids.size == 0:
        return None, None, 0

    peak_samples = np.empty(pix_ids.size, dtype=int)
    found_count = 0

    for i, pid in enumerate(pix_ids):
        pixel_trace = trace[pid]
        pks = trace_maximum(pixel_trace)

        if pks.size:
            found_count += 1
            peak_samples[i] = pks[np.argmax(pixel_trace[pks])]
        else:
            peak_samples[i] = np.argmax(pixel_trace)

    return peak_samples, pix_ids, found_count


def trace_maximum(trace):
    """
    Detect trace maximum.

    Parameters
    ----------
    trace : numpy.ndarray
        One-dimensional trace for a single pixel.

    Returns
    -------
    numpy.ndarray
        Array of integer indices of detected peaks (possibly empty).
    """
    try:
        peaks, _ = signal.find_peaks(trace, prominence=0.1 * np.max(trace))
        return peaks.astype(int)
    except (ValueError, TypeError):
        return np.empty(0, dtype=int)
