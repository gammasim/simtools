"""Measure baseline-subtracted digitized pulses."""

import numpy as np


def measure_waveform(samples, dt_ns, width_fraction=0.95, threshold_adc=None):
    """Return peak, sample sum, time integral, and width of the main pulse.

    Parameters
    ----------
    samples : array-like
        One baseline-subtracted waveform in ADC counts. Negative tails are retained.
    dt_ns : float
        Uniform sample interval in ns.
    width_fraction : float
        Fraction of the peak defining the width, unless an absolute threshold is supplied.
    threshold_adc : float, optional
        Absolute width threshold in ADC counts above pedestal.

    Returns
    -------
    dict
        Peak ADC, ADC sum, integral in ADC ns, first peak time in ns, width in ns,
        and a flag for a pulse whose threshold crossings leave the readout window.
        Width covers the contiguous component containing the first maximum, using
        linear interpolation at crossings. Width is zero if the peak is below threshold.

    Raises
    ------
    ValueError
        If the waveform, sampling interval, or threshold is invalid.
    """
    wave = np.asarray(samples, dtype=float)
    if wave.ndim != 1 or wave.size < 2:
        raise ValueError("A waveform requires at least two samples in one dimension")
    if not np.all(np.isfinite(wave)) or not np.isfinite(dt_ns) or dt_ns <= 0:
        raise ValueError("Waveform values must be finite and the sample interval positive")
    if not 0 < width_fraction < 1:
        raise ValueError("Width fraction must be between zero and one")
    if threshold_adc is not None and (not np.isfinite(threshold_adc) or threshold_adc <= 0):
        raise ValueError("Absolute width threshold must be finite and positive")
    peak_index = int(np.argmax(wave))
    peak = float(wave[peak_index])
    threshold = threshold_adc if threshold_adc is not None else width_fraction * peak
    width, truncated = _pulse_width(wave, peak_index, threshold, dt_ns)
    total = float(np.sum(wave))
    return {
        "peak_adc": peak,
        "adc_sum": total,
        "integral_adc_ns": total * dt_ns,
        "peak_time_ns": peak_index * dt_ns,
        "width_ns": width,
        "truncated": truncated,
    }


def _pulse_width(wave, peak_index, threshold, dt_ns):
    """Interpolate crossings around the main above-threshold component."""
    if wave[peak_index] <= 0 or wave[peak_index] < threshold:
        return 0.0, False
    left = right = peak_index
    while left > 0 and wave[left - 1] >= threshold:
        left -= 1
    while right < len(wave) - 1 and wave[right + 1] >= threshold:
        right += 1
    if left == 0 or right == len(wave) - 1:
        return float("nan"), True
    start = left - 1 + (threshold - wave[left - 1]) / (wave[left] - wave[left - 1])
    stop = right + (wave[right] - threshold) / (wave[right] - wave[right + 1])
    return float((stop - start) * dt_ns), False
