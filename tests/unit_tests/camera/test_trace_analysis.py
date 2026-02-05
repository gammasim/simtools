import numpy as np
import pytest
from astropy import units as u

import simtools.camera.trace_analysis as trace


# ============================================================================
# compute_integration_window tests - parametrized
# ============================================================================
@pytest.mark.parametrize(
    ("peak_idx", "n_samp", "half_width", "mode", "offset", "expected_a", "expected_b"),
    [
        (50, 100, 5, "signal", None, 45, 56),
        (2, 100, 5, "signal", None, 0, 8),
        (98, 100, 5, "signal", None, 93, 100),
        (50, 200, 2, "pedestal", None, 66, 71),
        (50, 200, 3, "pedestal", 10, 60, 67),
    ],
)
def test_compute_integration_window(
    peak_idx, n_samp, half_width, mode, offset, expected_a, expected_b
):
    if offset is None:
        a, b = trace.compute_integration_window(
            peak_idx=peak_idx, n_samp=n_samp, half_width=half_width, mode=mode
        )
    else:
        a, b = trace.compute_integration_window(
            peak_idx=peak_idx, n_samp=n_samp, half_width=half_width, mode=mode, offset=offset
        )
    assert a == expected_a
    assert b == expected_b


def test_compute_integration_window_pedestal_mode_near_end():
    """Test pedestal mode when window would exceed waveform length."""
    a, b = trace.compute_integration_window(
        peak_idx=190, n_samp=200, half_width=2, mode="pedestal", offset=16
    )
    assert b <= 200
    assert b - a == 5


def test_compute_integration_window_pedestal_mode_window_wraps():
    """Test pedestal mode fallback when offset window exceeds bounds."""
    a, b = trace.compute_integration_window(
        peak_idx=10, n_samp=100, half_width=10, mode="pedestal", offset=50
    )
    assert a >= 0
    assert b <= 100


# ============================================================================
# calculate_pedestals tests - parametrized
# ============================================================================
@pytest.mark.parametrize(
    ("waveform_shape", "peak_pos", "half_width", "offset", "expected_match"),
    [
        ((1, 20), (0, 0), 2, 2, lambda w: np.mean(w[0, 2:7])),
        ((1, 20), (0, 19), 2, 2, lambda w: np.mean(w[0, 14:19])),
        ((1, 2, 10), None, None, None, None),  # 3D case
    ],
)
def test_calculate_pedestals_dynamic_window(
    waveform_shape, peak_pos, half_width, offset, expected_match
):
    waveform = np.zeros(waveform_shape)
    if peak_pos:
        waveform[peak_pos] = 10
    if waveform_shape == (1, 2, 10):
        waveform[0, 0, 5] = 5
        waveform[0, 1, 7] = 7

    if half_width is None:
        ped = trace.calculate_pedestals(waveform)
        assert ped.shape == (2,)
    else:
        ped = trace.calculate_pedestals(waveform, half_width=half_width, offset=offset)
        expected = expected_match(waveform)
        assert np.allclose(ped[0], expected)


def test_calculate_pedestals_fixed_window():
    waveform = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    ped = trace.calculate_pedestals(waveform, start=2, end=5)
    expected = np.mean(waveform[:, 2:5], axis=-1)
    assert np.allclose(ped, expected)


def test_calculate_pedestals_dynamic_window_peak_centered():
    waveform = np.zeros((2, 20))
    waveform[0, 10] = 10
    waveform[1, 15] = 20
    ped = trace.calculate_pedestals(waveform, half_width=2, offset=3)
    assert np.allclose(ped[0], np.mean(waveform[0, 13:18]))
    assert np.allclose(ped[1], np.mean(waveform[1, 10:15]))


# ============================================================================
# trace_integration tests - parametrized
# ============================================================================
@pytest.mark.parametrize(
    ("waveform", "pedestals", "window", "expected_calc"),
    [
        (np.array([[1, 2, 3, 4, 5]]), np.array([1]), (1, 4), lambda w, p: np.sum(w[0, 1:4] - p[0])),
        (np.array([5, 7, 9, 11]), np.array([5]), (1, 3), lambda w, p: np.sum(w[1:3] - p[0])),
        (
            np.array([[1, 2, 3, 4], [10, 20, 30, 40]]),
            np.array([1, 10]),
            (0, 2),
            lambda w, p: np.array([np.sum(w[0, 0:2] - p[0]), np.sum(w[1, 0:2] - p[1])]),
        ),
    ],
)
def test_trace_integration(waveform, pedestals, window, expected_calc):
    result = trace.trace_integration(waveform, pedestals, window)
    expected = expected_calc(waveform, pedestals)
    assert np.allclose(result, expected)


def test_trace_integration_no_pedestals():
    waveform = np.array([[2, 4, 6, 8, 10]])
    result = trace.trace_integration(waveform, None, (0, 3))
    ped = trace.calculate_pedestals(waveform)
    expected = np.sum(waveform[0, 0:3] - ped[0])
    assert np.allclose(result, expected)


def test_trace_integration_with_3d_input():
    waveform = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    pedestals = np.array([1, 5])
    window = (1, 4)
    result = trace.trace_integration(waveform, pedestals, window)
    expected = np.array([np.sum(waveform[0, 0, 1:4] - 1), np.sum(waveform[0, 1, 1:4] - 5)])
    assert np.allclose(result, expected)


# ============================================================================
# find_signal_peaks tests - parametrized
# ============================================================================
@pytest.mark.parametrize(
    ("waveform_desc", "expected_peaks", "prominence"),
    [
        ("single_peak", [10], None),
        ("multiple_peaks", [5, 15, 25], None),
        ("negative_peak", [4], None),
        ("flat_trace", [0], None),
        ("prominence_filter", [3], 1.5),
    ],
)
def test_find_signal_peaks(waveform_desc, expected_peaks, prominence):
    if waveform_desc == "single_peak":
        waveform = np.zeros(20)
        waveform[10] = 5
    elif waveform_desc == "multiple_peaks":
        waveform = np.zeros(30)
        waveform[[5, 15, 25]] = [3, 7, 2]
    elif waveform_desc == "negative_peak":
        waveform = np.zeros(10)
        waveform[4] = -5
    elif waveform_desc == "flat_trace":
        waveform = np.full(8, 2.0)
    elif waveform_desc == "prominence_filter":
        waveform = np.array([0, 1, 0, 2, 0, 1, 0])

    if prominence:
        peaks = trace.find_signal_peaks(waveform, prominence=prominence)
    else:
        peaks = trace.find_signal_peaks(waveform)

    assert set(peaks) == set(expected_peaks)


def test_find_signal_peaks_no_peaks_returns_max():
    waveform = np.array([1, 1, 1, 1])
    peaks = trace.find_signal_peaks(waveform)
    assert peaks.size == 1
    assert peaks[0] == 0


def test_find_signal_peaks_empty_trace():
    waveform = np.array([])
    with pytest.raises(ValueError, match="zero-size array"):
        trace.find_signal_peaks(waveform)


# ============================================================================
# get_time_axis tests - parametrized
# ============================================================================
@pytest.mark.parametrize(
    ("sampling_rate_ns", "n_samples", "expected_max"),
    [
        (1, 5, 4),
        (2, 4, 1.5),
        (0, 3, 2),
        (4, 3, 0.5),
        (1, 1, 0),
    ],
)
def test_get_time_axis(sampling_rate_ns, n_samples, expected_max):
    sampling_rate = sampling_rate_ns * u.ns
    axis = trace.get_time_axis(sampling_rate, n_samples)
    expected = np.linspace(0, expected_max, n_samples)
    assert np.allclose(axis, expected)


# ============================================================================
# get_adc_samples_per_gain tests - parametrized
# ============================================================================
@pytest.mark.parametrize(
    ("adc_shape", "low_gain", "expected_result"),
    [
        ((2, 3), False, [1, 2, 3]),
        ((2, 3), True, [1, 2, 3]),
        ((2, 2, 3), False, [[1, 2, 3], [4, 5, 6]]),
        ((2, 2, 3), True, [10, 20, 30]),
    ],
)
def test_get_adc_samples_per_gain(adc_shape, low_gain, expected_result):
    if adc_shape == (2, 3):
        adc_samples = np.array([[1, 2, 3], [4, 5, 6]])
    else:  # (2, 2, 3)
        adc_samples = np.array([[[1, 2, 3], [4, 5, 6]], [[10, 20, 30], [40, 50, 60]]])

    result = trace.get_adc_samples_per_gain(adc_samples, low_gain=low_gain)
    assert np.allclose(result, expected_result)


def test_get_adc_samples_per_gain_high_gain_1d():
    adc_samples = np.array([7, 8, 9])
    result = trace.get_adc_samples_per_gain(adc_samples, low_gain=False)
    assert np.allclose(result, 7)


# ============================================================================
# trace_maxima tests - parametrized
# ============================================================================
@pytest.mark.parametrize(
    ("waveform_desc", "sum_threshold", "expected_count", "expected_pix_ids"),
    [
        ("basic", 5, 2, {0, 2}),
        ("all_below", 1, 0, None),
        ("multiple_peaks", 5, 1, {0}),
        ("with_3d", 1, 2, {0, 1}),
    ],
)
def test_trace_maxima(waveform_desc, sum_threshold, expected_count, expected_pix_ids):
    if waveform_desc == "basic":
        waveform = np.array([[0, 1, 3, 2, 0], [0, 0, 0, 0, 0], [2, 5, 1, 0, 2]])
    elif waveform_desc == "all_below":
        waveform = np.array([[0.1, 0.2, 0.1], [0.05, 0.05, 0.05]])
    elif waveform_desc == "multiple_peaks":
        waveform = np.array([[1, 3, 1, 3, 1], [0, 0, 0, 0, 0]])
    elif waveform_desc == "with_3d":
        waveform = np.zeros((1, 2, 6))
        waveform[0, 0, 2] = 5
        waveform[0, 1, 4] = 7
        waveform = waveform[0]

    peak_samples, pix_ids, found_count = trace.trace_maxima(waveform, sum_threshold=sum_threshold)

    assert found_count == expected_count
    if expected_count == 0:
        assert peak_samples is None
        assert pix_ids is None
    else:
        assert set(pix_ids) == expected_pix_ids


def test_trace_maxima_no_pixels_above_threshold():
    waveform = np.zeros((3, 10))
    peak_samples, pix_ids, found_count = trace.trace_maxima(waveform, sum_threshold=1)
    assert peak_samples is None
    assert pix_ids is None
    assert found_count == 0


def test_trace_maxima_peak_at_start_and_end():
    waveform = np.array([[0, 0, 10, 0, 0], [0, 0, 0, 0, 0]])
    peak_samples, pix_ids, found_count = trace.trace_maxima(waveform, sum_threshold=1)
    assert found_count == 1
    assert pix_ids[0] == 0
    assert peak_samples[0] == 2


def test_trace_maxima_single_pixel():
    waveform = np.array([[0, 2, 4, 3, 1]])
    peak_samples, pix_ids, found_count = trace.trace_maxima(waveform, sum_threshold=2)
    assert found_count == 1
    assert pix_ids[0] == 0
    assert peak_samples[0] == 2


def test_trace_maxima_empty_trace():
    waveform = np.empty((0, 5))
    peak_samples, pix_ids, found_count = trace.trace_maxima(waveform, sum_threshold=1)
    assert peak_samples is None
    assert pix_ids is None
    assert found_count == 0
