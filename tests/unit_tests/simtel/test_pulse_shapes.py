import numpy as np

from simtools.simtel.pulse_shapes import (
    generate_gauss_expconv_pulse,
    generate_pulse_from_rise_fall_times,
    solve_sigma_tau_from_rise_fall,
)


def test_solve_sigma_tau_basic():
    sigma, tau = solve_sigma_tau_from_rise_fall(2.5, 5.0, dt_ns=0.1)
    assert sigma > 0
    assert tau > 0
    assert np.isclose(sigma, 1.294, atol=0.05)
    assert np.isclose(tau, 2.094, atol=0.05)


def test_generate_gauss_expconv_pulse_shape():
    t, y = generate_gauss_expconv_pulse(1.2, 3.5, dt_ns=0.1)
    assert np.isclose(np.max(y), 1.0, atol=0.05)
    assert t[0] < 0 < t[-1]
    left = y[: len(y) // 2]
    assert np.max(left) <= 1.0


def test_generate_pulse_from_risefall_roundtrip():
    t, y = generate_pulse_from_rise_fall_times(2.5, 5.0, dt_ns=0.05)
    assert y.size == t.size
    assert np.isclose(y.max(), 1.0)


def _measure_rise_fall_widths(t, y):
    i_peak = int(np.argmax(y))
    tr = t[: i_peak + 1]
    yr = y[: i_peak + 1]
    t10r = np.interp(0.1, yr, tr)
    t90r = np.interp(0.9, yr, tr)
    rise = t90r - t10r
    tf = t[i_peak:]
    yf = y[i_peak:]
    t90f = np.interp(0.9, yf[::-1], tf[::-1])
    t10f = np.interp(0.1, yf[::-1], tf[::-1])
    fall = t10f - t90f
    return rise, fall


def test_center_on_peak_shifts_time_to_zero():
    t, y = generate_gauss_expconv_pulse(1.2, 3.0, dt_ns=0.05, center_on_peak=True)
    i_max = int(np.argmax(y))
    assert abs(t[i_max]) <= 1e-6
