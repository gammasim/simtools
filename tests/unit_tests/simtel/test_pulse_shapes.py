import numpy as np

from simtools.simtel.pulse_shapes import (
    _exp_decay,
    _fall_width,
    _gaussian,
    _rise_width,
    generate_gauss_expconv_pulse,
    generate_pulse_from_rise_fall_times,
    solve_sigma_tau_from_risefall,
)


def test_solve_sigma_tau_basic():
    sigma, tau = solve_sigma_tau_from_risefall(2.5, 5.0, dt_ns=0.1)
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


def test__gaussian_peak_and_symmetry():
    t = np.linspace(-5.0, 5.0, 2001)
    y = _gaussian(t, 1.0)
    i0 = np.argmin(np.abs(t))
    assert np.isclose(y[i0], 1.0, atol=1e-6)
    assert np.isclose(y[i0 - 100], y[i0 + 100])


def test__exp_decay_causality_and_decay():
    t = np.linspace(-5.0, 5.0, 2001)
    y = _exp_decay(t, 2.0)
    assert np.allclose(y[t < 0], 0.0, atol=1e-12)
    i0 = np.argmin(np.abs(t))
    assert np.isclose(y[i0], 1.0, atol=1e-6)
    post = y[i0 : i0 + 200]
    assert np.all(np.diff(post) <= 1e-12)


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


def test__rise_fall_width_helpers_match_manual():
    t, y = generate_gauss_expconv_pulse(1.2, 3.0, dt_ns=0.05)
    rise_h = _rise_width(t, y, y_low=0.1, y_high=0.9)
    fall_h = _fall_width(t, y, y_high=0.9, y_low=0.1)
    rise_m, fall_m = _measure_rise_fall_widths(t, y)
    assert np.isclose(rise_h, rise_m, atol=0.1)
    assert np.isclose(fall_h, fall_m, atol=0.1)


def test_pulse_matches_rise_fall_targets():
    target_rise = 2.5
    target_fall = 5.0
    dt = 0.05
    t, y = generate_pulse_from_rise_fall_times(target_rise, target_fall, dt_ns=dt)
    rise, fall = _measure_rise_fall_widths(t, y)
    atol = 5 * dt
    assert np.isclose(rise, target_rise, atol=atol)
    assert np.isclose(fall, target_fall, atol=atol)


def test_time_step_and_window():
    sigma = 1.0
    tau = 3.0
    dt = 0.1
    t, _ = generate_gauss_expconv_pulse(sigma, tau, dt_ns=dt)
    assert np.allclose(np.diff(t), dt)
    # Window should straddle zero and provide both negative and positive times
    assert t[0] < 0
    assert t[-1] > 0
    assert np.isclose(t[0], -10.0, atol=1.5 * dt)
    assert np.isclose(t[-1], 25.0, atol=1.5 * dt)


def test_parameter_sensitivity_sigma_tau():
    t1, y1 = generate_gauss_expconv_pulse(0.8, 3.0, dt_ns=0.05)
    t2, y2 = generate_gauss_expconv_pulse(1.6, 3.0, dt_ns=0.05)
    r1, _ = _measure_rise_fall_widths(t1, y1)
    r2, _ = _measure_rise_fall_widths(t2, y2)
    assert r2 > r1
    t3, y3 = generate_gauss_expconv_pulse(1.0, 2.0, dt_ns=0.05)
    t4, y4 = generate_gauss_expconv_pulse(1.0, 4.0, dt_ns=0.05)
    _, f3 = _measure_rise_fall_widths(t3, y3)
    _, f4 = _measure_rise_fall_widths(t4, y4)
    assert f4 > f3
