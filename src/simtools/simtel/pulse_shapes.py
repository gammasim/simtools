"""Pulse shape computations for light emission simulations for flasher."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import fftconvolve

_logger = logging.getLogger(__name__)


def _rise_width(t, y, y_low=0.1, y_high=0.9):
    """Width on the rising edge between fractional amplitudes y_low and y_high."""
    i_peak = int(np.argmax(y))
    tr = t[: i_peak + 1]
    yr = y[: i_peak + 1]
    t_low = np.interp(y_low, yr, tr)
    t_high = np.interp(y_high, yr, tr)
    return t_high - t_low


def _fall_width(t, y, y_high=0.9, y_low=0.1):
    """Width on the falling edge between fractional amplitudes y_high and y_low."""
    i_peak = int(np.argmax(y))
    tf = t[i_peak:]
    yf = y[i_peak:]
    t_rev = tf[::-1]
    y_rev = yf[::-1]
    t_hi = np.interp(y_high, y_rev, t_rev)
    t_lo = np.interp(y_low, y_rev, t_rev)
    return t_lo - t_hi


def _gaussian(t, sigma):
    return np.exp(-0.5 * (t / max(sigma, 1e-9)) ** 2)


def _exp_decay(t, tau):
    tau = max(tau, 1e-9)
    h = np.zeros_like(t)
    m = t >= 0
    h[m] = np.exp(-t[m] / tau)
    return h


def generate_gauss_expconv_pulse(sigma_ns, tau_ns, dt_ns=0.1, duration_sigma=8.0):
    """Return time and normalized pulse for Gaussian convolved with causal exponential."""
    left = -duration_sigma * sigma_ns
    right = duration_sigma * max(tau_ns, sigma_ns)
    t = np.arange(left, right + dt_ns, dt_ns, dtype=float)
    g = _gaussian(t, sigma_ns)
    e = _exp_decay(t, tau_ns)
    y = fftconvolve(g, e, mode="same")
    if y.max() > 0:
        y = y / y.max()
    return t, y


def solve_sigma_tau_from_risefall(
    rise_width_ns,
    fall_width_ns,
    dt_ns=0.1,
    rise_range=(0.1, 0.9),
    fall_range=(0.9, 0.1),
):
    """Solve (sigma, tau) so convolved pulse matches target rise/fall widths.

    Parameters
    ----------
    rise_width_ns : float
        Desired width on the rising edge in ns between rise_range=(low, high) fractions.
    fall_width_ns : float
        Desired width on the falling edge in ns between fall_range=(high, low) fractions.
    dt_ns : float
        Time step for internal pulse sampling in ns.
    rise_range : tuple[float, float]
        Fractional amplitudes (low, high) for the rising width, defaults to (0.1, 0.9).
    fall_range : tuple[float, float]
        Fractional amplitudes (high, low) for the falling width, defaults to (0.9, 0.1).
    """
    # Solving convolved expression, which is a transcendental equation
    # ~erf(1/tau - 1/sigma) * exp(sigma**2/(2*tau**2) - (t-mu)/tau)
    t = np.arange(-10.0, 25.0 + dt_ns, dt_ns, dtype=float)

    def pulse_for(sigma, tau):
        g = _gaussian(t, sigma)
        e = _exp_decay(t, tau)
        y = fftconvolve(g, e, mode="same")
        if y.max() > 0:
            y = y / y.max()
        return y

    # initial tau guess from exponential width between fall_range levels
    fall_hi, fall_lo = fall_range
    ratio = max(fall_hi / max(fall_lo, 1e-9), 1.000001)
    tau_guess = max(fall_width_ns / np.log(ratio), 1e-6)

    rise_lo, rise_hi = rise_range

    def rise_error(sigma):
        y = pulse_for(sigma, tau_guess)
        return abs(_rise_width(t, y, y_low=rise_lo, y_high=rise_hi) - rise_width_ns)

    res = minimize_scalar(rise_error, bounds=(0.3, 3.0), method="bounded")
    sigma = float(res.x)

    def fall_residual(tau):
        y = pulse_for(sigma, tau)
        return _fall_width(t, y, y_high=fall_hi, y_low=fall_lo) - fall_width_ns

    # bracket search for tau with expansion and fallback to minimization
    a, b = 0.01, 30.0
    fa, fb = fall_residual(a), fall_residual(b)
    expand = 0
    while fa * fb > 0 and expand < 10:
        b *= 2.0
        fb = fall_residual(b)
        expand += 1
    if fa * fb > 0:
        shrink = 0
        while fa * fb > 0 and shrink < 8 and a > 1e-6:
            a /= 2.0
            fa = fall_residual(a)
            shrink += 1
    if fa * fb > 0:
        res_tau = minimize_scalar(
            lambda x: abs(fall_residual(x)), bounds=(max(a, 1e-6), b), method="bounded"
        )
        tau = float(res_tau.x)
    else:
        tau = brentq(fall_residual, a, b, maxiter=200)

    _logger.info(f"Solved pulse parameters: sigma={sigma:.6g}, tau={tau:.6g}")
    return sigma, float(tau)


def generate_pulse_from_risefall(
    rise_width_ns,
    fall_width_ns,
    dt_ns=0.1,
    duration_sigma=8.0,
    rise_range=(0.1, 0.9),
    fall_range=(0.9, 0.1),
):
    """Get (t, y) by solving parameters from generic rise/fall specs and convolving.

    Defaults correspond to 10-90 rise and 90-10 fall widths.
    """
    sigma, tau = solve_sigma_tau_from_risefall(
        rise_width_ns, fall_width_ns, dt_ns=dt_ns, rise_range=rise_range, fall_range=fall_range
    )
    return generate_gauss_expconv_pulse(sigma, tau, dt_ns=dt_ns, duration_sigma=duration_sigma)
