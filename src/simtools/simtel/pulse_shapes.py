"""Pulse shape computations for light emission simulations for flasher."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import fftconvolve

_logger = logging.getLogger(__name__)


def _rise_10_90_width(t, y):
    i_peak = int(np.argmax(y))
    tr = t[: i_peak + 1]
    yr = y[: i_peak + 1]
    t10 = np.interp(0.1, yr, tr)
    t90 = np.interp(0.9, yr, tr)
    return t90 - t10


def _fall_90_10_width(t, y):
    i_peak = int(np.argmax(y))
    tf = t[i_peak:]
    yf = y[i_peak:]
    t90 = np.interp(0.9, yf[::-1], tf[::-1])
    t10 = np.interp(0.1, yf[::-1], tf[::-1])
    return t10 - t90


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


def solve_sigma_tau_from_risefall(rise_10_90_ns, fall_90_10_ns, dt_ns=0.1):
    """Solve (sigma, tau) so convolved pulse matches target 10-90 rise and 90-10 fall widths."""
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

    # use an initial tau guess from exponential 90-10 width
    tau_guess = max(fall_90_10_ns / np.log(9.0), 1e-6)

    def rise_error(sigma):
        y = pulse_for(sigma, tau_guess)
        return abs(_rise_10_90_width(t, y) - rise_10_90_ns)

    res = minimize_scalar(rise_error, bounds=(0.3, 3.0), method="bounded")
    sigma = float(res.x)

    def fall_residual(tau):
        y = pulse_for(sigma, tau)
        return _fall_90_10_width(t, y) - fall_90_10_ns

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


def generate_pulse_from_risefall(rise_10_90_ns, fall_90_10_ns, dt_ns=0.1, duration_sigma=8.0):
    """Get (t, y) by solving parameters from rise/fall and convolving Gaussian with exponential."""
    sigma, tau = solve_sigma_tau_from_risefall(rise_10_90_ns, fall_90_10_ns, dt_ns=dt_ns)
    return generate_gauss_expconv_pulse(sigma, tau, dt_ns=dt_ns, duration_sigma=duration_sigma)
