"""Pulse shape computations for light emission simulations for flasher."""

import logging

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

_logger = logging.getLogger(__name__)


def _rise_width(t, y, y_low=0.1, y_high=0.9):
    """Measure rise width between fractional amplitudes.

    Parameters
    ----------
    t : array-like
        Time samples in ns.
    y : array-like
        Pulse amplitude samples (normalized or arbitrary units).
    y_low : float, optional
        Lower fractional amplitude (0..1) on the rising edge. Default is 0.1.
    y_high : float, optional
        Upper fractional amplitude (0..1) on the rising edge. Default is 0.9.

    Returns
    -------
    float
        Width in ns between ``y_low`` and ``y_high`` on the rising edge.
    """
    i_peak = int(np.argmax(y))
    tr = t[: i_peak + 1]
    yr = y[: i_peak + 1]
    t_low = np.interp(y_low, yr, tr)
    t_high = np.interp(y_high, yr, tr)
    return t_high - t_low


def _fall_width(t, y, y_high=0.9, y_low=0.1):
    """Measure fall width between fractional amplitudes.

    Parameters
    ----------
    t : array-like
        Time samples in ns.
    y : array-like
        Pulse amplitude samples (normalized or arbitrary units).
    y_high : float, optional
        Higher fractional amplitude (0..1) on the falling edge. Default is 0.9.
    y_low : float, optional
        Lower fractional amplitude (0..1) on the falling edge. Default is 0.1.

    Returns
    -------
    float
        Width in ns between ``y_high`` and ``y_low`` on the falling edge.
    """
    i_peak = int(np.argmax(y))
    tf = t[i_peak:]
    yf = y[i_peak:]
    t_rev = tf[::-1]
    y_rev = yf[::-1]
    t_hi = np.interp(y_high, y_rev, t_rev)
    t_lo = np.interp(y_low, y_rev, t_rev)
    return t_lo - t_hi


def _gaussian(t, sigma):
    """Gaussian pulse shape.

    Parameters
    ----------
    t : array-like
        Time samples in ns.
    sigma : float
        Gaussian standard deviation in ns.

    Returns
    -------
    numpy.ndarray
        Gaussian values at ``t`` (unitless), with a small safeguard for ``sigma``.
    """
    return np.exp(-0.5 * (t / max(sigma, 1e-9)) ** 2)


def _exp_decay(t, tau):
    """Causal exponential decay shape.

    Parameters
    ----------
    t : array-like
        Time samples in ns.
    tau : float
        Exponential decay constant in ns.

    Returns
    -------
    numpy.ndarray
        Exponential values at ``t`` (unitless), zero for ``t < 0``.
    """
    tau = max(tau, 1e-9)
    return np.where(t >= 0, np.exp(-t / tau), 0.0)


def generate_gauss_expconv_pulse(
    sigma_ns,
    tau_ns,
    dt_ns=0.1,
    t_start_ns=-10,
    t_stop_ns=25,
    center_on_peak=False,
):
    """Generate a Gaussian convolved with a causal exponential.

    Parameters
    ----------
    sigma_ns : float
        Gaussian standard deviation (ns).
    tau_ns : float
        Exponential decay constant (ns).
    dt_ns : float, optional
        Time sampling step (ns). Default is 0.1 ns.
    t_start_ns : float
        Together with ``t_stop_ns``, defines the explicit start of the time grid
        for pulse generation (ns).
    t_stop_ns : float
        Together with ``t_start_ns``, defines the explicit end of the time grid
        for pulse generation (ns).
    center_on_peak : bool, optional
        If True, shift the returned time array so the pulse maximum is at t=0.
        Default is False.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Tuple ``(t, y)`` with time samples in ns and normalized pulse amplitude (peak 1).
    """
    left = float(t_start_ns)
    right = float(t_stop_ns)
    t = np.arange(left, right, dt_ns, dtype=float)
    g = _gaussian(t, sigma_ns)
    e = _exp_decay(t, tau_ns)
    y = fftconvolve(g, e, mode="same")
    if y.max() > 0:
        y = y / y.max()
    if center_on_peak:
        i_max = int(np.argmax(y))
        t = t - float(t[i_max])
    return t, y


def solve_sigma_tau_from_risefall(
    rise_width_ns,
    fall_width_ns,
    dt_ns=0.1,
    rise_range=(0.1, 0.9),
    fall_range=(0.9, 0.1),
    t_start_ns=-10,
    t_stop_ns=25,
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
    t_start_ns : float
        Optional start time (ns) for the internal sampling window. If provided together with
        ``t_stop_ns``, overrides the default window.
    t_stop_ns : float
        Optional stop time (ns) for the internal sampling window. If provided together with
        ``t_start_ns``, overrides the default window.

    Returns
    -------
    tuple[float, float]
        Tuple ``(sigma_ns, tau_ns)`` giving the Gaussian sigma and exponential tau in ns.
    """
    t = np.arange(float(t_start_ns), float(t_stop_ns) + dt_ns, dt_ns, dtype=float)

    def pulse(sigma, tau):
        g = _gaussian(t, sigma)
        e = _exp_decay(t, tau)
        y = fftconvolve(g, e, mode="same")
        return y / y.max() if y.max() > 0 else y

    rise_lo, rise_hi = rise_range
    fall_hi, fall_lo = fall_range

    def residuals(x):
        sigma, tau = x
        y = pulse(sigma, tau)
        r = _rise_width(t, y, y_low=rise_lo, y_high=rise_hi) - rise_width_ns
        f = _fall_width(t, y, y_high=fall_hi, y_low=fall_lo) - fall_width_ns
        return [r, f]

    res = least_squares(residuals, x0=[0.3, 10.0], bounds=(1e-6, 500))
    sigma, tau = float(res.x[0]), float(res.x[1])
    _logger.info(f"Solved pulse parameters (LSQ): sigma={sigma:.6g}, tau={tau:.6g}")
    return sigma, tau


def generate_pulse_from_rise_fall_times(
    rise_width_ns,
    fall_width_ns,
    dt_ns=0.1,
    rise_range=(0.1, 0.9),
    fall_range=(0.9, 0.1),
    t_start_ns=-10,
    t_stop_ns=25,
    center_on_peak=False,
):
    """Generate pulse from rise/fall time specifications.

    Parameters
    ----------
    rise_width_ns : float
    Target rise time (ns) between the fractional levels defined by ``rise_range``.
    Defaults correspond to 10-90% rise time.
    fall_width_ns : float
    Target fall time (ns) between the fractional levels defined by ``fall_range``.
    Defaults correspond to 90-10% fall time.
    dt_ns : float, optional
        Time sampling step (ns). Default is 0.1 ns.
    rise_range : tuple[float, float], optional
        Fractional amplitudes (low, high) for rise-time definition. Default (0.1, 0.9).
    fall_range : tuple[float, float], optional
        Fractional amplitudes (high, low) for fall-time definition. Default (0.9, 0.1).
    t_start_ns : float, optional
        Start time (ns) for the internal solver sampling window and output grid. Default -10.
    t_stop_ns : float, optional
        Stop time (ns) for the internal solver sampling window and output grid. Default 25.
    center_on_peak : bool, optional
        If True, shift the returned time array so the pulse maximum is at t=0.
        Default is False.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Tuple ``(t, y)`` with time samples in ns and normalized pulse amplitude (peak 1).

    Notes
    -----
    The model is a Gaussian convolved with an exponential. The parameters (sigma, tau)
    are solved via least-squares such that the resulting pulse matches the requested rise and
    fall times measured on monotonic segments relative to the peak.
    """
    sigma, tau = solve_sigma_tau_from_risefall(
        rise_width_ns,
        fall_width_ns,
        dt_ns=dt_ns,
        rise_range=rise_range,
        fall_range=fall_range,
        t_start_ns=t_start_ns,
        t_stop_ns=t_stop_ns,
    )

    return generate_gauss_expconv_pulse(
        sigma,
        tau,
        dt_ns=dt_ns,
        t_start_ns=t_start_ns,
        t_stop_ns=t_stop_ns,
        center_on_peak=center_on_peak,
    )
