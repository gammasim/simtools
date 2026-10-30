"""Angular and temporal emission profiles of calibration light sources.

The profiles follow the calibration-model parameters ``flasher_angular_distribution``
(with ``flasher_angular_distribution_width``) and ``flasher_pulse_shape``. They are
tables, linear between nodes, for ray-tracing backends that make the light themselves.
"""

import numpy as np

from simtools.simtel.pulse_shapes import generate_pulse_from_rise_fall_times

SAMPLES = 201  # Nodes of a tabulated smooth profile.
N_SIGMA = 5.0  # Range of a Gaussian profile in standard deviations.


def angular_profile(shape, width_deg, samples=SAMPLES, n_sigma=N_SIGMA):
    """
    Relative intensity per solid angle versus angle to the source axis.

    Parameters
    ----------
    shape: str
        ``Isotropic``, ``Gauss`` (standard deviation ``width_deg``), ``FilledCone``
        (half opening angle ``width_deg``) or ``Lambertian`` (cosine, up to
        ``width_deg``).
    width_deg: float
        Width of the distribution [deg].
    samples: int
        Number of nodes of the Gauss and Lambertian tables.
    n_sigma: float
        Range of the Gauss table in standard deviations.

    Returns
    -------
    tuple of numpy.ndarray
        Angles [deg] and relative intensities.
    """
    name = str(shape).lower()
    if name == "isotropic":
        return np.array([0.0, 180.0]), np.array([1.0, 1.0])
    if width_deg <= 0.0:
        raise ValueError(f"Angular distribution '{shape}' needs a positive width")
    if name == "filledcone":
        return np.array([0.0, width_deg, width_deg * (1.0 + 1e-9)]), np.array([1.0, 1.0, 0.0])
    if name == "gauss":
        angles = np.linspace(0.0, min(n_sigma * width_deg, 180.0), samples)
        return angles, np.exp(-0.5 * (angles / width_deg) ** 2)
    if name == "lambertian":
        angles = np.linspace(0.0, min(width_deg, 90.0), samples)
        return angles, np.cos(np.radians(angles))
    raise ValueError(f"Angular distribution '{shape}' has no tabulated profile")


def pulse_profile(shape, width_ns, decay_ns, samples=SAMPLES, n_sigma=N_SIGMA):
    """
    Relative emission rate versus time.

    Parameters
    ----------
    shape: str
        ``TopHat`` (full width ``width_ns``), ``Gauss`` (standard deviation
        ``width_ns``), ``Exponential`` (decay time ``decay_ns``) or
        ``Gauss-Exponential`` (10-90% rise time ``width_ns``, 90-10% fall time
        ``decay_ns``, a Gaussian convolved with an exponential).
    width_ns, decay_ns: float
        Width and decay time [ns].
    samples: int
        Number of nodes of the smooth profiles.
    n_sigma: float
        Range of the profiles: ``n_sigma`` standard deviations (Gauss), ``4 n_sigma``
        decay times (Exponential, down to exp(-20)), ``n_sigma`` times rise plus four
        fall times on either side of the peak (Gauss-Exponential).

    Returns
    -------
    tuple of numpy.ndarray or None
        Times [ns] and rates, or None for an instantaneous pulse.
    """
    name = str(shape).lower()
    if name == "exponential":
        if decay_ns <= 0.0:
            return None
        times = np.linspace(0.0, 4.0 * n_sigma * decay_ns, samples)
        return times, np.exp(-times / decay_ns)
    if name == "gauss-exponential":
        if width_ns <= 0.0 or decay_ns <= 0.0:
            raise ValueError("A Gauss-Exponential pulse needs positive rise and fall times")
        span = n_sigma * (width_ns + 4.0 * decay_ns)
        times, rates = generate_pulse_from_rise_fall_times(
            width_ns,
            decay_ns,
            dt_ns=2.0 * span / (samples - 1),
            t_start_ns=-span,
            t_stop_ns=span,
            center_on_peak=True,
        )
        return times, np.clip(rates, 0.0, None)  # The numerical convolution dips below zero.
    if width_ns <= 0.0:
        return None
    if name == "tophat":
        return np.array([-0.5 * width_ns, 0.5 * width_ns]), np.array([1.0, 1.0])
    if name == "gauss":
        times = np.linspace(-n_sigma * width_ns, n_sigma * width_ns, samples)
        return times, np.exp(-0.5 * (times / width_ns) ** 2)
    raise ValueError(f"Pulse shape '{shape}' has no tabulated profile")
