"""Tables of the light-source descriptions of the sim_telarray LightEmission programs.

simtools derives the angular distribution and the pulse shape of a calibration
light source from the calibration model as LightEmission arguments (tokens such
as ``gauss:11.0`` or table files). These functions turn the same arguments into
tables, for ray-tracing backends that make the light themselves.
"""

from pathlib import Path

import numpy as np

_GAUSS_NODES = 100  # LightEmission: nodes every 0.05 rms up to 5 rms.


def _table(path):
    """Two-column table (first column ascending) from a sim_telarray table file."""
    columns = np.loadtxt(path, comments="#", usecols=(0, 1), ndmin=2)
    return columns[:, 0], columns[:, 1]


def angular_profile(token):
    """
    Relative intensity per solid angle versus angle [deg] of a LightEmission angular distribution.

    Parameters
    ----------
    token: str
        ``isotropic``, ``gauss:<rms deg>``, ``filledcone:<half angle deg>`` or
        the name of a table file (angle [deg], relative intensity).

    Returns
    -------
    tuple of numpy.ndarray
        Angles [deg] and relative intensities.
    """
    name, _, value = str(token).lower().partition(":")
    if name == "isotropic":
        return np.array([0.0, 180.0]), np.array([1.0, 1.0])
    if name in ("gauss", "gauss-rms"):
        angles = np.minimum(np.arange(_GAUSS_NODES) * 0.05 * float(value), 180.0)
        return angles, np.exp(-0.5 * (angles / float(value)) ** 2)
    if name == "filledcone":
        cone = float(value)
        return np.array([0.0, cone, cone * (1.0 + 1e-9)]), np.array([1.0, 1.0, 0.0])
    if Path(token).is_file():
        return _table(token)
    raise ValueError(f"Angular distribution '{token}' is not supported")


def pulse_profile(token):
    """
    Relative emission rate versus time [ns] of a LightEmission pulse shape.

    Parameters
    ----------
    token: str
        ``simple:<full width ns>``, ``gauss:<rms ns>``, ``exponential:<decay ns>``
        or the name of a table file (time [ns], rate).

    Returns
    -------
    tuple of numpy.ndarray or None
        Times [ns] and rates, or None for an instantaneous pulse.
    """
    name, _, value = str(token).lower().partition(":")
    width = float(value) if value else 0.0
    if name in ("simple", "gauss", "exponential") and width <= 0.0:
        return None
    if name == "simple":
        return np.array([-0.5 * width, 0.5 * width]), np.array([1.0, 1.0])
    if name == "gauss":
        times = np.linspace(-5.0 * width, 5.0 * width, 201)
        return times, np.exp(-0.5 * (times / width) ** 2)
    if name == "exponential":  # Mean time zero, as in sim_telarray.
        times = np.linspace(0.0, 20.0 * width, 401)
        return times - width, np.exp(-times / width)
    if Path(token).is_file():
        return _table(token)
    raise ValueError(f"Pulse shape '{token}' is not supported")
