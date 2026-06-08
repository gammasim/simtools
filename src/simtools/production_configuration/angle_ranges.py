"""Helpers for directed circular angle ranges used in production grids."""

import numpy as np


def directed_circular_span_degrees(axis_range):
    """Return directed circular span (degrees) from start to end.

    Parameters
    ----------
    axis_range : tuple or list
        Two-element range ``(start, end)`` in degrees.

    Returns
    -------
    float
        Directed circular span in degrees, where full-circle ranges are represented as ``360.0``.
    """
    start, end = axis_range
    raw_span = abs(float(end) - float(start))
    if raw_span > 0.0 and np.isclose(raw_span % 360.0, 0.0):
        return 360.0
    return float((end - start) % 360.0)


def ceil_with_tolerance(value):
    """Ceil a float while avoiding near-integer floating-point artifacts.

    Parameters
    ----------
    value : float
        Input value.

    Returns
    -------
    int
        Ceiled integer with near-integer tolerance handling.
    """
    nearest_integer = round(value)
    if np.isclose(value, nearest_integer):
        return int(nearest_integer)
    return int(np.ceil(value))
