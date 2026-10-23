"""Internal calculations for atmospheric characterization."""

import astropy.units as u
import numpy as np

_ALTITUDE = "altitude"
_WAVELENGTH = "wavelength"
_EXTINCTION = "extinction"
_PROFILE_COLUMNS = (
    "density",
    "thickness",
    "refractive_index",
    "temperature",
    "pressure",
    "water_vapor_partial_pressure",
)


def _column_values(table, name, unit=None):
    """Read a numeric column, converting its unit when requested."""
    column = table[name]
    if unit is not None:
        if getattr(column, "unit", None) is None:
            raise ValueError(f"Column '{name}' requires units compatible with {unit}.")
        return np.asarray(u.Quantity(column).to_value(unit), dtype=float)
    return np.asarray(column, dtype=float)


def _altitude_km(altitude, kind):
    """Return a finite requested altitude in kilometres."""
    target = altitude.to_value(u.km)
    if not np.isfinite(target):
        raise ValueError(f"{kind} altitude must be finite.")
    return target


def _profile_samples(table, altitude):
    """Sample profile quantities at a given altitude without extrapolation."""
    heights = _column_values(table, _ALTITUDE, u.km)
    target = _altitude_km(altitude, "Profile")
    if len(heights) < 2 or not np.all(np.isfinite(heights)):
        raise ValueError("Profile altitude must contain at least two finite rows.")
    order = np.argsort(heights)
    heights = heights[order]
    if np.any(np.diff(heights) <= 0):
        raise ValueError("Profile altitude values must be unique.")
    if target < heights[0] or target > heights[-1]:
        raise ValueError("Requested altitude is outside the profile range.")
    result = {}
    for name in _PROFILE_COLUMNS:
        column = _column_values(table, name)[order]
        if not np.all(np.isfinite(column)):
            raise ValueError(f"Profile column '{name}' contains non-finite values.")
        result[name] = float(np.interp(target, heights, column))
    return result


def _transmission_columns(table):
    """Return validated altitude (km), wavelength (nm), and optical depth columns."""
    heights = _column_values(table, _ALTITUDE, u.km)
    wavelengths = _column_values(table, _WAVELENGTH, u.nm)
    depth = _column_values(table, _EXTINCTION)
    if not (np.all(np.isfinite(heights)) and np.all(np.isfinite(wavelengths))):
        raise ValueError("Transmission altitude and wavelength must be finite.")
    if not np.all(np.isfinite(depth)):
        raise ValueError("Transmission optical depth must be finite.")
    if np.any(depth < 0):
        raise ValueError("Transmission optical depth must be nonnegative.")
    return heights, wavelengths, depth


def _transmission_slice(table, altitude):
    """Return wavelengths and direct transmission at a tabulated altitude."""
    target = _altitude_km(altitude, "Transmission")
    heights, wavelengths, depth = _transmission_columns(table)
    selected = np.isclose(heights, target, rtol=0, atol=1e-9)
    if not np.any(selected):
        raise ValueError("Requested transmission altitude is not tabulated.")
    wavelengths = wavelengths[selected]
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Transmission wavelengths at the requested altitude must be unique.")
    return wavelengths * u.nm, np.exp(-depth[selected][order])


def _interpolate_depth(heights, depth, target):
    """Interpolate or linearly extrapolate optical depth at one altitude."""
    if target < heights[0]:
        slope = (depth[1] - depth[0]) / (heights[1] - heights[0])
        return depth[0] + slope * (target - heights[0])
    if target > heights[-1]:
        slope = (depth[-1] - depth[-2]) / (heights[-1] - heights[-2])
        return depth[-1] + slope * (target - heights[-1])
    return np.interp(target, heights, depth)


def _transmission_at_altitude(table, altitude):
    """
    Return direct transmission interpolated at an altitude for all wavelengths.

    Wavelengths tabulated at fewer than two altitudes are skipped. Extrapolated
    optical depths are clipped at zero.
    """
    target = _altitude_km(altitude, "Transmission")
    heights, wavelengths, depth = _transmission_columns(table)
    valid_wavelengths = []
    result = []
    for wavelength in np.unique(wavelengths):
        selected = wavelengths == wavelength
        if np.count_nonzero(selected) < 2:
            continue
        order = np.argsort(heights[selected])
        sample_heights = heights[selected][order]
        if np.any(np.diff(sample_heights) <= 0):
            raise ValueError("Transmission wavelengths must be present once per altitude.")
        valid_wavelengths.append(wavelength)
        result.append(_interpolate_depth(sample_heights, depth[selected][order], target))
    if not result:
        raise ValueError("Transmission requires at least two altitudes per wavelength.")
    return np.asarray(valid_wavelengths) * u.nm, np.exp(-np.maximum(result, 0))
