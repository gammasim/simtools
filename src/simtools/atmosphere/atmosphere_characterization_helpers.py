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


def _profile_samples(table, altitude):
    """Sample profile quantities at a given altitude without extrapolation."""
    heights = _column_values(table, _ALTITUDE, u.km)
    target = altitude.to_value(u.km)
    if not np.isfinite(target):
        raise ValueError("Profile altitude must be finite.")
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


def _transmission_slice(table, altitude):
    """Return wavelengths and direct transmission at a tabulated altitude."""
    heights = _column_values(table, _ALTITUDE, u.km)
    target = altitude.to_value(u.km)
    if not np.isfinite(target):
        raise ValueError("Transmission altitude must be finite.")
    selected = np.isclose(heights, target, rtol=0, atol=1e-9)
    if not np.any(selected):
        raise ValueError("Requested transmission altitude is not tabulated.")
    wavelengths = _column_values(table, _WAVELENGTH, u.nm)[selected]
    depth = _column_values(table, _EXTINCTION)[selected]
    if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(depth)):
        raise ValueError("Transmission wavelength and optical depth must be finite.")
    if np.any(depth < 0):
        raise ValueError("Transmission optical depth must be nonnegative.")
    order = np.argsort(wavelengths)
    wavelengths = wavelengths[order]
    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Transmission wavelengths at the requested altitude must be unique.")
    return wavelengths * u.nm, np.exp(-depth[order])


def _interpolate_depth(heights, depth, target):
    """Interpolate or linearly extrapolate optical depth at one altitude."""
    if target < heights[0]:
        slope = np.diff(depth[:2]) / np.diff(heights[:2])
        return depth[0] + slope[0] * (target - heights[0])
    if target > heights[-1]:
        slope = np.diff(depth[-2:]) / np.diff(heights[-2:])
        return depth[-1] + slope[0] * (target - heights[-1])
    return np.interp(target, heights, depth)


def _transmission_at_altitude(table, altitude):
    """Return direct transmission interpolated at an altitude for all wavelengths."""
    heights = _column_values(table, _ALTITUDE, u.km)
    target = altitude.to_value(u.km)
    if not np.isfinite(target):
        raise ValueError("Transmission altitude must be finite.")
    wavelengths = _column_values(table, _WAVELENGTH, u.nm)
    depth = _column_values(table, _EXTINCTION)
    if not np.all(np.isfinite(heights)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError("Transmission altitude and wavelength must be finite.")
    if not np.all(np.isfinite(depth)):
        raise ValueError("Transmission optical depth must be finite.")
    if np.any(depth < 0):
        raise ValueError("Transmission optical depth must be nonnegative.")
    wavelength_values = np.unique(wavelengths)
    result = []
    valid_wavelengths = []
    for wavelength in wavelength_values:
        selected = wavelengths == wavelength
        sample_heights = heights[selected]
        sample_depth = depth[selected]
        order = np.argsort(sample_heights)
        sample_heights = sample_heights[order]
        sample_depth = sample_depth[order]
        if len(sample_heights) < 2:
            continue
        if np.any(np.diff(sample_heights) <= 0):
            raise ValueError("Transmission wavelengths must be present once per altitude.")
        value = _interpolate_depth(sample_heights, sample_depth, target)
        valid_wavelengths.append(wavelength)
        result.append(value)
    if not result:
        raise ValueError("Transmission requires at least two altitudes per wavelength.")
    result = np.maximum(np.asarray(result), 0)
    wavelength_values = np.asarray(valid_wavelengths)
    if np.any(result < 0):
        raise ValueError("Interpolated transmission optical depth must be nonnegative.")
    return wavelength_values * u.nm, np.exp(-result)
