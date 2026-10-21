"""Characterize and compare atmospheric profile and transmission tables."""

import json
from pathlib import Path

import astropy.units as u
import numpy as np

from simtools.atmosphere.atmosphere_characterization_helpers import (
    _ALTITUDE,
    _PROFILE_COLUMNS,
    _column_values,
    _profile_samples,
    _transmission_at_altitude,
    _transmission_slice,
)
from simtools.visualization import visualize
from simtools.visualization.matplotlib_backend import pyplot as plt

_SITE = "site"
_DENSITY = "density"
_THICKNESS = "thickness"
_PROFILE_VERSION = "profile_version"
_TRANSMISSION_VERSION = "transmission_version"
_OBSERVATION_LEVEL = "observation_level_km"
_OBSERVATORY_LEVEL = "observatory_level"
_HEIGHT = "height"
_SAMPLE = "sample"
_WAVELENGTH = "wavelength"
_SMALL = "small"

_PROFILE = "profile"
_TRANSMISSION = "transmission"
_TRANSMISSIONS = "transmissions"
_LABEL = "label"
_SUMMARY = "atmosphere_summary.json"
_SAMPLE_HEIGHT = 10 * u.km
_TOP_OF_ATMOSPHERE = "Top of atmosphere (a.s.l.)"
_DEFAULT_RELATIVE_DIFFERENCE_LIMIT = 35.0
_RELATIVE_DIFFERENCE_LABEL = "Relative difference [%]"
_GRID_ALPHA = 0.25
_UNITS = {
    _DENSITY: u.g / u.cm**3,
    _THICKNESS: u.g / u.cm**2,
    "temperature": u.K,
    "pressure": u.mbar,
}


def characterize_atmospheres(
    atmospheres, output_path, relative_difference_limit=_DEFAULT_RELATIVE_DIFFERENCE_LIMIT
):
    """
    Characterize atmospheric inputs and compare them to the first input.

    Parameters
    ----------
    atmospheres : list of dict
        Each input contains a unique ``label``, an Astropy ``profile`` table,
        and a ``transmission`` table. Profile altitude is in km, density in
        mass per volume, thickness in mass per area, and refractive index is
        dimensionless n-1. Transmission tables contain wavelength, emission
        altitude and dimensionless optical depth. Optional ``site``,
        ``profile_version`` and ``transmission_version`` entries are retained.
    output_path : str or pathlib.Path
        Existing output directory for figures and the JSON summary.
    relative_difference_limit : float, optional
        Positive percentage limit for relative-difference axes. The default is 35 percent.

    Returns
    -------
    dict
        JSON-serializable characterization with 10 km a.s.l. samples and
        transmission differences at common tabulated wavelengths.

    Raises
    ------
    ValueError
        If inputs are empty, labels are duplicated, or data are invalid.
    """
    if not atmospheres:
        raise ValueError("At least one atmosphere is required.")
    if not np.isfinite(relative_difference_limit) or relative_difference_limit <= 0:
        raise ValueError("Relative difference limit must be positive.")
    labels = [item[_LABEL] for item in atmospheres]
    if len(set(labels)) != len(labels):
        raise ValueError("Atmosphere labels must be unique.")
    output_path = Path(output_path)
    highest_observation_level = max(_observation_level(item) for item in atmospheres)
    transmission_levels = _transmission_levels(highest_observation_level)
    inputs = [_prepare_atmosphere(item, transmission_levels) for item in atmospheres]
    summary = _summarize(inputs)
    _plot_profiles(inputs, output_path, relative_difference_limit)
    _plot_transmission(inputs, output_path, relative_difference_limit)
    (output_path / _SUMMARY).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _observation_level(item):
    """Return the recorded observatory level in kilometres."""
    endpoint = item[_TRANSMISSION].meta.get(_OBSERVATORY_LEVEL)
    if endpoint is None:
        raise ValueError(f"Transmission metadata must define {_OBSERVATORY_LEVEL}.")
    return u.Quantity(endpoint).to(u.km)


def _transmission_levels(highest_observation_level):
    """Return common transmission levels above sea level."""
    levels = (
        20 * u.km,
        _SAMPLE_HEIGHT,
        5 * u.km,
        highest_observation_level + 1.5 * u.km,
        highest_observation_level + 0.5 * u.km,
    )
    return (
        (_TOP_OF_ATMOSPHERE, None),
        *((f"{level.to_value(u.km):g} km a.s.l.", level) for level in levels),
    )


def _prepare_atmosphere(item, transmission_levels):
    """Validate and convert a pair of atmospheric tables for plotting."""
    profile = item[_PROFILE].copy()
    for name, unit in _UNITS.items():
        profile[name] = _column_values(profile, name, unit) * unit
    samples = _profile_samples(profile, _SAMPLE_HEIGHT)
    transmission_table = item[_TRANSMISSION]
    sample_wavelength, sample_transmission = _transmission_slice(transmission_table, _SAMPLE_HEIGHT)
    heights = _column_values(profile, _ALTITUDE, u.km)
    order = np.argsort(heights)
    top_of_atmosphere = np.max(_column_values(transmission_table, _ALTITUDE, u.km)) * u.km
    prepared_transmissions = {}
    for label, level in transmission_levels:
        wavelength, transmission = _transmission_at_altitude(
            transmission_table, top_of_atmosphere if level is None else level
        )
        prepared_transmissions[label] = {
            _WAVELENGTH: wavelength.to_value(u.nm),
            _TRANSMISSION: transmission,
        }
    return {
        _LABEL: item[_LABEL],
        _SITE: item.get(_SITE),
        _PROFILE_VERSION: item.get(_PROFILE_VERSION),
        _TRANSMISSION_VERSION: item.get(_TRANSMISSION_VERSION),
        _OBSERVATION_LEVEL: float(_observation_level(item).to_value(u.km)),
        _HEIGHT: heights[order],
        _PROFILE: {name: _column_values(profile, name)[order] for name in _PROFILE_COLUMNS},
        _SAMPLE: samples,
        _WAVELENGTH: sample_wavelength.to_value(u.nm),
        _TRANSMISSION: sample_transmission,
        _TRANSMISSIONS: prepared_transmissions,
    }


def _summarize(inputs):
    """Build a reproducible numerical summary using common wavelengths."""
    reference = inputs[0]
    data = []
    for item in inputs:
        common, ref_indices, item_indices = np.intersect1d(
            reference[_WAVELENGTH], item[_WAVELENGTH], return_indices=True
        )
        difference = item[_TRANSMISSION][item_indices] - reference[_TRANSMISSION][ref_indices]
        comparison = None
        if len(common):
            comparison = {
                "common_wavelength_count": len(common),
                "wavelength_min_nm": float(common.min()),
                "wavelength_max_nm": float(common.max()),
                "max_absolute_difference": float(np.max(np.abs(difference))),
                "rms_difference": float(np.sqrt(np.mean(difference**2))),
            }
        data.append(
            {
                key: item[key]
                for key in (
                    _LABEL,
                    _SITE,
                    _PROFILE_VERSION,
                    _TRANSMISSION_VERSION,
                    _OBSERVATION_LEVEL,
                )
            }
            | {"profile_at_10_km": item[_SAMPLE], "transmission_difference": comparison}
        )
    return {
        "reference": reference[_LABEL],
        "emission_altitude_km_asl": float(_SAMPLE_HEIGHT.to_value(u.km)),
        "profile_units": {name: unit.to_string() for name, unit in _UNITS.items()}
        | {"refractive_index": "n-1", "water_vapor_partial_pressure": "p_w/p"},
        "transmission_definition": "exp(-vertical optical depth) to each table observation level",
        "atmospheres": data,
    }


def _profile_label(item):
    """Identify a profile without including its transmission model."""
    if item.get(_SITE) and item.get(_PROFILE_VERSION):
        return f"{item[_SITE]} profile {item[_PROFILE_VERSION]}"
    return item[_LABEL]


def _profile_difference(reference, candidate, name):
    """Compute absolute profile differences on the overlapping altitude grid."""
    ref_height = reference[_HEIGHT]
    height = candidate[_HEIGHT]
    lower = max(ref_height[0], height[0])
    upper = min(ref_height[-1], height[-1])
    common = np.union1d(ref_height, height)
    common = common[(common >= lower) & (common <= upper)]
    return common, np.interp(common, height, candidate[_PROFILE][name]) - np.interp(
        common, ref_height, reference[_PROFILE][name]
    )


def _profile_relative_difference(reference, candidate, name):
    """Compute percentage differences on the overlapping altitude grid."""
    height, difference = _profile_difference(reference, candidate, name)
    reference_values = np.interp(height, reference[_HEIGHT], reference[_PROFILE][name])
    relative = np.full_like(difference, np.nan)
    valid = reference_values != 0
    relative[valid] = 100 * difference[valid] / reference_values[valid]
    return height, relative


def _transmission_relative_difference(reference, candidate):
    """Compute percentage transmission differences on common wavelengths."""
    common, ref_indices, candidate_indices = np.intersect1d(
        reference[_WAVELENGTH], candidate[_WAVELENGTH], return_indices=True
    )
    reference_values = reference[_TRANSMISSION][ref_indices]
    relative = np.full_like(reference_values, np.nan)
    valid = reference_values != 0
    relative[valid] = (
        100
        * (candidate[_TRANSMISSION][candidate_indices][valid] - reference_values[valid])
        / reference_values[valid]
    )
    return common, relative


def _plot_profiles(inputs, output_path, relative_difference_limit):
    """Plot profiles and absolute and relative differences from the first profile."""
    fig, axes = plt.subplots(
        3,
        len(_PROFILE_COLUMNS),
        figsize=(24, 10),
        sharex="col",
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2, 1, 1)},
    )
    titles = (
        r"Density [g cm$^{-3}$]",
        r"Depth [g cm$^{-2}$]",
        "Refractivity n - 1",
        "Temperature [K]",
        "Pressure [mbar]",
        "Water vapor partial pressure [p$_w$/p]",
    )
    reference = inputs[0]
    for column, (name, title) in enumerate(zip(_PROFILE_COLUMNS, titles, strict=True)):
        upper, difference_axis, relative = axes[:, column]
        for item in inputs:
            positive = item[_PROFILE][name] > 0
            upper.semilogy(
                item[_HEIGHT][positive],
                item[_PROFILE][name][positive],
                label=_profile_label(item),
            )
            height, difference = _profile_difference(reference, item, name)
            difference_axis.plot(height, difference, label=_profile_label(item))
            height, percentage = _profile_relative_difference(reference, item, name)
            relative.plot(height, percentage, label=_profile_label(item))
        upper.set(ylabel=title)
        difference_axis.set_ylabel(f"Difference: {title}")
        relative.set(
            xlabel="Altitude a.s.l. [km]",
            ylabel=_RELATIVE_DIFFERENCE_LABEL,
            ylim=(-relative_difference_limit, relative_difference_limit),
        )
        upper.grid(True, alpha=_GRID_ALPHA)
        difference_axis.grid(True, alpha=_GRID_ALPHA)
        relative.grid(True, alpha=_GRID_ALPHA)
    axes[0, 0].legend(fontsize=_SMALL)
    visualize.save_figure(fig, output_path / "atmosphere_profiles", close=True)


def _plot_transmission(inputs, output_path, relative_difference_limit):
    """Plot transmission and common-wavelength differences at requested levels."""
    levels = tuple(inputs[0][_TRANSMISSIONS])
    fig, axes = plt.subplots(
        len(levels), 2, sharex="col", figsize=(12, 14), constrained_layout=True
    )
    reference = inputs[0]
    for row, level in enumerate(levels):
        direct_axis, difference_axis = axes[row]
        reference_transmission = reference[_TRANSMISSIONS][level]
        for item in inputs:
            transmission = item[_TRANSMISSIONS][level]
            direct_axis.scatter(
                transmission[_WAVELENGTH], transmission[_TRANSMISSION], label=item[_LABEL], s=3
            )
            common, relative_difference = _transmission_relative_difference(
                reference_transmission, transmission
            )
            difference_axis.scatter(common, relative_difference, label=item[_LABEL], s=3)
        direct_axis.set(ylabel="Direct transmission", ylim=(0, 1.02), title=level)
        difference_axis.set(
            ylabel=_RELATIVE_DIFFERENCE_LABEL,
            ylim=(-relative_difference_limit, relative_difference_limit),
        )
        for axis in (direct_axis, difference_axis):
            axis.legend(fontsize=_SMALL)
            axis.grid(True, alpha=_GRID_ALPHA)
    for axis in axes[-1]:
        axis.set_xlabel("Wavelength [nm]")
    visualize.save_figure(fig, output_path / "atmosphere_transmission", close=True)
