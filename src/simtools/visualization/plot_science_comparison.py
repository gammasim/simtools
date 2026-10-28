"""Plot telescope-throughput science-result comparisons into PDF pages."""

import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools.visualization.matplotlib_backend import pyplot as plt

_WAVELENGTH_COLUMN = "wl"
_EFFICIENCY_LABEL = "Efficiency"
_RELATIVE_CHANGE_LABEL = "Relative change (%)"
_DEFAULT_DETECTED_COLUMN = "cherenkov_detected"
# Optical throughput, compared when tables have no detected spectrum.
_THROUGHPUT_COLUMN = "eff"
_NSB_DETECTED_COLUMN = "nsb_detected"
_DETECTED_COLUMNS = {"nsb": _NSB_DETECTED_COLUMN}
# Column names written by earlier sim_telarray versions.
_LEGACY_COLUMNS = {"C4": _DEFAULT_DETECTED_COLUMN, "N4": _NSB_DETECTED_COLUMN}
_ZERO_LINE = {"color": "black", "linewidth": 0.7}
_SHADOWING_COLUMN = "masts"
_PIXEL_COLUMN = "pixel"
_COMBINED_LABEL = "Combined throughput"
# Optical stages and the table columns whose product forms each plotted curve.
EFFICIENCY_COMPONENT_GROUPS = {
    "Mirrors": {
        "Reflectivity": ("ref",),
        "Shadowing": (_SHADOWING_COLUMN,),
        _COMBINED_LABEL: ("ref", _SHADOWING_COLUMN),
    },
    "Camera": {
        "Filter": ("filt",),
        "Pixel and lightguide": (_PIXEL_COLUMN,),
        _COMBINED_LABEL: ("filt", _PIXEL_COLUMN),
    },
    "Photodetector": {"Quantum efficiency": ("qe",)},
    "Atmosphere": {"Transmission": ("atm_trans",)},
    "Telescope total": {"Without atmosphere": ("eff",), "With atmosphere": ("eff_atm",)},
}


def plot_metric_changes(changes, title, pdf_pages):
    """Add a page with scalar values and their relative change.

    Parameters
    ----------
    changes : dict
        Metric changes as returned by ``telescope_throughput.compare_results``.
    title : str
        Page title.
    pdf_pages : PDF report object
        Open PDF report.
    """
    names = list(changes)
    positions = np.arange(len(names))
    labels = [_metric_label(name, changes[name]["unit"]) for name in names]
    relative = [
        np.nan if relative is None else 100 * relative
        for relative in (changes[name]["relative"] for name in names)
    ]
    figure, (values_axis, change_axis) = plt.subplots(1, 2, figsize=(12, max(4, len(names) * 0.55)))
    values_axis.axis("off")
    values_axis.set_title("Scalar values")
    for position, name, label in zip(positions, names, labels, strict=True):
        change = changes[name]
        values_axis.text(
            0, position, f"{label}: {change['baseline']:g} -> {change['candidate']:g}", va="center"
        )
    values_axis.set_ylim(len(names) - 0.5, -0.5)
    change_axis.barh(positions, relative)
    change_axis.axvline(0, **_ZERO_LINE)
    change_axis.set_yticks(positions, labels)
    change_axis.invert_yaxis()
    change_axis.set_xlabel(_RELATIVE_CHANGE_LABEL)
    figure.suptitle(title)
    figure.tight_layout()
    pdf_pages.add(figure)


def plot_spectra(baseline_tables, candidate_tables, title, labels, pdf_pages):
    """Add a page comparing the detected spectrum of each common channel.

    Parameters
    ----------
    baseline_tables, candidate_tables : dict
        Wavelength-table paths by channel.
    title : str
        Page title.
    labels : tuple of str
        Baseline and candidate legend labels.
    pdf_pages : PDF report object
        Open PDF report.

    Returns
    -------
    list of str
        Channels that could not be compared.
    """
    channels = sorted(set(baseline_tables) & set(candidate_tables))
    curves, diagnostics = {}, []
    for channel in channels:
        try:
            wavelength, unit, before, after = _read_pair(
                baseline_tables[channel], candidate_tables[channel]
            )
            column = _DETECTED_COLUMNS.get(channel, _DEFAULT_DETECTED_COLUMN)
            if column not in before.colnames and column not in after.colnames:
                column = _THROUGHPUT_COLUMN
            curves[channel] = (wavelength, unit, _curve(before, column), _curve(after, column))
        except (KeyError, OSError, ValueError, u.UnitsError) as exc:
            diagnostics.append(f"Cannot compare {channel} spectrum: {exc}")
    if not curves:
        return diagnostics
    figure, axes = plt.subplots(2, len(curves), sharex="col", figsize=(5 * len(curves), 8))
    axes = np.asarray(axes).reshape(2, -1)
    for (channel, (wavelength, unit, before, after)), column in zip(
        curves.items(), axes.T, strict=True
    ):
        column[0].set_title(channel)
        column[0].plot(wavelength, before, label=labels[0])
        column[0].plot(wavelength, after, label=labels[1])
        column[0].set_ylabel("Detected spectrum")
        column[0].legend()
        column[1].plot(wavelength, _relative_percent(before, after))
        column[1].axhline(0, **_ZERO_LINE)
        column[1].set_ylabel(_RELATIVE_CHANGE_LABEL)
        for axis in column:
            _label_wavelength_axis(axis, unit)
    figure.suptitle(title)
    figure.tight_layout()
    pdf_pages.add(figure)
    return diagnostics


def plot_components(baseline_table, candidate_table, title, labels, pdf_pages):
    """Add absolute and relative pages for each optical efficiency component.

    Parameters
    ----------
    baseline_table, candidate_table : str or pathlib.Path
        Wavelength tables of one channel.
    title : str
        Page title prefix.
    labels : tuple of str
        Baseline and candidate legend labels.
    pdf_pages : PDF report object
        Open PDF report.

    Returns
    -------
    str or None
        Reason the components could not be compared.
    """
    try:
        wavelength, unit, baseline, candidate = _read_pair(baseline_table, candidate_table)
        groups = {
            group: {
                label: (_curve(baseline, *columns), _curve(candidate, *columns))
                for label, columns in components.items()
            }
            for group, components in EFFICIENCY_COMPONENT_GROUPS.items()
        }
    except (KeyError, OSError, ValueError, u.UnitsError) as exc:
        return f"Cannot compare efficiency components: {exc}"
    for relative in (False, True):
        figure = _component_page(wavelength, unit, groups, relative, labels)
        page = "relative change by component" if relative else "efficiency components"
        figure.suptitle(f"{title}: {page}")
        figure.tight_layout()
        pdf_pages.add(figure)
    return None


def _component_page(wavelength, unit, groups, relative, labels):
    """Return a figure with a panel per optical stage, absolute or relative to the baseline."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for axis, (group, curves) in zip(axes.flat, groups.items()):
        axis.set_title(group)
        for label, (before, after) in curves.items():
            if relative:
                axis.plot(wavelength, _relative_percent(before, after), label=label)
            else:
                axis.plot(wavelength, before, linestyle="--", label=f"{label} {labels[0]}")
                axis.plot(wavelength, after, label=f"{label} {labels[1]}")
        axis.set_ylabel(_RELATIVE_CHANGE_LABEL if relative else _EFFICIENCY_LABEL)
        if not relative:
            axis.set_ylim(top=1.0)
        axis.legend(fontsize="small")
        _label_wavelength_axis(axis, unit)
    axes.flat[-1].axis("off")
    return figure


def _read_pair(baseline_path, candidate_path):
    """Read two wavelength tables and require a common wavelength grid."""
    baseline, candidate = _read_table(baseline_path), _read_table(candidate_path)
    unit = baseline[_WAVELENGTH_COLUMN].unit or u.nm
    wavelength = np.asarray(baseline[_WAVELENGTH_COLUMN], dtype=float)
    if wavelength.ndim != 1 or not wavelength.size or not np.all(np.isfinite(wavelength)):
        raise ValueError("wavelength grid must be a finite, non-empty list")
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("wavelength grid must be strictly increasing")
    other = u.Quantity(candidate[_WAVELENGTH_COLUMN], candidate[_WAVELENGTH_COLUMN].unit or u.nm)
    other = other.to_value(unit)
    if wavelength.shape != other.shape or not np.allclose(wavelength, other, rtol=1e-12, atol=0):
        raise ValueError("wavelength grids differ")
    return wavelength, unit, baseline, candidate


def _curve(table, *columns):
    """Return the product of finite numeric table columns."""
    arrays = [np.asarray(table[column], dtype=float) for column in columns]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError(f"column(s) {', '.join(columns)} must be one-dimensional")
    values = np.prod(arrays, axis=0)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite values in column(s) {', '.join(columns)}")
    return values


def _read_table(path):
    """Read an ECSV or plain whitespace-separated table, mapping legacy column names."""
    table = Table.read(path, format="ascii")
    for legacy, name in _LEGACY_COLUMNS.items():
        if legacy in table.colnames and name not in table.colnames:
            table.rename_column(legacy, name)
    return table


def _relative_percent(before, after):
    """Return the relative change in percent, NaN where the baseline is zero."""
    return 100 * np.divide(
        after - before, before, out=np.full_like(after, np.nan, dtype=float), where=before != 0
    )


def _metric_label(name, unit):
    return f"{name} ({unit})" if unit else name


def _label_wavelength_axis(axis, unit):
    axis.set_xlabel(f"Wavelength ({unit})")
    axis.tick_params(axis="x", labelbottom=True)
