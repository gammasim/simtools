"""Plot scalar and wavelength science-result comparisons."""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools.visualization.matplotlib_backend import pyplot as plt

_ECSV_FORMAT = "ascii.ecsv"
_BASIC_TABLE_FORMAT = "ascii.basic"
_BASELINE_KEY = "baseline"
_CANDIDATE_KEY = "candidate"
_CASE_KEY = "case"
_SOFTWARE_KEY = "software"
_SIMTOOLS_VERSION_KEY = "simtools_version"
_RELATIVE_KEY = "relative"
_LABEL_KEY = "label"
_VALUE_KEY = "value"
_UNKNOWN_VALUE = "unknown"
_SHOWER_CHANNEL = "shower"
_NSB_CHANNEL = "nsb"
_TELESCOPE_KEY = "telescope"
_WAVELENGTH_CHANNELS = (_SHOWER_CHANNEL, "muon", _NSB_CHANNEL)
_RELATIVE_CHANGE_LABEL = "Relative change (%)"
_REFERENCE_LINE_COLOR = "black"
_CENTER_ALIGNMENT = "center"
_WAVELENGTH_COLUMN = "wl"
_WAVELENGTH_FALLBACK_UNIT = "nm"
_TELESCOPE_EFFICIENCY_TITLE = "Telescope efficiency"
_SHADOWING_COLUMN = "masts"
_PIXEL_COLUMN = "pixel"
_COMBINED_THROUGHPUT_LABEL = "Combined throughput"
_EFFICIENCY_COMPONENT_GROUPS = (
    (
        "Mirrors",
        (
            ("Reflectivity", ("ref",)),
            ("Shadowing", (_SHADOWING_COLUMN,)),
            (_COMBINED_THROUGHPUT_LABEL, ("ref", _SHADOWING_COLUMN)),
        ),
    ),
    (
        "Camera",
        (
            ("Filter", ("filt",)),
            ("Pixel and lightguide", (_PIXEL_COLUMN,)),
            (_COMBINED_THROUGHPUT_LABEL, ("filt", _PIXEL_COLUMN)),
        ),
    ),
    ("Photodetector", (("Quantum efficiency", ("qe",)),)),
    ("Atmosphere", (("Transmission", ("atm_trans",)),)),
    (
        "Telescope total",
        (("Without atmosphere", ("eff",)), ("With atmosphere", ("eff_atm",))),
    ),
)


def plot_comparison(changes, baseline, candidate, output_path, figure_format=None, pdf_pages=None):
    """Write scalar value and relative-change panels."""
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    metrics = list(changes)
    if not metrics:
        raise ValueError("Cannot plot a science comparison without comparable metrics.")
    figure, (values_axis, change_axis) = plt.subplots(
        1, 2, figsize=(12, max(4, len(metrics) * 0.55))
    )
    positions = np.arange(len(metrics))
    changes_values = [
        np.nan if changes[name].get(_RELATIVE_KEY) is None else changes[name][_RELATIVE_KEY] * 100
        for name in metrics
    ]
    values_axis.axis("off")
    values_axis.set_title("Scalar values")
    for position, name in zip(positions, metrics, strict=True):
        item = changes[name]
        unit = item[_BASELINE_KEY].get("unit", "")
        unit_label = f" {unit}" if unit else ""
        values_axis.text(
            0,
            position,
            f"{name}: {item[_BASELINE_KEY][_VALUE_KEY]:g}{unit_label}  ->  "
            f"{item[_CANDIDATE_KEY][_VALUE_KEY]:g}{unit_label}",
            va=_CENTER_ALIGNMENT,
        )
    values_axis.set_ylim(len(metrics) - 0.5, -0.5)
    change_axis.set_title(
        f"Relative change: {baseline.get(_LABEL_KEY, _BASELINE_KEY)} to "
        f"{candidate.get(_LABEL_KEY, _CANDIDATE_KEY)}"
    )
    baseline_version = baseline.get(_SOFTWARE_KEY, {}).get(_SIMTOOLS_VERSION_KEY)
    candidate_version = candidate.get(_SOFTWARE_KEY, {}).get(_SIMTOOLS_VERSION_KEY)
    telescope = baseline.get(_CASE_KEY, {}).get(_TELESCOPE_KEY)
    if baseline_version or candidate_version:
        comparison_title = (
            f"simtools {baseline_version or _UNKNOWN_VALUE} vs "
            f"{candidate_version or _UNKNOWN_VALUE}"
        )
    else:
        comparison_title = (
            f"{baseline.get(_LABEL_KEY, _BASELINE_KEY)} vs "
            f"{candidate.get(_LABEL_KEY, _CANDIDATE_KEY)}"
        )
    report_title = f"{_TELESCOPE_EFFICIENCY_TITLE}: {comparison_title}"
    if telescope:
        report_title = f"{report_title} ({telescope})"
    figure.suptitle(report_title)
    change_axis.barh(positions, changes_values)
    change_axis.axvline(0, color=_REFERENCE_LINE_COLOR, linewidth=0.7)
    metric_labels = [
        f"{name} ({changes[name][_BASELINE_KEY].get('unit')})"
        if changes[name][_BASELINE_KEY].get("unit")
        else name
        for name in metrics
    ]
    change_axis.set_yticks(positions, metric_labels)
    change_axis.set_xlabel(_RELATIVE_CHANGE_LABEL)
    figure.tight_layout()
    if pdf_pages is not None:
        pdf_pages.savefig(figure)
    formats = ("png", "pdf") if figure_format is None else tuple(figure_format)
    products = {}
    for image_format in formats:
        path = output / f"science-comparison.{image_format}"
        figure.savefig(path)
        products[image_format] = str(path)
    plt.close(figure)
    return products, []


def plot_wavelength_comparison(baseline_path, candidate_path, output_path):
    """Overlay wavelength tables only when their grids match after unit conversion."""
    return plot_wavelength_comparisons(
        {_SHOWER_CHANNEL: baseline_path}, {_SHOWER_CHANNEL: candidate_path}, output_path
    )


def plot_wavelength_comparisons(
    baseline_paths,
    candidate_paths,
    output_path,
    baseline_label=_BASELINE_KEY,
    candidate_label=_CANDIDATE_KEY,
    pdf_pages=None,
    telescope=None,
):
    """Plot shower, muon, and NSB spectra with absolute and relative values."""
    channels = _WAVELENGTH_CHANNELS
    paired = {
        channel: (baseline_paths.get(channel), candidate_paths.get(channel)) for channel in channels
    }
    available = {name: paths for name, paths in paired.items() if all(paths)}
    if not available:
        return None, "No paired wavelength tables are available."
    tables = {}
    for channel, (baseline_path, candidate_path) in available.items():
        baseline = _read_wavelength_table(baseline_path)
        candidate = _read_wavelength_table(candidate_path)
        arrays, diagnostic = _wavelength_arrays(baseline, candidate, channel)
        if diagnostic:
            return None, f"{channel.title()} spectrum: {diagnostic}"
        tables[channel] = arrays
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, len(channels), sharex="col", figsize=(15, 8))
    for column, channel in enumerate(channels):
        absolute_axis, relative_axis = axes[:, column]
        absolute_axis.set_title(channel.title())
        if channel not in tables:
            absolute_axis.text(
                0.5, 0.5, "No paired table", ha=_CENTER_ALIGNMENT, va=_CENTER_ALIGNMENT
            )
            relative_axis.axis("off")
            continue
        wavelength, baseline, candidate, unit, ordinate = tables[channel]
        absolute_axis.plot(wavelength, baseline, label=baseline_label)
        absolute_axis.plot(wavelength, candidate, label=candidate_label)
        absolute_axis.legend()
        absolute_axis.set_ylabel(ordinate)
        _label_wavelength_axis(absolute_axis, unit)
        relative = np.divide(
            candidate - baseline,
            baseline,
            out=np.full_like(candidate, np.nan, dtype=float),
            where=baseline != 0,
        )
        relative_axis.plot(wavelength, relative * 100)
        relative_axis.axhline(0, color=_REFERENCE_LINE_COLOR, linewidth=0.7)
        _label_wavelength_axis(relative_axis, unit)
    axes[1, 0].set_ylabel(_RELATIVE_CHANGE_LABEL)
    title = f"{baseline_label} vs {candidate_label}"
    figure.suptitle(f"{telescope}: {title}" if telescope else title)
    figure.tight_layout()
    if pdf_pages is not None:
        pdf_pages.savefig(figure)
    product = None
    if pdf_pages is None:
        product = output / "wavelength-comparison.png"
        figure.savefig(product)
    plt.close(figure)
    return str(product) if product else None, None


def plot_efficiency_component_comparisons(
    baseline_path,
    candidate_path,
    output_path,
    baseline_label=_BASELINE_KEY,
    candidate_label=_CANDIDATE_KEY,
    pdf_pages=None,
    telescope=None,
):
    """Plot wavelength-dependent throughput components for each telescope."""
    return _plot_efficiency_components_for_channel(
        baseline_path,
        candidate_path,
        output_path,
        baseline_label,
        candidate_label,
        pdf_pages,
        telescope,
        channel=None,
    )


def _plot_efficiency_components_for_channel(
    baseline_path,
    candidate_path,
    output_path,
    baseline_label,
    candidate_label,
    pdf_pages,
    telescope,
    channel,
):
    """Plot component comparisons for one wavelength-table channel."""
    baseline = _read_wavelength_table(baseline_path)
    candidate = _read_wavelength_table(candidate_path)
    aligned, diagnostic = _aligned_wavelengths(baseline, candidate)
    if diagnostic:
        return None, diagnostic
    missing = sorted(
        {
            column
            for _, components in _EFFICIENCY_COMPONENT_GROUPS
            for _, columns in components
            for column in columns
            if column not in baseline.colnames or column not in candidate.colnames
        }
    )
    if missing:
        return None, f"Efficiency tables are missing component columns: {', '.join(missing)}."

    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    wavelength, unit = aligned
    try:
        component_values = _efficiency_component_values(baseline, candidate)
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    title = f"{baseline_label} vs {candidate_label}"
    title = f"{telescope}: {title}" if telescope else title
    if channel:
        title = f"{title}: {channel.upper() if channel == _NSB_CHANNEL else channel.title()}"
    _plot_efficiency_component_page(
        wavelength, unit, component_values, title, output_path, pdf_pages, relative=False
    )
    _plot_efficiency_component_page(
        wavelength, unit, component_values, title, output_path, pdf_pages, relative=True
    )
    return None, None


def _aligned_wavelengths(baseline, candidate):
    """Return wavelength arrays aligned to baseline units or a diagnostic."""
    if _WAVELENGTH_COLUMN not in baseline.colnames or _WAVELENGTH_COLUMN not in candidate.colnames:
        return None, "Wavelength table has no wl column."
    baseline_wavelength = baseline[_WAVELENGTH_COLUMN]
    candidate_wavelength = candidate[_WAVELENGTH_COLUMN]
    baseline_unit = baseline_wavelength.unit or u.Unit(_WAVELENGTH_FALLBACK_UNIT)
    candidate_unit = candidate_wavelength.unit or u.Unit(_WAVELENGTH_FALLBACK_UNIT)
    try:
        baseline_unit.to(u.nm)
        baseline_values = np.asarray(baseline_wavelength, dtype=float)
        candidate_values = (np.asarray(candidate_wavelength) * candidate_unit).to_value(
            baseline_unit
        )
    except (TypeError, ValueError) as exc:
        return None, f"Wavelength units are incompatible: {exc}"
    if (
        baseline_values.size == 0
        or not np.all(np.isfinite(baseline_values))
        or np.any(np.diff(baseline_values) <= 0)
    ):
        return None, "Wavelength grid must be finite, non-empty, and strictly increasing."
    if baseline_values.shape != candidate_values.shape or not np.allclose(
        baseline_values, candidate_values, rtol=1e-12, atol=0
    ):
        return None, "Wavelength grids differ; tables were not resampled."
    return (baseline_values, baseline_unit), None


def _efficiency_component_values(baseline, candidate):
    """Return component curves grouped by optical stage."""
    values = {}
    for group, components in _EFFICIENCY_COMPONENT_GROUPS:
        values[group] = {}
        for label, columns in components:
            baseline_values = [_curve_values(baseline, column) for column in columns]
            candidate_values = [_curve_values(candidate, column) for column in columns]
            values[group][label] = (
                np.prod(baseline_values, axis=0),
                np.prod(candidate_values, axis=0),
            )
    return values


def _plot_efficiency_component_page(
    wavelength, unit, components, title, output_path, pdf_pages, relative
):
    """Write one absolute or relative optical-component comparison page."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for axis, (group, values) in zip(axes.flat, components.items(), strict=False):
        axis.set_title(group)
        _plot_component_group(axis, wavelength, unit, values, relative)
    axes.flat[-1].axis("off")
    page_title = "Relative change by efficiency component" if relative else "Efficiency components"
    figure.suptitle(f"{title}: {page_title}")
    figure.tight_layout()
    if pdf_pages is not None:
        pdf_pages.savefig(figure)
    else:
        filename = "efficiency-components-relative.png" if relative else "efficiency-components.png"
        figure.savefig(Path(output_path) / filename)
    plt.close(figure)


def _plot_component_group(axis, wavelength, unit, values, relative):
    """Plot all optical components for one efficiency stage."""
    for label, (baseline, candidate) in values.items():
        if relative:
            curve = np.divide(
                candidate - baseline,
                baseline,
                out=np.full_like(candidate, np.nan, dtype=float),
                where=baseline != 0,
            )
            axis.plot(wavelength, curve * 100, label=label)
        else:
            axis.plot(wavelength, baseline, linestyle="--", label=f"{label} baseline")
            axis.plot(wavelength, candidate, label=f"{label} candidate")
    if relative:
        axis.axhline(0, color=_REFERENCE_LINE_COLOR, linewidth=0.7)
    _label_wavelength_axis(axis, unit)
    axis.set_ylabel(_RELATIVE_CHANGE_LABEL if relative else "Efficiency")
    axis.legend(fontsize="small")


def _label_wavelength_axis(axis, unit):
    """Show wavelength units and tick labels on every row of a shared-axis plot."""
    axis.set_xlabel(f"Wavelength ({unit})")
    axis.tick_params(axis="x", labelbottom=True)


def _wavelength_arrays(baseline, candidate, channel):
    """Return aligned detected spectra or a reason they cannot be compared."""
    aligned, diagnostic = _aligned_wavelengths(baseline, candidate)
    if diagnostic:
        return None, diagnostic
    wavelength, unit = aligned
    if channel == _NSB_CHANNEL:
        columns = ("nsb_detected", "N4")
        ordinate = "Detected NSB [1e9 ph / (nm s m2 sr)]"
    else:
        columns = ("cherenkov_detected", "C4")
        ordinate = "Detected Cherenkov spectrum (relative units)"
    selected = [
        next((column for column in columns if column in table.colnames), None)
        for table in (baseline, candidate)
    ]
    if not any(selected):
        selected = ["eff", "eff"]
        ordinate = "Optical throughput"
    if any(column not in table.colnames for table, column in zip((baseline, candidate), selected)):
        return (
            None,
            f"Wavelength table has no eff column or paired detected spectrum for {channel}.",
        )
    try:
        before, after = [
            _curve_values(table, column)
            for table, column in zip((baseline, candidate), selected, strict=True)
        ]
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    return (wavelength, before, after, unit, ordinate), None


def _curve_values(table, column):
    """Require a numeric, finite curve for each wavelength."""
    values = np.asarray(table[column], dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"Wavelength table column {column} must contain finite scalar values.")
    return values


def _read_wavelength_table(path):
    """Read ECSV output or a legacy whitespace table without hiding ECSV errors."""
    with Path(path).open(encoding="utf-8") as source:
        is_ecsv = source.readline().startswith("# %ECSV")
    table_format = _ECSV_FORMAT if is_ecsv else _BASIC_TABLE_FORMAT
    return Table.read(path, format=table_format)
