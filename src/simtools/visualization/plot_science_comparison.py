"""Plot scalar and wavelength science-result comparisons."""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools.visualization.matplotlib_backend import pyplot as plt


def plot_comparison(changes, baseline, candidate, output_path, figure_format=None):
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
        np.nan if changes[name].get("relative") is None else changes[name]["relative"] * 100
        for name in metrics
    ]
    values_axis.axis("off")
    values_axis.set_title("Scalar values")
    for position, name in zip(positions, metrics, strict=True):
        item = changes[name]
        unit = item["baseline"].get("unit", "")
        unit_label = f" {unit}" if unit else ""
        values_axis.text(
            0,
            position,
            f"{name}: {item['baseline']['value']:g}{unit_label}  ->  "
            f"{item['candidate']['value']:g}{unit_label}",
            va="center",
        )
    values_axis.set_ylim(len(metrics) - 0.5, -0.5)
    change_axis.set_title(
        f"Relative change: {baseline.get('label', 'baseline')} to "
        f"{candidate.get('label', 'candidate')}"
    )
    baseline_version = baseline.get("software", {}).get("simtools_version")
    candidate_version = candidate.get("software", {}).get("simtools_version")
    if baseline_version or candidate_version:
        figure.suptitle(
            f"simtools {baseline_version or 'unknown'} vs {candidate_version or 'unknown'}"
        )
    change_axis.barh(positions, changes_values)
    change_axis.axvline(0, color="black", linewidth=0.7)
    metric_labels = [
        f"{name} ({changes[name]['baseline'].get('unit')})"
        if changes[name]["baseline"].get("unit")
        else name
        for name in metrics
    ]
    change_axis.set_yticks(positions, metric_labels)
    change_axis.set_xlabel("Relative change (%)")
    figure.tight_layout()
    formats = list(figure_format or ("png", "pdf"))
    products = {}
    for image_format in formats:
        path = output / f"science-comparison.{image_format}"
        figure.savefig(path)
        products[image_format] = str(path)
    plt.close(figure)
    return products, []


def plot_wavelength_comparison(baseline_path, candidate_path, output_path):
    """Overlay wavelength tables only when their grids match after unit conversion."""
    baseline = Table.read(baseline_path, format="ascii.ecsv")
    candidate = Table.read(candidate_path, format="ascii.ecsv")
    if "wl" not in baseline.colnames or "wl" not in candidate.colnames:
        return None, "Wavelength table has no wl column."
    baseline_wl, candidate_wl = baseline["wl"], candidate["wl"]
    try:
        if getattr(candidate_wl, "unit", None) and getattr(baseline_wl, "unit", None):
            candidate_values = candidate_wl.to(baseline_wl.unit).value
        else:
            candidate_values = np.asarray(candidate_wl)
    except u.UnitConversionError as exc:
        return None, f"Wavelength units are incompatible: {exc}"
    baseline_values = np.asarray(baseline_wl)
    if baseline_values.shape != candidate_values.shape or not np.array_equal(
        baseline_values, candidate_values
    ):
        return None, "Wavelength grids differ; tables were not resampled."
    if "eff" not in baseline.colnames or "eff" not in candidate.colnames:
        return None, "Wavelength table has no eff column."
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    figure, (spectrum_axis, difference_axis) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    left = np.asarray(baseline["eff"])
    right = np.asarray(candidate["eff"])
    spectrum_axis.plot(baseline_values, left, label="baseline")
    spectrum_axis.plot(baseline_values, right, label="candidate")
    spectrum_axis.set_ylabel("Efficiency")
    spectrum_axis.legend()
    difference_axis.plot(baseline_values, right - left)
    difference_axis.set_xlabel(f"Wavelength ({baseline_wl.unit or 'native unit'})")
    difference_axis.set_ylabel("Candidate - baseline")
    figure.tight_layout()
    product = output / "wavelength-comparison.png"
    figure.savefig(product)
    plt.close(figure)
    return str(product), None
