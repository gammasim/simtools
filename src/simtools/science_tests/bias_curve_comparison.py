"""Load and compare derived bias-curve ECSV tables."""

from pathlib import Path

import numpy as np
from astropy.table import Table

from simtools.visualization.matplotlib_backend import pyplot as plt

THRESHOLD = "threshold"
RATE_COLUMNS = ("NSB rate (Hz)", "Proton rate (Hz)")
CURVE_FILE = "bias_curve.ecsv"


def load_results(path, telescopes=None):
    """Load bias-curve tables keyed by telescope name."""
    path = Path(path)
    selected = set(telescopes or [])
    if path.is_file():
        if path.name != CURVE_FILE:
            raise ValueError(f"Expected {CURVE_FILE}: {path}")
        if len(selected) != 1:
            raise ValueError("Select exactly one telescope when a bias-curve file is given.")
        curves = {next(iter(selected)): [path]}
    else:
        files = sorted(path.rglob(CURVE_FILE))
        curves = {}
        for file in files:
            curves.setdefault(file.parent.name, []).append(file)
        if selected:
            curves = {name: items for name, items in curves.items() if name in selected}
    if not curves:
        raise ValueError(f"No {CURVE_FILE} files found in {path}.")
    duplicates = sorted(name for name, files in curves.items() if len(files) > 1)
    if duplicates:
        raise ValueError(
            f"Multiple bias-curve files found for telescope(s): {', '.join(duplicates)}."
        )
    return {name: _read_curve(files[0]) for name, files in curves.items()}


def compare_results(baseline, candidate, tolerances=None):
    """Compare NSB and proton rates on their shared threshold ranges."""
    tolerances = tolerances or {}
    unknown_tolerances = set(tolerances) - set(RATE_COLUMNS)
    if unknown_tolerances:
        names = ", ".join(sorted(unknown_tolerances))
        raise ValueError(f"Tolerance given for unknown bias-curve metric(s): {names}.")
    changes = {}
    for column in RATE_COLUMNS:
        deviation, common_range = _max_relative_deviation(baseline, candidate, column)
        tolerance = tolerances.get(column)
        changes[column] = {
            "max_relative_deviation": deviation,
            "common_threshold_range": common_range,
            "tolerance": tolerance,
            "advisory": bool(tolerance is not None and deviation > tolerance),
        }
    return changes


def plot_comparison(baseline, candidate, telescope, labels, pdf_pages):
    """Add the NSB and proton rate curves for one telescope to a PDF report."""
    figure, axes = plt.subplots(1, len(RATE_COLUMNS), figsize=(12, 5))
    for axis, column in zip(axes, RATE_COLUMNS, strict=True):
        has_positive_rates = False
        for table, label in ((baseline, labels[0]), (candidate, labels[1])):
            threshold, rates = _finite_curve(table, column)
            axis.plot(threshold, rates, marker="o", label=label)
            has_positive_rates = has_positive_rates or bool(np.any(rates > 0))
        axis.set_title(column.removesuffix(" (Hz)"))
        axis.set_xlabel("Trigger threshold")
        axis.set_ylabel("Rate (Hz)")
        if has_positive_rates:
            axis.set_yscale("log")
        axis.legend()
    figure.suptitle(f"{telescope}: bias curves")
    figure.tight_layout()
    pdf_pages.add(figure)


def _read_curve(path):
    """Read and validate one bias-curve ECSV table."""
    table = Table.read(path, format="ascii.ecsv")
    missing = sorted({THRESHOLD, *RATE_COLUMNS} - set(table.colnames))
    if missing:
        raise ValueError(f"Bias-curve table {path} is missing columns: {', '.join(missing)}.")
    threshold = np.asarray(table[THRESHOLD], dtype=float)
    if not np.all(np.isfinite(threshold)) or len(np.unique(threshold)) != len(threshold):
        raise ValueError(f"Bias-curve table {path} must contain finite, unique threshold values.")
    for column in RATE_COLUMNS:
        values = np.asarray(table[column], dtype=float)
        if np.any(values[np.isfinite(values)] < 0):
            raise ValueError(f"Bias-curve rates in {path} must be non-negative.")
    return table


def _max_relative_deviation(baseline, candidate, column):
    """Return the largest relative curve deviation and shared x range."""
    base_x, base_y = _finite_curve(baseline, column)
    candidate_x, candidate_y = _finite_curve(candidate, column)
    lower = max(base_x[0], candidate_x[0])
    upper = min(base_x[-1], candidate_x[-1])
    if lower >= upper:
        raise ValueError(f"{column} curves have no common threshold range.")
    points = np.unique(
        np.concatenate(
            (
                [lower, upper],
                base_x[(base_x > lower) & (base_x < upper)],
                candidate_x[(candidate_x > lower) & (candidate_x < upper)],
            )
        )
    )
    base_rates = np.interp(points, base_x, base_y)
    candidate_rates = np.interp(points, candidate_x, candidate_y)
    zero_baseline = base_rates == 0
    if np.any(zero_baseline & (candidate_rates != 0)):
        raise ValueError(f"{column} relative deviation is undefined for zero baseline rate.")
    relative = np.zeros_like(base_rates)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(
            np.abs(candidate_rates - base_rates),
            np.abs(base_rates),
            out=relative,
            where=~zero_baseline,
        )
    if not np.all(np.isfinite(relative)):
        raise ValueError(f"{column} relative deviation is not finite.")
    return float(np.max(relative)), [float(lower), float(upper)]


def _finite_curve(table, column):
    """Return sorted finite threshold and rate values for a curve."""
    threshold = np.asarray(table[THRESHOLD], dtype=float)
    rates = np.asarray(table[column], dtype=float)
    if not np.all(np.isfinite(threshold)) or not np.all(np.isfinite(rates)):
        raise ValueError(f"{column} curve contains missing or non-finite values.")
    order = np.argsort(threshold)
    threshold, rates = threshold[order], rates[order]
    if len(threshold) < 2:
        raise ValueError(f"{column} curve needs at least two finite rate values.")
    return threshold, rates
