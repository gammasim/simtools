"""Bookkeeping helpers for streaming production job generation."""

import astropy.units as u
import numpy as np


def format_quantity_range(quantity_min, quantity_max, summary_unit=None):
    """Format quantity min/max as a single value or range with explicit unit."""
    summary_unit = summary_unit or quantity_max.unit
    min_value = quantity_min.to_value(summary_unit)
    max_value = quantity_max.to_value(summary_unit)

    if np.isclose(min_value, max_value):
        return f"{max_value:.6g} {summary_unit}"
    return f"[{min_value:.6g}, {max_value:.6g}] {summary_unit}"


def format_quantity_summary(quantity_values, summary_unit=None):
    """Format a quantity series as a single value or range with explicit unit."""
    quantity_values = u.Quantity(quantity_values)
    return format_quantity_range(
        quantity_values.min(),
        quantity_values.max(),
        summary_unit=summary_unit,
    )


def format_integer_summary(values):
    """Format integer min/max as a single value or compact range."""
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return f"{max_value:d}"
    return f"[{min_value:d}, {max_value:d}]"
