"""Build summaries for generated production job grids."""

import numpy as np
from astropy import units as u


def format_quantity_summary(quantity_values, summary_unit=None):
    """Format quantity min/max as a single value or range with explicit unit."""
    quantity_min = quantity_values.min()
    quantity_max = quantity_values.max()

    summary_unit = summary_unit or quantity_max.unit
    min_value = quantity_min.to_value(summary_unit)
    max_value = quantity_max.to_value(summary_unit)

    if np.isclose(min_value, max_value):
        return f"{max_value:.6g} {summary_unit}"
    return f"[{min_value:.6g}, {max_value:.6g}] {summary_unit}"


def _format_integer_summary(values):
    """Format integer min/max as a single value or compact range."""
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return f"{max_value:d}"
    return f"[{min_value:d}, {max_value:d}]"


def _get_quantity_summary(rows, key, summary_unit=None):
    """Return formatted quantity summary for available row values."""
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return "not available"
    return format_quantity_summary(u.Quantity(values), summary_unit=summary_unit)


def build_job_grid_summary(rows):
    """Build a user-facing summary of generated job-grid values."""
    if not rows:
        return {
            "simulation_rows": 0,
            "total_showers": 0,
        }

    showers_per_run_values = [int(row["showers_per_run"]) for row in rows]
    configured_showers_per_run_values = [
        int(row.get("configured_showers_per_run", row["showers_per_run"])) for row in rows
    ]
    return {
        "simulation_rows": len(rows),
        "energy_min_used": _get_quantity_summary(rows, "energy_min"),
        "energy_max_used": _get_quantity_summary(rows, "energy_max"),
        "energy_min_configured": _get_quantity_summary(rows, "configured_energy_min"),
        "energy_max_configured": _get_quantity_summary(rows, "configured_energy_max"),
        "energy_min_lookup_limit": _get_quantity_summary(
            rows, "energy_min_lookup_limit", summary_unit=u.GeV
        ),
        "core_scatter_max_used": _get_quantity_summary(rows, "core_scatter_max"),
        "core_scatter_max_configured": _get_quantity_summary(rows, "configured_core_scatter_max"),
        "core_scatter_max_lookup_limit": _get_quantity_summary(rows, "lookup_core_scatter_max"),
        "view_cone_min_used": _get_quantity_summary(rows, "view_cone_min"),
        "view_cone_max_used": _get_quantity_summary(rows, "view_cone_max"),
        "view_cone_min_configured": _get_quantity_summary(rows, "configured_view_cone_min"),
        "view_cone_max_configured": _get_quantity_summary(rows, "configured_view_cone_max"),
        "view_cone_max_lookup_limit": _get_quantity_summary(rows, "lookup_view_cone_max"),
        "showers_per_run_used": _format_integer_summary(showers_per_run_values),
        "showers_per_run_configured": _format_integer_summary(configured_showers_per_run_values),
        "total_showers": int(sum(showers_per_run_values)),
    }
