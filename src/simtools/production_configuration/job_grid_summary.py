"""Build summaries for generated production job grids."""

from astropy import units as u

from simtools.production_configuration.job_generation_summary import (
    format_integer_summary,
    format_quantity_summary,
)


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
        "showers_per_run_min": min(showers_per_run_values),
        "showers_per_run_max": max(showers_per_run_values),
        "showers_per_run_used": format_integer_summary(showers_per_run_values),
        "showers_per_run_configured_max": max(configured_showers_per_run_values),
        "showers_per_run_configured": format_integer_summary(configured_showers_per_run_values),
        "total_showers": int(sum(showers_per_run_values)),
    }
