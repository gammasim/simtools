import astropy.units as u

from simtools.production_configuration.job_grid_summary import (
    build_job_grid_summary,
    format_quantity_summary,
)


def test_format_quantity_summary_returns_single_value_for_constant_series():
    summary = format_quantity_summary(u.Quantity([200, 200, 200], u.TeV))

    assert summary == "200 TeV"


def test_format_quantity_summary_returns_range_with_explicit_unit():
    summary = format_quantity_summary(u.Quantity([30 * u.GeV, 1 * u.TeV]))

    assert summary == "[30, 1000] GeV"


def test_build_job_grid_summary_includes_used_configured_and_lookup_values():
    rows = [
        {
            "energy_min": 40 * u.GeV,
            "energy_max": 100 * u.GeV,
            "configured_energy_min": 30 * u.GeV,
            "configured_energy_max": 100 * u.GeV,
            "energy_min_lookup_limit": 40 * u.GeV,
            "core_scatter_max": 150 * u.m,
            "configured_core_scatter_max": 200 * u.m,
            "lookup_core_scatter_max": 150 * u.m,
            "view_cone_min": 0 * u.deg,
            "view_cone_max": 2 * u.deg,
            "configured_view_cone_min": 0 * u.deg,
            "configured_view_cone_max": 5 * u.deg,
            "lookup_view_cone_max": 2 * u.deg,
            "showers_per_run": 500,
            "configured_showers_per_run": 1000,
        },
        {
            "energy_min": 50 * u.GeV,
            "energy_max": 100 * u.GeV,
            "configured_energy_min": 30 * u.GeV,
            "configured_energy_max": 100 * u.GeV,
            "energy_min_lookup_limit": 50 * u.GeV,
            "core_scatter_max": 200 * u.m,
            "configured_core_scatter_max": 200 * u.m,
            "lookup_core_scatter_max": 250 * u.m,
            "view_cone_min": 1 * u.deg,
            "view_cone_max": 5 * u.deg,
            "configured_view_cone_min": 1 * u.deg,
            "configured_view_cone_max": 5 * u.deg,
            "lookup_view_cone_max": 10 * u.deg,
            "showers_per_run": 1000,
            "configured_showers_per_run": 1000,
        },
    ]

    summary = build_job_grid_summary(rows)

    assert summary["simulation_rows"] == 2
    assert summary["energy_min_used"] == "[40, 50] GeV"
    assert summary["core_scatter_max_used"] == "[150, 200] m"
    assert summary["core_scatter_max_configured"] == "200 m"
    assert summary["view_cone_max_lookup_limit"] == "[2, 10] deg"
    assert summary["showers_per_run_used"] == "[500, 1000]"
    assert summary["showers_per_run_configured"] == "1000"
    assert summary["total_showers"] == 1500
