"""Tests for validated production definitions."""

import pytest

from simtools.production_configuration.production_definition import (
    production_definition_from_arguments,
)


def test_production_definition_from_grid_arguments():
    definition = production_definition_from_arguments(
        {
            "axis": [["zenith", "20", "deg", "40", "deg", "3"]],
            "corsika_limits": "limits.ecsv",
            "run_number_offset": 4,
            "number_of_runs": 5,
            "showers_per_run": 100,
            "total_showers": None,
            "max_showers_per_run": None,
            "simulation_software": "corsika_sim_telarray",
        }
    )

    assert definition["axes"] == [["zenith", "20", "deg", "40", "deg", "3"]]
    assert definition["limits_lookup"] == {"file": "limits.ecsv", "interpolation": "point"}
    assert definition["run_number_allocation"] == {"offset": 4, "runs": 5}
    assert definition["job_sizes"]["showers_per_run"] == 100
    assert definition["simulation_stages"] == ["corsika", "sim_telarray"]


def test_production_definition_rejects_unknown_simulation_stage():
    with pytest.raises(KeyError, match="unknown"):
        production_definition_from_arguments(
            {
                "simulation_software": "unknown",
            }
        )
