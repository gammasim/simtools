from simtools.production_configuration.simulation_jobs import (
    build_job_grid_metadata,
    resolve_single_model_version,
)


def test_resolve_single_model_version_uses_first_list_entry():
    assert resolve_single_model_version(["7.0.0", "7.1.0"]) == "7.0.0"
    assert resolve_single_model_version("7.0.0") == "7.0.0"


def test_build_job_grid_metadata_includes_job_context():
    metadata = build_job_grid_metadata(
        {
            "site": "North",
            "simulation_software": "corsika_sim_telarray",
            "coordinate_system": "ra_dec",
            "observing_time": "2017-09-16 00:00:00",
            "lookup_table": "limits.ecsv",
        }
    )

    assert metadata["site"] == "North"
    assert metadata["simulation_software"] == "corsika_sim_telarray"
    assert metadata["coordinate_system"] == "ra_dec"
    assert metadata["observing_time_utc"].startswith("2017-09-16T00:00:00")
    assert metadata["lookup_table"] == "limits.ecsv"
