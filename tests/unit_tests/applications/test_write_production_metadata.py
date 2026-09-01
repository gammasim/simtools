"""Tests for the production metadata backfill and check application."""

from pathlib import Path

import astropy.units as u
import pytest

from simtools.applications import write_production_metadata


def _job_grid_row():
    """Return one complete authoritative job-grid row."""
    return {
        "run_number": 1,
        "primary": "gamma",
        "azimuth_angle": 180 * u.deg,
        "zenith_angle": 20 * u.deg,
        "energy_min": 0.03 * u.TeV,
        "energy_max": 300 * u.TeV,
        "cores_per_shower": 10,
        "core_scatter_max": 500 * u.m,
        "view_cone_min": 0 * u.deg,
        "view_cone_max": 10 * u.deg,
        "showers_per_run": 100,
        "model_version": "7.0.0",
        "array_layout_name": "CTAO-North-Alpha",
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "qgs3",
    }


def test_check_existing_manifests_reports_missing_job_manifest(tmp_test_directory):
    production_path = Path(tmp_test_directory)
    (production_path / "job-000001").mkdir()

    with pytest.raises(FileNotFoundError, match="Missing production metadata manifest"):
        write_production_metadata._check_existing_manifests(production_path)


def test_write_manifests_does_not_mark_incomplete_job_complete(
    mocker,
    tmp_test_directory,
):
    production_path = Path(tmp_test_directory)
    (production_path / "job-000001").mkdir()
    mocker.patch(
        "simtools.applications.write_production_metadata.read_job_grid",
        return_value=(
            [_job_grid_row()],
            {"site": "North", "simulation_software": "corsika_sim_telarray"},
        ),
    )
    write_file = mocker.patch("simtools.applications.write_production_metadata.write_data_to_file")

    with pytest.raises(ValueError, match="Incomplete production job"):
        write_production_metadata._write_manifests(
            production_path,
            "job-grid.ecsv",
            {"overwrite": False},
        )

    write_file.assert_not_called()


def test_write_manifests_omits_unknown_catalog_and_atmosphere_values(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000001"
    job_directory.mkdir()
    (job_directory / "gamma_run000001.simtel.zst").touch()
    mocker.patch(
        "simtools.applications.write_production_metadata.read_job_grid",
        return_value=(
            [_job_grid_row()],
            {"site": "North", "simulation_software": "corsika_sim_telarray"},
        ),
    )
    write_file = mocker.patch("simtools.applications.write_production_metadata.write_data_to_file")

    write_production_metadata._write_manifests(
        production_path,
        "job-grid.ecsv",
        {"overwrite": False},
    )

    manifest = write_file.call_args.args[0]
    assert "sct" not in manifest["catalog_metadata"]
    assert manifest["configuration"]["atmosphere"] == {}
