"""Tests for production metadata backfill and validation workflows."""

from pathlib import Path

import astropy.units as u
import pytest

from simtools.production_configuration import production_metadata


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
    with pytest.raises(FileNotFoundError, match="No production job directories"):
        production_metadata._check_existing_manifests(production_path)

    (production_path / "job-000001").mkdir()

    with pytest.raises(FileNotFoundError, match="Missing production metadata manifest"):
        production_metadata._check_existing_manifests(production_path)


def test_write_production_metadata_dispatches_check_and_write_modes(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    mock_check = mocker.patch(
        "simtools.production_configuration.production_metadata._check_existing_manifests"
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.production_metadata._write_manifests"
    )

    production_metadata.write_production_metadata(
        {"production_path": str(production_path), "check": True}
    )
    production_metadata.write_production_metadata(
        {"production_path": str(production_path), "job_grid_file": "job-grid.ecsv"}
    )

    mock_check.assert_called_once_with(production_path)
    mock_write.assert_called_once_with(production_path, "job-grid.ecsv", mocker.ANY)


def test_check_existing_manifests_validates_each_manifest(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000001"
    job_directory.mkdir()
    manifest_path = job_directory / production_metadata.SIMULATE_PROD_JOB_METADATA
    manifest_path.touch()
    mock_check = mocker.patch(
        "simtools.production_configuration.production_metadata.check_manifest"
    )

    production_metadata._check_existing_manifests(production_path)

    mock_check.assert_called_once_with(manifest_path)


def test_write_manifests_does_not_mark_incomplete_job_complete(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    (production_path / "job-000001").mkdir()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=(
            [_job_grid_row()],
            {"site": "North", "simulation_software": "corsika_sim_telarray"},
        ),
    )
    write_file = mocker.patch(
        "simtools.production_configuration.production_metadata.write_data_to_file"
    )

    with pytest.raises(ValueError, match="Incomplete production job"):
        production_metadata._write_manifests(
            production_path,
            "job-grid.ecsv",
            {"overwrite": False},
        )

    write_file.assert_not_called()


def test_write_manifests_rejects_missing_job_directory(mocker, tmp_test_directory):
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=([_job_grid_row()], {}),
    )

    with pytest.raises(FileNotFoundError, match="Production job directory not found"):
        production_metadata._write_manifests(Path(tmp_test_directory), "job-grid.ecsv", {})


def test_write_manifests_supports_zero_based_job_directories(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000000"
    job_directory.mkdir()
    (job_directory / "gamma_run000001.simtel.zst").touch()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=([_job_grid_row()], {"site": "North", "simulation_software": "sim_telarray"}),
    )
    write_file = mocker.patch(
        "simtools.production_configuration.production_metadata.write_data_to_file"
    )

    production_metadata._write_manifests(production_path, "job-grid.ecsv", {"overwrite": False})

    assert write_file.call_args.args[1] == job_directory / "simulate_prod_job_metadata.yml"


def test_write_manifests_requires_overwrite_for_existing_manifest(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000001"
    job_directory.mkdir()
    (job_directory / production_metadata.SIMULATE_PROD_JOB_METADATA).touch()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=([_job_grid_row()], {}),
    )

    with pytest.raises(FileExistsError, match="Use --overwrite"):
        production_metadata._write_manifests(production_path, "job-grid.ecsv", {})


def test_write_manifests_omits_unknown_catalog_and_atmosphere_values(mocker, tmp_test_directory):
    production_path = Path(tmp_test_directory)
    job_directory = production_path / "job-000001"
    job_directory.mkdir()
    (job_directory / "gamma_run000001.simtel.zst").touch()
    mocker.patch(
        "simtools.production_configuration.production_metadata.read_job_grid",
        return_value=(
            [_job_grid_row()],
            {"site": "North", "simulation_software": "corsika_sim_telarray"},
        ),
    )
    write_file = mocker.patch(
        "simtools.production_configuration.production_metadata.write_data_to_file"
    )

    production_metadata._write_manifests(
        production_path,
        "job-grid.ecsv",
        {"overwrite": False},
    )

    manifest = write_file.call_args.args[0]
    assert "sct" not in manifest["catalog_metadata"]
    assert manifest["configuration"]["atmosphere"] == {}


def test_backfilled_atmosphere_and_configuration_validation(tmp_test_directory):
    assert production_metadata._backfilled_atmosphere_configuration(
        {"curved_atmosphere_min_zenith_angle": 70 * u.deg}
    ) == {"curved_atmosphere_min_zenith_angle": 70 * u.deg}

    with pytest.raises(ValueError, match="primary"):
        production_metadata._validate_resolved_configuration({}, "job-grid.ecsv")
    with pytest.raises(FileNotFoundError, match="Production path not found"):
        production_metadata._job_directories(Path(tmp_test_directory) / "missing")
