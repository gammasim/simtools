"""Tests for production metadata file discovery, filtering, and grouping."""

from pathlib import Path

import astropy.units as u
import pytest

from simtools.io.ascii_handler import write_data_to_file
from simtools.production_configuration.production_file_selection import (
    SIMULATE_PROD_JOB_METADATA,
    _embedded_metadata_matches,
    check_manifest,
    inventory_production_files,
    select_file_groups,
    validate_required_production_outputs,
)


def _write_manifest(base, job_name, run_number, he_interaction="qgs3", zenith=None):
    base = Path(base)
    job_directory = base / job_name
    job_directory.mkdir()
    data_file = job_directory / f"gamma_run{run_number:06d}.reduced_event_data.hdf5"
    simtel_file = job_directory / f"gamma_run{run_number:06d}.simtel.zst"
    data_file.touch()
    simtel_file.touch()
    manifest = {
        "schema_name": "simulate_prod_job_metadata",
        "schema_version": "1.0.0",
        "product_type": "simulate_prod_job",
        "job_id": job_name,
        "status": "complete",
        "catalog_metadata": {"runNumber": run_number},
        "configuration": {
            "run_number": run_number,
            "primary": "gamma",
            "site": "North",
            "array_layout_name": "CTAO-North-Alpha",
            "model_version": "7.0.0",
            "simulation_software": "corsika_sim_telarray",
            "corsika_he_interaction": he_interaction,
            "azimuth_angle": {"value": 180.0, "unit": "deg"},
            "zenith_angle": zenith or {"value": 20.0, "unit": "deg"},
            "energy_min": {"value": 0.03, "unit": "TeV"},
            "energy_max": {"value": 300.0, "unit": "TeV"},
            "view_cone_min": {"value": 0.0, "unit": "deg"},
            "view_cone_max": {"value": 10.0, "unit": "deg"},
            "cores_per_shower": 10,
            "core_scatter_max": {"value": 500.0, "unit": "m"},
            "showers_per_run": 100,
            "model_parameter_overrides": {},
            "atmosphere": {},
        },
        "files": {
            "reduced_event_data": [data_file.name],
            "sim_telarray": [simtel_file.name],
        },
    }
    manifest_path = job_directory / SIMULATE_PROD_JOB_METADATA
    write_data_to_file(manifest, manifest_path)
    return manifest_path


def test_select_file_groups_filters_quantities_and_orders_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000002", 2, he_interaction="qgs3")
    _write_manifest(tmp_test_directory, "job-000001", 1, he_interaction="qgs3")
    _write_manifest(tmp_test_directory, "job-000003", 3, he_interaction="epos")

    result = select_file_groups(
        tmp_test_directory,
        selections=[
            "configuration.corsika_he_interaction=qgs3",
            "configuration.zenith_angle=20 deg",
        ],
        file_type="reduced_event_data",
    )

    assert result["metadata_files_read"] == 3
    assert result["matching_jobs"] == 2
    assert result["configuration_groups"] == 1
    assert result["groups"][0].run_numbers == [1, 2]
    assert [path.name for path in result["groups"][0].file_paths] == [
        "gamma_run000001.reduced_event_data.hdf5",
        "gamma_run000002.reduced_event_data.hdf5",
    ]


def test_select_file_groups_separates_different_configurations(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1, he_interaction="qgs3")
    _write_manifest(
        tmp_test_directory,
        "job-000002",
        2,
        he_interaction="qgs3",
        zenith={"value": 0.3490658503988659, "unit": "rad"},
    )

    result = select_file_groups(tmp_test_directory)

    assert result["configuration_groups"] == 1
    assert result["groups"][0].run_numbers == [1, 2]


def test_select_file_groups_reports_missing_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1)
    _write_manifest(tmp_test_directory, "job-000003", 3)

    result = select_file_groups(tmp_test_directory)

    assert result["groups"][0].missing_run_numbers == [2]


def test_select_file_groups_rejects_duplicate_runs(tmp_test_directory):
    _write_manifest(tmp_test_directory, "job-000001", 1)
    _write_manifest(tmp_test_directory, "job-000002", 1)

    with pytest.raises(ValueError, match="Duplicate run numbers"):
        select_file_groups(tmp_test_directory)


def test_check_manifest_rejects_paths_outside_job_directory(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    manifest = {
        "schema_name": "simulate_prod_job_metadata",
        "schema_version": "1.0.0",
        "product_type": "simulate_prod_job",
        "job_id": "job-000001",
        "status": "complete",
        "catalog_metadata": {"runNumber": 1},
        "configuration": {
            "run_number": 1,
            "primary": "gamma",
            "site": "North",
            "array_layout_name": "CTAO-North-Alpha",
            "model_version": "7.0.0",
            "simulation_software": "corsika_sim_telarray",
            "azimuth_angle": {"value": 180.0, "unit": "deg"},
            "zenith_angle": {"value": 20.0, "unit": "deg"},
            "energy_min": {"value": 0.03, "unit": "TeV"},
            "energy_max": {"value": 300.0, "unit": "TeV"},
            "view_cone_min": {"value": 0.0, "unit": "deg"},
            "view_cone_max": {"value": 10.0, "unit": "deg"},
            "cores_per_shower": 10,
            "core_scatter_max": {"value": 500.0, "unit": "m"},
            "showers_per_run": 100,
            "model_parameter_overrides": {},
            "atmosphere": {},
        },
        "files": {"sim_telarray": ["../gamma_run000001.simtel.zst"]},
    }
    write_data_to_file(manifest, manifest_path)

    with pytest.raises(ValueError, match="escapes job directory"):
        check_manifest(manifest_path)


def test_check_manifest_rejects_missing_files(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    (manifest_path.parent / "gamma_run000001.reduced_event_data.hdf5").unlink()

    with pytest.raises(FileNotFoundError, match="references missing file"):
        check_manifest(manifest_path)


def test_check_manifest_rejects_unlisted_production_files(tmp_test_directory):
    manifest_path = _write_manifest(tmp_test_directory, "job-000001", 1)
    (manifest_path.parent / "gamma_run000001.simtel.log.gz").touch()

    with pytest.raises(ValueError, match="Unexpected production files"):
        check_manifest(manifest_path)


def test_inventory_production_files_discovers_nested_simulation_outputs(tmp_test_directory):
    job_directory = Path(tmp_test_directory)
    simtel_file = job_directory / "sim_telarray" / "run000001" / "gamma_run000001.simtel.zst"
    log_file = job_directory / "sim_telarray" / "run000001" / "gamma_run000001.simtel.log.gz"
    reduced_file = (
        job_directory / "sim_telarray" / "run000001" / "gamma_run000001.reduced_event_data.hdf5"
    )
    simtel_file.parent.mkdir(parents=True)
    simtel_file.touch()
    log_file.touch()
    reduced_file.touch()

    assert inventory_production_files(job_directory) == {
        "reduced_event_data": ["sim_telarray/run000001/gamma_run000001.reduced_event_data.hdf5"],
        "sim_telarray": ["sim_telarray/run000001/gamma_run000001.simtel.zst"],
        "sim_telarray_log": ["sim_telarray/run000001/gamma_run000001.simtel.log.gz"],
    }


def test_embedded_metadata_matches_in_memory_and_serialized_quantities():
    assert _embedded_metadata_matches(0 * u.deg, "0.0")
    assert _embedded_metadata_matches({"value": 20.0, "unit": "deg"}, "20")
    assert not _embedded_metadata_matches(20 * u.deg, "21")


def test_validate_required_production_outputs_rejects_incomplete_job(tmp_test_directory):
    with pytest.raises(ValueError, match="no 'sim_telarray' output"):
        validate_required_production_outputs(
            {"reduced_event_data": ["gamma.reduced_event_data.hdf5"]},
            "corsika_sim_telarray",
            tmp_test_directory,
        )
