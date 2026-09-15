"""Tests for production resource requirement collection."""

import json
from pathlib import Path

import pytest

from simtools.production_configuration import resource_requirements
from simtools.production_configuration.production_file_selection import ProductionManifest


def _manifest(tmp_test_directory):
    directory = Path(tmp_test_directory) / "job-000001"
    corsika_directory = directory / "corsika" / "run000001"
    simtel_directory = directory / "sim_telarray" / "run000001"
    corsika_directory.mkdir(parents=True)
    simtel_directory.mkdir(parents=True)
    for role, directory_part in (
        ("corsika", corsika_directory),
        ("sim_telarray", simtel_directory),
    ):
        suffix = "simtel" if role == "sim_telarray" else role
        (directory_part / f"run000001.{suffix}.resources.json").write_text(
            json.dumps(
                {
                    "schema_name": "process_resource_record",
                    "schema_version": "1.0.0",
                    "run_id": "1",
                    "role": role,
                    "return_code": 0,
                    "wall_time_seconds": 20.0,
                    "user_cpu_seconds": 10.0,
                    "system_cpu_seconds": 2.0,
                    "peak_rss_bytes": 1000,
                }
            ),
            encoding="utf-8",
        )
    outputs = {
        "sim_telarray": "event.simtel.zst",
        "reduced_event_data": "events.hdf5",
        "sim_telarray_histogram": "events.hdata.zst",
    }
    for output_type, file_name in outputs.items():
        (simtel_directory / file_name).write_bytes(b"x" * 10)
        outputs[output_type] = f"sim_telarray/run000001/{file_name}"
    return ProductionManifest(
        path=directory / "simulate_prod_job_metadata.yml",
        data={
            "job_id": "job-000001",
            "configuration": {
                "run_number": 1,
                "primary": "gamma",
                "site": "North",
                "array_layout_name": ["CTAO-North-Alpha"],
                "model_version": "7.0.0",
                "energy_min": {"value": 100.0, "unit": "GeV"},
                "energy_max": {"value": 100.0, "unit": "GeV"},
                "zenith_angle": {"value": 20.0, "unit": "deg"},
                "azimuth_angle": {"value": 0.0, "unit": "deg"},
                "showers_per_run": 10,
            },
            "files": {key: [value] for key, value in outputs.items()},
        },
    )


def test_collect_resource_requirements_normalizes_and_measures_outputs(mocker, tmp_test_directory):
    manifest = _manifest(tmp_test_directory)
    mocker.patch.object(resource_requirements, "discover_manifests", return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements("production")

    assert diagnostics == []
    assert len(rows) == 2
    assert {row["role"] for row in rows} == {"corsika", "sim_telarray"}
    assert rows[0]["wall_time_seconds_per_event"] == pytest.approx(2.0)
    assert rows[0]["cpu_time_seconds_per_event"] == pytest.approx(1.2)
    corsika_row = next(row for row in rows if row["role"] == "corsika")
    simtel_row = next(row for row in rows if row["role"] == "sim_telarray")
    assert corsika_row["corsika_output_bytes"] is None
    assert corsika_row["sim_telarray_storage_bytes"] is None
    assert simtel_row["sim_telarray_storage_bytes"] == 30
    assert simtel_row["sim_telarray_storage_bytes_per_event"] == pytest.approx(3.0)


def test_collect_resource_requirements_reports_invalid_record_and_keeps_other_role(
    mocker, tmp_test_directory
):
    manifest = _manifest(tmp_test_directory)
    record = next((manifest.directory / "corsika").rglob("*.resources.json"))
    data = json.loads(record.read_text(encoding="utf-8"))
    data["return_code"] = 1
    record.write_text(json.dumps(data), encoding="utf-8")
    mocker.patch.object(resource_requirements, "discover_manifests", return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements("production")

    assert [row["role"] for row in rows] == ["sim_telarray"]
    assert diagnostics[0]["role"] == "corsika"
    assert "return code 1" in diagnostics[0]["message"]


def test_write_resource_requirements_writes_table_report_and_plots(mocker, tmp_test_directory):
    manifest = _manifest(tmp_test_directory)
    mocker.patch.object(resource_requirements, "discover_manifests", return_value=[manifest])
    plotter = mocker.Mock(return_value=[Path(tmp_test_directory) / "resource_wall_time"])

    result = resource_requirements.write_resource_requirements(
        {"baseline_path": "production", "select": [], "figure_format": ["png"]},
        tmp_test_directory,
        plotter,
    )

    assert result["table_file"].is_file()
    assert result["report_file"].is_file()
    assert "Storage" in result["report_file"].read_text(encoding="utf-8")
    plotter.assert_called_once()
