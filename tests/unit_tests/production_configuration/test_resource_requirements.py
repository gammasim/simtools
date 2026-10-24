"""Tests for production resource requirement collection."""

import json
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from simtools.job_execution import process_accounting
from simtools.production_configuration import resource_requirements
from simtools.production_configuration.production_file_selection import (
    ProductionManifest,
)

CORSIKA = "corsika"
SIM_TELARRAY = "sim_telarray"
BASELINE = "baseline"
CANDIDATE = "candidate"
ENCODING = "utf-8"
RESOURCE_RECORD_PATTERN = "*.resources.json"
UNKNOWN = "unknown"
RUN_DIRECTORY = "run000001"
DISCOVER_MANIFESTS = "discover_manifests"
PRODUCTION = "production"
COVERAGE_DIFFERS = "coverage differs"
BASELINE_PATH = "baseline_path"
CANDIDATE_PATH = "candidate_path"
MODEL_VERSION = "7.0.0"
FIXTURE_START_TIME = "2026-01-01T00:00:00+00:00"
TRIGGERED_EVENTS = "triggered_events"
SIMULATED_EVENTS = "simulated_events"
INITIAL_SCHEMA_VERSION = "1.0.0"
PRODUCTION_LABEL = "production_label"
MESSAGE = "message"
CONFIGURATION = "configuration"
FILES = "files"
REPORT_FILE = "report_file"
VALUE = "value"
START_TIME = "start_time"
HOSTNAME = "hostname"
CPU_MODEL = "cpu_model"
LOGICAL_CPU_COUNT = "logical_cpu_count"
AVAILABLE_CPU_COUNT = "available_cpu_count"
CONTAINER_IMAGE = "container_image"
CORSIKA_OUTPUT_BYTES = "corsika_output_bytes"
SIM_TELARRAY_STORAGE_BYTES = "sim_telarray_storage_bytes"
STORAGE_BYTES_PER_TRIGGERED_EVENT = "sim_telarray_storage_bytes_per_triggered_event"
SIMTEL = "simtel"
TEST_HOSTNAME = "node-1"
TEST_CPU_MODEL = "Test CPU"
ARM_CPU_MODEL = "ARM server"
EXECUTION_ENVIRONMENTS = "execution_environments"
RECORD_COUNT = "record_count"
HOST_COUNT = "host_count"
TEST_CONTAINER_IMAGE = "simtools.sif"


def _record_name(suffix):
    return f"{RUN_DIRECTORY}.{suffix}.resources.json"


def _manifest(tmp_test_directory):
    directory = Path(tmp_test_directory) / "job-000001"
    corsika_directory = directory / CORSIKA / RUN_DIRECTORY
    simtel_directory = directory / SIM_TELARRAY / RUN_DIRECTORY
    corsika_directory.mkdir(parents=True)
    simtel_directory.mkdir(parents=True)
    for role, directory_part in (
        (CORSIKA, corsika_directory),
        (SIM_TELARRAY, simtel_directory),
    ):
        suffix = SIMTEL if role == SIM_TELARRAY else role
        (directory_part / _record_name(suffix)).write_text(
            json.dumps(
                {
                    "run_id": "1",
                    "role": role,
                    "return_code": 0,
                    "wall_time_seconds": 20.0,
                    "user_cpu_seconds": 10.0,
                    "system_cpu_seconds": 2.0,
                    "peak_rss_bytes": 1000,
                    START_TIME: FIXTURE_START_TIME,
                    "end_time": "2026-01-01T00:00:20+00:00",
                    "measurement_method": "linux_procfs_direct_pid",
                    "platform": "Linux",
                }
            ),
            encoding=ENCODING,
        )
    outputs = {
        SIM_TELARRAY: "event.simtel.zst",
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
            CONFIGURATION: {
                "run_number": 1,
                "primary": "gamma",
                "site": "North",
                "array_layout_name": ["CTAO-North-Alpha"],
                "model_version": MODEL_VERSION,
                "energy_min": {VALUE: 100.0, "unit": "GeV"},
                "energy_max": {VALUE: 100.0, "unit": "GeV"},
                "zenith_angle": {VALUE: 20.0, "unit": "deg"},
                "azimuth_angle": {VALUE: 0.0, "unit": "deg"},
                "showers_per_run": 10,
            },
            FILES: {key: [value] for key, value in outputs.items()},
        },
    )


def test_collect_resource_requirements_normalizes_and_measures_outputs(mocker, tmp_test_directory):
    manifest = _manifest(tmp_test_directory)
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements(PRODUCTION)

    assert diagnostics == []
    assert len(rows) == 2
    assert {row["role"] for row in rows} == {CORSIKA, SIM_TELARRAY}
    assert rows[0]["wall_time_seconds_per_event"] == pytest.approx(2.0)
    assert rows[0]["cpu_time_seconds_per_event"] == pytest.approx(1.2)
    assert rows[0]["wall_time_seconds"] == pytest.approx(20.0)
    assert rows[0]["cpu_time_seconds"] == pytest.approx(12.0)
    corsika_row = next(row for row in rows if row["role"] == CORSIKA)
    simtel_row = next(row for row in rows if row["role"] == SIM_TELARRAY)
    assert corsika_row[CORSIKA_OUTPUT_BYTES] is None
    assert corsika_row[SIM_TELARRAY_STORAGE_BYTES] is None
    assert simtel_row[SIM_TELARRAY_STORAGE_BYTES] == 30
    assert simtel_row["sim_telarray_storage_bytes_per_event"] == pytest.approx(3.0)


def test_collect_resource_requirements_reports_invalid_record_and_keeps_other_role(
    mocker, tmp_test_directory
):
    manifest = _manifest(tmp_test_directory)
    record = next((manifest.directory / CORSIKA).rglob(RESOURCE_RECORD_PATTERN))
    data = json.loads(record.read_text(encoding=ENCODING))
    data["return_code"] = 1
    record.write_text(json.dumps(data), encoding=ENCODING)
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements(PRODUCTION)

    assert [row["role"] for row in rows] == [SIM_TELARRAY]
    assert diagnostics[0]["role"] == CORSIKA
    assert "return code 1" in diagnostics[0][MESSAGE]


def test_collect_resource_requirements_normalizes_simtel_by_triggered_events(
    mocker, tmp_test_directory
):
    manifest = _manifest(tmp_test_directory)
    manifest.data["statistics"] = {SIMULATED_EVENTS: 10, TRIGGERED_EVENTS: 5}
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements(PRODUCTION)

    assert diagnostics == []
    simtel_row = next(row for row in rows if row["role"] == SIM_TELARRAY)
    assert simtel_row[TRIGGERED_EVENTS] == 5
    assert simtel_row["wall_time_seconds_per_triggered_event"] == pytest.approx(4.0)
    assert simtel_row[STORAGE_BYTES_PER_TRIGGERED_EVENT] == pytest.approx(6.0)
    corsika_row = next(row for row in rows if row["role"] == CORSIKA)
    assert corsika_row[TRIGGERED_EVENTS] is None
    assert corsika_row["wall_time_seconds_per_triggered_event"] is None


def test_collect_resource_requirements_recovers_trigger_count_from_reduced_data(
    mocker, tmp_test_directory
):
    manifest = _manifest(tmp_test_directory)
    reduced_file = manifest.directory / manifest.data[FILES]["reduced_event_data"][0]
    with h5py.File(reduced_file, "w") as data_file:
        data_file.create_dataset("SHOWERS", shape=(10,), dtype="i8")
        data_file.create_dataset("TRIGGERS", shape=(4,), dtype="i8")
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements(PRODUCTION)

    assert diagnostics == []
    simtel_row = next(row for row in rows if row["role"] == SIM_TELARRAY)
    assert simtel_row[TRIGGERED_EVENTS] == 4


def test_write_resource_requirements_writes_table_report_and_plots(mocker, tmp_test_directory):
    manifest = _manifest(tmp_test_directory)
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])
    plotter = mocker.Mock(return_value=[Path(tmp_test_directory) / "resource_wall_time"])

    result = resource_requirements.write_resource_requirements(
        {BASELINE_PATH: PRODUCTION, "select": [], "figure_format": ["png"]},
        tmp_test_directory,
        plotter,
    )

    assert result["table_file"].is_file()
    assert result[REPORT_FILE].is_file()
    metrics = json.loads(result["json_file"].read_text(encoding=ENCODING))
    assert metrics["resource_row_count"] == 2
    assert metrics["diagnostic_count"] == 0
    assert metrics["summary"][0]["peak_rss_bytes_p05"] == pytest.approx(1000)
    assert metrics["summary"][0]["peak_rss_bytes_p95"] == pytest.approx(1000)
    assert metrics[EXECUTION_ENVIRONMENTS] == [
        {
            PRODUCTION_LABEL: BASELINE,
            "role": role,
            CPU_MODEL: UNKNOWN,
            LOGICAL_CPU_COUNT: None,
            AVAILABLE_CPU_COUNT: None,
            CONTAINER_IMAGE: UNKNOWN,
            HOST_COUNT: 1,
            RECORD_COUNT: 1,
        }
        for role in (CORSIKA, SIM_TELARRAY)
    ]
    report = result[REPORT_FILE].read_text(encoding=ENCODING)
    assert "Storage" in report
    assert "Wall time (s/run)" in report
    assert "CPU time (s/run)" in report
    assert "sim_telarray Storage (bytes/run)" in report
    assert all(
        f"{statistic}=" in report
        for statistic in ("mean", "median", "std", "min", "max", "p05", "p95")
    )
    assert "| baseline | corsika | unknown | - | - | unknown | 1 | 1 |" in report
    plotter.assert_called_once()


def test_collect_resource_requirements_keeps_execution_environment(mocker, tmp_test_directory):
    manifest = _manifest(tmp_test_directory)
    for record_path in manifest.directory.rglob(RESOURCE_RECORD_PATTERN):
        record = json.loads(record_path.read_text(encoding=ENCODING))
        record.update(
            hostname=TEST_HOSTNAME,
            cpu_model=TEST_CPU_MODEL,
            logical_cpu_count=8,
            available_cpu_count=2,
            container_image=TEST_CONTAINER_IMAGE,
        )
        record_path.write_text(json.dumps(record), encoding=ENCODING)
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements(PRODUCTION)

    assert diagnostics == []
    assert {
        (row[HOSTNAME], row[CPU_MODEL], row[AVAILABLE_CPU_COUNT], row[CONTAINER_IMAGE])
        for row in rows
    } == {(TEST_HOSTNAME, TEST_CPU_MODEL, 2, TEST_CONTAINER_IMAGE)}


@pytest.mark.parametrize("different_cpu", [False, True])
def test_split_records_keep_all_environments(mocker, tmp_test_directory, different_cpu):
    manifest = _manifest(tmp_test_directory)
    record_path = next((manifest.directory / CORSIKA).rglob(RESOURCE_RECORD_PATTERN))
    record = json.loads(record_path.read_text(encoding=ENCODING))
    record[HOSTNAME] = TEST_HOSTNAME
    record[CPU_MODEL] = TEST_CPU_MODEL
    record_path.write_text(json.dumps(record), encoding=ENCODING)
    record[HOSTNAME] = "node-2"
    if different_cpu:
        record[CPU_MODEL] = ARM_CPU_MODEL
    record_path.with_name("split.corsika.resources.json").write_text(
        json.dumps(record), encoding=ENCODING
    )
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])

    result = resource_requirements.write_resource_requirements(
        {BASELINE_PATH: PRODUCTION, "select": [], "figure_format": []},
        tmp_test_directory,
        mocker.Mock(return_value=[]),
    )

    metrics = json.loads(result["json_file"].read_text(encoding=ENCODING))
    environments = [item for item in metrics[EXECUTION_ENVIRONMENTS] if item["role"] == CORSIKA]
    assert sum(item[RECORD_COUNT] for item in environments) == 2
    assert sum(item[HOST_COUNT] for item in environments) == 2
    assert {item[CPU_MODEL] for item in environments} == (
        {TEST_CPU_MODEL, ARM_CPU_MODEL} if different_cpu else {TEST_CPU_MODEL}
    )


@pytest.mark.parametrize("legacy_metadata", [False, True])
def test_collect_resource_requirements_ignores_schema_metadata(
    mocker, tmp_test_directory, legacy_metadata
):
    manifest = _manifest(tmp_test_directory)
    if legacy_metadata:
        for record_path in manifest.directory.rglob(RESOURCE_RECORD_PATTERN):
            record = json.loads(record_path.read_text(encoding=ENCODING))
            record.update(schema_name="process_resource_record", schema_version="unregistered")
            record_path.write_text(json.dumps(record), encoding=ENCODING)
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])

    rows, diagnostics = resource_requirements.collect_resource_requirements(PRODUCTION)

    assert len(rows) == 2
    assert diagnostics == []


def test_write_resource_requirements_compact_skips_table_and_writes_summary(
    mocker, tmp_test_directory
):
    manifest = _manifest(tmp_test_directory)
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[manifest])
    plotter = mocker.Mock(return_value=[])

    result = resource_requirements.write_resource_requirements(
        {BASELINE_PATH: PRODUCTION, "report_mode": "compact"},
        tmp_test_directory,
        plotter,
    )

    assert result["table_file"] is None
    assert result[REPORT_FILE].is_file()
    report = result[REPORT_FILE].read_text(encoding=ENCODING)
    assert "Wall time (s/event)" in report
    assert "| baseline | corsika | 1 |" in report
    plotter.assert_called_once_with(
        plotter.call_args.args[0],
        tmp_test_directory,
        figure_format=None,
        report_mode="compact",
    )


def test_compact_resource_summary_handles_missing_metrics(tmp_test_directory):
    report = Path(tmp_test_directory) / "compact.md"
    summary = resource_requirements._summarize_compact_resource_requirements(
        [
            {
                PRODUCTION_LABEL: CANDIDATE,
                "role": SIM_TELARRAY,
                "wall_time_seconds_per_event": None,
                "cpu_time_seconds_per_event": 2.0,
                "peak_rss_bytes": 10,
                "sim_telarray_storage_bytes_per_event": None,
            }
        ]
    )

    assert summary[0]["wall_time_seconds_per_event_median"] is None
    assert summary[0]["cpu_time_seconds_per_event_median"] == pytest.approx(2.0)
    resource_requirements._write_compact_markdown_report(
        summary, [{"job_id": "job-1", "role": CORSIKA, MESSAGE: "bad record"}], report
    )
    assert "Skipped records" in report.read_text(encoding=ENCODING)


def test_resource_percentiles_are_interpolated():
    metrics = resource_requirements._metric_statistics("wall", [{"wall": v} for v in (0, 10, 20)])
    assert metrics["wall_p05"] == pytest.approx(1)
    assert metrics["wall_p95"] == pytest.approx(19)


def test_collect_and_write_apply_selection_and_candidate_labels(mocker, tmp_test_directory):
    baseline = _manifest(tmp_test_directory / BASELINE)
    candidate = _manifest(tmp_test_directory / CANDIDATE)
    baseline.data[CONFIGURATION]["primary"] = "gamma"
    candidate.data[CONFIGURATION]["primary"] = "proton"
    mocker.patch.object(resource_requirements, DISCOVER_MANIFESTS, return_value=[baseline])

    rows, _ = resource_requirements.collect_resource_requirements(
        PRODUCTION, selections=["configuration.primary=gamma"]
    )
    assert len(rows) == 2
    with pytest.raises(ValueError, match="No production jobs matched"):
        resource_requirements.collect_resource_requirements(
            PRODUCTION, selections=["configuration.primary=proton"]
        )

    mocker.patch.object(
        resource_requirements,
        DISCOVER_MANIFESTS,
        side_effect=[[baseline], [candidate]],
    )
    candidate.data[CONFIGURATION]["primary"] = "gamma"
    plotter = mocker.Mock(return_value=[])
    result = resource_requirements.write_resource_requirements(
        {
            BASELINE_PATH: BASELINE,
            CANDIDATE_PATH: CANDIDATE,
            "baseline_label": "reference",
            "candidate_label": "optimized",
            "select": [],
        },
        tmp_test_directory,
        plotter,
    )
    assert {row[PRODUCTION_LABEL] for row in plotter.call_args.args[0]} == {
        "reference",
        "optimized",
    }
    assert {row["comparison_role"] for row in plotter.call_args.args[0]} == {
        BASELINE,
        CANDIDATE,
    }
    assert result["table_file"].is_file()


def _make_discoverable(manifest):
    """Add the job-metadata fields required by manifest discovery and write the manifest."""
    manifest.data.update(
        schema_name="simulate_prod_job_metadata",
        schema_version=INITIAL_SCHEMA_VERSION,
        product_type="simulate_prod_job",
        status="complete",
        catalog_metadata={},
        statistics={SIMULATED_EVENTS: 10, TRIGGERED_EVENTS: 5},
    )
    manifest.data[CONFIGURATION].update(
        simulation_software="corsika_sim_telarray",
        view_cone_min={VALUE: 0.0, "unit": "deg"},
        view_cone_max={VALUE: 0.0, "unit": "deg"},
        core_scatter_max={VALUE: 100.0, "unit": "m"},
        cores_per_shower=1,
        model_parameter_overrides={},
        atmosphere={},
    )
    manifest.path.write_text(json.dumps(manifest.data), encoding=ENCODING)


def test_collect_resource_requirements_reads_records_from_accounting_command(tmp_test_directory):
    manifest = _manifest(tmp_test_directory)
    sim_telarray_outputs = [
        manifest.directory / files[0] for files in manifest.data[FILES].values()
    ]
    corsika_output = manifest.directory / CORSIKA / RUN_DIRECTORY / "showers.corsika.zst"
    manifest.data[FILES][CORSIKA] = [str(corsika_output.relative_to(manifest.directory))]
    _make_discoverable(manifest)
    for role, suffix, outputs, size in (
        (CORSIKA, CORSIKA, [corsika_output], 7),
        (SIM_TELARRAY, SIMTEL, sim_telarray_outputs, 20),
    ):
        record = manifest.directory / role / RUN_DIRECTORY / _record_name(suffix)
        child = (
            "from pathlib import Path; "
            f"[Path(path).write_bytes(b'x' * {size}) for path in {[str(p) for p in outputs]!r}]"
        )
        command = process_accounting.build_accounting_command(
            [sys.executable, "-c", child], record, role, 1, model_version=MODEL_VERSION
        )
        subprocess.run(command, check=True, timeout=60)

    rows, diagnostics = resource_requirements.collect_resource_requirements(
        manifest.directory.parent
    )

    assert diagnostics == []
    rows_by_role = {row["role"]: row for row in rows}
    assert rows_by_role[CORSIKA][CORSIKA_OUTPUT_BYTES] == 7
    assert rows_by_role[SIM_TELARRAY][SIM_TELARRAY_STORAGE_BYTES] == 60
    assert rows_by_role[SIM_TELARRAY][STORAGE_BYTES_PER_TRIGGERED_EVENT] == (pytest.approx(12.0))
    for row in rows:
        assert row[START_TIME] != FIXTURE_START_TIME
        assert row["simtools_version"] != UNKNOWN
        assert row[LOGICAL_CPU_COUNT] >= 1
        assert row[HOSTNAME] != UNKNOWN


def test_write_resource_requirements_reports_skipped_record_with_matching_coverage(
    mocker, tmp_test_directory
):
    baseline = _manifest(tmp_test_directory / BASELINE)
    candidate = _manifest(tmp_test_directory / CANDIDATE)
    incomplete = _manifest(tmp_test_directory / "incomplete")
    next((incomplete.directory / CORSIKA).rglob(RESOURCE_RECORD_PATTERN)).unlink()
    mocker.patch.object(
        resource_requirements,
        DISCOVER_MANIFESTS,
        side_effect=[[baseline], [candidate, incomplete]],
    )

    result = resource_requirements.write_resource_requirements(
        {BASELINE_PATH: BASELINE, CANDIDATE_PATH: CANDIDATE},
        tmp_test_directory,
        mocker.Mock(return_value=[]),
    )

    assert [item["role"] for item in result["diagnostics"]] == [CORSIKA]
    report = result[REPORT_FILE].read_text(encoding=ENCODING)
    assert "Skipped process records: 1" in report
    assert "## Skipped records" in report


def test_write_resource_requirements_coverage_error_names_cause(mocker, tmp_test_directory):
    baseline = _manifest(tmp_test_directory / BASELINE)
    candidate = _manifest(tmp_test_directory / CANDIDATE)
    next((candidate.directory / CORSIKA).rglob(RESOURCE_RECORD_PATTERN)).unlink()
    mocker.patch.object(
        resource_requirements, DISCOVER_MANIFESTS, side_effect=[[baseline], [candidate]]
    )

    with pytest.raises(ValueError, match=COVERAGE_DIFFERS) as error:
        resource_requirements.write_resource_requirements(
            {BASELINE_PATH: BASELINE, CANDIDATE_PATH: CANDIDATE},
            tmp_test_directory,
            mocker.Mock(return_value=[]),
        )

    message = str(error.value)
    assert "baseline only: {role=corsika, primary=gamma" in message
    assert "candidate only" not in message
    assert "skipped records: candidate job-000001 corsika: Expected one corsika" in message


def test_resource_coverage_rejects_missing_configuration_or_role():
    common = {"energy_min_gev": 30, "energy_max_gev": 30, "zenith_angle_deg": 20}
    rows = [
        common | {"comparison_role": BASELINE, "role": CORSIKA},
        common | {"comparison_role": CANDIDATE, "role": CORSIKA},
    ]
    resource_requirements._validate_resource_coverage(rows)
    with pytest.raises(ValueError, match=COVERAGE_DIFFERS):
        resource_requirements._validate_resource_coverage(
            [*rows, common | {"comparison_role": BASELINE, "role": SIM_TELARRAY}]
        )


def test_configuration_resource_summary_preserves_costs_and_ratios():
    common = {
        "role": CORSIKA,
        PRODUCTION_LABEL: PRODUCTION,
        "energy_midpoint_gev": 30,
        "zenith_angle_deg": 20,
    }
    rows = [
        common | {"comparison_role": BASELINE, "cpu_time_seconds_per_event": 2},
        common | {"comparison_role": CANDIDATE, "cpu_time_seconds_per_event": 3},
    ]
    result = resource_requirements._configuration_resource_summary(rows)[0]
    assert result["ratios"]["cpu_time_seconds_per_event"] == pytest.approx(1.5)
    assert result["productions"][BASELINE]["count"] == 1
