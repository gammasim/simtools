import json
import os
from pathlib import Path

import jsonschema
import pytest
import yaml

from simtools.science_tests import campaign as campaign_runner
from simtools.science_tests import definitions as science_test_definitions
from simtools.science_tests import execution as science_test_execution


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_failed_dependency_blocks_comparison(campaign, monkeypatch):
    calls = []

    def fail(args, replacements):
        calls.append(args)
        raise RuntimeError("simulator failed")

    monkeypatch.setattr(science_test_execution, "run_applications", fail)
    with pytest.raises(RuntimeError, match="Science tests failed"):
        campaign_runner.run_release(**campaign, sites=["north"], tests=["derive", "compare"])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    assert len(calls) == 1
    assert {r["test"]: r["status"] for r in summary["results"] if r["site"] == "north"} == {
        "derive": "incomplete",
        "compare": "blocked",
    }


def test_missing_product_is_recorded(campaign, monkeypatch):
    monkeypatch.setattr(science_test_execution, "run_applications", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Missing or empty required product"):
        campaign_runner.run_release(**campaign, sites=["north"])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    failed = next(r for r in summary["results"] if r["id"] == "compare.north")
    assert "Missing or empty required product" in failed["error"]
    assert failed["status"] == "incomplete"


def test_acceptance_failure_has_completed_execution(campaign, successful_workflow):
    _write_yaml(
        campaign["template_dir"] / "acceptance/thresholds.yml",
        {
            "rules": {
                "comparison": {
                    "mode": "blocking",
                    "metrics": [
                        {
                            "file": "__SCIENCE_REPORT_PATH__/metrics.json",
                            "key": "count",
                            "maximum": 2,
                        }
                    ],
                },
            }
        },
    )
    with pytest.raises(RuntimeError, match="Science tests failed"):
        campaign_runner.run_release(**campaign, sites=["north"])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    failed = next(r for r in summary["results"] if r["id"] == "compare.north")
    assert failed["status"] == "fail"
    assert failed["execution"] == "completed"


def test_existing_submission_prevents_resubmission(tmp_test_directory):
    root = Path(tmp_test_directory)
    (root / "submission.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Production submission blocked") as error:
        science_test_definitions._check_inputs(
            {"produces_production": True}, {"__SCIENCE_CANDIDATE_SITE_PATH__": str(root)}
        )

    assert str(root / "submission.json") in str(error.value)
    assert "--allow_production does not override it" in str(error.value)


def test_reuse_rejects_unrecorded_partial_output(tmp_test_directory):
    output = Path(tmp_test_directory) / "derived"
    output.mkdir()
    (output / "partial.hdf5").write_bytes(b"not HDF5")
    definition = {"derived_output": str(output)}
    args = {
        "science_workflow": {"applications": []},
        "provenance_path": str(Path(tmp_test_directory) / "workflow.yml"),
    }
    with pytest.raises(ValueError, match="--overwrite"):
        science_test_execution._reuse_derived_product(definition, {}, args)
    assert not science_test_execution._reuse_derived_product(
        definition, {}, args | {"science_overwrite": True}
    )


@pytest.mark.parametrize(("writes_log", "raises"), [(False, False), (True, False), (True, True)])
def test_execution_log_is_recorded_only_for_current_execution(
    tmp_test_directory, monkeypatch, writes_log, raises
):
    work = Path(tmp_test_directory) / "work"
    work.mkdir()
    log_path = work / science_test_definitions._EXECUTION_LOG_FILE_NAME
    log_path.write_text("stale log", encoding="utf-8")

    def execute(_definition, args, _replacements):
        if writes_log:
            log_path.write_text("current log", encoding="utf-8")
        if raises:
            raise RuntimeError("execution failed")
        return {"status": "pass", "execution": "completed"}

    monkeypatch.setattr(science_test_execution, "_execute_test_workflow", execute)
    result = {}
    if raises:
        with pytest.raises(RuntimeError, match="execution failed"):
            science_test_execution._run_or_reuse_test(
                result, {}, {}, {}, work, {"applications": []}
            )
    else:
        science_test_execution._run_or_reuse_test(result, {}, {}, {}, work, {"applications": []})

    assert result.get("execution_log") == (str(log_path) if writes_log else None)


def test_test_output_paths_deduplicate_and_fall_back_to_candidate():
    candidate = "/candidate/site"
    replacements = {"__SCIENCE_CANDIDATE_SITE_PATH__": candidate}

    assert science_test_execution._test_output_paths(
        [
            {"configuration": {"output_path": "/output/one"}},
            {"configuration": {"output_path": "/output/one"}},
            {"configuration": {"output_path": "/output/two"}},
        ],
        replacements,
    ) == ["/output/one", "/output/two"]
    assert science_test_execution._test_output_paths([], replacements) == [candidate]


@pytest.mark.parametrize(
    "collection",
    [
        {"write_inventory": True},
        [{"write_inventory": True}, {"write_inventory": True}],
        None,
    ],
)
def test_science_test_collection_does_not_write_inventory(collection):
    resolved = {"collection": collection}

    science_test_execution._disable_collection_inventory(resolved)

    if isinstance(collection, dict):
        assert collection["write_inventory"] is False
    elif isinstance(collection, list):
        assert all(entry["write_inventory"] is False for entry in collection)


@pytest.mark.parametrize("present", [False, True])
def test_shared_runtime_environment(tmp_test_directory, present):
    template = Path(tmp_test_directory)
    if present:
        _write_yaml(
            template / "run_time.yml",
            {
                "runtime_environment": {
                    "container_engine": "apptainer",
                    "image": "__SCIENCE_CONTAINER_IMAGE_PATH__",
                    "environment_file": "__CONFIG_DIRECTORY__/profiles/htcondor.env",
                },
            },
        )
    runtime = science_test_definitions._shared_runtime_environment(
        template,
        {
            "__SCIENCE_CONTAINER_IMAGE_PATH__": "/images/simtools.sif",
            "__CONFIG_DIRECTORY__": "/other/workflow",
        },
    )
    if present:
        assert runtime == {
            "container_engine": "apptainer",
            "image": "/images/simtools.sif",
            "environment_file": str(template / "profiles/htcondor.env"),
        }
    else:
        assert runtime is None


def test_shared_runtime_environment_invalid(tmp_test_directory):
    template = Path(tmp_test_directory)
    _write_yaml(template / "run_time.yml", {"runtime_environment": {"image": ""}})
    with pytest.raises(jsonschema.ValidationError):
        science_test_definitions._shared_runtime_environment(template, {})


@pytest.mark.parametrize(
    ("produces_production", "inline"), [(False, False), (True, False), (False, True)]
)
def test_run_test_uses_shared_runtime(campaign, monkeypatch, produces_production, inline):
    template = campaign["template_dir"]
    shared = {"container_engine": "apptainer", "image": "__SCIENCE_CONTAINER_IMAGE_PATH__"}
    _write_yaml(template / "run_time.yml", {"runtime_environment": shared})
    if inline:
        workflow = science_test_definitions._load_yaml(template / "workflow.yml")
        workflow["runtime_environment"] = {"container_engine": "apptainer", "image": "inline.sif"}
        _write_yaml(template / "workflow.yml", workflow)
    monkeypatch.setattr(science_test_execution, "_preview", lambda *_: {})
    captured = []
    prepare = science_test_execution.prepare_workflow

    def capture(args, replacements):
        result = prepare(args, replacements=replacements)
        captured.append(result[1])
        return result

    monkeypatch.setattr(science_test_execution, "prepare_workflow", capture)
    science_test_execution._run_test(
        campaign["release_dir"],
        template,
        {
            "__SCIENCE_CANDIDATE_PATH__": str(template / "candidate"),
            "__SCIENCE_CONTAINER_IMAGE_PATH__": "/images/shared.sif",
        },
        "north",
        {"array_layout_name": "CTAO-North-Alpha"},
        "example",
        {"workflow": "workflow.yml", "produces_production": produces_production},
        dry_run=True,
    )
    expected = None
    if inline:
        expected = {"container_engine": "apptainer", "image": "inline.sif"}
    elif not produces_production:
        expected = {"container_engine": "apptainer", "image": "/images/shared.sif"}
    assert captured == [expected]


def test_named_artifacts_on_rerun(campaign, successful_workflow, monkeypatch):
    args = {**campaign, "sites": ["north"], "tests": ["derive"]}
    initial = campaign_runner.run_release(**args)
    report = campaign["release_dir"] / "reports/north/derive"
    previous = next(result for result in initial["results"] if result["id"] == "derive.north")
    assert previous["execution_provenance"].endswith("/candidate/work/science-tests/north/derive")
    context = science_test_definitions._load_yaml(campaign["context_file"])
    work = Path(context["__SCIENCE_CANDIDATE_PATH__"]) / previous["execution_provenance"]
    stale = [report / "stale.txt", work / "stale.txt"]
    for path in stale:
        path.touch()
    campaign_runner.run_release(**args, dry_run=True, overwrite=True)
    assert all(path.exists() for path in stale)
    current = next(
        result
        for result in campaign_runner.run_release(**args)["results"]
        if result["id"] == "derive.north"
    )
    assert current["run_id"] != previous["run_id"]
    assert (work / "workflow.yml").is_file()
    assert not (work / "provenance").exists()
    assert not (work / "output").exists()
    assert not any(path.exists() for path in stale)
    assert not any((report / file_name).exists() for file_name in ("result.json", "summary.md"))
    monkeypatch.setattr(science_test_execution, "run_applications", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Science tests failed"):
        campaign_runner.run_release(**args, overwrite=True)
    failed = next(
        result
        for result in json.loads(
            (campaign["release_dir"] / "reports/release-summary.json").read_text()
        )["results"]
        if result["id"] == "derive.north"
    )
    assert failed["status"] == "incomplete"
    assert failed["report"] is None
    assert failed["previous_report"] == "reports/north/derive"
    assert (report / "metrics.json").exists()


@pytest.mark.parametrize(
    ("state", "finished"),
    [("failed", True), ("failed", False), ("completed", True), ("submitted", False)],
)
def test_production_retry_requires_finished_failed_jobs(mocker, state, finished):
    mocker.patch.object(
        science_test_execution,
        "load_submission",
        return_value=mocker.Mock(metadata={"state": state}, backend="htcondor"),
    )
    backend = mocker.patch.object(science_test_execution, "get_backend").return_value
    backend.is_finished.return_value = finished
    if state == "failed" and finished:
        science_test_execution._validate_production_retry(Path("submission.json"))
    else:
        with pytest.raises(ValueError, match="Cannot overwrite"):
            science_test_execution._validate_production_retry(Path("submission.json"))


@pytest.mark.parametrize("overwrite", [False, True])
def test_production_retry_archives_artifacts(tmp_test_directory, mocker, overwrite):
    root = Path(tmp_test_directory)
    candidate = root / "candidate/north"
    work = root / "candidate/work/science-tests/north/production.gamma"
    report = root / "release/reports/north/production.gamma"
    for source in (candidate, work, report):
        source.mkdir(parents=True)
        (source / "keep.txt").write_text("previous run")
    (candidate / "submission.json").write_text("{}")
    validate = mocker.patch.object(science_test_execution, "_validate_production_retry")
    science_test_execution._prepare_test_retry(
        {"produces_production": True},
        {"__SCIENCE_CANDIDATE_SITE_PATH__": str(candidate)},
        {"science_overwrite": overwrite},
        "retry-id",
        work,
        report,
    )
    assert candidate.exists() is not overwrite
    assert work.exists() is not overwrite
    assert report.exists() is not overwrite
    assert validate.call_count == int(overwrite)
    if overwrite:
        assert len(list(root.rglob("keep.txt"))) == 3
        assert all("archive" in path.parts for path in root.rglob("keep.txt"))


@pytest.mark.parametrize("pending", [False, True])
def test_collection_waits_for_outputs_without_submitting(tmp_test_directory, mocker, pending):
    root = Path(tmp_test_directory)
    product = root / "event.dat"
    product.write_text("event")
    manifest = root / "submission.json"
    payload = {
        "backend": "htcondor",
        "work_dir": str(root),
        "job_ids": ["job"],
        "metadata": {"state": "submitted", "expected_outputs": {"job": [str(product)]}},
    }
    manifest.write_text(json.dumps(payload))

    def collect(submission):
        if pending:
            return None
        payload["metadata"]["state"] = "completed"
        manifest.write_text(json.dumps(payload))
        return []

    mocker.patch.object(science_test_execution, "collect_submission", side_effect=collect)
    run = mocker.patch.object(science_test_execution, "run_applications")
    result = science_test_execution._execute_test_workflow(
        {"collect_production": True, "products": [str(product)]},
        {},
        {"__SCIENCE_CANDIDATE_SITE_PATH__": str(root)},
    )
    assert result["status"] == ("pending" if pending else "pass")
    run.assert_not_called()


def test_submission_does_not_validate_completed_production(mocker):
    run = mocker.patch.object(science_test_execution, "run_applications")
    mocker.patch.object(
        science_test_execution, "_evaluate_products", return_value={"status": "pass"}
    )
    result = science_test_execution._execute_test_workflow({"produces_production": True}, {}, {})
    assert result["status"] == result["execution"] == "submitted"
    run.assert_called_once()


@pytest.mark.parametrize(
    ("source", "inactive_environment"),
    [
        ({"simulation_models_git_path": "/models.git"}, "SIMTOOLS_SIMULATION_MODELS_PATH"),
        ({"simulation_models_path": "/models"}, "SIMTOOLS_SIMULATION_MODELS_GIT_PATH"),
    ],
)
def test_execution_hides_inactive_model_source_environment(
    source, inactive_environment, monkeypatch, mocker
):
    monkeypatch.setenv(inactive_environment, "/inherited")
    observed = []
    mocker.patch.object(
        science_test_execution,
        "run_applications",
        side_effect=lambda *_args, **_kwargs: observed.append(os.environ[inactive_environment]),
    )
    mocker.patch.object(
        science_test_execution, "_evaluate_products", return_value={"status": "pass"}
    )
    science_test_execution._execute_test_workflow({}, source, {})
    assert observed == [""]
    assert os.environ[inactive_environment] == "/inherited"


def test_execution_restores_absent_model_source_environment_after_failure(monkeypatch, mocker):
    inactive = "SIMTOOLS_SIMULATION_MODELS_PATH"
    monkeypatch.delenv(inactive, raising=False)

    def fail(*_args, **_kwargs):
        assert os.environ[inactive] == ""
        raise RuntimeError("application failed")

    mocker.patch.object(science_test_execution, "run_applications", side_effect=fail)
    with pytest.raises(RuntimeError, match="application failed"):
        science_test_execution._execute_test_workflow(
            {}, {"simulation_models_git_path": "/models.git"}, {}
        )
    assert inactive not in os.environ


def test_pending_collection_defers_dependent_tests(mocker):
    collect = {"id": "collect.north", "site": "north", "test": "collect"}
    compare = {"id": "compare.north", "site": "north", "test": "compare"}
    execute = mocker.patch.object(
        science_test_execution, "_run_test", return_value={**collect, "status": "pending"}
    )
    mocker.patch.object(science_test_execution, "_log_test_execution")
    results = science_test_execution._execute_selection(
        [(({},), collect), (({"depends_on": ["collect"]},), compare)],
        False,
        "run",
        {},
        {"collect": {}, "compare": {"depends_on": ["collect"]}},
    )
    assert [result["status"] for result in results] == ["pending", "pending"]
    execute.assert_called_once()


def test_derived_group_checks_readability_runs_settings_and_freshness(tmp_test_directory):
    from types import SimpleNamespace

    from astropy.table import Table

    from simtools.io import table_handler
    from simtools.production_configuration.production_file_selection import ProductionManifest
    from simtools.production_configuration.trigger_histograms import (
        TRIGGER_HISTOGRAM_BINS_TABLE,
        TRIGGER_HISTOGRAM_METADATA_TABLE,
    )

    root = Path(tmp_test_directory)
    source = root / "events.hdf5"
    source.write_bytes(b"events")
    output = root / "product.hdf5"
    tables = [
        Table({"count": [1]}, meta={"EXTNAME": name})
        for name in (TRIGGER_HISTOGRAM_METADATA_TABLE, TRIGGER_HISTOGRAM_BINS_TABLE)
    ]
    table_handler.write_tables(tables, output, file_type="HDF5")
    group = SimpleNamespace(run_numbers=[1], file_paths=[source])
    product = ProductionManifest(
        root / "product.yml",
        {
            "input_run_numbers": [1],
            "input_files": {"reduced_event_data": ["events.hdf5"]},
            "files": {"trigger_histograms": ["product.hdf5"]},
            "histogram_settings": {"angular_distance_bin_width": {"value": 0.5, "unit": "deg"}},
        },
    )
    config = {"angular_distance_bin_width": "0.5 deg"}
    science_test_execution._validate_derived_group(group, product, config)
    with pytest.raises(ValueError, match="setting changed"):
        science_test_execution._validate_derived_group(
            group, product, {"angular_distance_bin_width": "1 deg"}
        )
    product.data["input_run_numbers"] = [1, 2]
    with pytest.raises(ValueError, match="run coverage"):
        science_test_execution._validate_derived_group(group, product, config)
    product.data["input_run_numbers"] = [1]
    output.write_bytes(b"not HDF5")
    with pytest.raises(ValueError, match="Unreadable trigger histogram"):
        science_test_execution._validate_derived_group(group, product, config)


def test_derived_coverage_rejects_a_partial_configuration_set(mocker, tmp_test_directory):
    from types import SimpleNamespace

    mocker.patch.object(
        science_test_execution,
        "select_file_groups",
        return_value={
            "groups": [
                SimpleNamespace(configuration={"energy_min": 30}),
                SimpleNamespace(configuration={"energy_min": 3000}),
            ],
        },
    )
    mocker.patch.object(
        science_test_execution,
        "discover_product_manifests",
        return_value=[
            SimpleNamespace(data={"configuration": {"energy_min": 30}}),
        ],
    )
    with pytest.raises(ValueError, match="coverage differs"):
        science_test_execution._validate_derived_products(
            {
                "production_path": str(tmp_test_directory),
                "output_path": str(tmp_test_directory),
            }
        )


def test_materialize_effective_workflow_serializes_model_paths(tmp_test_directory):
    work = Path(tmp_test_directory) / "work"
    models = Path(tmp_test_directory) / "models"
    resolved = {
        "applications": [
            {
                "application": "simtools-write-trigger-histograms",
                "configuration": {"simulation_models_path": models},
            }
        ]
    }
    path = science_test_execution._materialize_effective_workflow(work, resolved)
    recorded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert recorded["applications"][0]["configuration"]["simulation_models_path"] == str(models)
