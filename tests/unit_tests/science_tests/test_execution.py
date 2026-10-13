import json
import os
import time
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


def test_reuse_derived_product_requires_complete_products(mocker):
    definition = {"derived_output": "__OUTPUT__"}
    replacements = {"__OUTPUT__": "/existing/output"}

    mocker.patch.object(science_test_execution, "_evaluate_products")
    assert science_test_execution._reuse_derived_product(definition, replacements, {})
    assert not science_test_execution._reuse_derived_product(
        definition, replacements, {"science_overwrite": True}
    )

    mocker.patch.object(
        science_test_execution, "_evaluate_products", side_effect=ValueError("incomplete")
    )
    assert not science_test_execution._reuse_derived_product(definition, replacements, {})


def test_science_comparison_receives_release_parameters():
    workflow = {
        "applications": [
            {
                "application": "simtools-compare-science-results",
                "configuration": {"output_path": "/tmp/report"},
            }
        ]
    }
    definition = {
        "comparison_parameters": {"compare_by": ["case.azimuth_angle"]},
        "baseline": "v0.37.0",
        "release_label": "v0.38.0",
        "expected_changes": ["case.azimuth_angle"],
    }

    science_test_execution._configure_comparison_applications(workflow, definition)

    configuration = workflow["applications"][0]["configuration"]
    assert configuration["compare_by"] == ["case.azimuth_angle"]
    assert configuration["baseline_label"] == "v0.37.0"
    assert configuration["candidate_label"] == "v0.38.0"


def test_simtools_version_uses_runtime_identity_when_log_has_none():
    assert (
        science_test_execution._simtools_version_from_log(
            None, {"options": ["--env", "SIMTOOLS_VERSION=0.38.1"]}
        )
        == "0.38.1"
    )


def test_camera_collection_copies_native_plots_into_report_work(tmp_test_directory, mocker):
    root = Path(tmp_test_directory)
    output = root / "candidate/north/camera_efficiency"
    (output / "plots").mkdir(parents=True)
    (output / "plots/native.png").write_bytes(b"plot")
    work = root / "work/science-tests/north/camera_efficiency.candidate"
    mocker.patch.object(
        science_test_execution,
        "_load_yaml",
        return_value={
            "applications": [
                {
                    "application": "simtools-validate-camera-efficiency",
                    "configuration": {"output_path": str(output)},
                }
            ]
        },
    )
    mocker.patch.object(
        science_test_execution,
        "collect_legacy_results",
        return_value={"artifacts": {"native_plots": ["plots/native.png"]}},
    )

    science_test_execution._collect_camera_results(
        {"result_type": "camera_efficiency"},
        {"config_file": "workflow.yml", "log_file": str(work / "execution.log")},
    )

    assert (work / "native_artifacts/native.png").read_bytes() == b"plot"


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


def test_role_specific_runtime_environment(tmp_test_directory):
    template = Path(tmp_test_directory)
    _write_yaml(
        template / "run_time.yml",
        {
            "runtime_environments": {
                "baseline": {"container_engine": "apptainer", "image": "base.sif"},
                "candidate": {"container_engine": "apptainer", "image": "candidate.sif"},
                "analysis": {"container_engine": "apptainer", "image": "analysis.sif"},
            }
        },
    )

    runtime = science_test_definitions._shared_runtime_environment(template, {}, role="candidate")

    assert runtime["image"] == "candidate.sif"


def test_role_runtime_template_keeps_default_for_existing_tests(tmp_test_directory):
    template = Path(tmp_test_directory)
    _write_yaml(
        template / "run_time.yml",
        {
            "runtime_environment": {
                "container_engine": "apptainer",
                "image": "default.sif",
            },
            "runtime_environments": {
                "candidate": {"container_engine": "apptainer", "image": "candidate.sif"}
            },
        },
    )

    runtime = science_test_definitions._shared_runtime_environment(template, {})

    assert runtime["image"] == "default.sif"


def test_role_specific_runtime_requires_known_role(tmp_test_directory):
    template = Path(tmp_test_directory)
    _write_yaml(
        template / "run_time.yml",
        {"runtime_environments": {"analysis": {"image": "analysis.sif"}}},
    )

    with pytest.raises(ValueError, match="No runtime environment"):
        science_test_definitions._shared_runtime_environment(template, {}, role="baseline")


def test_compute_summary_records_timing_and_unavailable_memory(tmp_test_directory, monkeypatch):
    path = Path(tmp_test_directory) / "compute-summary.json"
    monkeypatch.setattr(science_test_execution, "_peak_memory_bytes", lambda: None)
    before = os.times()

    science_test_execution._write_compute_summary(
        path,
        time.monotonic() - 0.01,
        before,
        {"log_file": "execution.log"},
        {"status": "pass"},
    )

    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["wall_seconds"] > 0
    assert summary["peak_memory_bytes"] is None
    assert summary["memory_accounting"] == "unavailable"


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
    assert previous["execution_provenance"] == "work/science-tests/north/derive"
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
