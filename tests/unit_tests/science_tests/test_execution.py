import json
import os
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import yaml

from simtools.science_tests import campaign as campaign_runner
from simtools.science_tests import definitions as science_test_definitions
from simtools.science_tests import execution as science_test_execution

_COMPLETE_STATUS = "complete"
_ENCODING = "utf-8"
_VALIDATOR_FAILURE = "Validator failed"
_RESULT_TYPE_KEY = "result_type"
_CAMERA_RESULT_TYPE = "camera_efficiency"
_CAMERA_VALIDATOR_APPLICATION = "simtools-validate-telescope-throughput"
_TELESCOPE_KEY = "telescope"
_SECOND_TELESCOPE = "LSTN-02"

_SCIENCE_CANDIDATE_SITE_PATH = "__SCIENCE_CANDIDATE_SITE_PATH__"

_PRODUCES_PRODUCTION = "produces_production"

_DEPENDS_ON = "depends_on"
_HTCONDOR = "htcondor"
_WORKFLOW_YML = "workflow.yml"
_CAMERA_METRICS = (
    "reflectivity",
    "cam_eff",
    "tel_eff",
    "tot_sens",
    "nsb_rate",
    "nsb_ref",
    "muon_frac",
)
_RUNTIME_IMAGE = "runtime.sif"
_OLD_SIMTOOLS_VERSION = "0.37.0"
_CURRENT_SIMTOOLS_VERSION = "0.38.1"
_MISSING_DEPENDENCY_MANIFEST = "manifest missing"
_CAMERA_MANIFEST_NAME = "camera_efficiency_manifest.json"
_APPTAINER = "apptainer"
_SIMTOOLS_PACKAGE = "simtools"

_COLLECT = "collect"
_COMPARE = "compare"
_COMPARE_NORTH = "compare.north"
_DERIVE_NORTH = "derive.north"
_FAILED = "failed"
_METADATA = "metadata"
_PROVENANCE_PATH = "provenance_path"
_STATE = "state"

_OUTPUT_ONE = "/output/one"
_OUTPUT_TWO = "/output/two"
_MISSING_OR_EMPTY_REQUIRED_PRODUCT = "Missing or empty required product"
_EVALUATE_PRODUCTS = "_evaluate_products"
_COMPLETED = "completed"
_DERIVE = "derive"
_EXECUTION = "execution"
_NORTH = "north"
_OUTPUT_PATH_KEY = "output_path"
_PENDING = "pending"
_RESULTS = "results"
_SCIENCE_WORKFLOW = "science_workflow"
_SIMULATION_MODELS_PATH = "simulation_models_path"
_SUBMITTED = "submitted"

_MISSING_DERIVED_CONFIGURATION = "Missing derived configuration"
_EXECUTE_TEST_WORKFLOW = "_execute_test_workflow"
_VALIDATE_DERIVED_WORKFLOW = "_validate_derived_workflow"
_APPLICATION_KEY = "application"
_APPLICATIONS_KEY = "applications"
_CONFIGURATION_KEY = "configuration"
_DERIVED = "derived"
_DERIVED_OUTPUT = "derived_output"
_ERROR = "error"
_INCOMPLETE = "incomplete"
_REPORT = "report"
_STATUS_KEY = "status"


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding=_ENCODING)


def test_new_derivations_must_validate_products_before_passing(tmp_test_directory, mocker):
    root = Path(tmp_test_directory)
    result = {_STATUS_KEY: "pass"}
    mocker.patch.object(science_test_execution, _EXECUTE_TEST_WORKFLOW, return_value=result)
    validate = mocker.patch.object(
        science_test_execution,
        _VALIDATE_DERIVED_WORKFLOW,
        side_effect=ValueError(_MISSING_DERIVED_CONFIGURATION),
    )
    applications = [{_APPLICATION_KEY: "other-derivation", _CONFIGURATION_KEY: {}}]
    science_test_execution._execute_and_record_test(
        result,
        {_DERIVED_OUTPUT: str(root / _DERIVED)},
        {},
        {},
        "run",
        root / "work",
        root / _REPORT,
        {_APPLICATIONS_KEY: applications},
    )
    assert result[_STATUS_KEY] == _INCOMPLETE
    assert _MISSING_DERIVED_CONFIGURATION in result[_ERROR]
    validate.assert_called_once_with(applications)


def test_failed_dependency_blocks_comparison(campaign, monkeypatch):
    calls = []

    def fail(args, replacements):
        calls.append(args)
        raise RuntimeError("simulator failed")

    monkeypatch.setattr(science_test_execution, "run_applications", fail)
    with pytest.raises(RuntimeError, match="Science tests failed"):
        campaign_runner.run_release(**campaign, sites=[_NORTH], tests=[_DERIVE, _COMPARE])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    assert len(calls) == 1
    assert {r["test"]: r[_STATUS_KEY] for r in summary[_RESULTS] if r["site"] == _NORTH} == {
        _DERIVE: _INCOMPLETE,
        _COMPARE: "blocked",
    }


def test_missing_product_is_recorded(campaign, monkeypatch):
    monkeypatch.setattr(science_test_execution, "run_applications", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match=_MISSING_OR_EMPTY_REQUIRED_PRODUCT):
        campaign_runner.run_release(**campaign, sites=[_NORTH])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    failed = next(r for r in summary[_RESULTS] if r["id"] == _COMPARE_NORTH)
    assert _MISSING_OR_EMPTY_REQUIRED_PRODUCT in failed[_ERROR]
    assert failed[_STATUS_KEY] == _INCOMPLETE


def test_acceptance_limit_is_a_warning_after_completed_execution(campaign, successful_workflow):
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
    campaign_runner.run_release(**campaign, sites=[_NORTH])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    warned = next(r for r in summary[_RESULTS] if r["id"] == _COMPARE_NORTH)
    assert warned[_STATUS_KEY] == "warn"
    assert warned[_EXECUTION] == _COMPLETED
    assert "Acceptance warning" in warned["reason"]


def test_existing_submission_prevents_resubmission(tmp_test_directory):
    root = Path(tmp_test_directory)
    (root / "submission.json").write_text("{}", encoding=_ENCODING)
    with pytest.raises(ValueError, match="Production submission blocked") as error:
        science_test_definitions._check_inputs(
            {_PRODUCES_PRODUCTION: True}, {_SCIENCE_CANDIDATE_SITE_PATH: str(root)}
        )

    assert str(root / "submission.json") in str(error.value)
    assert "--allow_production does not override it" in str(error.value)


def test_reuse_rejects_unrecorded_partial_output(tmp_test_directory):
    output = Path(tmp_test_directory) / _DERIVED
    output.mkdir()
    (output / "partial.hdf5").write_bytes(b"not HDF5")
    definition = {_DERIVED_OUTPUT: str(output)}
    args = {
        _SCIENCE_WORKFLOW: {_APPLICATIONS_KEY: []},
        _PROVENANCE_PATH: str(Path(tmp_test_directory) / _WORKFLOW_YML),
    }
    with pytest.raises(ValueError, match="--overwrite"):
        science_test_execution._reuse_derived_product(definition, {}, args)
    assert not science_test_execution._reuse_derived_product(
        definition, {}, args | {"science_overwrite": True}
    )


def test_reused_products_keep_acceptance_breaches_as_warnings(tmp_test_directory, mocker):
    output = Path(tmp_test_directory) / _DERIVED
    output.mkdir()
    (output / "trigger_histograms.hdf5").touch()
    provenance = Path(tmp_test_directory) / _WORKFLOW_YML
    provenance.touch()
    workflow = {_APPLICATIONS_KEY: [{_CONFIGURATION_KEY: {"site": "North"}}]}
    args = {_SCIENCE_WORKFLOW: workflow, _PROVENANCE_PATH: str(provenance)}
    mocker.patch.object(science_test_execution, "_load_yaml", return_value=workflow)
    mocker.patch.object(science_test_execution, _VALIDATE_DERIVED_WORKFLOW)
    mocker.patch.object(
        science_test_execution,
        _EVALUATE_PRODUCTS,
        side_effect=science_test_definitions._AcceptanceError("threshold exceeded"),
    )

    with pytest.raises(science_test_definitions._AcceptanceError, match="threshold exceeded"):
        science_test_execution._reuse_derived_product({_DERIVED_OUTPUT: str(output)}, {}, args)


@pytest.mark.parametrize(("writes_log", "raises"), [(False, False), (True, False), (True, True)])
def test_execution_log_is_recorded_only_for_current_execution(
    tmp_test_directory, monkeypatch, writes_log, raises
):
    work = Path(tmp_test_directory) / "work"
    work.mkdir()
    log_path = work / science_test_definitions._EXECUTION_LOG_FILE_NAME
    log_path.write_text("stale log", encoding=_ENCODING)

    def execute(_definition, args, _replacements):
        if writes_log:
            log_path.write_text("current log", encoding=_ENCODING)
        if raises:
            raise RuntimeError("execution failed")
        return {_STATUS_KEY: "pass", _EXECUTION: _COMPLETED}

    monkeypatch.setattr(science_test_execution, _EXECUTE_TEST_WORKFLOW, execute)
    result = {}
    if raises:
        with pytest.raises(RuntimeError, match="execution failed"):
            science_test_execution._run_or_reuse_test(
                result, {}, {}, {}, work, {_APPLICATIONS_KEY: []}
            )
    else:
        science_test_execution._run_or_reuse_test(result, {}, {}, {}, work, {_APPLICATIONS_KEY: []})

    assert result.get("execution_log") == (str(log_path) if writes_log else None)


def test_test_output_paths_deduplicate_and_fall_back_to_candidate():
    candidate = "/candidate/site"
    replacements = {_SCIENCE_CANDIDATE_SITE_PATH: candidate}

    assert science_test_execution._test_output_paths(
        [
            {_CONFIGURATION_KEY: {_OUTPUT_PATH_KEY: _OUTPUT_ONE}},
            {_CONFIGURATION_KEY: {_OUTPUT_PATH_KEY: _OUTPUT_ONE}},
            {_CONFIGURATION_KEY: {_OUTPUT_PATH_KEY: _OUTPUT_TWO}},
        ],
        replacements,
    ) == [_OUTPUT_ONE, _OUTPUT_TWO]
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


def test_shared_runtime_environment_selects_role(tmp_test_directory):
    template = Path(tmp_test_directory)
    _write_yaml(
        template / "run_time.yml",
        {
            "runtime_environment": {"image": "default.sif"},
            "runtime_environments": {
                "baseline": {"image": "baseline.sif"},
                "candidate": {"image": "candidate.sif"},
            },
        },
    )

    assert science_test_definitions._shared_runtime_environment(template, {}, "baseline") == {
        "image": "baseline.sif"
    }


def test_shared_runtime_environment_rejects_missing_role(tmp_test_directory):
    template = Path(tmp_test_directory)
    _write_yaml(
        template / "run_time.yml",
        {"runtime_environments": {"candidate": {"image": "candidate.sif"}}},
    )

    with pytest.raises(ValueError, match="No runtime environment configured"):
        science_test_definitions._shared_runtime_environment(template, {}, "baseline")


def test_collect_camera_efficiency_results_records_runtime(tmp_test_directory, monkeypatch):
    image = Path(tmp_test_directory) / _RUNTIME_IMAGE
    image.write_bytes(b"container image")
    monkeypatch.setattr(
        science_test_execution,
        "_runtime_simtools_version",
        lambda _runtime: _CURRENT_SIMTOOLS_VERSION,
    )
    output = Path(tmp_test_directory) / "camera"
    output.mkdir()
    (output / "camera_efficiency_summary_legacy.yml").write_text(
        yaml.safe_dump(
            {
                "meta": {"tel": "MSTN-01", "model_version": "7.0.0"},
                **{metric: {"value": 0.5} for metric in _CAMERA_METRICS},
            }
        ),
        encoding=_ENCODING,
    )
    configuration = {
        "output_path": str(output),
        "site": "North",
        _TELESCOPE_KEY: "MSTN-01",
        "model_version": "7.0.0",
        "zenith_angle": 0,
        "azimuth_angle": 0,
    }
    args = {
        "science_workflow": {
            "applications": [
                {
                    "application": _CAMERA_VALIDATOR_APPLICATION,
                    "configuration": configuration,
                }
            ]
        },
        "runtime_environment": {"image": str(image)},
        "provenance_path": str(Path(tmp_test_directory) / "work" / "resolved.yml"),
    }

    science_test_execution._collect_camera_efficiency_results(args)

    manifest_path = output / _CAMERA_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding=_ENCODING))
    assert manifest["status"] == _COMPLETE_STATUS
    assert manifest["image"] == f"sha256:{science_test_execution.sha256_file(image)}"
    assert manifest["software"]["simtools_version"] == _CURRENT_SIMTOOLS_VERSION
    report_work = Path(args["provenance_path"]).parent
    assert not (report_work / "camera-efficiency").exists()


def test_execute_camera_workflow_collects_results_after_application(monkeypatch):
    calls = []
    monkeypatch.setattr(
        science_test_execution, "run_applications", lambda *_args, **_kwargs: calls.append("run")
    )
    monkeypatch.setattr(
        science_test_execution,
        "_collect_camera_efficiency_results",
        lambda _args: calls.append("collect"),
    )
    monkeypatch.setattr(
        science_test_execution,
        "_evaluate_products",
        lambda *_args: {"status": "pass"},
    )

    result = science_test_execution._execute_test_workflow(
        {_RESULT_TYPE_KEY: _CAMERA_RESULT_TYPE},
        {"science_workflow": {}, "ignore_runtime_environment": True},
        {},
    )

    assert result == {"status": "pass"}
    assert calls == ["run", "collect"]


def test_collect_camera_efficiency_requires_validator_application():
    with pytest.raises(ValueError, match="requires its validation application"):
        science_test_execution._collect_camera_efficiency_results(
            {"science_workflow": {"applications": []}}
        )


def test_camera_workflow_expands_one_application_per_telescope():
    workflow = {
        "applications": [
            {
                "application": _CAMERA_VALIDATOR_APPLICATION,
                "configuration": {"output_path": "/results/telescope_throughput"},
            },
            {
                "application": "simtools-compare-science-results",
                "configuration": {},
            },
        ]
    }

    science_test_execution._expand_camera_efficiency_telescopes(
        workflow, {"camera_telescopes": ["LSTN-01", _SECOND_TELESCOPE]}, {}
    )

    applications = workflow["applications"]
    validators = applications[:2]
    assert [item["configuration"][_TELESCOPE_KEY] for item in validators] == [
        "LSTN-01",
        _SECOND_TELESCOPE,
    ]
    assert [item["configuration"]["output_path"] for item in validators] == [
        "/results/telescope_throughput/LSTN-01",
        "/results/telescope_throughput/LSTN-02",
    ]
    assert applications[-1]["configuration"]["telescopes"] == ["LSTN-01", _SECOND_TELESCOPE]


def test_camera_workflow_keeps_single_legacy_output_path():
    workflow = {
        "applications": [
            {
                "application": _CAMERA_VALIDATOR_APPLICATION,
                "configuration": {"output_path": "/results/telescope_throughput"},
            }
        ]
    }

    science_test_execution._expand_camera_efficiency_telescopes(
        workflow, {}, {"__SCIENCE_CAMERA_TELESCOPE__": _SECOND_TELESCOPE}
    )

    configuration = workflow["applications"][0]["configuration"]
    assert configuration[_TELESCOPE_KEY] == _SECOND_TELESCOPE
    assert configuration["output_path"] == "/results/telescope_throughput"


def test_comparison_workflow_receives_site_telescope_list():
    workflow = {
        "applications": [
            {
                "application": "simtools-compare-science-results",
                "configuration": {},
            }
        ]
    }

    science_test_execution._expand_camera_efficiency_telescopes(
        workflow, {"camera_telescopes": ["LSTN-01", _SECOND_TELESCOPE]}, {}
    )

    assert workflow["applications"][0]["configuration"]["telescopes"] == [
        "LSTN-01",
        _SECOND_TELESCOPE,
    ]


def test_camera_workflow_rejects_invalid_telescope_list():
    workflow = {"applications": [{"application": _CAMERA_VALIDATOR_APPLICATION}]}

    with pytest.raises(ValueError, match="non-empty camera_telescopes list"):
        science_test_execution._expand_camera_efficiency_telescopes(
            workflow, {"camera_telescopes": []}, {}
        )


def test_runtime_simtools_version_uses_container_manifest(monkeypatch):
    monkeypatch.setattr(
        science_test_execution,
        "read_runtime_environment",
        lambda _runtime: [_APPTAINER, "exec", _RUNTIME_IMAGE],
    )
    monkeypatch.setattr(
        science_test_execution.dependencies,
        "get_dependency_manifest",
        lambda _command: {_SIMTOOLS_PACKAGE: {"version": _CURRENT_SIMTOOLS_VERSION}},
    )

    assert (
        science_test_execution._runtime_simtools_version({"image": _RUNTIME_IMAGE})
        == _CURRENT_SIMTOOLS_VERSION
    )


def test_runtime_simtools_version_probes_old_image(monkeypatch):
    def missing_manifest(_command):
        raise FileNotFoundError(_MISSING_DEPENDENCY_MANIFEST)

    monkeypatch.setattr(
        science_test_execution,
        "read_runtime_environment",
        lambda _runtime: [_APPTAINER, "exec", _RUNTIME_IMAGE],
    )
    monkeypatch.setattr(
        science_test_execution.dependencies,
        "get_dependency_manifest",
        missing_manifest,
    )
    monkeypatch.setattr(
        science_test_execution.subprocess,
        "run",
        lambda command, **_kwargs: science_test_execution.subprocess.CompletedProcess(
            command, 0, stdout=f"{_OLD_SIMTOOLS_VERSION}\n", stderr=""
        ),
    )

    assert (
        science_test_execution._runtime_simtools_version({"image": _RUNTIME_IMAGE})
        == _OLD_SIMTOOLS_VERSION
    )


def test_runtime_simtools_version_uses_host_when_no_container(monkeypatch):
    def missing_manifest(_command):
        raise FileNotFoundError(_MISSING_DEPENDENCY_MANIFEST)

    monkeypatch.setattr(
        science_test_execution.dependencies, "get_dependency_manifest", missing_manifest
    )

    assert (
        science_test_execution._runtime_simtools_version(None) == science_test_execution.__version__
    )


def test_runtime_simtools_version_returns_none_when_old_image_cannot_be_probed(monkeypatch):
    def missing_manifest(_command):
        raise FileNotFoundError(_MISSING_DEPENDENCY_MANIFEST)

    monkeypatch.setattr(
        science_test_execution,
        "read_runtime_environment",
        lambda _runtime: [_APPTAINER, "exec", _RUNTIME_IMAGE],
    )
    monkeypatch.setattr(
        science_test_execution.dependencies,
        "get_dependency_manifest",
        missing_manifest,
    )
    monkeypatch.setattr(
        science_test_execution.subprocess,
        "run",
        lambda command, **_kwargs: science_test_execution.subprocess.CompletedProcess(
            command, 1, stdout="", stderr="probe failed"
        ),
    )

    assert science_test_execution._runtime_simtools_version({"image": _RUNTIME_IMAGE}) is None


def test_runtime_simtools_version_accepts_legacy_manifest_value(monkeypatch):
    monkeypatch.setattr(
        science_test_execution,
        "read_runtime_environment",
        lambda _runtime: [_APPTAINER, "exec", _RUNTIME_IMAGE],
    )
    monkeypatch.setattr(
        science_test_execution.dependencies,
        "get_dependency_manifest",
        lambda _command: {_SIMTOOLS_PACKAGE: _OLD_SIMTOOLS_VERSION},
    )

    assert (
        science_test_execution._runtime_simtools_version({"image": _RUNTIME_IMAGE})
        == _OLD_SIMTOOLS_VERSION
    )


def test_local_image_identity_returns_none_for_nonlocal_path(tmp_test_directory):
    assert (
        science_test_execution._local_image_identity(str(Path(tmp_test_directory) / "missing.sif"))
        is None
    )


def test_shared_runtime_environment_invalid(tmp_test_directory):
    template = Path(tmp_test_directory)
    _write_yaml(template / "run_time.yml", {"runtime_environment": {"image": ""}})
    with pytest.raises(jsonschema.ValidationError):
        science_test_definitions._shared_runtime_environment(template, {})


@pytest.mark.parametrize(
    (_PRODUCES_PRODUCTION, "inline"), [(False, False), (True, False), (False, True)]
)
def test_run_test_uses_shared_runtime(campaign, monkeypatch, produces_production, inline):
    template = campaign["template_dir"]
    shared = {"container_engine": "apptainer", "image": "__SCIENCE_CONTAINER_IMAGE_PATH__"}
    _write_yaml(template / "run_time.yml", {"runtime_environment": shared})
    if inline:
        workflow = science_test_definitions._load_yaml(template / _WORKFLOW_YML)
        workflow["runtime_environment"] = {"container_engine": "apptainer", "image": "inline.sif"}
        _write_yaml(template / _WORKFLOW_YML, workflow)
    preview_images = []
    monkeypatch.setattr(
        science_test_execution,
        "_preview",
        lambda args, image: preview_images.append(image) or {},
    )
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
        _NORTH,
        {"array_layout_name": "CTAO-North-Alpha"},
        "example",
        {"workflow": _WORKFLOW_YML, _PRODUCES_PRODUCTION: produces_production},
        dry_run=True,
    )
    expected = None
    if inline:
        expected = {"container_engine": "apptainer", "image": "inline.sif"}
    elif not produces_production:
        expected = {"container_engine": "apptainer", "image": "/images/shared.sif"}
    assert captured == [expected]
    if produces_production:
        assert preview_images == ["inline.sif" if inline else "/images/shared.sif"]


def test_named_artifacts_on_rerun(campaign, successful_workflow, monkeypatch):
    args = {**campaign, "sites": [_NORTH], "tests": [_DERIVE]}
    initial = campaign_runner.run_release(**args)
    report = campaign["release_dir"] / "reports/north/derive"
    previous = next(result for result in initial[_RESULTS] if result["id"] == _DERIVE_NORTH)
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
        for result in campaign_runner.run_release(**args)[_RESULTS]
        if result["id"] == _DERIVE_NORTH
    )
    assert current["run_id"] != previous["run_id"]
    assert (work / _WORKFLOW_YML).is_file()
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
        )[_RESULTS]
        if result["id"] == _DERIVE_NORTH
    )
    assert failed[_STATUS_KEY] == _INCOMPLETE
    assert failed[_REPORT] is None
    assert failed["previous_report"] == "reports/north/derive"
    assert (report / "metrics.json").exists()


@pytest.mark.parametrize(
    (_STATE, "finished"),
    [(_FAILED, True), (_FAILED, False), (_COMPLETED, True), (_SUBMITTED, False)],
)
def test_production_retry_requires_finished_failed_jobs(mocker, state, finished):
    mocker.patch.object(
        science_test_execution,
        "load_submission",
        return_value=mocker.Mock(metadata={_STATE: state}, backend=_HTCONDOR),
    )
    backend = mocker.patch.object(science_test_execution, "get_backend").return_value
    backend.is_finished.return_value = finished
    if state == _FAILED and finished:
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
        {_PRODUCES_PRODUCTION: True},
        {_SCIENCE_CANDIDATE_SITE_PATH: str(candidate)},
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


@pytest.mark.parametrize(_PENDING, [False, True])
def test_collection_waits_for_outputs_without_submitting(tmp_test_directory, mocker, pending):
    root = Path(tmp_test_directory)
    product = root / "event.dat"
    product.write_text("event")
    manifest = root / "submission.json"
    payload = {
        "backend": _HTCONDOR,
        "work_dir": str(root),
        "job_ids": ["job"],
        _METADATA: {_STATE: _SUBMITTED, "expected_outputs": {"job": [str(product)]}},
    }
    manifest.write_text(json.dumps(payload))

    def collect(submission):
        if pending:
            return None
        payload[_METADATA][_STATE] = _COMPLETED
        manifest.write_text(json.dumps(payload))
        return []

    mocker.patch.object(science_test_execution, "collect_submission", side_effect=collect)
    run = mocker.patch.object(science_test_execution, "run_applications")
    result = science_test_execution._execute_test_workflow(
        {"collect_production": True, "products": [str(product)]},
        {},
        {_SCIENCE_CANDIDATE_SITE_PATH: str(root)},
    )
    assert result[_STATUS_KEY] == (_PENDING if pending else "pass")
    run.assert_not_called()


def test_submission_does_not_validate_completed_production(mocker):
    run = mocker.patch.object(science_test_execution, "run_applications")
    mocker.patch.object(
        science_test_execution, _EVALUATE_PRODUCTS, return_value={_STATUS_KEY: "pass"}
    )
    result = science_test_execution._execute_test_workflow({_PRODUCES_PRODUCTION: True}, {}, {})
    assert result[_STATUS_KEY] == result[_EXECUTION] == _SUBMITTED
    run.assert_called_once()


@pytest.mark.parametrize(
    ("source", "inactive_environment"),
    [
        ({"simulation_models_git_path": "/models.git"}, "SIMTOOLS_SIMULATION_MODELS_PATH"),
        ({_SIMULATION_MODELS_PATH: "/models"}, "SIMTOOLS_SIMULATION_MODELS_GIT_PATH"),
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
        science_test_execution, _EVALUATE_PRODUCTS, return_value={_STATUS_KEY: "pass"}
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


def test_execution_preserves_model_source_environment_when_no_source_is_selected(
    monkeypatch, mocker
):
    variable = "SIMTOOLS_SIMULATION_MODELS_PATH"
    monkeypatch.setenv(variable, "/inherited")
    observed = []
    mocker.patch.object(
        science_test_execution,
        "run_applications",
        side_effect=lambda *_args, **_kwargs: observed.append(os.environ[variable]),
    )
    mocker.patch.object(science_test_execution, _EVALUATE_PRODUCTS, return_value={})

    science_test_execution._execute_test_workflow({}, {}, {})

    assert observed == ["/inherited"]
    assert os.environ[variable] == "/inherited"


def test_pending_collection_defers_dependent_tests(mocker):
    collect = {"id": "collect.north", "site": _NORTH, "test": _COLLECT}
    compare = {"id": _COMPARE_NORTH, "site": _NORTH, "test": _COMPARE}
    execute = mocker.patch.object(
        science_test_execution, "_run_test", return_value={**collect, _STATUS_KEY: _PENDING}
    )
    mocker.patch.object(science_test_execution, "_log_test_execution")
    results = science_test_execution._execute_selection(
        [(({},), collect), (({_DEPENDS_ON: [_COLLECT]},), compare)],
        False,
        "run",
        {},
        {_COLLECT: {}, _COMPARE: {_DEPENDS_ON: [_COLLECT]}},
    )
    assert [result[_STATUS_KEY] for result in results] == [_PENDING, _PENDING]
    execute.assert_called_once()


def test_materialize_effective_workflow_serializes_model_paths(tmp_test_directory):
    work = Path(tmp_test_directory) / "work"
    models = Path(tmp_test_directory) / "models"
    resolved = {
        _APPLICATIONS_KEY: [
            {
                _APPLICATION_KEY: "simtools-write-trigger-histograms",
                _CONFIGURATION_KEY: {_SIMULATION_MODELS_PATH: models},
            }
        ]
    }
    path = science_test_execution._materialize_effective_workflow(work, resolved)
    recorded = yaml.safe_load(path.read_text(encoding=_ENCODING))
    assert recorded[_APPLICATIONS_KEY][0][_CONFIGURATION_KEY][_SIMULATION_MODELS_PATH] == str(
        models
    )


@pytest.mark.parametrize("manifest", [{}, {_SIMTOOLS_PACKAGE: {}}])
def test_runtime_version_probes_image_when_manifest_has_no_version(monkeypatch, manifest):
    monkeypatch.setattr(
        science_test_execution, "read_runtime_environment", lambda _runtime: [_APPTAINER]
    )
    monkeypatch.setattr(
        science_test_execution.dependencies, "get_dependency_manifest", lambda _command: manifest
    )
    monkeypatch.setattr(
        science_test_execution.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=_CURRENT_SIMTOOLS_VERSION),
    )

    assert science_test_execution._runtime_simtools_version({"image": _RUNTIME_IMAGE}) == (
        _CURRENT_SIMTOOLS_VERSION
    )


def test_science_comparison_labels_follow_release_definition():
    workflow = {
        _APPLICATIONS_KEY: [
            {_APPLICATION_KEY: "simtools-compare-science-results", _CONFIGURATION_KEY: {}}
        ]
    }
    science_test_execution._configure_comparison_applications(
        workflow, {"baseline": "reference", "release_label": "candidate"}
    )

    assert workflow[_APPLICATIONS_KEY][0][_CONFIGURATION_KEY] == {
        "baseline_label": "reference",
        "candidate_label": "candidate",
    }


def test_failed_validator_invalidates_previous_manifest(tmp_test_directory, monkeypatch):
    output = Path(tmp_test_directory)
    manifest = output / _CAMERA_MANIFEST_NAME
    manifest.write_text(json.dumps({_STATUS_KEY: _COMPLETE_STATUS}), encoding=_ENCODING)
    args = {
        _SCIENCE_WORKFLOW: {
            _APPLICATIONS_KEY: [
                {
                    _APPLICATION_KEY: _CAMERA_VALIDATOR_APPLICATION,
                    _CONFIGURATION_KEY: {
                        _OUTPUT_PATH_KEY: str(output),
                        _TELESCOPE_KEY: _SECOND_TELESCOPE,
                    },
                }
            ]
        }
    }

    def fail_validation(*_args, **_kwargs):
        assert json.loads(manifest.read_text(encoding=_ENCODING))[_STATUS_KEY] == _INCOMPLETE
        raise RuntimeError(_VALIDATOR_FAILURE)

    monkeypatch.setattr(science_test_execution, "run_applications", fail_validation)
    with pytest.raises(RuntimeError, match=_VALIDATOR_FAILURE):
        science_test_execution._execute_test_workflow(
            {_RESULT_TYPE_KEY: _CAMERA_RESULT_TYPE}, args, {}
        )

    assert json.loads(manifest.read_text(encoding=_ENCODING))[_STATUS_KEY] == _INCOMPLETE
