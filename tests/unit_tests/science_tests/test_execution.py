import os
from pathlib import Path

import pytest
import yaml

from simtools.science_tests import definitions as defs
from simtools.science_tests import execution

MODELS = "/models"
PRODUCTION_SIF = "/production.sif"
LSTN_01 = "LSTN-01"
MSTN_01 = "MSTN-01"
SIMTOOLS_SIMULATION_MODELS_GIT_PATH = "SIMTOOLS_SIMULATION_MODELS_GIT_PATH"
APPLICATION = "application"
APPLICATIONS = "applications"
C_ATMOSPHERE = "c.atmosphere"
COLLECT_PRODUCTION = "collect_production"
CONFIGURATION = "configuration"
COVERAGE = "coverage"
FIGURE_FORMAT = "figure_format"
INCOMPLETE = "incomplete"
OUTPUT_PATH = "output_path"
PENDING = "pending"
PRODUCES_PRODUCTION = "produces_production"
RUNTIME = "runtime"
RUNTIME_ENVIRONMENT = "runtime_environment"
SUBMITTED = "submitted"
TELESCOPE = "telescope"
UTF_8 = "utf-8"
WORKFLOW_YML = "workflow.yml"

NORTH = "north"
DERIVE = "derive"
COMPARE = "compare"
RUN_ID = "run"
STATUS = "status"
VALIDATOR = "simtools-validate-telescope-throughput"
MODELS_PATH_ENV = "SIMTOOLS_SIMULATION_MODELS_PATH"


def write_workflow(bundle, name, *applications, **extra):
    path = bundle.template_dir / name
    data = {
        "schema_version": "0.7.0",
        "schema_name": "application_workflow.metaschema",
        APPLICATIONS: list(applications),
        **extra,
    }
    path.write_text(yaml.safe_dump(data), encoding=UTF_8)


def prepare(bundle, test_id=DERIVE, args=None):
    return execution.prepare_test(bundle, NORTH, test_id, args or {}, RUN_ID)


def test_prepare_test_plans_without_runtime(bundle):
    test = prepare(bundle)

    assert test.result[STATUS] == "planned"
    assert test.result[RUNTIME] == "host (no container)"
    assert test.result["output_paths"] == [str(test.work)]
    assert test.work.parts[-2:] == (NORTH, DERIVE)


def test_prepare_test_expands_telescopes_and_applies_overrides(bundle):
    bundle.suites[NORTH]["camera_telescopes"] = [LSTN_01, MSTN_01]
    bundle.release["configuration_overrides"] = {DERIVE: {FIGURE_FORMAT: ["pdf"]}}
    write_workflow(
        bundle,
        WORKFLOW_YML,
        {
            APPLICATION: VALIDATOR,
            CONFIGURATION: {
                TELESCOPE: defs.TELESCOPE,
                OUTPUT_PATH: f"{defs.CANDIDATE_SITE_PATH}/{defs.TELESCOPE}",
            },
        },
        {APPLICATION: "simtools-other", CONFIGURATION: {}},
    )

    applications = prepare(bundle).workflow[APPLICATIONS]

    assert [app[CONFIGURATION].get(TELESCOPE) for app in applications] == [
        LSTN_01,
        MSTN_01,
        None,
    ]
    assert applications[1][CONFIGURATION][OUTPUT_PATH].endswith("north/MSTN-01")
    assert all(app[CONFIGURATION][FIGURE_FORMAT] == ["pdf"] for app in applications)


def test_prepare_test_requires_telescopes_and_declared_changes(bundle):
    write_workflow(
        bundle, WORKFLOW_YML, {APPLICATION: VALIDATOR, CONFIGURATION: {"t": defs.TELESCOPE}}
    )
    with pytest.raises(ValueError, match="set camera_telescopes"):
        prepare(bundle)
    write_workflow(
        bundle,
        WORKFLOW_YML,
        {APPLICATION: "simtools-compare", CONFIGURATION: {"compare_by": [C_ATMOSPHERE]}},
    )
    with pytest.raises(ValueError, match="not in expected_changes"):
        prepare(bundle)
    bundle.release["expected_changes"] = [C_ATMOSPHERE]
    prepare(bundle)


def test_prepare_test_rejects_unsupported_site(bundle):
    bundle.catalogue[DERIVE]["sites"] = ["south"]
    with pytest.raises(ValueError, match="does not support site"):
        prepare(bundle)


def test_prepare_test_adds_shared_runtime_and_production_preview(bundle, mocker):
    (bundle.template_dir / defs.RUNTIME_FILE).write_text(
        yaml.safe_dump(
            {RUNTIME_ENVIRONMENT: {"container_engine": "apptainer", "image": "/image.sif"}}
        ),
        encoding=UTF_8,
    )
    assert "apptainer (image: /image.sif)" == prepare(bundle).result[RUNTIME]

    preview = mocker.patch.object(execution, "preview", return_value={"ok": True})
    bundle.catalogue[DERIVE][PRODUCES_PRODUCTION] = True
    backend = bundle.template_dir / "backend.yml"
    backend.write_text(yaml.safe_dump({"container_image": PRODUCTION_SIF}), encoding=UTF_8)
    write_workflow(
        bundle,
        WORKFLOW_YML,
        {
            APPLICATION: "simtools-simulate-prod",
            CONFIGURATION: {"backend_config": str(backend)},
        },
    )
    test = prepare(bundle)

    assert test.result["dependency_compatibility"] == {"ok": True}
    assert preview.call_args.args[1] == PRODUCTION_SIF
    assert RUNTIME_ENVIRONMENT not in test.workflow


def test_run_test_executes_workflow_and_publishes_report(bundle, successful_workflow):
    test = prepare(bundle)
    execution.run_test(test, bundle.release_dir, RUN_ID)

    assert test.result[STATUS] == "pass"
    assert successful_workflow[0][1] == str(test.work / execution.WORKFLOW_RECORD)
    assert (bundle.release_dir / "reports" / NORTH / DERIVE / "metrics.json").is_file()


def test_run_test_records_failures_and_acceptance_errors(bundle, mocker):
    test = prepare(bundle)
    mocker.patch.object(execution, "run_applications", side_effect=RuntimeError("boom"))
    execution.run_test(test, bundle.release_dir, RUN_ID)
    assert test.result[STATUS] == INCOMPLETE
    assert test.result["error"] == "boom"

    mocker.patch.object(execution, "check_inputs", side_effect=execution.AcceptanceError("limit"))
    execution.run_test(test, bundle.release_dir, RUN_ID)
    assert test.result[STATUS] == "warn"


def test_execute_tests_runs_dependencies_first(bundle, successful_workflow):
    tests = [prepare(bundle), prepare(bundle, COMPARE)]
    results = execution.execute_tests(tests, bundle.catalogue, bundle.release_dir, RUN_ID)
    assert [result[STATUS] for result in results] == ["pass", "warn"]
    assert [call[0] for call in successful_workflow] == [NORTH, NORTH]


def test_execute_tests_blocks_dependents(bundle, mocker):
    derive, compare = prepare(bundle), prepare(bundle, COMPARE)

    def fail(test, *_args):
        test.result[STATUS] = SUBMITTED if test.test_id == DERIVE else "pass"

    mocker.patch.object(execution, "run_test", side_effect=fail)
    results = execution.execute_tests([derive, compare], bundle.catalogue, None, RUN_ID)
    assert results[1][STATUS] == PENDING

    def incomplete(test, *_args):
        test.result[STATUS] = INCOMPLETE

    execution.run_test.side_effect = incomplete
    results = execution.execute_tests([derive, compare], bundle.catalogue, None, RUN_ID)
    assert results[1][STATUS] == "blocked"


def test_execute_collects_production(bundle, mocker):
    bundle.catalogue[DERIVE][COLLECT_PRODUCTION] = True
    test = prepare(bundle)
    collect = mocker.patch.object(execution, COLLECT_PRODUCTION, return_value={STATUS: PENDING})
    assert execution._execute(test, overwrite=False) == {STATUS: PENDING}
    collect.return_value = {COVERAGE: {}}
    mocker.patch.object(execution, "evaluate_products", return_value={STATUS: "pass"})
    assert execution._execute(test, overwrite=False) == {STATUS: "pass", COVERAGE: {}}


def test_execute_reports_submitted_production(bundle, successful_workflow):
    test = prepare(bundle)
    test.definition[PRODUCES_PRODUCTION] = True
    assert execution._execute(test, overwrite=False)[STATUS] == SUBMITTED


def test_derived_products_are_reused_only_when_current(bundle, successful_workflow, mocker):
    mocker.patch.object(execution, "validate_derived_workflow")
    bundle.catalogue[DERIVE]["derived_output"] = defs.REPORT_PATH
    test = prepare(bundle)
    execution._execute(test, overwrite=False)
    assert len(successful_workflow) == 1

    assert execution._execute(test, overwrite=False)["execution"] == "reused"
    test.workflow[APPLICATIONS][0][CONFIGURATION]["changed"] = True
    with pytest.raises(ValueError, match="stale"):
        execution._execute(test, overwrite=False)
    execution._execute(test, overwrite=True)
    assert len(successful_workflow) == 2


def test_selected_model_source_environment_hides_inactive_source(monkeypatch):
    monkeypatch.setenv(MODELS_PATH_ENV, MODELS)
    with execution._selected_model_source_environment({defs.MODELS_GIT_PATH: "/git"}):
        assert os.environ[MODELS_PATH_ENV] == ""
    assert os.environ[MODELS_PATH_ENV] == MODELS
    monkeypatch.delenv(SIMTOOLS_SIMULATION_MODELS_GIT_PATH, raising=False)
    with execution._selected_model_source_environment({defs.MODELS_PATH: MODELS}):
        assert os.environ[SIMTOOLS_SIMULATION_MODELS_GIT_PATH] == ""
    assert SIMTOOLS_SIMULATION_MODELS_GIT_PATH not in os.environ
    with execution._selected_model_source_environment({}):
        assert os.environ[MODELS_PATH_ENV] == MODELS


def test_log_planned(bundle, caplog):
    with caplog.at_level("INFO"):
        execution.log_planned(prepare(bundle))
    assert "Planned science test: derive.north" in caplog.text
    assert Path(caplog.records[-1].getMessage().split(": ")[-1]).name == DERIVE
