import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from simtools.science_tests import campaign as campaign_runner
from simtools.science_tests import execution as science_test_execution

_SCIENCE_CANDIDATE_PATH = "__SCIENCE_CANDIDATE_PATH__"
_EXPECTED_PRODUCTS_FILE = "acceptance/expected-products.yml"
_APPLICATIONS_KEY = "applications"
_CATALOGUE_FILE = "catalogue.yml"
_COLLECT = "collect"
_COLLECTION_WORKFLOW_FILE = "collect.yml"
_COMPARE = "compare"
_COMPLETED = "completed"
_CONTEXT_FILE = "context_file"
_DERIVE = "derive"
_JOB_1 = "job-1"
_METADATA = "metadata"
_NORTH = "north"
_RELEASE_LIGHT = "release-light"
_RELEASE_DIR = "release_dir"
_REPORT = "report"
_RESULTS = "results"
_RUN_APPLICATIONS = "run_applications"
_SITES = "sites"
_SOUTH = "south"
_STATE = "state"
_TEMPLATE_DIR = "template_dir"
_TESTS = "tests"
_WORKFLOW = "workflow"
RESULT_STATUS_KEY = "status"


def test_staged_campaign_preserves_collection_and_recovers_partial_derivation(
    campaign, successful_workflow, monkeypatch, write_science_yaml
):
    template = campaign[_TEMPLATE_DIR]
    catalogue_path = template / _CATALOGUE_FILE
    catalogue = yaml.safe_load(catalogue_path.read_text())
    catalogue[_TESTS][_COLLECT] = {
        _WORKFLOW: _COLLECTION_WORKFLOW_FILE,
        "tier": _RELEASE_LIGHT,
        _SITES: [_NORTH, _SOUTH],
        "collect_production": True,
    }
    catalogue[_TESTS][_DERIVE]["depends_on"] = [_COLLECT]
    write_science_yaml(catalogue_path, catalogue)
    write_science_yaml(
        template / _COLLECTION_WORKFLOW_FILE,
        {
            _APPLICATIONS_KEY: [],
            "schema_name": "application_workflow.metaschema",
            "schema_version": "0.7.0",
        },
    )
    products_path = template / _EXPECTED_PRODUCTS_FILE
    products = yaml.safe_load(products_path.read_text())
    products[_TESTS][_COLLECT] = ["__SCIENCE_CANDIDATE_SITE_PATH__/submission.json"]
    write_science_yaml(products_path, products)
    context = yaml.safe_load(campaign[_CONTEXT_FILE].read_text())
    production = Path(context[_SCIENCE_CANDIDATE_PATH]) / _NORTH
    production.mkdir(parents=True)
    manifest = production / "submission.json"
    event_file = production / "events.dat"
    payload = {
        "job_ids": [_JOB_1],
        _METADATA: {
            _STATE: "submitted",
            "expected_outputs": {_JOB_1: [str(event_file)]},
        },
    }
    manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(
        science_test_execution,
        "load_submission",
        lambda _path: SimpleNamespace(backend="htcondor", metadata=payload[_METADATA]),
    )
    monkeypatch.setattr(science_test_execution, "collect_submission", lambda _submission: None)

    pending = campaign_runner.run_release(
        **campaign, sites=[_NORTH], tests=[_COLLECT, _DERIVE, _COMPARE]
    )
    assert {
        record[RESULT_STATUS_KEY] for record in pending[_RESULTS] if record["site"] == _NORTH
    } == {"pending"}

    event_file.write_bytes(b"completed event data")
    payload[_METADATA][_STATE] = _COMPLETED
    manifest.write_text(json.dumps(payload))
    execute = science_test_execution.run_applications
    monkeypatch.setattr(science_test_execution, _RUN_APPLICATIONS, lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="Missing or empty required product"):
        campaign_runner.run_release(**campaign, sites=[_NORTH], tests=[_COLLECT, _DERIVE, _COMPARE])
    monkeypatch.setattr(science_test_execution, _RUN_APPLICATIONS, execute)
    campaign_runner.run_release(**campaign, sites=[_NORTH], tests=[_DERIVE])
    summary = campaign_runner.run_release(**campaign, sites=[_NORTH], tests=[_COMPARE])
    records = {record["test"]: record for record in summary[_RESULTS] if record["site"] == _NORTH}
    assert records[_COLLECT][RESULT_STATUS_KEY] == "pass"
    assert records[_DERIVE][RESULT_STATUS_KEY] == "pass"
    assert records[_COMPARE][RESULT_STATUS_KEY] == "warn"
    assert (campaign[_RELEASE_DIR] / records[_COMPARE][_REPORT] / "acceptance.md").is_file()
