import json
import shutil
from pathlib import Path

import pytest
import yaml

from simtools import science_tests


def test_run_test_dry_run_validates_resolved_workflow(tmp_test_directory):
    root = Path(tmp_test_directory)
    release_dir = root / "release"
    template_dir = root / "science-test-template"
    release_dir.mkdir()
    template_dir.mkdir()
    workflow = template_dir / "workflow.yml"
    workflow.write_text(
        yaml.safe_dump(
            {
                "schema_version": "0.5.0",
                "schema_name": "application_workflow.metaschema",
                "applications": [
                    {
                        "application": "simtools-run-application",
                        "configuration": {},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = science_tests._run_test(
        release_dir,
        template_dir,
        {"__SCIENCE_CANDIDATE_ROOT__": str(root / "candidate")},
        "north",
        {"array_layout_name": "CTAO-North-Alpha"},
        "example",
        {"workflow": "workflow.yml", "sites": ["north"]},
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert not (release_dir / "reports").exists()
    assert not (root / "candidate").exists()


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def campaign(tmp_test_directory):
    root = Path(tmp_test_directory)
    release = root / "release"
    template = root / "science-test-template"
    _write_yaml(
        release / "release.yml",
        {
            "release_label": "candidate",
            "baseline": "baseline",
            "required_sites": ["north", "south"],
        },
    )
    for site in ("north", "south"):
        _write_yaml(
            release / "sites" / f"{site}.yml",
            {
                "site": site,
                "array_layout_name": f"CTAO-{site.title()}-Alpha",
                "required_tests": ["compare"],
            },
        )
    _write_yaml(
        template / "catalogue.yml",
        {
            "tests": {
                "derive": {
                    "workflow": "workflow.yml",
                    "tier": "release-light",
                    "sites": ["north", "south"],
                },
                "compare": {
                    "workflow": "workflow.yml",
                    "tier": "release-light",
                    "sites": ["north", "south"],
                    "depends_on": ["derive"],
                    "acceptance_rule": "comparison",
                },
            }
        },
    )
    _write_yaml(
        template / "workflow.yml",
        {
            "schema_version": "0.7.0",
            "schema_name": "application_workflow.metaschema",
            "applications": [
                {
                    "application": "simtools-compare-productions",
                    "configuration": {"output_path": "__SCIENCE_REPORT_ROOT__"},
                }
            ],
            "collection": {
                "output_path": "__SCIENCE_COLLECTION_ROOT__",
                "files": ["metrics.json"],
                "write_inventory": True,
                "preserve_relative_paths": True,
            },
        },
    )
    _write_yaml(
        template / "acceptance" / "expected-products.yml",
        {
            "tests": {
                "derive": ["__SCIENCE_REPORT_ROOT__/metrics.json"],
                "compare": ["__SCIENCE_REPORT_ROOT__/metrics.json"],
            }
        },
    )
    _write_yaml(
        template / "acceptance" / "thresholds.yml",
        {
            "rules": {
                "comparison": {
                    "mode": "advisory",
                    "metrics": [
                        {
                            "file": "__SCIENCE_REPORT_ROOT__/metrics.json",
                            "key": "count",
                            "minimum": 2,
                        }
                    ],
                },
            }
        },
    )
    context = root / "context.yml"
    _write_yaml(
        context,
        {
            key: str(root / name)
            for key, name in zip(science_tests._ROOT_KEYS, ("candidate", "baseline", "reference"))
        },
    )
    return {"release_dir": release, "template_dir": template, "context_file": context}


@pytest.fixture
def successful_workflow(monkeypatch):
    calls = []

    def execute(args, replacements):
        from simtools.runners.simtools_runner import _copy_collection_files

        calls.append(replacements["__SCIENCE_SITE_KEY__"])
        output = Path(replacements["__SCIENCE_REPORT_ROOT__"])
        output.mkdir(parents=True)
        (output / "metrics.json").write_text('{"count": 3}', encoding="utf-8")
        _copy_collection_files(
            [{"configuration": {"output_path": str(output)}}],
            {
                "output_path": replacements["__SCIENCE_COLLECTION_ROOT__"],
                "files": ["metrics.json"],
                "write_inventory": True,
            },
        )

    monkeypatch.setattr(science_tests, "run_applications", execute)
    return calls


def test_dry_run_is_read_only(campaign, successful_workflow):
    summary = science_tests.run_release(**campaign, dry_run=True)
    assert len(summary["results"]) == 4
    assert all(result["status"] == "planned" for result in summary["results"])
    assert not summary["qualified"]
    assert successful_workflow == []
    assert not (campaign["release_dir"] / "reports").exists()


def test_run_collects_evidence_and_preserves_subset_results(campaign, successful_workflow):
    north = science_tests.run_release(**campaign, sites=["north"], tests=["compare"])
    assert {r["id"]: r["status"] for r in north["results"]} == {
        "compare.north": "warn",
        "derive.north": "pass",
        "compare.south": "not_run",
    }
    south = science_tests.run_release(**campaign, sites=["south"], tests=["compare"])
    assert len(south["results"]) == 4
    assert successful_workflow == ["north", "north", "south", "south"]
    result = south["results"][0]
    report = campaign["release_dir"] / result["report"]
    assert json.loads((report / "metrics.json").read_text()) == {"count": 3}
    inventory = json.loads((report / "inventory.json").read_text())
    assert inventory[0]["source"] == "metrics.json"
    assert len(inventory[0]["sha256"]) == 64
    assert not south["qualified"]
    assert science_tests.run_release(**campaign, dry_run=True)["results"][0]["status"] == "planned"
    assert (
        json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text()) == south
    )


@pytest.mark.parametrize(
    "selection", [{"tests": ["missing"]}, {"sites": ["missing"]}, {"tests": []}, {"sites": []}]
)
def test_invalid_selection_fails_without_execution(campaign, successful_workflow, selection):
    with pytest.raises(ValueError, match="selection"):
        science_tests.run_release(**campaign, **selection)
    assert successful_workflow == []


def test_missing_unselected_site_fails_preflight(campaign, successful_workflow):
    (campaign["release_dir"] / "sites/south.yml").unlink()
    with pytest.raises(FileNotFoundError):
        science_tests.run_release(**campaign, sites=["north"])
    assert successful_workflow == []


def test_failed_dependency_blocks_comparison(campaign, monkeypatch):
    calls = []

    def fail(args, replacements):
        calls.append(args)
        raise RuntimeError("simulator failed")

    monkeypatch.setattr(science_tests, "run_applications", fail)
    with pytest.raises(RuntimeError, match="Science tests failed"):
        science_tests.run_release(**campaign, sites=["north"])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    assert len(calls) == 1
    assert {r["test"]: r["status"] for r in summary["results"] if r["site"] == "north"} == {
        "derive": "incomplete",
        "compare": "blocked",
    }


@pytest.mark.parametrize("value", [0, float("nan"), float("inf"), True, "3"])
def test_invalid_or_insufficient_metric_fails(value):
    with pytest.raises(ValueError, match="count"):
        science_tests._check_metric(value, {"key": "count", "minimum": 2})


def test_dependency_order_and_invalid_definitions():
    assert science_tests._dependency_order(["b", "a"], {"a": {}, "b": {"depends_on": ["a"]}}) == [
        "a",
        "b",
    ]
    with pytest.raises(ValueError, match="Circular"):
        science_tests._dependency_order(["a"], {"a": {"depends_on": ["a"]}})
    with pytest.raises(ValueError, match="Unknown"):
        science_tests._dependency_order(["a"], {})
    with pytest.raises(ValueError, match="Invalid definition"):
        science_tests._dependency_order(["a"], {"a": []})


@pytest.mark.parametrize(
    ("allow", "tests"), [(False, ["production"]), (True, None), (True, ["compare"])]
)
def test_production_must_be_explicit(allow, tests):
    with pytest.raises(ValueError, match="Select production explicitly"):
        science_tests._check_production_selection(
            "production", {"tier": "release-prod"}, tests, allow, False
        )


def test_completed_production_gate(tmp_test_directory):
    root = Path(tmp_test_directory)
    submission = root / "submission.json"
    product = root / "event.dat"
    product.write_text("event", encoding="utf-8")
    payload = {
        "job_ids": ["job"],
        "metadata": {"state": "submitted", "expected_outputs": {"job": [str(product)]}},
    }
    submission.write_text(json.dumps(payload), encoding="utf-8")
    definition = {"requires_completed_production": True}
    replacements = {"__SCIENCE_CANDIDATE_SITE_ROOT__": str(root)}
    with pytest.raises(ValueError, match="not completed"):
        science_tests._check_inputs(definition, replacements)
    payload["metadata"]["state"] = "completed"
    submission.write_text(json.dumps(payload), encoding="utf-8")
    science_tests._check_inputs(definition, replacements)
    product.unlink()
    with pytest.raises(ValueError, match="missing expected outputs"):
        science_tests._check_inputs(definition, replacements)


@pytest.mark.parametrize(
    ("release", "context", "message"),
    [
        ({"release_label": "../invalid"}, {}, "release_label"),
        ({"release_label": "candidate"}, {"__SCIENCE_RELEASE_LABEL__": "other"}, "labels differ"),
        (
            {"release_label": "candidate"},
            {"__SCIENCE_RELEASE_LABEL__": "candidate"},
            "absolute path",
        ),
    ],
)
def test_invalid_release_context(release, context, message):
    with pytest.raises(ValueError, match=message):
        science_tests._validate_release_context(release, context)


@pytest.mark.parametrize(
    ("test_id", "definition", "message"),
    [
        ("../test", {}, "Invalid science-test ID"),
        ("test", {"sites": ["south"]}, "does not support"),
        ("test", {"sites": ["north"], "tier": "invalid"}, "Invalid tier"),
        ("test", {"sites": ["north"], "tier": "smoke"}, "Missing workflow"),
    ],
)
def test_invalid_test_definition(test_id, definition, message):
    with pytest.raises(ValueError, match=message):
        science_tests._validate_definition(test_id, "north", definition)


@pytest.mark.parametrize("required", [[], ["north", "north"], "north"])
def test_invalid_required_sites(campaign, required):
    _write_yaml(
        campaign["release_dir"] / "release.yml",
        {
            "release_label": "candidate",
            "required_sites": required,
        },
    )
    with pytest.raises(ValueError, match="required_sites"):
        science_tests.run_release(**campaign, dry_run=True)


def test_configuration_change_invalidates_old_results(campaign, successful_workflow):
    science_tests.run_release(**campaign, sites=["north"])
    path = campaign["template_dir"] / "workflow.yml"
    path.write_text(path.read_text() + "# changed workflow\n", encoding="utf-8")
    summary = science_tests.run_release(**campaign, sites=["south"])
    assert next(r for r in summary["results"] if r["id"] == "compare.north")["status"] == "not_run"


def test_rerunning_dependency_invalidates_comparison(campaign, successful_workflow):
    science_tests.run_release(**campaign, sites=["north"])
    summary = science_tests.run_release(**campaign, sites=["north"], tests=["derive"])
    record = next(r for r in summary["results"] if r["id"] == "compare.north")
    assert record["status"] == "not_run"
    assert "Upstream" in record["reason"]


def test_two_checkouts_have_portable_equivalent_plans(campaign, tmp_test_directory):
    root = Path(tmp_test_directory) / "second-checkout"
    shutil.copytree(campaign["release_dir"], root / "release")
    shutil.copytree(campaign["template_dir"], root / "science-test-template")
    first = science_tests.run_release(**campaign, dry_run=True)
    second = science_tests.run_release(
        release_dir=root / "release", context_file=campaign["context_file"], dry_run=True
    )
    assert first["campaign_sha256"] == second["campaign_sha256"]
    assert [(r["id"], r["status"]) for r in first["results"]] == [
        (r["id"], r["status"]) for r in second["results"]
    ]
    assert str(root) not in json.dumps(second)


def test_missing_product_is_recorded(campaign, monkeypatch):
    monkeypatch.setattr(science_tests, "run_applications", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Science tests failed"):
        science_tests.run_release(**campaign, sites=["north"])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    failed = next(r for r in summary["results"] if r["id"] == "derive.north")
    report = campaign["release_dir"] / failed["report"]
    assert "Missing or empty required product" in (report / "summary.md").read_text()
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
                            "file": "__SCIENCE_REPORT_ROOT__/metrics.json",
                            "key": "count",
                            "maximum": 2,
                        }
                    ],
                },
            }
        },
    )
    with pytest.raises(RuntimeError, match="Science tests failed"):
        science_tests.run_release(**campaign, sites=["north"])
    summary = json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text())
    failed = next(r for r in summary["results"] if r["id"] == "compare.north")
    assert failed["status"] == "fail"
    assert failed["execution"] == "completed"


def test_missing_metric_and_empty_file(tmp_test_directory):
    root = Path(tmp_test_directory)
    path = root / "metrics.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing metric"):
        science_tests._evaluate_products(
            {"rule": {"metrics": [{"file": str(path), "key": "missing"}]}}, {}
        )
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or empty"):
        science_tests._matched_files(str(path))


def test_existing_submission_prevents_resubmission(tmp_test_directory):
    root = Path(tmp_test_directory)
    (root / "submission.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="do not resubmit"):
        science_tests._check_inputs(
            {"produces_production": True}, {"__SCIENCE_CANDIDATE_SITE_ROOT__": str(root)}
        )


def test_unknown_acceptance_rule_fails_preflight(campaign, successful_workflow):
    _write_yaml(campaign["template_dir"] / "acceptance/thresholds.yml", {"rules": {}})
    with pytest.raises(ValueError, match="Missing acceptance rule"):
        science_tests.run_release(**campaign, dry_run=True)
    assert successful_workflow == []


def test_missing_catalogue_does_not_fallback(campaign):
    with pytest.raises(FileNotFoundError):
        science_tests._load_catalogue(
            {"catalogue": "missing.yml"}, campaign["release_dir"], campaign["template_dir"]
        )


def test_invalid_workflow_and_context(tmp_test_directory):
    root = Path(tmp_test_directory)
    with pytest.raises(ValueError, match="relative template path"):
        science_tests._resolve_workflow(root, root, "../workflow.yml")
    assert science_tests._load_context(None) == {}
    path = root / "context.yml"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a mapping"):
        science_tests._load_context(path)


def test_site_cannot_override_context_root(tmp_test_directory):
    root = Path(tmp_test_directory)
    context = {"__SCIENCE_CANDIDATE_ROOT__": str(root)}
    suite = {
        "array_layout_name": "CTAO-North-Alpha",
        "replacements": {"__SCIENCE_CANDIDATE_ROOT__": "other"},
    }
    with pytest.raises(ValueError, match="reserved context key"):
        science_tests._site_replacements(context, "north", suite, root)


def test_optional_catalogue_test_can_be_selected(campaign, successful_workflow):
    path = campaign["template_dir"] / "catalogue.yml"
    catalogue = yaml.safe_load(path.read_text())
    catalogue["tests"]["grid"] = {
        "workflow": "workflow.yml",
        "sites": ["north"],
        "tier": "release-light",
    }
    _write_yaml(path, catalogue)
    products = campaign["template_dir"] / "acceptance/expected-products.yml"
    contracts = yaml.safe_load(products.read_text())
    contracts["tests"]["grid"] = ["__SCIENCE_REPORT_ROOT__/metrics.json"]
    _write_yaml(products, contracts)
    summary = science_tests.run_release(**campaign, sites=["north"], tests=["grid"], dry_run=True)
    assert next(r for r in summary["results"] if r["id"] == "grid.north")["status"] == "planned"
    with pytest.raises(ValueError, match="does not support"):
        science_tests.run_release(**campaign, sites=["south"], tests=["grid"], dry_run=True)
    assert successful_workflow == []


@pytest.mark.parametrize(
    "rule",
    [
        [],
        {"mode": "unknown"},
        {"mode": "advisory", "metrics": {}},
        {"mode": "advisory", "metrics": [{}]},
        {
            "mode": "advisory",
            "metrics": [{"file": "metrics.json", "key": "count", "minimum": True}],
        },
    ],
)
def test_malformed_acceptance_rules_fail_preflight(rule):
    with pytest.raises(ValueError, match=r"acceptance|Acceptance"):
        science_tests._validate_rule(rule, "comparison")
