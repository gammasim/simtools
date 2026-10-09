import json
import logging
import shutil
from pathlib import Path

import jsonschema
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
        {"__SCIENCE_CANDIDATE_PATH__": str(root / "candidate")},
        "north",
        {"array_layout_name": "CTAO-North-Alpha"},
        "example",
        {"workflow": "workflow.yml", "sites": ["north"]},
        dry_run=True,
    )

    assert result["status"] == "planned"
    assert not (release_dir / "reports").exists()
    assert not (root / "candidate").exists()


def test_setup_release_creates_minimal_release_from_templates(tmp_test_directory, caplog):
    root = Path(tmp_test_directory)
    template = root / "science-test-template"
    release = root / "v1.2.3" / "science_tests"
    for relative, contents in {
        "release.yml": "release_label: __SCIENCE_RELEASE_LABEL__\n",
        "sites/north.yml": "site: north\n",
        "sites/south.yml": "site: south\n",
        "context.example.yml": "__SCIENCE_CANDIDATE_PATH__: /path/candidate\n",
    }.items():
        path = template / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    with caplog.at_level(logging.INFO):
        science_tests.setup_release(release, template)

    assert (release / "release.yml").read_text(encoding="utf-8") == "---\nrelease_label: v1.2.3\n"
    assert all(
        path.read_text(encoding="utf-8").startswith("---") for path in release.rglob("*.yml")
    )
    assert "Edit these files before the dry run" in caplog.text
    assert sorted(path.relative_to(release).as_posix() for path in release.rglob("*")) == [
        "context.yml",
        "release.yml",
        "sites",
        "sites/north.yml",
        "sites/south.yml",
    ]


def test_setup_release_does_not_overwrite_existing_files(tmp_test_directory):
    root = Path(tmp_test_directory)
    template = root / "science-test-template"
    release = root / "v1.2.3" / "science_tests"
    (template / "release.yml").parent.mkdir(parents=True)
    (template / "release.yml").write_text("release_label: __SCIENCE_RELEASE_LABEL__\n")
    (template / "context.example.yml").write_text("context\n")
    (release / "release.yml").parent.mkdir(parents=True)
    (release / "release.yml").write_text("existing\n")

    with pytest.raises(FileExistsError, match="already exist"):
        science_tests.setup_release(release, template)


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
                    "configuration": {"output_path": "__SCIENCE_REPORT_PATH__"},
                }
            ],
            "collection": {
                "output_path": "__SCIENCE_COLLECTION_PATH__",
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
                "derive": ["__SCIENCE_REPORT_PATH__/metrics.json"],
                "compare": ["__SCIENCE_REPORT_PATH__/metrics.json"],
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
                            "file": "__SCIENCE_REPORT_PATH__/metrics.json",
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
            for key, name in zip(science_tests._PATH_KEYS, ("candidate", "baseline", "reference"))
        },
    )
    return {"release_dir": release, "template_dir": template, "context_file": context}


@pytest.fixture
def successful_workflow(monkeypatch):
    calls = []

    def execute(args, replacements):
        from simtools.runners.simtools_runner import _copy_collection_files

        calls.append(replacements["__SCIENCE_SITE_KEY__"])
        output = Path(replacements["__SCIENCE_REPORT_PATH__"])
        output.mkdir(parents=True)
        (output / "metrics.json").write_text('{"count": 3}', encoding="utf-8")
        _copy_collection_files(
            [{"configuration": {"output_path": str(output)}}],
            {
                "output_path": replacements["__SCIENCE_COLLECTION_PATH__"],
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
    product_directory = root / "events"
    product.write_text("event", encoding="utf-8")
    product_directory.mkdir()
    payload = {
        "job_ids": ["job"],
        "metadata": {
            "state": "submitted",
            "expected_outputs": {"job": [str(product), str(product_directory)]},
        },
    }
    submission.write_text(json.dumps(payload), encoding="utf-8")
    definition = {"requires_completed_production": True}
    replacements = {"__SCIENCE_CANDIDATE_SITE_PATH__": str(root)}
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
        (
            {"release_label": "candidate"},
            {},
            "absolute path",
        ),
    ],
)
def test_invalid_release_context(release, context, message):
    with pytest.raises(ValueError, match=message):
        science_tests._validate_release_context(release, context)


def test_identical_release_context_paths_identifies_conflicting_entries():
    context = {
        "__SCIENCE_CANDIDATE_PATH__": "/data/production",
        "__SCIENCE_BASELINE_PATH__": "/data/production",
        "__PRODUCTION_CONFIGURATION_PATH__": "/data/configuration",
    }

    with pytest.raises(ValueError, match="__SCIENCE_CANDIDATE_PATH__=/data/production") as error:
        science_tests._validate_release_context({"release_label": "candidate"}, context)

    assert "__SCIENCE_BASELINE_PATH__=/data/production" in str(error.value)


def test_identical_release_context_paths_are_allowed_when_distinctness_not_required():
    context = {
        "__SCIENCE_CANDIDATE_PATH__": "/data/production",
        "__SCIENCE_BASELINE_PATH__": "/data/production",
        "__PRODUCTION_CONFIGURATION_PATH__": "/data/configuration",
    }

    science_tests._validate_release_context(
        {"release_label": "candidate"}, context, require_distinct_paths=False
    )


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


@pytest.mark.parametrize("configuration_file", ["workflow.yml", "run_time.yml"])
def test_configuration_change_invalidates_old_results(
    campaign, successful_workflow, configuration_file
):
    science_tests.run_release(**campaign, sites=["north"])
    path = campaign["template_dir"] / configuration_file
    if configuration_file == "run_time.yml":
        _write_yaml(path, {"runtime_environment": {"image": "test"}})
    else:
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
    with pytest.raises(RuntimeError, match="Missing or empty required product"):
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
    with pytest.raises(ValueError, match="Production submission blocked") as error:
        science_tests._check_inputs(
            {"produces_production": True}, {"__SCIENCE_CANDIDATE_SITE_PATH__": str(root)}
        )

    assert str(root / "submission.json") in str(error.value)
    assert "--allow_production does not override it" in str(error.value)


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
    context = {"__SCIENCE_CANDIDATE_PATH__": str(root)}
    suite = {
        "array_layout_name": "CTAO-North-Alpha",
        "replacements": {"__SCIENCE_CANDIDATE_PATH__": "other"},
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
    contracts["tests"]["grid"] = ["__SCIENCE_REPORT_PATH__/metrics.json"]
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
    runtime = science_tests._shared_runtime_environment(
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
        science_tests._shared_runtime_environment(template, {})


@pytest.mark.parametrize(
    ("produces_production", "inline"), [(False, False), (True, False), (False, True)]
)
def test_run_test_uses_shared_runtime(campaign, monkeypatch, produces_production, inline):
    template = campaign["template_dir"]
    shared = {"container_engine": "apptainer", "image": "__SCIENCE_CONTAINER_IMAGE_PATH__"}
    _write_yaml(template / "run_time.yml", {"runtime_environment": shared})
    if inline:
        workflow = science_tests._load_yaml(template / "workflow.yml")
        workflow["runtime_environment"] = {"container_engine": "apptainer", "image": "inline.sif"}
        _write_yaml(template / "workflow.yml", workflow)
    captured = []
    prepare = science_tests.prepare_workflow

    def capture(args, replacements):
        result = prepare(args, replacements=replacements)
        captured.append(result[1])
        return result

    monkeypatch.setattr(science_tests, "prepare_workflow", capture)
    science_tests._run_test(
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


@pytest.mark.parametrize("dry_run", [False, True])
def test_science_test_output(campaign, successful_workflow, caplog, dry_run):
    with caplog.at_level(logging.INFO):
        science_tests.run_release(**campaign, sites=["north"], dry_run=dry_run)
    action = "Planned" if dry_run else "Running"
    assert caplog.messages.count(f"{action} science test: derive.north") == 1
    assert caplog.messages.count(f"{action} science test: compare.north") == 1
    assert caplog.messages.count("  Runtime: host (no container)") == 2
    assert len([message for message in caplog.messages if message.startswith("  Output:")]) == 2
    assert not any("Setting workflow output path" in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("runtime", "ignored", "expected"),
    [
        (None, False, "host (no container)"),
        (
            {"container_engine": "apptainer", "image": "/images/test.sif"},
            True,
            "host (no container)",
        ),
        (
            {"container_engine": "apptainer", "image": "/images/test.sif"},
            False,
            "apptainer (image: /images/test.sif)",
        ),
        ({"image": "test:latest"}, False, "docker (image: test:latest)"),
    ],
)
def test_runtime_description(runtime, ignored, expected):
    assert science_tests._runtime_description(runtime, ignored) == expected


@pytest.mark.parametrize("sites", [["north"], ["North"], ["NORTH"], ["nOrTh"], ["North", "NORTH"]])
def test_site_selection_is_case_insensitive(campaign, successful_workflow, sites):
    summary = science_tests.run_release(**campaign, sites=sites)
    assert successful_workflow == ["north", "north"]
    assert {
        result["site"] for result in summary["results"] if result["execution"] == "completed"
    } == {"north"}


@pytest.mark.parametrize("sites", [["SOUTH"], ["sOuTh"]])
def test_south_selection_is_case_insensitive(campaign, sites):
    summary = science_tests.run_release(**campaign, sites=sites, dry_run=True)
    assert {result["site"] for result in summary["results"] if result["status"] == "planned"} == {
        "south"
    }


def test_test_names_remain_case_sensitive(campaign, successful_workflow):
    with pytest.raises(ValueError, match="Invalid test selection"):
        science_tests.run_release(**campaign, sites=["North"], tests=["COMPARE"])
    assert successful_workflow == []


def test_unknown_mixed_case_site_is_rejected(campaign, successful_workflow):
    with pytest.raises(ValueError, match="Invalid site selection"):
        science_tests.run_release(**campaign, sites=["Unknown"])
    assert successful_workflow == []


def test_named_artifacts_on_rerun(campaign, successful_workflow, monkeypatch):
    args = {**campaign, "sites": ["north"], "tests": ["derive"]}
    science_tests.run_release(**args)
    report = campaign["release_dir"] / "reports/north/derive"
    previous = json.loads((report / "result.json").read_text())
    assert previous["execution_provenance"] == "work/science-tests/north/derive"
    context = science_tests._load_yaml(campaign["context_file"])
    work = Path(context["__SCIENCE_CANDIDATE_PATH__"]) / previous["execution_provenance"]
    stale = [report / "stale.txt", work / "output/stale.txt"]
    for path in stale:
        path.touch()
    science_tests.run_release(**args, dry_run=True, overwrite=True)
    assert all(path.exists() for path in stale)
    assert json.loads((report / "result.json").read_text()) == previous
    science_tests.run_release(**args)
    current = json.loads((report / "result.json").read_text())
    assert current["run_id"] != previous["run_id"]
    assert current["run_id"] in (report / "summary.md").read_text()
    assert not any(path.exists() for path in stale)
    monkeypatch.setattr(science_tests, "run_applications", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="Science tests failed"):
        science_tests.run_release(**args)
    assert json.loads((report / "result.json").read_text())["status"] == "incomplete"
    assert not (report / "metrics.json").exists()


def test_campaign_model_settings(campaign, monkeypatch):
    context = science_tests._load_yaml(campaign["context_file"])
    context["simulation_models_git_path"] = str(campaign["template_dir"] / "models")
    _write_yaml(campaign["context_file"], context)
    release_file = campaign["release_dir"] / "release.yml"
    release = science_tests._load_yaml(release_file)
    release["simulation_models_git_revision"] = "v0.18.0"
    _write_yaml(release_file, release)
    captured = []
    prepare = science_tests.prepare_workflow

    def capture(args, replacements):
        captured.append(
            (args["simulation_models_git_path"], args["simulation_models_git_revision"])
        )
        return prepare(args, replacements=replacements)

    monkeypatch.setattr(science_tests, "prepare_workflow", capture)
    science_tests.run_release(
        **campaign,
        sites=["north"],
        dry_run=True,
        application_args={"simulation_models_git_revision": "main"},
    )
    assert captured == [(context["simulation_models_git_path"], "v0.18.0")] * 2


@pytest.mark.parametrize(
    ("state", "finished"),
    [("failed", True), ("failed", False), ("completed", True), ("submitted", False)],
)
def test_production_retry_requires_finished_failed_jobs(mocker, state, finished):
    mocker.patch.object(
        science_tests,
        "load_submission",
        return_value=mocker.Mock(metadata={"state": state}, backend="htcondor"),
    )
    backend = mocker.patch.object(science_tests, "get_backend").return_value
    backend.is_finished.return_value = finished
    if state == "failed" and finished:
        science_tests._validate_production_retry(Path("submission.json"))
    else:
        with pytest.raises(ValueError, match="Cannot overwrite"):
            science_tests._validate_production_retry(Path("submission.json"))


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
    validate = mocker.patch.object(science_tests, "_validate_production_retry")
    science_tests._prepare_test_retry(
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

    mocker.patch.object(science_tests, "collect_submission", side_effect=collect)
    run = mocker.patch.object(science_tests, "run_applications")
    result = science_tests._execute_test_workflow(
        {"collect_production": True, "products": [str(product)]},
        {},
        {"__SCIENCE_CANDIDATE_SITE_PATH__": str(root)},
    )
    assert result["status"] == ("pending" if pending else "pass")
    run.assert_not_called()


def test_submission_does_not_validate_completed_production(mocker):
    run = mocker.patch.object(science_tests, "run_applications")
    mocker.patch.object(science_tests, "_evaluate_products", return_value={"status": "pass"})
    result = science_tests._execute_test_workflow({"produces_production": True}, {}, {})
    assert result["status"] == result["execution"] == "submitted"
    run.assert_called_once()


def test_pending_collection_defers_dependent_tests(mocker):
    collect = {"id": "collect.north", "site": "north"}
    compare = {"id": "compare.north", "site": "north"}
    execute = mocker.patch.object(
        science_tests, "_run_test", return_value={**collect, "status": "pending"}
    )
    mocker.patch.object(science_tests, "_log_test_execution")
    results = science_tests._execute_selection(
        [(({},), collect), (({"depends_on": ["collect"]},), compare)], False, "run", {}
    )
    assert [result["status"] for result in results] == ["pending", "pending"]
    execute.assert_called_once()
