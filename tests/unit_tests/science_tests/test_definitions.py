import json
from pathlib import Path

import pytest
import yaml

from simtools.science_tests import definitions as science_test_definitions


def test_dependency_order_and_invalid_definitions():
    assert science_test_definitions._dependency_order(
        ["b", "a"], {"a": {}, "b": {"depends_on": ["a"]}}
    ) == [
        "a",
        "b",
    ]
    with pytest.raises(ValueError, match="Circular"):
        science_test_definitions._dependency_order(["a"], {"a": {"depends_on": ["a"]}})
    with pytest.raises(ValueError, match="Unknown"):
        science_test_definitions._dependency_order(["a"], {})
    with pytest.raises(ValueError, match="Invalid definition"):
        science_test_definitions._dependency_order(["a"], {"a": []})


@pytest.mark.parametrize(
    ("allow", "tests"), [(False, ["production"]), (True, None), (True, ["compare"])]
)
def test_production_must_be_explicit(allow, tests):
    with pytest.raises(ValueError, match="Select production explicitly"):
        science_test_definitions._check_production_selection(
            "production", {"tier": "release-prod"}, tests, allow, False
        )


def test_production_selection_allows_dry_runs_and_non_production_tests():
    science_test_definitions._check_production_selection(
        "production", {"tier": "release-prod"}, None, False, True
    )
    science_test_definitions._check_production_selection(
        "compare", {"tier": "release-light"}, None, False, False
    )


@pytest.mark.parametrize(
    ("test_id", "site", "definition", "message"),
    [
        ("bad/name", "north", {"sites": ["north"], "tier": "smoke", "workflow": "x"}, "ID"),
        ("test", "south", {"sites": ["north"], "tier": "smoke", "workflow": "x"}, "support"),
        ("test", "north", {"sites": ["north"], "tier": "large", "workflow": "x"}, "tier"),
        ("test", "north", {"sites": ["north"], "tier": "smoke"}, "workflow"),
    ],
)
def test_validate_definition_rejects_invalid_catalogue_entries(test_id, site, definition, message):
    with pytest.raises(ValueError, match=message):
        science_test_definitions._validate_definition(test_id, site, definition)


def test_load_suites_rejects_site_that_disagrees_with_its_file(campaign):
    site_path = campaign["release_dir"] / "sites/north.yml"
    suite = yaml.safe_load(site_path.read_text(encoding="utf-8"))
    suite["site"] = "south"
    site_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    release = science_test_definitions._load_yaml(campaign["release_dir"] / "release.yml")
    catalogue = science_test_definitions._load_catalogue(
        release, campaign["release_dir"], campaign["template_dir"]
    )

    with pytest.raises(ValueError, match="Invalid site or missing layout"):
        science_test_definitions._load_suites(campaign["release_dir"], release, catalogue)


def test_test_definition_applies_release_comparison_contract():
    test_id = "compare.events"
    definition = science_test_definitions._test_definition(
        test_id,
        {test_id: {"acceptance_rule": "limits"}},
        {"products": {test_id: ["metrics.json"]}, "rules": {"limits": {"mode": "advisory"}}},
        {
            "expected_changes": ["configuration.azimuth_angle"],
            "comparison_parameters": {"events": {"compare_by": ["configuration.azimuth_angle"]}},
        },
    )

    assert definition["comparison_parameters"] == {"compare_by": ["configuration.azimuth_angle"]}
    assert definition["expected_changes"] == ["configuration.azimuth_angle"]
    assert definition["rule"]["mode"] == "advisory"


def test_expected_pair_count_blocks_an_incomplete_comparison(tmp_test_directory):
    report = Path(tmp_test_directory)
    inventory = report / "comparison_inventory.json"
    inventory.write_text(json.dumps({"matched_pair_count": 2}), encoding="utf-8")
    definition = {"expected_pair_count": 2}
    replacements = {"__SCIENCE_REPORT_PATH__": str(report)}

    science_test_definitions._check_expected_pair_count(definition, replacements)
    inventory.write_text(json.dumps({"matched_pair_count": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 2 matched pairs, observed 1"):
        science_test_definitions._check_expected_pair_count(definition, replacements)


def test_evaluate_products_checks_nested_metrics_and_advisory_limits(tmp_test_directory):
    report = Path(tmp_test_directory)
    metrics = report / "metrics.json"
    metrics.write_text(json.dumps({"results": [{"count": 3}]}), encoding="utf-8")
    definition = {
        "products": [str(metrics)],
        "rule": {
            "mode": "advisory",
            "metrics": [{"file": str(metrics), "key": "results.0.count", "minimum": 2}],
        },
    }

    assert science_test_definitions._evaluate_products(definition, {})["status"] == "warn"
    metrics.write_text(json.dumps({"results": [{"count": 1}]}), encoding="utf-8")
    with pytest.raises(science_test_definitions._AcceptanceError, match="Acceptance failed"):
        science_test_definitions._evaluate_products(definition, {})
    metrics.write_text(json.dumps({"results": [{}]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Missing metric results.0.count"):
        science_test_definitions._evaluate_products(definition, {})
    metrics.write_text(json.dumps({"results": [{"count": True}]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Invalid metric results.0.count"):
        science_test_definitions._evaluate_products(definition, {})


def test_release_acceptance_override_is_rejected(campaign):
    override = campaign["release_dir"] / "acceptance/overrides.yml"
    override.parent.mkdir(parents=True)
    override.write_text(yaml.safe_dump({"overrides": {"comparison": {}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="overrides are not supported"):
        science_test_definitions._load_acceptance(campaign["release_dir"], campaign["template_dir"])


@pytest.mark.parametrize("workflow", ["/outside.yml", "../outside.yml"])
def test_workflow_path_cannot_escape_template(campaign, workflow):
    with pytest.raises(ValueError, match="relative template path"):
        science_test_definitions._resolve_workflow(
            campaign["template_dir"], campaign["release_dir"], workflow
        )


@pytest.mark.parametrize("value", [True, float("nan")])
def test_metric_rules_reject_non_numeric_or_non_finite_limits(value):
    with pytest.raises(ValueError, match="Invalid acceptance minimum"):
        science_test_definitions._validate_metric_rule(
            {"file": "metrics.json", "key": "count", "minimum": value}
        )


def test_completed_production_gate(tmp_test_directory):
    root = Path(tmp_test_directory)
    submission = root / "submission.json"
    product = root / "event.dat"
    product_directory = root / "events"
    product.write_text("event", encoding="utf-8")
    product_directory.mkdir()
    (product_directory / "data").write_bytes(b"events")
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
        science_test_definitions._check_inputs(definition, replacements)
    payload["metadata"]["state"] = "completed"
    submission.write_text(json.dumps(payload), encoding="utf-8")
    science_test_definitions._check_inputs(definition, replacements)
    product.unlink()
    with pytest.raises(ValueError, match="missing expected outputs"):
        science_test_definitions._check_inputs(definition, replacements)


def test_required_inputs_must_exist_and_be_nonempty(tmp_test_directory):
    candidate = Path(tmp_test_directory) / "candidate"
    product = candidate / "nested/input.json"
    product.parent.mkdir(parents=True)
    product.write_text("{}", encoding="utf-8")
    definition = {"requires": ["__SCIENCE_CANDIDATE_SITE_PATH__/**/*.json"]}
    replacements = {"__SCIENCE_CANDIDATE_SITE_PATH__": str(candidate)}

    science_test_definitions._check_inputs(definition, replacements)
    product.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing or empty required product"):
        science_test_definitions._check_inputs(definition, replacements)


@pytest.mark.parametrize("expected", [True, 0])
def test_expected_pair_count_must_be_a_positive_integer(expected, tmp_test_directory):
    with pytest.raises(ValueError, match="positive integer"):
        science_test_definitions._check_expected_pair_count(
            {"expected_pair_count": expected}, {"__SCIENCE_REPORT_PATH__": tmp_test_directory}
        )


@pytest.mark.parametrize("contents", ["- item\n", ""])
def test_yaml_loader_requires_a_mapping(contents, tmp_test_directory):
    path = Path(tmp_test_directory) / "invalid.yml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a mapping"):
        science_test_definitions._load_yaml(path)


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
        science_test_definitions._validate_release_context(release, context)


def test_identical_release_context_paths_identifies_conflicting_entries():
    context = {
        "__SCIENCE_CANDIDATE_PATH__": "/data/production",
        "__SCIENCE_BASELINE_PATH__": "/data/production",
        "__PRODUCTION_CONFIGURATION_PATH__": "/data/configuration",
    }

    with pytest.raises(ValueError, match="__SCIENCE_CANDIDATE_PATH__=/data/production") as error:
        science_test_definitions._validate_release_context({"release_label": "candidate"}, context)

    assert "__SCIENCE_BASELINE_PATH__=/data/production" in str(error.value)


def test_nested_release_context_paths_are_rejected():
    context = {
        "__SCIENCE_CANDIDATE_PATH__": "/data/production/candidate",
        "__SCIENCE_BASELINE_PATH__": "/data/production",
        "__PRODUCTION_CONFIGURATION_PATH__": "/data/configuration",
    }

    with pytest.raises(ValueError, match="non-overlapping"):
        science_test_definitions._validate_release_context({"release_label": "candidate"}, context)


def test_site_cannot_override_context_root(tmp_test_directory):
    root = Path(tmp_test_directory)
    context = {"__SCIENCE_CANDIDATE_PATH__": str(root)}
    suite = {
        "array_layout_name": "CTAO-North-Alpha",
        "replacements": {"__SCIENCE_CANDIDATE_PATH__": "other"},
    }
    with pytest.raises(ValueError, match="reserved context key"):
        science_test_definitions._site_replacements(context, "north", suite, root)


@pytest.mark.parametrize(
    "observed", [["job-000001"], ["job-000001", "job-000001"], ["job-000001", "job-000003"]]
)
def test_reviewed_ids_reject_missing_extra_and_duplicate_jobs(observed):
    with pytest.raises(ValueError, match="Incomplete production jobs"):
        science_test_definitions._require_same_ids(
            {"job-000001", "job-000002"}, observed, "production jobs"
        )


def test_grid_configuration_checks_quantities_and_each_run():
    import astropy.units as u

    row = {"run_number": 2, "energy_min": 1 * u.TeV, "model_version": "7.0.0"}
    config = {
        "run_number": 2,
        "energy_min": {"value": 1000, "unit": "GeV"},
        "model_version": ["7.0.0"],
    }
    science_test_definitions._check_grid_configuration(row, {}, config)
    with pytest.raises(ValueError, match="run_number"):
        science_test_definitions._check_grid_configuration(row, {}, config | {"run_number": 1})


def test_reviewed_grid_rejects_missing_whole_submission(mocker, tmp_test_directory):
    from simtools.production_configuration.production_file_selection import ProductionManifest

    root = Path(tmp_test_directory)
    (root / "events.hdf5").write_bytes(b"events")
    manifest = ProductionManifest(
        root / "metadata.yml",
        {
            "job_id": "job-000001",
            "configuration": {"run_number": 1},
            "files": {"reduced_event_data": ["events.hdf5"]},
        },
    )
    mocker.patch.object(
        science_test_definitions, "read_job_grid", return_value=([{"run_number": 1}], {})
    )
    mocker.patch.object(science_test_definitions, "discover_manifests", return_value=[manifest])
    submission = {
        "job_ids": ["job-000001"],
        "metadata": {
            "state": "completed",
            "expected_outputs": {"job-000001": [str(root / "events.hdf5")]},
        },
    }
    (root / "submission.json").write_text(json.dumps(submission), encoding="utf-8")
    definition = {"reviewed_grid": "grid.ecsv"}
    replacements = {"__SCIENCE_CANDIDATE_SITE_PATH__": str(root)}
    assert science_test_definitions._check_reviewed_production(definition, replacements) == {
        "expected_jobs": 1,
        "observed_jobs": 1,
    }
    mocker.patch.object(
        science_test_definitions,
        "read_job_grid",
        return_value=([{"run_number": 1}, {"run_number": 2}], {}),
    )
    with pytest.raises(ValueError, match="job-000002"):
        science_test_definitions._check_reviewed_production(definition, replacements)
