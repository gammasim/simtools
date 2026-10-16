import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from simtools.science_tests import definitions as science_test_definitions

_REASON_KEY = "reason"
_STATUS_KEY = "status"

_CONTEXT_FILE_NAME = "context.yml"
_SCIENCE_TESTS_ROOT_ENV = "SIMTOOLS_SCIENCE_TESTS_ROOT"
_STORAGE_ROOT_PLACEHOLDER = f"${{{_SCIENCE_TESTS_ROOT_ENV}}}"
_ENCODING = "utf-8"
_RELEASE_LABEL_KEY = "release_label"
_REPORT_PAGE_COLUMN = "report_page"


def test_science_test_config_schema_rejects_unknown_site_fields():
    with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
        science_test_definitions._validate_science_test_config(
            {
                "site": "north",
                "required_tests": ["compare"],
                "array_layout_name": "CTAO-North-Alpha",
                "unexpected": True,
            },
            science_test_definitions._SITE_SCHEMA_FILE,
        )


def test_htcondor_environment_schema_accepts_interaction_table_path():
    science_test_definitions._validate_science_test_config(
        {
            "SIMTOOLS_CORSIKA_PATH": "/workdir/simulation_software/corsika7",
            "SIMTOOLS_SIM_TELARRAY_PATH": "/workdir/simulation_software/sim_telarray",
            "SIMTOOLS_CORSIKA_INTERACTION_TABLE_PATH": (
                "/corsika7-interaction-tables/interaction-tables"
            ),
        },
        "htcondor_environment.schema.yml",
    )


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
    suite = yaml.safe_load(site_path.read_text(encoding=_ENCODING))
    suite["site"] = "south"
    site_path.write_text(yaml.safe_dump(suite), encoding=_ENCODING)
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
    inventory.write_text(json.dumps({"matched_pair_count": 2}), encoding=_ENCODING)
    definition = {"expected_pair_count": 2}
    replacements = {"__SCIENCE_REPORT_PATH__": str(report)}

    science_test_definitions._check_expected_pair_count(definition, replacements)
    inventory.write_text(json.dumps({"matched_pair_count": 1}), encoding=_ENCODING)
    with pytest.raises(ValueError, match="expected 2 matched pairs, observed 1"):
        science_test_definitions._check_expected_pair_count(definition, replacements)


def test_evaluate_products_checks_nested_metrics_and_advisory_limits(tmp_test_directory):
    report = Path(tmp_test_directory)
    metrics = report / "metrics.json"
    metrics.write_text(json.dumps({"results": [{"count": 3}]}), encoding=_ENCODING)
    definition = {
        "products": [str(metrics)],
        "rule": {
            "mode": "advisory",
            "metrics": [{"file": str(metrics), "key": "results.0.count", "minimum": 2}],
        },
    }
    assert science_test_definitions._evaluate_products(definition, {})[_STATUS_KEY] == "warn"
    metrics.write_text(json.dumps({"results": [{"count": 1}]}), encoding=_ENCODING)
    result = science_test_definitions._evaluate_products(definition, {})
    assert result[_STATUS_KEY] == "warn"
    assert any("Acceptance failed" in warning for warning in result["acceptance_warnings"])
    metrics.write_text(json.dumps({"results": [{}]}), encoding=_ENCODING)
    with pytest.raises(ValueError, match=r"Missing metric results\.0\.count"):
        science_test_definitions._evaluate_products(definition, {})
    metrics.write_text(json.dumps({"results": [{"count": True}]}), encoding=_ENCODING)
    with pytest.raises(ValueError, match=r"Invalid metric results\.0\.count"):
        science_test_definitions._evaluate_products(definition, {})


def test_csv_acceptance_warns_for_small_samples_and_fails_above_maximum(tmp_test_directory):
    report = Path(tmp_test_directory)
    metrics_file = report / "comparison_metrics.csv"
    metrics_file.write_text(
        f"plot,scope,metric,value,normalized_value,valid,reason,baseline_count,candidate_count,{_REPORT_PAGE_COLUMN}\ndistribution_core_distance,whole_array,ks,0.04,,True,ok,50,120,12\ndistribution_core_distance_LSTN,telescope_type,ks,0.02,,True,ok,500,510,14\n",
        encoding=_ENCODING,
    )
    metric = {
        "file": str(metrics_file),
        "key": "value",
        "format": "csv",
        "select": {
            "metric": "ks",
            "plot": ["distribution_core_distance", "distribution_core_distance_LST?"],
        },
        "minimum_entries": 100,
        "warning_maximum": 0.03,
        "maximum": 0.05,
    }
    definition = {"products": [str(metrics_file)], "rule": {"metrics": [metric]}}
    result = science_test_definitions._evaluate_products(definition, {})
    assert result[_STATUS_KEY] == "warn"
    assert "insufficient statistics" in result[_REASON_KEY]
    metrics_file.write_text(
        f"plot,scope,metric,value,normalized_value,valid,reason,baseline_count,candidate_count,{_REPORT_PAGE_COLUMN}\ndistribution_core_distance_LSTN,telescope_type,ks,0.06853146853146852,,True,ok,500,510,27\n",
        encoding=_ENCODING,
    )
    result = science_test_definitions._evaluate_products(definition, {})
    assert result[_STATUS_KEY] == "warn"
    assert "value=0.06853 exceeds 0.05 (PDF page 27)" in result[_REASON_KEY]


def test_release_acceptance_override_is_rejected(campaign):
    override = campaign["release_dir"] / "acceptance/overrides.yml"
    override.parent.mkdir(parents=True)
    override.write_text(yaml.safe_dump({"overrides": {"comparison": {}}}), encoding=_ENCODING)

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


@pytest.mark.parametrize("metric", [None, {}, {"file": "metrics.json"}, {"key": "count"}])
def test_metric_rules_reject_incomplete_selector(metric):
    with pytest.raises(ValueError, match="Each acceptance metric needs file and key"):
        science_test_definitions._validate_metric_rule(metric)


def test_completed_production_gate(tmp_test_directory):
    root = Path(tmp_test_directory)
    submission = root / "submission.json"
    product = root / "event.dat"
    product_directory = root / "events"
    product.write_text("event", encoding=_ENCODING)
    product_directory.mkdir()
    (product_directory / "data").write_bytes(b"events")
    payload = {
        "job_ids": ["job"],
        "metadata": {
            "state": "submitted",
            "expected_outputs": {"job": [str(product), str(product_directory)]},
        },
    }
    submission.write_text(json.dumps(payload), encoding=_ENCODING)
    definition = {"requires_completed_production": True}
    replacements = {"__SCIENCE_CANDIDATE_SITE_PATH__": str(root)}
    with pytest.raises(ValueError, match="not completed"):
        science_test_definitions._check_inputs(definition, replacements)
    payload["metadata"]["state"] = "completed"
    submission.write_text(json.dumps(payload), encoding=_ENCODING)
    science_test_definitions._check_inputs(definition, replacements)
    product.unlink()
    with pytest.raises(ValueError, match="missing expected outputs"):
        science_test_definitions._check_inputs(definition, replacements)


def test_required_inputs_must_exist_and_be_nonempty(tmp_test_directory):
    candidate = Path(tmp_test_directory) / "candidate"
    product = candidate / "nested/input.json"
    product.parent.mkdir(parents=True)
    product.write_text("{}", encoding=_ENCODING)
    definition = {"requires": ["__SCIENCE_CANDIDATE_SITE_PATH__/**/*.json"]}
    replacements = {"__SCIENCE_CANDIDATE_SITE_PATH__": str(candidate)}

    science_test_definitions._check_inputs(definition, replacements)
    product.write_text("", encoding=_ENCODING)
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
    path.write_text(contents, encoding=_ENCODING)
    with pytest.raises(ValueError, match="Expected a mapping"):
        science_test_definitions._load_yaml(path)


@pytest.mark.parametrize(
    ("release", "context", "message"),
    [
        ({_RELEASE_LABEL_KEY: "../invalid"}, {}, _RELEASE_LABEL_KEY),
        (
            {_RELEASE_LABEL_KEY: "candidate"},
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
        science_test_definitions._validate_release_context(
            {_RELEASE_LABEL_KEY: "candidate"}, context
        )

    assert "__SCIENCE_BASELINE_PATH__=/data/production" in str(error.value)


def test_nested_release_context_paths_are_rejected():
    context = {
        "__SCIENCE_CANDIDATE_PATH__": "/data/production/candidate",
        "__SCIENCE_BASELINE_PATH__": "/data/production",
        "__PRODUCTION_CONFIGURATION_PATH__": "/data/configuration",
    }

    with pytest.raises(ValueError, match="non-overlapping"):
        science_test_definitions._validate_release_context(
            {_RELEASE_LABEL_KEY: "candidate"}, context
        )


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


@pytest.mark.parametrize("layout", ["CTAO-North-Alpha", ["CTAO-North-Alpha"]])
def test_grid_configuration_accepts_single_array_layout(layout):
    row = {"run_number": 1, "array_layout_name": "CTAO-North-Alpha"}
    config = row | {"array_layout_name": layout}
    science_test_definitions._check_grid_configuration(row, {}, config)


@pytest.mark.parametrize(
    "layout", [[], ["CTAO-South-Alpha"], ["CTAO-North-Alpha", "CTAO-South-Alpha"]]
)
def test_grid_configuration_rejects_different_or_multiple_array_layouts(layout):
    row = {"run_number": 1, "array_layout_name": "CTAO-North-Alpha"}
    config = row | {"array_layout_name": layout}
    with pytest.raises(ValueError, match="array_layout_name"):
        science_test_definitions._check_grid_configuration(row, {}, config)


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
    (root / "submission.json").write_text(json.dumps(submission), encoding=_ENCODING)
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


def test_baseline_check_preserves_paths_containing_candidate_name(mocker):
    check = mocker.patch.object(science_test_definitions, "_check_reviewed_production")
    replacements = {
        "__SCIENCE_CANDIDATE_PATH__": "/data/v1",
        "__SCIENCE_BASELINE_PATH__": "/data/v10",
        "__SCIENCE_CANDIDATE_SITE_PATH__": "/data/v1/north",
        "__SCIENCE_BASELINE_SITE_PATH__": "/data/v10/north",
    }
    science_test_definitions._check_inputs(
        {"reviewed_grid": "grid.ecsv", "acceptance_rule": "events"}, replacements
    )
    baseline = check.call_args.args[1]
    assert baseline["__SCIENCE_CANDIDATE_PATH__"] == "/data/v10"
    assert baseline["__SCIENCE_CANDIDATE_SITE_PATH__"] == "/data/v10/north"
    assert baseline["__SCIENCE_BASELINE_PATH__"] == "/data/v10"


def test_load_context_resolves_storage_root_before_path_validation(tmp_test_directory, monkeypatch):
    root = Path(tmp_test_directory)
    storage = root / "shared storage"
    monkeypatch.setenv(_SCIENCE_TESTS_ROOT_ENV, f"{storage}/")
    context_file = root / _CONTEXT_FILE_NAME
    paths = dict(zip(science_test_definitions._PATH_KEYS, ("v0.39.0", "v0.37.0", "reference-data")))
    source = {key: f"{_STORAGE_ROOT_PLACEHOLDER}/{name}" for key, name in paths.items()}
    source["simulation_models_git_revision"] = "v0.18.0"
    source["optional_path"] = None
    context_file.write_text(yaml.safe_dump(source), encoding=_ENCODING)

    context = science_test_definitions._load_context(context_file)

    assert context == {**source, **{key: str(storage / name) for key, name in paths.items()}}
    science_test_definitions._validate_release_context(
        {_RELEASE_LABEL_KEY: "test-release"}, context
    )
    assert yaml.safe_load(context_file.read_text(encoding=_ENCODING)) == source
    assert not storage.exists()


@pytest.mark.parametrize("storage_root", [None, "", "relative/path"])
def test_load_context_rejects_missing_or_relative_storage_root(
    tmp_test_directory, monkeypatch, storage_root
):
    if storage_root is None:
        monkeypatch.delenv(_SCIENCE_TESTS_ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(_SCIENCE_TESTS_ROOT_ENV, storage_root)
    context_file = Path(tmp_test_directory) / _CONTEXT_FILE_NAME
    context_file.write_text(
        yaml.safe_dump(
            {science_test_definitions._PATH_KEYS[0]: f"{_STORAGE_ROOT_PLACEHOLDER}/candidate"}
        ),
        encoding=_ENCODING,
    )
    with pytest.raises(ValueError, match=f"Set {_SCIENCE_TESTS_ROOT_ENV}.*absolute path"):
        science_test_definitions._load_context(context_file)


def test_load_context_keeps_explicit_paths_without_storage_root(tmp_test_directory, monkeypatch):
    root = Path(tmp_test_directory)
    monkeypatch.delenv(_SCIENCE_TESTS_ROOT_ENV, raising=False)
    context_file = root / _CONTEXT_FILE_NAME
    context = {science_test_definitions._PATH_KEYS[0]: str(root / "absolute-candidate")}
    context_file.write_text(yaml.safe_dump(context), encoding=_ENCODING)
    assert science_test_definitions._load_context(context_file) == context
    assert science_test_definitions._load_context(None) == {}
