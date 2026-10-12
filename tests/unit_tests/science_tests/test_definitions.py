import json
from pathlib import Path

import pytest

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
        science_test_definitions._check_inputs(definition, replacements)
    payload["metadata"]["state"] = "completed"
    submission.write_text(json.dumps(payload), encoding="utf-8")
    science_test_definitions._check_inputs(definition, replacements)
    product.unlink()
    with pytest.raises(ValueError, match="missing expected outputs"):
        science_test_definitions._check_inputs(definition, replacements)


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


def test_non_comparison_context_does_not_require_a_baseline():
    context = {
        "__SCIENCE_CANDIDATE_PATH__": "/data/production",
        "__PRODUCTION_CONFIGURATION_PATH__": "/data/configuration",
    }

    science_test_definitions._validate_release_context(
        {"release_label": "candidate"}, context, require_baseline=False
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


def test_report_and_collection_share_output_directory(tmp_test_directory):
    root = Path(tmp_test_directory)
    context = {key: str(root) for key in science_test_definitions._PATH_KEYS}
    site_config = {"array_layout_name": "CTAO-North-Alpha"}

    replacements = science_test_definitions._site_replacements(context, "north", site_config, root)

    assert replacements["__SCIENCE_COLLECTION_PATH__"] == replacements["__SCIENCE_REPORT_PATH__"]
    assert replacements["__SCIENCE_REPORT_PATH__"] == str(root)
