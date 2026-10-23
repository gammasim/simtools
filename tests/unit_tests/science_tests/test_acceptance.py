import csv
import json
from pathlib import Path

import pytest

from simtools.science_tests import acceptance

LSTN_01 = "LSTN-01"
MISSING_OR_EMPTY = "Missing or empty"
ADVISORY = "advisory"
CHANGES = "changes"
COMPARISONS = "comparisons"
COMPLETE = "complete"
DATA_TXT = "data.txt"
RESULT_STATUS_FILE = "result_status_file"

_REASON_KEY = "reason"

_VALID = "valid"

_ACCEPTANCE_WARNINGS_KEY = "acceptance_warnings"
_ARRAY = "array"
_COMPARISON_INVENTORY_FILE = "comparison_inventory.json"
_COMPARISON_METRICS_FILE = "comparison_metrics.csv"
_LST_CORE_PLOT_PATTERN = "distribution_core_distance_LST?"
_FORMAT = "format"
_MAXIMUM = "maximum"
_METRIC = "metric"
_METRICS = "metrics"
_POPULATION = "population"
_PRODUCTS = "products"
_SCOPE_KEY = "scope"
_SELECT_KEY = "select"
_STATUS_KEY = "status"
_TELESCOPE_TYPE_KEY = "telescope_type"
_TRIGGERED_POPULATION = "triggered"
_VALUE = "value"
_ENCODING = "utf-8"
_REPORT_PAGE_COLUMN = "report_page"


@pytest.fixture
def scalar_comparison(tmp_test_directory):
    directory = Path(tmp_test_directory) / "acceptance"
    directory.mkdir()
    path = directory / _COMPARISON_METRICS_FILE
    scopes = {pair: {_ARRAY: ["LSTN", "MSTN"]} for pair in ("first", "second")}
    (directory / _COMPARISON_INVENTORY_FILE).write_text(
        json.dumps({"metric_scopes": scopes}), encoding=_ENCODING
    )
    rows = [
        {
            "pair": pair,
            "array_layout": _ARRAY,
            _SCOPE_KEY: _TELESCOPE_TYPE_KEY,
            _TELESCOPE_TYPE_KEY: telescope,
            "plot": f"distribution_core_distance_{telescope}",
            _POPULATION: _TRIGGERED_POPULATION,
            _METRIC: "ks",
            _VALUE: "0.01",
            _VALID: "true",
            _REASON_KEY: "ok",
            "baseline_count": "500",
            "candidate_count": "500",
            _REPORT_PAGE_COLUMN: str(index + 1),
        }
        for index, (pair, telescope) in enumerate(
            (pair, telescope) for pair in scopes for telescope in ("LSTN", "MSTN")
        )
    ]

    def write(selected):
        with path.open("w", encoding=_ENCODING, newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
        return path

    metric = {
        "key": _VALUE,
        _FORMAT: "csv",
        _MAXIMUM: 0.05,
        _SELECT_KEY: {
            _SCOPE_KEY: _TELESCOPE_TYPE_KEY,
            "plot": [_LST_CORE_PLOT_PATTERN, "distribution_core_distance_MST?"],
            _POPULATION: _TRIGGERED_POPULATION,
            _METRIC: "ks",
        },
    }
    return rows, write, metric


@pytest.mark.parametrize("missing", [[0], [0, 2], [0, 1, 2, 3]])
def test_csv_acceptance_warns_for_missing_configurations_and_types(scalar_comparison, missing):
    rows, write, metric = scalar_comparison
    path = write([row for index, row in enumerate(rows) if index not in missing])
    warnings = acceptance._evaluate_csv_metric(path, metric)
    assert warnings
    if len(missing) < len(rows):
        assert len(warnings) == len(missing)
        assert all("Missing comparison metric rows" in warning for warning in warnings)


def test_csv_acceptance_rejects_duplicate_measurements(scalar_comparison):
    rows, write, metric = scalar_comparison
    path = write([*rows, rows[0] | {_VALUE: "0.09"}])
    with pytest.raises(ValueError, match="Duplicate comparison metric rows"):
        acceptance._evaluate_csv_metric(path, metric)


def test_csv_acceptance_reports_every_deviation(scalar_comparison):
    rows, write, metric = scalar_comparison
    path = write([row | {_VALUE: "0.09"} for row in rows])
    result = acceptance.evaluate_products(
        {_PRODUCTS: [str(path)], "rule": {_METRICS: [metric | {"file": str(path)}]}}, {}
    )
    assert result[_STATUS_KEY] == "warn"
    assert len(result[_ACCEPTANCE_WARNINGS_KEY]) == len(rows)
    for index, warning in enumerate(result[_ACCEPTANCE_WARNINGS_KEY]):
        assert "=0.09: outside [-inf, 0.05]" in warning
        assert f"PDF page {index + 1}" in warning


def test_csv_acceptance_does_not_require_other_telescope_types(scalar_comparison):
    rows, write, metric = scalar_comparison
    path = write([row for row in rows if row[_TELESCOPE_TYPE_KEY] == "LSTN"])
    metric[_SELECT_KEY]["plot"] = _LST_CORE_PLOT_PATTERN
    assert acceptance._evaluate_csv_metric(path, metric) == []


@pytest.mark.parametrize(
    ("updates", "limits", "message"),
    [
        ({_VALID: "false", _REASON_KEY: "empty"}, {}, "invalid statistic"),
        ({_VALUE: "0.01"}, {"minimum": 0.02}, "outside [0.02, 0.05]"),
        ({_VALUE: "0.04"}, {"warning_maximum": 0.03}, "warning limit"),
    ],
)
def test_csv_acceptance_collects_invalid_and_boundary_warnings(
    scalar_comparison, updates, limits, message
):
    rows, write, metric = scalar_comparison
    path = write([row | updates for row in rows])
    warnings = acceptance._evaluate_csv_metric(path, metric | limits)
    assert len(warnings) == len(rows)
    assert all(message in warning for warning in warnings)


@pytest.mark.parametrize(_VALUE, ["nan", "malformed"])
def test_csv_acceptance_rejects_unusable_values(scalar_comparison, value):
    rows, write, metric = scalar_comparison
    path = write([row | {_VALUE: value} for row in rows])
    with pytest.raises(ValueError, match="Invalid acceptance value"):
        acceptance._evaluate_csv_metric(path, metric)


def test_csv_acceptance_without_matching_rows_or_inventory(scalar_comparison):
    rows, write, metric = scalar_comparison
    path = write(rows)
    path.with_name(_COMPARISON_INVENTORY_FILE).unlink()
    metric[_SELECT_KEY]["plot"] = "other"
    warnings = acceptance._evaluate_csv_metric(path, metric)
    assert warnings == [f"No comparison rows match {metric[_SELECT_KEY]} in {path}."]
    insufficient = acceptance._evaluate_csv_metric(
        path, metric | {_SELECT_KEY: {}, "minimum_entries": 1000}
    )
    assert all("insufficient statistics" in warning for warning in insufficient)


def _write_json(directory, name, payload):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding=_ENCODING)
    return path


def test_json_metrics_and_advisory_rules(tmp_test_directory):
    path = _write_json(tmp_test_directory, "json/metrics.json", {"a": {"b": [1, 5]}})
    rule = {"mode": ADVISORY, _METRICS: [{"file": str(path), "key": "a.b.1", _MAXIMUM: 4}]}

    result = acceptance.evaluate_products({_PRODUCTS: [str(path)], "rule": rule}, {})

    assert result[_STATUS_KEY] == "warn"
    assert "a.b.1=5: outside [-inf, 4]" in result[_ACCEPTANCE_WARNINGS_KEY][0]
    assert ADVISORY in result[_ACCEPTANCE_WARNINGS_KEY][1]
    assert acceptance.evaluate_products({_PRODUCTS: [str(path)]}, {}) == {_STATUS_KEY: "pass"}
    for key, message in (("a.c", "Missing metric"), ("a", "Invalid metric")):
        with pytest.raises(ValueError, match=message):
            acceptance._evaluate_metric(path, {"key": key})


def test_matched_files_requires_nonempty_products(tmp_test_directory):
    directory = Path(tmp_test_directory) / _PRODUCTS
    directory.mkdir()
    (directory / "empty.txt").touch()
    with pytest.raises(ValueError, match=MISSING_OR_EMPTY):
        acceptance.matched_files(str(directory / "*.txt"))
    with pytest.raises(ValueError, match=MISSING_OR_EMPTY):
        acceptance.matched_files(str(directory / "none.txt"))
    assert not acceptance.nonempty(directory)
    (directory / DATA_TXT).write_text("x", encoding=_ENCODING)
    assert acceptance.nonempty(directory)
    assert acceptance.nonempty(directory / DATA_TXT)


def test_result_status_file_requires_complete_comparisons(tmp_test_directory):
    def evaluate(payload):
        path = _write_json(tmp_test_directory, "status/result.json", payload)
        return acceptance.evaluate_products({RESULT_STATUS_FILE: str(path)}, {})

    complete = {_STATUS_KEY: COMPLETE, COMPARISONS: {LSTN_01: {CHANGES: {"x": {}}}}}
    assert evaluate(complete) == {_STATUS_KEY: "pass"}
    complete[COMPARISONS][LSTN_01][CHANGES]["x"][ADVISORY] = True
    assert "advisory tolerance exceeded for x" in evaluate(complete)[_REASON_KEY]
    with pytest.raises(ValueError, match="is incomplete"):
        evaluate({_STATUS_KEY: "incomplete"})
    with pytest.raises(ValueError, match="no metric changes"):
        evaluate({_STATUS_KEY: COMPLETE, COMPARISONS: {LSTN_01: {CHANGES: {}}}})
    with pytest.raises(ValueError, match="Cannot read result status file"):
        acceptance.evaluate_products({RESULT_STATUS_FILE: "/missing.json"}, {})
