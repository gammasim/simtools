import json
import logging
from pathlib import Path

import pytest
import yaml

from simtools.science_tests import reports as science_test_reports

_RUN_1 = "run-1"
_RUN_ID = "run_id"

_ACCEPTANCE_WARNINGS_KEY = "acceptance_warnings"
_ACCEPTANCE_REPORT_KEY = "acceptance_report"
_PRODUCTION_HISTORY = "production_history"
_RELEASE = "release"
_REPORTS_DIRECTORY = "reports"

DERIVE_TEST = "derive"
COMPUTING_TEST = "computing"
CONTAINER_RUNTIME = "container"
REQUIRED_TESTS_KEY = "required_tests"
COMPLETED_EXECUTION = "completed"
EXECUTION_KEY = "execution"

REPORT_KEY = "report"
STATUS_KEY = "status"

SUMMARY_PATH = "reports/release-summary.json"
CAMPAIGN_DIGEST_KEY = "campaign_sha256"
UNCHANGED_SIGNATURE = "unchanged"
RESULTS_KEY = "results"
RELEASE_LABEL = "candidate"
RELEASE_LABEL_KEY = "release_label"
SITE = "north"
COMPARE_TEST = "compare"
COMPARE_ID = "compare.north"
QUALIFIED_KEY = "qualified"
TEXT_ENCODING = "utf-8"


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding=TEXT_ENCODING)


def test_acceptance_deviations_are_published_once_and_linked(tmp_test_directory):
    root = Path(tmp_test_directory)
    work = root / "work"
    report = root / _REPORTS_DIRECTORY / SITE / COMPARE_TEST
    result = {
        "id": COMPARE_ID,
        "test": COMPARE_TEST,
        "site": SITE,
        STATUS_KEY: "warn",
        EXECUTION_KEY: COMPLETED_EXECUTION,
        _ACCEPTANCE_WARNINGS_KEY: [
            "core whole-array configuration-a: value=0.09 exceeds 0.05 (PDF page 2).",
            "multiplicity LSTN configuration-b: value=0.08 exceeds 0.05 (PDF page 5).",
        ],
    }
    science_test_reports._write_test_report(work, report, root, result)
    deviations = (report / "acceptance.md").read_text(encoding=TEXT_ENCODING)
    assert "configuration-a" in deviations
    assert "configuration-b" in deviations
    assert "trigger-report.pdf#page=2" in deviations
    assert "trigger-report.pdf#page=5" in deviations
    assert _ACCEPTANCE_WARNINGS_KEY not in result
    assert len(list(report.iterdir())) == 1
    science_test_reports._write_summary(
        root / _REPORTS_DIRECTORY,
        {
            _RELEASE: RELEASE_LABEL,
            QUALIFIED_KEY: False,
            RESULTS_KEY: [result],
            _PRODUCTION_HISTORY: [],
        },
    )
    summary = (root / "reports/release-summary.md").read_text(encoding=TEXT_ENCODING)
    assert "[deviations](north/compare/acceptance.md)" in summary


def test_reused_report_removes_stale_acceptance_deviations(tmp_test_directory):
    root = Path(tmp_test_directory)
    work = root / "work"
    work.mkdir()
    (work / "acceptance.md").write_text("stale warning\n", encoding=TEXT_ENCODING)
    (work / "metrics.json").write_text("{}", encoding=TEXT_ENCODING)
    report = root / _REPORTS_DIRECTORY / SITE / COMPARE_TEST
    result = {
        STATUS_KEY: "pass",
        _ACCEPTANCE_REPORT_KEY: "reports/north/compare/acceptance.md",
    }

    science_test_reports._write_test_report(work, report, root, result)

    assert not (work / "acceptance.md").exists()
    assert not (report / "acceptance.md").exists()
    assert _ACCEPTANCE_REPORT_KEY not in result


def test_production_history_survives_a_non_production_configuration_change(campaign):
    catalogue_path = campaign["template_dir"] / "catalogue.yml"
    catalogue = yaml.safe_load(catalogue_path.read_text(encoding=TEXT_ENCODING))
    catalogue["tests"]["production"] = {"affects_production": True}
    _write_yaml(catalogue_path, catalogue)
    report_path = campaign["release_dir"] / SUMMARY_PATH
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                CAMPAIGN_DIGEST_KEY: "old",
                RESULTS_KEY: [],
                _PRODUCTION_HISTORY: [
                    {
                        "id": "production.north",
                        "test": "production",
                        "site": SITE,
                        EXECUTION_KEY: "submitted",
                        STATUS_KEY: "submitted",
                        _RUN_ID: _RUN_1,
                    }
                ],
            }
        ),
        encoding=TEXT_ENCODING,
    )

    summary = science_test_reports._release_summary(
        campaign["release_dir"],
        {RELEASE_LABEL_KEY: RELEASE_LABEL},
        {SITE: {REQUIRED_TESTS_KEY: [COMPARE_TEST]}},
        [],
        False,
        "new",
        catalogue["tests"],
    )

    assert summary[_PRODUCTION_HISTORY][0][_RUN_ID] == _RUN_1


def test_write_summary_includes_failures_and_production_history(tmp_test_directory):
    report_dir = Path(tmp_test_directory)
    summary = {
        _RELEASE: RELEASE_LABEL,
        QUALIFIED_KEY: False,
        RESULTS_KEY: [
            {
                "site": SITE,
                "test": COMPARE_TEST,
                EXECUTION_KEY: "failed",
                STATUS_KEY: "incomplete",
                "error": "missing report",
            }
        ],
        _PRODUCTION_HISTORY: [
            {
                "id": "production.north",
                EXECUTION_KEY: "submitted",
                STATUS_KEY: "submitted",
                _RUN_ID: _RUN_1,
            }
        ],
    }

    science_test_reports._write_summary(report_dir, summary)

    markdown = (report_dir / "release-summary.md").read_text(encoding=TEXT_ENCODING)
    assert "incomplete | missing report" in markdown
    assert "Production history" in markdown
    assert _RUN_1 in markdown


def test_publication_filters_evidence_replaces_stale_output_and_warns_when_large(
    tmp_test_directory, monkeypatch, caplog
):
    root = Path(tmp_test_directory)
    work = root / "work"
    (work / "pair").mkdir(parents=True)
    unused = Path("unused/layout")
    (work / unused).mkdir(parents=True)
    (work / "pair/metrics.json").write_text("{}", encoding=TEXT_ENCODING)
    provenance = work / "provenance"
    provenance.mkdir()
    (provenance / "effective-settings.json").write_text("{}", encoding=TEXT_ENCODING)
    (provenance / "resolved-workflow.yml").write_text("workflow: {}\n", encoding=TEXT_ENCODING)
    (work / "inventory.json").write_text("[]", encoding=TEXT_ENCODING)
    (work / "comparison_inventory.json").write_text(
        '{"matched_pair_count": 1}', encoding=TEXT_ENCODING
    )
    release_dir = root / _RELEASE
    report_root = release_dir / "reports/north/compare"
    report_root.mkdir(parents=True)
    (report_root / "stale.json").touch()
    result = {STATUS_KEY: "pass"}
    monkeypatch.setattr(science_test_reports, "_PUBLICATION_WARNING_FILES", 0)

    with caplog.at_level(logging.WARNING):
        science_test_reports._write_test_report(work, report_root, release_dir, result)

    assert (report_root / "pair/metrics.json").read_text() == "{}"
    assert not (report_root / "stale.json").exists()
    assert result["publication"]["warning"]
    assert "Science-test publication contains" in caplog.text
    assert not (report_root / "provenance/effective-settings.json").exists()
    assert (report_root / "provenance/resolved-workflow.yml").is_file()
    assert not (report_root / "inventory.json").exists()
    assert (report_root / "comparison_inventory.json").is_file()
    assert not (report_root / unused).exists()


@pytest.mark.parametrize("has_work", [False, True])
def test_success_without_report_products_does_not_create_directories(tmp_test_directory, has_work):
    root = Path(tmp_test_directory)
    work = root / "work"
    if has_work:
        (work / "empty").mkdir(parents=True)
        (work / "execution.log").write_text("execution log", encoding=TEXT_ENCODING)
    report_root = root / "reports/test"
    result = {STATUS_KEY: "pass"}
    science_test_reports._write_test_report(work, report_root, root, result)
    assert result[REPORT_KEY] is None
    assert not report_root.exists()
    assert not report_root.with_name(f".{report_root.name}.new").exists()


def test_separate_steps_preserve_results_across_different_runtimes(tmp_test_directory):
    root = Path(tmp_test_directory)
    path = root / SUMMARY_PATH
    path.parent.mkdir()
    catalogue = {DERIVE_TEST: {}, COMPARE_TEST: {"depends_on": [DERIVE_TEST]}, COMPUTING_TEST: {}}
    completed = {}
    for test, runtime, status in (
        (DERIVE_TEST, CONTAINER_RUNTIME, "pass"),
        (COMPARE_TEST, "host", "warn"),
        (COMPUTING_TEST, "host", "pass"),
        (COMPARE_TEST, CONTAINER_RUNTIME, "pass"),
    ):
        result = {
            "id": f"{test}.{SITE}",
            "test": test,
            "site": SITE,
            "runtime": runtime,
            STATUS_KEY: status,
            EXECUTION_KEY: COMPLETED_EXECUTION,
        }
        completed[result["id"]] = dict(result)
        summary = science_test_reports._release_summary(
            root,
            {RELEASE_LABEL_KEY: RELEASE_LABEL},
            {SITE: {REQUIRED_TESTS_KEY: list(catalogue)}},
            [result],
            False,
            UNCHANGED_SIGNATURE,
            catalogue,
        )
        records = {record["id"]: record for record in summary[RESULTS_KEY]}
        for test_id, expected in completed.items():
            assert records[test_id] == expected
        path.write_text(json.dumps(summary), encoding=TEXT_ENCODING)
    assert summary[QUALIFIED_KEY] is True


def test_release_summary_reuses_results_without_scanning_storage(tmp_test_directory, monkeypatch):
    root = Path(tmp_test_directory)
    report_path = root / SUMMARY_PATH
    report_path.parent.mkdir()
    record = {"id": COMPARE_ID, "test": COMPARE_TEST, "site": SITE, STATUS_KEY: "pass"}
    report_path.write_text(
        json.dumps({CAMPAIGN_DIGEST_KEY: UNCHANGED_SIGNATURE, RESULTS_KEY: [record]}),
        encoding=TEXT_ENCODING,
    )

    def reject_scan(*_args, **_kwargs):
        raise AssertionError("Release summaries must not recursively scan storage")

    monkeypatch.setattr(Path, "rglob", reject_scan)
    summary = science_test_reports._release_summary(
        root,
        {RELEASE_LABEL_KEY: RELEASE_LABEL},
        {SITE: {REQUIRED_TESTS_KEY: [COMPARE_TEST]}},
        [],
        False,
        UNCHANGED_SIGNATURE,
        {COMPARE_TEST: {}},
    )
    assert summary[RESULTS_KEY] == [record]
    assert summary[QUALIFIED_KEY] is True
    assert summary["evidence_sha256"] is None


def test_failed_test_retains_previous_report(tmp_test_directory):
    root = Path(tmp_test_directory)
    report_root = root / "reports/north/previous"
    report_root.mkdir(parents=True)
    result = {STATUS_KEY: "fail"}
    science_test_reports._write_test_report(root / "work", report_root, root, result)
    assert result[REPORT_KEY] is None
    assert result["previous_report"] == report_root.relative_to(root).as_posix()
    assert report_root.is_dir()


def test_write_summary_links_report_paths(tmp_test_directory):
    report_dir = Path(tmp_test_directory)
    summary = {
        _RELEASE: RELEASE_LABEL,
        QUALIFIED_KEY: False,
        _PRODUCTION_HISTORY: [],
        RESULTS_KEY: [
            {
                "site": SITE,
                "test": COMPARE_TEST,
                EXECUTION_KEY: COMPLETED_EXECUTION,
                STATUS_KEY: "warn",
                REPORT_KEY: "reports/north/compare",
            }
        ],
    }
    science_test_reports._write_summary(report_dir, summary)
    markdown = (report_dir / "release-summary.md").read_text(encoding=TEXT_ENCODING)
    assert "[report](north/compare)" in markdown
