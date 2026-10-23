import json
from pathlib import Path

import pytest

from simtools.science_tests import reports

ACCEPTANCE_MD = "acceptance.md"
COMPARE = "compare"
COMPARE_NORTH = "compare.north"
DATA_JSON = "data.json"
DERIVE = "derive"
INCOMPLETE = "incomplete"
NESTED = "nested"
NOT_RUN = "not_run"
PREVIOUS_REPORT = "previous_report"
PRODUCTION_HISTORY = "production_history"
QUALIFIED = "qualified"
RELEASE = "release"
RELEASE_SUMMARY_JSON = "release-summary.json"
REPORT_PDF = "report.pdf"
REPORTS = "reports"
REPORTS_NORTH_TEST = "reports/north/test"
RESULTS = "results"
SUBMITTED = "submitted"

ENCODING = "utf-8"
STATUS = "status"
REPORT = "report"
NORTH = "north"


@pytest.fixture
def report_dirs(tmp_test_directory):
    root = Path(tmp_test_directory) / "reports-test"
    work = root / "work"
    work.mkdir(parents=True)
    (work / REPORT_PDF).write_text("pdf", encoding=ENCODING)
    (work / "execution.log").write_text("log", encoding=ENCODING)
    (work / NESTED).mkdir()
    (work / NESTED / DATA_JSON).write_text("{}", encoding=ENCODING)
    release = root / RELEASE
    return work, release / REPORTS / NORTH / "test", release


def result(status="pass", **extra):
    return {"id": f"test.{NORTH}", "test": "test", "site": NORTH, STATUS: status} | extra


def test_write_test_report_publishes_products_and_deviations(report_dirs):
    work, report_root, release = report_dirs
    record = result("warn", acceptance_warnings=["value high (PDF page 3)"])

    reports.write_test_report(work, report_root, release, record)

    assert record[REPORT] == REPORTS_NORTH_TEST
    assert sorted(path.name for path in report_root.rglob("*")) == [
        ACCEPTANCE_MD,
        DATA_JSON,
        NESTED,
        REPORT_PDF,
    ]
    assert "[PDF page 3](report.pdf#page=3)" in (report_root / ACCEPTANCE_MD).read_text(
        encoding=ENCODING
    )
    assert record["publication"]["file_count"] == 3
    assert "acceptance_warnings" not in record

    (work / REPORT_PDF).unlink()
    reports.write_test_report(work, report_root, release, result())
    assert not (report_root / REPORT_PDF).exists()
    assert not (report_root / ACCEPTANCE_MD).exists()


def test_write_test_report_keeps_previous_report_on_failure(report_dirs):
    work, report_root, release = report_dirs
    reports.write_test_report(work, report_root, release, result())
    failed = result(INCOMPLETE)

    reports.write_test_report(work, report_root, release, failed)

    assert failed[REPORT] is None
    assert failed[PREVIOUS_REPORT] == REPORTS_NORTH_TEST
    assert (report_root / REPORT_PDF).is_file()


def test_write_test_report_without_products_or_publication(report_dirs):
    work, report_root, release = report_dirs
    reports.write_test_report(work, report_root, release, result())
    unpublished = result()
    reports.write_test_report(work, report_root, release, unpublished, publish=False)
    assert unpublished[REPORT] is None
    assert not report_root.exists()

    empty = result()
    reports.write_test_report(work.parent / "missing", report_root, release, empty)
    assert empty[REPORT] is None
    assert not report_root.exists()

    only_directories = work.parent / "only-directories"
    (only_directories / NESTED).mkdir(parents=True)
    reports.write_test_report(only_directories, report_root, release, empty)
    assert empty[REPORT] is None
    assert not report_root.exists()


def test_publication_size_warns_for_large_reports(tmp_test_directory, caplog, monkeypatch):
    monkeypatch.setattr(reports, "_PUBLICATION_WARNING_FILES", 0)
    directory = Path(tmp_test_directory) / "large"
    directory.mkdir()
    (directory / "file.txt").write_text("x", encoding=ENCODING)
    with caplog.at_level("WARNING"):
        size = reports._publication_size(directory)
    assert "warning" in size
    assert "is large" in caplog.text


def test_release_summary_merges_results_of_same_campaign(bundle):
    summary_path = bundle.release_dir / REPORTS / RELEASE_SUMMARY_JSON
    first = reports.release_summary(
        bundle, [result_for(DERIVE, "pass"), result_for(COMPARE, "pass")], False, "a"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(first), encoding=ENCODING)

    same = reports.release_summary(bundle, [], False, "a")
    changed = reports.release_summary(bundle, [], False, "b")
    rerun = reports.release_summary(bundle, [result_for(DERIVE, "pass")], False, "a")

    assert by_id(same)[COMPARE_NORTH][STATUS] == "pass"
    assert by_id(same)["compare.south"][STATUS] == NOT_RUN
    assert by_id(changed)[COMPARE_NORTH][STATUS] == NOT_RUN
    assert by_id(rerun)[COMPARE_NORTH][STATUS] == NOT_RUN
    assert same[QUALIFIED] is False


def test_release_summary_keeps_production_history(bundle):
    bundle.catalogue[DERIVE]["produces_production"] = True
    summary_path = bundle.release_dir / REPORTS / RELEASE_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True)
    first = reports.release_summary(bundle, [result_for(DERIVE, SUBMITTED)], False, "a")
    summary_path.write_text(json.dumps(first), encoding=ENCODING)

    summary = reports.release_summary(bundle, [result_for(COMPARE, "pass")], False, "b")

    assert [record["id"] for record in summary[PRODUCTION_HISTORY]] == ["derive.north"]


def test_write_summary_writes_json_and_markdown(tmp_test_directory):
    summary = {
        RELEASE: "v2",
        QUALIFIED: False,
        PRODUCTION_HISTORY: [result_for(DERIVE, SUBMITTED) | {"run_id": "run"}],
        RESULTS: [
            result_for(COMPARE, "warn")
            | {
                REPORT: "reports/north/compare",
                "acceptance_report": "reports/north/compare/acceptance.md",
                "execution_log": "/work/execution.log",
                "execution_provenance": "/work",
            },
            result_for(DERIVE, INCOMPLETE)
            | {"error": "boom", PREVIOUS_REPORT: "reports/north/derive"},
        ],
    }
    directory = Path(tmp_test_directory) / "summary"
    reports.write_summary(directory, summary)

    markdown = (directory / "release-summary.md").read_text(encoding=ENCODING)
    assert "[report](north/compare); [deviations](north/compare/acceptance.md)" in markdown
    assert "| boom | - |" in markdown
    assert "## Execution details" not in markdown
    assert "/work/execution.log" not in markdown
    assert "Previous successful output" not in markdown
    assert "run ID `run`" in markdown
    assert json.loads((directory / RELEASE_SUMMARY_JSON).read_text(encoding=ENCODING))


def result_for(test, status):
    return {
        "id": f"{test}.{NORTH}",
        "test": test,
        "site": NORTH,
        STATUS: status,
        "execution": status,
    }


def by_id(summary):
    return {record["id"]: record for record in summary[RESULTS]}
