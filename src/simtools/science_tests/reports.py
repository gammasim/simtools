"""Publish per-test reports and the release summary of science tests."""

import json
import logging
import re
import shutil
from pathlib import Path

from simtools.science_tests.definitions import dependency_order

REPORTS_DIRECTORY = "reports"
SUMMARY_FILE = "release-summary"
ACCEPTANCE_FILE = "acceptance.md"
_REPORT = "report"
_STATUS = "status"
_EXECUTION = "execution"
_REASON = "reason"
_ACCEPTANCE_REPORT = "acceptance_report"
_PREVIOUS_REPORT = "previous_report"
_NOT_RUN = "not_run"
_JSON_SUFFIX = ".json"
_RELEASE = "release"
_BASELINE = "baseline"
_SIGNATURE = "campaign_sha256"
_QUALIFIED = "qualified"
_RESULTS = "results"
_HISTORY = "production_history"
_EXECUTION_LOG = "execution_log"
_UNKNOWN = "unknown"
_FAILED = {"fail", "incomplete", "blocked"}
_ENCODING = "utf-8"
# Work-directory files kept with the execution record but not published.
_UNPUBLISHED_FILES = {
    "execution.log",
    "workflow.yml",
    "effective-settings.json",
    "inventory.json",
}
_PUBLICATION_WARNING_FILES = 30
_PUBLICATION_WARNING_BYTES = 20 * 1024 * 1024

logger = logging.getLogger(__name__)


def write_test_report(work, report_root, release_dir, result, publish=True):
    """Publish the products of a successful test to the release report directory.

    Parameters
    ----------
    work : pathlib.Path
        Work directory with the test products.
    report_root : pathlib.Path
        Report directory of this site/test.
    release_dir : pathlib.Path
        Release directory; report paths are recorded relative to it.
    result : dict
        Test result, updated with report paths and publication size.
    publish : bool
        Publish products; False removes any previous report.
    """
    warnings = result.pop("acceptance_warnings", [])
    result[_REPORT] = None
    if not publish:
        _remove(report_root)
        return
    report = report_root.relative_to(release_dir).as_posix()
    if result[_STATUS] in _FAILED:
        if report_root.exists():
            result[_PREVIOUS_REPORT] = report
        return
    staging = report_root.with_name(f".{report_root.name}.new")
    _remove(staging)
    if work.is_dir():
        shutil.copytree(work, staging, ignore=shutil.ignore_patterns(*_UNPUBLISHED_FILES))
    if warnings:
        staging.mkdir(parents=True, exist_ok=True)
        _write_acceptance_report(staging, warnings)
        result[_ACCEPTANCE_REPORT] = f"{report}/{ACCEPTANCE_FILE}"
    _remove(report_root)
    if not staging.is_dir() or not any(path.is_file() for path in staging.rglob("*")):
        _remove(staging)
        return
    staging.rename(report_root)
    result[_REPORT] = report
    result["publication"] = _publication_size(report_root)


def release_summary(bundle, results, dry_run, signature):
    """Merge new results into the previous summary of the same campaign definition.

    Parameters
    ----------
    bundle : ReleaseDefinition
        Release definitions.
    results : list of dict
        Results of this runner call.
    dry_run : bool
        Whether the results are planned only.
    signature : str
        Campaign digest; previous results are kept only for the same digest.

    Returns
    -------
    dict
        Release summary with results for all required site/test pairs.
    """
    catalogue = bundle.catalogue
    path = bundle.release_dir / REPORTS_DIRECTORY / f"{SUMMARY_FILE}{_JSON_SUFFIX}"
    previous = {}
    if path.is_file() and not dry_run:
        previous = json.loads(path.read_text(encoding=_ENCODING))
    records = {}
    if previous.get(_SIGNATURE) == signature:
        records = {record["id"]: record for record in previous.get(_RESULTS, [])}
    _invalidate_dependents(records, results, catalogue)
    records.update({result["id"]: result for result in results})
    required = []
    for site, suite in bundle.suites.items():
        for test_id in suite["required_tests"]:
            key = f"{test_id}.{site}"
            required.append(key)
            records.setdefault(
                key,
                {
                    "id": key,
                    "test": test_id,
                    "site": site,
                    _REPORT: None,
                    _STATUS: _NOT_RUN,
                    _EXECUTION: _NOT_RUN,
                    _REASON: "Not selected or not executed.",
                },
            )
    history = {record["id"]: record for record in previous.get(_HISTORY, [])}
    history.update(
        {result["id"]: result for result in results if _affects_production(catalogue, result)}
    )
    return {
        _RELEASE: bundle.release["release_label"],
        _BASELINE: bundle.release.get(_BASELINE),
        "required_sites": list(bundle.suites),
        "dry_run": dry_run,
        _SIGNATURE: signature,
        _QUALIFIED: all(records[key][_STATUS] == "pass" for key in required),
        _RESULTS: sorted(records.values(), key=_site_test),
        _HISTORY: sorted(history.values(), key=_site_test),
    }


def write_summary(report_dir, summary):
    """Write the release summary as JSON and Markdown.

    Parameters
    ----------
    report_dir : pathlib.Path
        Release report directory.
    summary : dict
        Release summary returned by :func:`release_summary`.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{SUMMARY_FILE}{_JSON_SUFFIX}").write_text(
        json.dumps(summary, indent=2) + "\n", encoding=_ENCODING
    )
    lines = [
        f"# Science-test summary: {summary[_RELEASE]}",
        "",
        f"Qualified: {summary[_QUALIFIED]}",
        "",
        "| Site | Test | Execution | Status | Reason | Report |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary[_RESULTS]:
        reason = result.get("error") or result.get(_REASON) or "-"
        links = [
            f"[{label}]({_relative_link(result[key])})"
            for key, label in ((_REPORT, _REPORT), (_ACCEPTANCE_REPORT, "deviations"))
            if result.get(key)
        ]
        lines.append(
            f"| {result['site']} | {result['test']} | {result[_EXECUTION]} | "
            f"{result[_STATUS]} | {reason} | {'; '.join(links) or '-'} |"
        )
    lines.extend(["", "## Execution details", ""])
    for result in summary[_RESULTS]:
        if result.get(_EXECUTION_LOG):
            lines.append(
                f"- {result['site']}/{result['test']}: {result.get('runtime', _UNKNOWN)}; "
                f"[log]({result[_EXECUTION_LOG]}); work: `{result['execution_provenance']}`."
            )
        if result.get(_PREVIOUS_REPORT):
            lines.append(
                f"  Previous successful output: `{result[_PREVIOUS_REPORT]}` "
                "(not evidence for this failed attempt)."
            )
    if summary[_HISTORY]:
        lines.extend(["", "## Production history", ""])
        lines.extend(
            f"- {record['id']}: {record[_EXECUTION]} ({record[_STATUS]}); "
            f"run ID `{record.get('run_id', _UNKNOWN)}`."
            for record in summary[_HISTORY]
        )
    (report_dir / f"{SUMMARY_FILE}.md").write_text("\n".join(lines) + "\n", encoding=_ENCODING)


def _invalidate_dependents(records, results, catalogue):
    """Mark kept results as not run when upstream evidence of the same site was rerun."""
    for record in records.values():
        upstream = set(dependency_order([record["test"]], catalogue))
        for result in results:
            if (
                result["site"] == record["site"]
                and result["id"] != record["id"]
                and (result["test"] in upstream or _affects_production(catalogue, result))
            ):
                record.update(
                    {
                        _STATUS: _NOT_RUN,
                        _EXECUTION: _NOT_RUN,
                        _REASON: "Upstream evidence changed; rerun required.",
                    }
                )


def _affects_production(catalogue, result):
    definition = catalogue[result["test"]]
    return bool(definition.get("produces_production") or definition.get("affects_production"))


def _write_acceptance_report(directory, warnings):
    """Write acceptance deviations, linking PDF page references to the report's PDF."""
    pdfs = sorted(directory.glob("*.pdf"))
    lines = ["# Acceptance deviations", ""]
    for warning in warnings:
        if len(pdfs) == 1:
            warning = re.sub(r"PDF page (\d+)", rf"[PDF page \1]({pdfs[0].name}#page=\1)", warning)
        lines.append(f"- {warning}")
    (directory / ACCEPTANCE_FILE).write_text("\n".join(lines) + "\n", encoding=_ENCODING)


def _publication_size(report_root):
    """Return the published file count and size, warning for unusually large reports."""
    files = [path for path in report_root.rglob("*") if path.is_file()]
    byte_count = sum(path.stat().st_size for path in files)
    size = {"file_count": len(files), "byte_count": byte_count}
    if len(files) > _PUBLICATION_WARNING_FILES or byte_count > _PUBLICATION_WARNING_BYTES:
        size["warning"] = "Publication is larger than the compact-report guidance."
        logger.warning("Science-test publication %s is large: %s", report_root, size)
    return size


def _relative_link(path):
    return Path(path).relative_to(REPORTS_DIRECTORY).as_posix()


def _site_test(record):
    return record["site"], record["test"]


def _remove(path):
    if path.exists():
        shutil.rmtree(path)
