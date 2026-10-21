"""Build release and per-test science-test reports."""

import json
import logging
import re
import shutil
from pathlib import Path

from simtools.science_tests.definitions import (
    _EXECUTION_LOG_FILE_NAME,
    _PUBLICATION_WARNING_BYTES,
    _PUBLICATION_WARNING_FILES,
    _WORKFLOW_FILE_NAME,
    _dependency_order,
)

_RELEASE_SUMMARY_JSON = "release-summary.json"
_REPORT = "report"

_ACCEPTANCE_REPORT_KEY = "acceptance_report"
_PREVIOUS_REPORT_KEY = "previous_report"
_REPORTS_DIRECTORY = "reports"
_ENCODING = "utf-8"

logger = logging.getLogger(__name__)
_REDUNDANT_PUBLICATION_FILES = {"effective-settings.json", "inventory.json"}


def _release_summary(release_dir, release, suites, results, dry_run, signature, catalogue):
    """Merge compatible subset results into the complete required-site matrix."""
    path = release_dir / _REPORTS_DIRECTORY / _RELEASE_SUMMARY_JSON
    previous = (
        json.loads(path.read_text(encoding=_ENCODING)) if path.is_file() and not dry_run else {}
    )
    records = (
        {r["id"]: r for r in previous.get("results", [])}
        if previous.get("campaign_sha256") == signature
        else {}
    )
    production_history = _production_history(previous, results, catalogue)
    _invalidate_dependents(records, results, catalogue)
    records.update({r["id"]: r for r in results})
    required = []
    for site, suite in suites.items():
        for test_id in suite["required_tests"]:
            key = f"{test_id}.{site}"
            records.setdefault(
                key,
                {
                    "id": key,
                    "test": test_id,
                    "site": site,
                    _REPORT: None,
                    "status": "not_run",
                    "execution": "not_run",
                    "reason": "Not selected or not executed.",
                },
            )
            required.append(key)
    return {
        "release": release["release_label"],
        "baseline": release.get("baseline"),
        "required_sites": list(suites),
        "dry_run": dry_run,
        "campaign_sha256": signature,
        "evidence_sha256": None,
        "qualified": all(records[key]["status"] == "pass" for key in required),
        "results": sorted(records.values(), key=lambda r: (r["site"], r["test"])),
        "production_history": production_history,
    }


def _production_history(previous, results, catalogue):
    """Keep submitted-production provenance when non-production checks are rerun."""
    records = {record["id"]: record for record in previous.get("production_history", [])}
    for result in results:
        definition = catalogue[result["test"]]
        if definition.get("produces_production") or definition.get("affects_production"):
            records[result["id"]] = result
    return sorted(records.values(), key=lambda record: (record["site"], record["test"]))


def _invalidate_dependents(records, results, catalogue):
    """Retain subset results only when their upstream evidence was not rerun."""
    for record in records.values():
        dependencies = set(_dependency_order([record["test"]], catalogue))
        for result in results:
            changed = result["site"] == record["site"] and result["id"] != record["id"]
            definition = catalogue[result["test"]]
            upstream = (
                result["test"] in dependencies
                or definition.get("produces_production")
                or definition.get("affects_production")
            )
            if changed and upstream:
                record.update(
                    status="not_run",
                    execution="not_run",
                    reason="Upstream evidence changed; rerun required.",
                )


def _clear_test_directory(path):
    """Remove artifacts from the previous execution of this named test."""
    if path.exists():
        shutil.rmtree(path)


def _write_test_report(work, report_root, release_dir, result, publish=True):
    """Publish successful collected evidence atomically without per-test wrappers."""
    if not publish:
        result[_REPORT] = None
        result.pop(_PREVIOUS_REPORT_KEY, None)
        _clear_test_directory(report_root)
        return
    report = report_root.relative_to(release_dir).as_posix()
    if result["status"] in {"fail", "incomplete", "blocked"}:
        result[_REPORT] = None
        if report_root.exists():
            result[_PREVIOUS_REPORT_KEY] = report
        return
    result[_REPORT] = report
    _publish_test_report(work, report_root, result)


def _publish_test_report(work, report_root, result):
    """Publish successful evidence and its acceptance notes using a staging directory."""
    report = result[_REPORT]
    warnings = result.pop("acceptance_warnings", [])
    if warnings:
        work.mkdir(parents=True, exist_ok=True)
        lines = ["# Acceptance deviations", ""]
        for warning in warnings:
            warning = re.sub(
                r"PDF page (\d+)", r"[PDF page \1](trigger-report.pdf#page=\1)", warning
            )
            lines.append(f"- {warning}")
        (work / "acceptance.md").write_text("\n".join(lines) + "\n", encoding=_ENCODING)
        result[_ACCEPTANCE_REPORT_KEY] = f"{report}/acceptance.md"
    else:
        (work / "acceptance.md").unlink(missing_ok=True)
        result.pop(_ACCEPTANCE_REPORT_KEY, None)
    staging = report_root.with_name(f".{report_root.name}.new")
    _clear_test_directory(staging)
    if work.exists():
        for source in work.iterdir():
            if source.name in {
                _EXECUTION_LOG_FILE_NAME,
                _WORKFLOW_FILE_NAME,
                *_REDUNDANT_PUBLICATION_FILES,
            }:
                continue
            destination = staging / source.name
            _copy_report_products(source, destination)
    result["publication"] = _publication_inventory(staging)
    if not staging.exists():
        result[_REPORT] = None
        _clear_test_directory(report_root)
        return
    previous = report_root.with_name(f".{report_root.name}.previous")
    _clear_test_directory(previous)
    if report_root.exists():
        report_root.rename(previous)
    staging.rename(report_root)
    _clear_test_directory(previous)


def _publication_inventory(report_root):
    """Summarize compact publication size without making it an acceptance rule."""
    files = [path for path in report_root.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    inventory = {"file_count": len(files), "byte_count": total_bytes}
    if len(files) > _PUBLICATION_WARNING_FILES or total_bytes > _PUBLICATION_WARNING_BYTES:
        inventory["warning"] = "Publication is larger than the compact-report guidance."
        logger.warning(
            "Science-test publication contains %d files and %d bytes.", len(files), total_bytes
        )
    return inventory


def _copy_report_products(source, destination):
    """Copy report files, creating parent directories only when a file is copied."""
    if source.name in _REDUNDANT_PUBLICATION_FILES:
        return
    if source.is_dir():
        for child in source.iterdir():
            _copy_report_products(child, destination / child.name)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _write_summary(report_dir, summary):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / _RELEASE_SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding=_ENCODING,
    )
    lines = [
        f"# Science-test summary: {summary['release']}",
        "",
        f"Qualified: {summary['qualified']}",
        "",
        "| Site | Test | Execution | Status | Reason | Report |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        reason = result.get("error") or result.get("reason") or "-"
        report = "-"
        if result.get(_REPORT):
            report_path = Path(result[_REPORT]).relative_to(_REPORTS_DIRECTORY).as_posix()
            report = f"[report]({report_path})"
        if result.get(_ACCEPTANCE_REPORT_KEY):
            acceptance_path = (
                Path(result[_ACCEPTANCE_REPORT_KEY]).relative_to(_REPORTS_DIRECTORY).as_posix()
            )
            report += f"; [deviations]({acceptance_path})"
        lines.append(
            f"| {result['site']} | {result['test']} | {result['execution']} | "
            f"{result['status']} | {reason} | {report} |"
        )
    lines.extend(["", "## Execution details", ""])
    for result in summary["results"]:
        if not result.get("execution_log"):
            continue
        lines.append(
            f"- {result['site']}/{result['test']}: {result.get('runtime', 'unknown')}; "
            f"[log]({result['execution_log']}); "
            f"work: `{result['execution_provenance']}`."
        )
        if result.get(_PREVIOUS_REPORT_KEY):
            lines.append(
                f"  Previous successful output: `{result[_PREVIOUS_REPORT_KEY]}` "
                "(not evidence for this failed attempt)."
            )
    if summary["production_history"]:
        lines.extend(["", "## Production history", ""])
        for record in summary["production_history"]:
            lines.append(
                f"- {record['id']}: {record['execution']} ({record['status']}); "
                f"run ID `{record.get('run_id', 'unknown')}`."
            )
    (report_dir / "release-summary.md").write_text("\n".join(lines) + "\n", encoding=_ENCODING)
