"""Build release and per-test science-test reports."""

import json
import logging
import shutil

from simtools.science_tests.definitions import (
    _EXECUTION_LOG_FILE_NAME,
    _PUBLICATION_WARNING_BYTES,
    _PUBLICATION_WARNING_FILES,
    _WORKFLOW_FILE_NAME,
    _dependency_order,
)

logger = logging.getLogger(__name__)
_REDUNDANT_PUBLICATION_FILES = {"effective-settings.json", "inventory.json"}


def _release_summary(release_dir, release, suites, results, dry_run, signature, catalogue):
    """Merge compatible subset results into the complete required-site matrix."""
    path = release_dir / "reports" / "release-summary.json"
    previous = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not dry_run else {}
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
                    "report": None,
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


def _write_test_report(work, report_root, release_dir, result):
    """Publish successful collected evidence atomically without per-test wrappers."""
    result["report"] = report_root.relative_to(release_dir).as_posix()
    if result["status"] in {"fail", "incomplete", "blocked"}:
        return
    staging = report_root.with_name(f".{report_root.name}.new")
    _clear_test_directory(staging)
    if work.exists():
        staging.mkdir(parents=True, exist_ok=True)
        for source in work.iterdir():
            if source.name in {
                _EXECUTION_LOG_FILE_NAME,
                _WORKFLOW_FILE_NAME,
                *_REDUNDANT_PUBLICATION_FILES,
            }:
                continue
            destination = staging / source.name
            if source.is_dir():
                shutil.copytree(source, destination, ignore=_ignore_redundant_files)
            else:
                shutil.copyfile(source, destination)
    staging.mkdir(parents=True, exist_ok=True)
    result["publication"] = _publication_inventory(staging)
    result["artifacts"] = sorted(
        path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()
    )
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


def _ignore_redundant_files(_directory, names):
    """Skip duplicate settings and checksum files within published evidence."""
    return set(names) & _REDUNDANT_PUBLICATION_FILES


def _write_summary(report_dir, summary):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "release-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Science-test summary: {summary['release']}",
        "",
        f"Qualified: {summary['qualified']}",
        "",
        "| Site | Test | Execution | Status | Reason | Artifacts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        reason = result.get("error") or result.get("reason") or "-"
        artifacts = _artifact_links(result)
        lines.append(
            f"| {result['site']} | {result['test']} | {result['execution']} | "
            f"{result['status']} | {reason} | {artifacts} |"
        )
    if summary["production_history"]:
        lines.extend(["", "## Production history", ""])
        for record in summary["production_history"]:
            lines.append(
                f"- {record['id']}: {record['execution']} ({record['status']}); "
                f"run ID `{record.get('run_id', 'unknown')}`."
            )
    (report_dir / "release-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _artifact_links(result):
    """Link published products from the release summary table."""
    report = result.get("report")
    artifacts = result.get("artifacts", [])
    if not report or not artifacts:
        return "-"
    relative_report = report.removeprefix("reports/")
    return ", ".join(
        f"[{path.rsplit('/', maxsplit=1)[-1]}]({relative_report}/{path})" for path in artifacts
    )
