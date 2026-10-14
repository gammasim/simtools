"""Build release and per-test science-test reports."""

import hashlib
import json
import logging
import shutil
from pathlib import Path

from simtools.science_tests.definitions import (
    _EXECUTION_LOG_FILE_NAME,
    _PUBLICATION_WARNING_BYTES,
    _PUBLICATION_WARNING_FILES,
    _WORKFLOW_FILE_NAME,
    _dependency_order,
)

_RESULTS = "results"
_STATUS = "status"
_DEPENDENCY_COMPATIBILITY = "dependency_compatibility"
_COMPATIBLE = "compatible"
_UNKNOWN = "unknown"

logger = logging.getLogger(__name__)
_REDUNDANT_PUBLICATION_FILES = {"effective-settings.json", "inventory.json"}


def _release_summary(
    release_dir, release, suites, results, dry_run, signature, catalogue, context=None
):
    """Merge compatible subset results into the complete required-site matrix."""
    path = release_dir / "reports" / "release-summary.json"
    previous = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not dry_run else {}
    )
    records = (
        {r["id"]: r for r in previous.get(_RESULTS, [])}
        if previous.get("campaign_sha256") == signature
        else {}
    )
    evidence = (
        {site: _evidence_fingerprints(context, site) for site in suites}
        if context is not None and not dry_run
        else None
    )
    if evidence is not None:
        _invalidate_evidence_changes(
            records, previous.get("evidence_sha256", {}), evidence, catalogue
        )
    _invalidate_runtime_changes(records, results, catalogue)
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
                    _STATUS: "not_run",
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
        "evidence_sha256": evidence,
        "qualified": all(records[key][_STATUS] == "pass" for key in required)
        and _dependencies_qualified(records.values()),
        _RESULTS: sorted(records.values(), key=lambda r: (r["site"], r["test"])),
        "production_history": production_history,
    }


def _dependencies_qualified(records):
    """Keep development production inputs unqualified even after collection passes."""
    return all(
        (record.get(_DEPENDENCY_COMPATIBILITY) or {}).get(_STATUS, _COMPATIBLE) == _COMPATIBLE
        for record in records
    )


def _evidence_fingerprints(context, site):
    """Hash small metadata and product file identities, excluding analysis work directories."""
    hashes = {kind: hashlib.sha256() for kind in ("production", "analysis")}
    for role in ("candidate", "baseline"):
        path = context.get(f"__SCIENCE_{role.upper()}_PATH__")
        if not path:
            continue
        root = Path(path) / site
        grid = root.parent / f"{site}_job_grid.ecsv"
        files = sorted(file for file in root.rglob("*") if file.is_file())
        if grid.is_file():
            files.append(grid)
        for file in files:
            stat = file.stat()
            content = (
                hashlib.sha256(file.read_bytes()).hexdigest()
                if stat.st_size <= 1024 * 1024 or file.suffix in {".yml", ".yaml", ".json", ".ecsv"}
                else None
            )
            relative = file.relative_to(root.parent)
            entry = json.dumps(
                [role, str(relative), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, content]
            ).encode()
            hashes["analysis"].update(entry)
            if role == "candidate" and relative.parts[:2] != (site, "derived"):
                hashes["production"].update(entry)
    return {kind: digest.hexdigest() for kind, digest in hashes.items()}


def _invalidate_evidence_changes(records, previous, current, catalogue):
    """Discard stale evidence while preserving native steps after derived-only changes."""
    for record in records.values():
        definition = catalogue[record["test"]]
        native = definition.get("collect_production") or definition.get("produces_production")
        kind = "production" if native else "analysis"
        site = record["site"]
        if previous.get(site, {}).get(kind) != current[site][kind]:
            record.update(
                status="not_run",
                execution="not_run",
                reason="Production evidence changed; rerun required.",
            )


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
    report = report_root.relative_to(release_dir).as_posix()
    if result[_STATUS] in {"fail", "incomplete", "blocked"}:
        result["report"] = None
        if report_root.exists():
            result["previous_report"] = report
        return
    result["report"] = report
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
        "| Site | Test | Execution | Status | Reason | Report |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary[_RESULTS]:
        reason = result.get("error") or result.get("reason") or "-"
        report = "-"
        if result.get("report"):
            report_path = Path(result["report"]).relative_to("reports").as_posix()
            report = f"[report]({report_path})"
        lines.append(
            f"| {result['site']} | {result['test']} | {result['execution']} | "
            f"{result[_STATUS]} | {reason} | {report} |"
        )
    lines.extend(["", "## Dependency compatibility", ""])
    for result in summary[_RESULTS]:
        proof = result.get(_DEPENDENCY_COMPATIBILITY)
        if proof:
            lines.append(
                f"- {result['id']}: target simtools {proof.get('target_simtools')}; "
                f"models {proof.get('version', _UNKNOWN)} at `{proof.get('source')}`; "
                f"allowed `{proof.get('allowed')}` ({proof.get('policy')}); "
                f"{proof.get(_STATUS)}; initialization: {proof.get('initialization')}."
            )
    lines.extend(["", "## Execution details", ""])
    for result in summary[_RESULTS]:
        if not result.get("execution_log"):
            continue
        lines.append(
            f"- {result['site']}/{result['test']}: {result.get('runtime', _UNKNOWN)}; "
            f"[log]({result['execution_log']}); "
            f"work: `{result['execution_provenance']}`."
        )
        if result.get("previous_report"):
            lines.append(
                f"  Previous successful output: `{result['previous_report']}` "
                "(not evidence for this failed attempt)."
            )
    if summary["production_history"]:
        lines.extend(["", "## Production history", ""])
        for record in summary["production_history"]:
            lines.append(
                f"- {record['id']}: {record['execution']} ({record[_STATUS]}); "
                f"run ID `{record.get('run_id', _UNKNOWN)}`."
            )
    (report_dir / "release-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _invalidate_runtime_changes(records, results, catalogue):
    """Invalidate analysis results when their actual runtime changes, retaining native steps."""
    native = {
        name
        for name, definition in catalogue.items()
        if definition.get("collect_production") or definition.get("produces_production")
    }
    for result in results:
        if result["test"] in native or "runtime" not in result:
            continue
        for record in records.values():
            if (
                record["site"] == result["site"]
                and record["test"] not in native
                and record.get("runtime") != result["runtime"]
            ):
                record.update(
                    status="not_run",
                    execution="not_run",
                    reason="Analysis runtime changed; rerun required.",
                )
