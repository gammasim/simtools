"""Compare, sync, and report obsolete versioned integration-test resources."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from simtools import constants
from simtools.testing import resource_generation

logger = logging.getLogger(__name__)
_IGNORED_DIRECTORIES = {"config_files", "log_files", "tmp_application_output"}


def get_destination_directories(resources_path=None):
    """Return the configured local test-resource directories by resource class."""
    if resources_path is not None:
        root = Path(resources_path).expanduser().resolve()
        return {
            "static": root / "static",
            "generated": root / "generated",
            "downloaded": root / "downloaded",
        }
    return {
        "static": Path(constants.TEST_RESOURCES_STATIC).expanduser().resolve(),
        "generated": Path(constants.TEST_RESOURCES_GENERATED).expanduser().resolve(),
        "downloaded": Path(constants.TEST_RESOURCES_DOWNLOADED).expanduser().resolve(),
    }


def get_destination_directory(resources_path=None):
    """Return the common local test-resources root directory."""
    return get_destination_directories(resources_path)["static"].parent


def _selected_resource_directories(args_dict):
    """Return the selected resource classes to compare or sync."""
    selected = []

    def _is_selected(resource_name):
        return not args_dict.get(f"exclude_{resource_name}", False)

    if _is_selected("static"):
        selected.append("static")
    if _is_selected("generated"):
        selected.append("generated")
    if _is_selected("downloaded"):
        selected.append("downloaded")
    if not selected:
        raise ValueError("Select at least one resource class to compare.")
    return tuple(selected)


def _calculate_sha256(file_path):
    """Return the SHA-256 checksum of a file."""
    checksum = hashlib.sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _collect_files(directory):
    """Return all files below a directory keyed by relative path."""
    directory = Path(directory)
    if not directory.exists():
        return {}

    files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(directory)
        if relative_path.parts and relative_path.parts[0] in _IGNORED_DIRECTORIES:
            continue
        files[relative_path.as_posix()] = path
    return files


def _validate_source_directory(directory):
    """Raise if a selected source resource directory does not exist."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Source resource directory does not exist: {directory}")


def compare_resource_directories(source_dir, destination_dir):
    """Compare two resource directories and classify their files."""
    source_files = _collect_files(source_dir)
    destination_files = _collect_files(destination_dir)

    new_files = []
    changed_files = []
    unchanged_files = []
    obsolete_files = []

    for relative_path, source_path in source_files.items():
        destination_path = destination_files.get(relative_path)
        if destination_path is None:
            new_files.append(relative_path)
            continue
        source_hash = _calculate_sha256(source_path)
        destination_hash = _calculate_sha256(destination_path)
        if source_hash == destination_hash:
            unchanged_files.append(relative_path)
        else:
            changed_files.append(relative_path)

    for relative_path in destination_files:
        if relative_path not in source_files:
            obsolete_files.append(relative_path)

    return {
        "new": sorted(new_files),
        "changed": sorted(changed_files),
        "unchanged": sorted(unchanged_files),
        "obsolete": sorted(obsolete_files),
        "source_files": source_files,
        "destination_files": destination_files,
    }


def build_sync_report(test_directory, simtools_version, selected_directories, resources_path=None):
    """Build a sync report for the selected resource directories."""
    source_root = resource_generation.get_integration_test_directory(
        test_directory, simtools_version
    )
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source integration-test directory does not exist: {source_root}")
    destination_directories = get_destination_directories(resources_path)

    report = {
        "source_root": source_root,
        "destination_root": get_destination_directory(resources_path),
        "destination_directories": destination_directories,
        "directories": {},
        "summary": {"new": 0, "changed": 0, "unchanged": 0, "obsolete": 0},
    }

    for resource_dir in selected_directories:
        _validate_source_directory(source_root / resource_dir)
        comparison = compare_resource_directories(
            source_root / resource_dir,
            destination_directories[resource_dir],
        )
        report["directories"][resource_dir] = comparison
        for key in report["summary"]:
            report["summary"][key] += len(comparison[key])

    return report


def _format_file_group(directory_name, group_name, relative_paths):
    """Render one report section for a resource class."""
    if not relative_paths:
        return []
    lines = [f"{group_name} ({directory_name}):"]
    lines.extend(f"  {path}" for path in relative_paths)
    return lines


def render_sync_report(report):
    """Render a human-readable sync report with actionable differences only."""
    lines = [
        f"Source: {report['source_root']}",
        f"Destination: {report['destination_root']}",
    ]
    summary = report["summary"]
    lines.append(
        "Summary: "
        f"new={summary['new']}, changed={summary['changed']}, obsolete={summary['obsolete']}"
    )

    for directory_name, comparison in report["directories"].items():
        lines.extend(_format_file_group(directory_name, "new", comparison["new"]))
        lines.extend(_format_file_group(directory_name, "changed", comparison["changed"]))
        lines.extend(_format_file_group(directory_name, "obsolete", comparison["obsolete"]))

    return "\n".join(lines)


def _copy_file(source_path, destination_path):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    logger.info("Synced %s", destination_path)


def _resolve_delete_target(path, root):
    """Return a validated delete target below the given root directory."""
    path = Path(path)
    root = Path(root).resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete file outside test resources: {path}") from exc
    if not resolved_path.is_file():
        raise ValueError(f"Refusing to delete non-file path: {path}")
    return path


def _deletion_plan(report):
    """Return validated obsolete-file delete actions without modifying files."""
    plan = []
    for directory_name, comparison in report["directories"].items():
        destination_root = report["destination_directories"][directory_name]
        for relative_path in comparison["obsolete"]:
            destination_path = comparison["destination_files"][relative_path]
            plan.append(
                (
                    directory_name,
                    relative_path,
                    _resolve_delete_target(destination_path, destination_root),
                    destination_root,
                )
            )
    return plan


def apply_sync_actions(report, sync=False, delete_missing=False):
    """Apply sync actions and collect obsolete files for manual removal."""
    if not sync and not delete_missing:
        return {"copied": [], "remove_candidates": []}

    copied = []
    remove_candidates = []
    destination_directories = report["destination_directories"]
    delete_plan = _deletion_plan(report) if delete_missing else []

    for directory_name, comparison in report["directories"].items():
        if sync:
            for relative_path in comparison["new"] + comparison["changed"]:
                source_path = comparison["source_files"][relative_path]
                destination_path = destination_directories[directory_name] / relative_path
                _copy_file(source_path, destination_path)
                copied.append(f"{directory_name}/{relative_path}")

    for directory_name, relative_path, _, _ in delete_plan:
        remove_candidates.append(f"{directory_name}/{relative_path}")

    return {"copied": copied, "remove_candidates": remove_candidates}


def sync_test_resources(args_dict):
    """Compare, optionally sync, and optionally report obsolete test resources."""
    selected_directories = _selected_resource_directories(args_dict)
    report = build_sync_report(
        args_dict["test_directory"],
        args_dict["simtools_version"],
        selected_directories,
        resources_path=args_dict.get("resources_path"),
    )
    report_text = render_sync_report(report)
    if report_text:
        logger.info("%s", report_text)

    actions = apply_sync_actions(
        report,
        sync=args_dict.get("sync", False),
        delete_missing=args_dict.get("delete_missing", False),
    )
    if actions["copied"]:
        logger.info("Copied %d file(s).", len(actions["copied"]))
    if actions["remove_candidates"]:
        logger.info(
            "Obsolete test resources were not removed automatically. Remove these file(s) "
            "manually if they should be deleted:"
        )
        for relative_path in actions["remove_candidates"]:
            logger.info("  %s", relative_path)
    return report, actions
