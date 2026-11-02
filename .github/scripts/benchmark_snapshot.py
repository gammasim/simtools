"""Manage resource benchmark snapshots stored as assets of a draft GitHub release."""

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

from publish_docs import extract_archive

RELEASE_TAG = "benchmark-data"
ASSET_PATTERN = re.compile(r"resource-benchmark-snapshot-\d+-\d+\.tar\.gz\Z")
RETAINED_SNAPSHOTS = 2
REPOSITORY_URL = "https://github.com/gammasim/simtools"
DASHBOARD_DIRECTORY = Path("dev/test-resources")


def empty_state():
    """Return the benchmark history of a new installation."""
    return {"lastUpdate": 0, "repoUrl": REPOSITORY_URL, "entries": {}}


def validate_state(state):
    """Raise ValueError unless the benchmark history has the expected structure."""
    if not isinstance(state, dict) or not isinstance(state.get("entries"), dict):
        raise ValueError("Benchmark history state is invalid")
    return state


def _benchmark_releases(releases_path):
    """Return the draft benchmark releases from the paginated output of 'gh api --slurp'."""
    pages = json.loads(Path(releases_path).read_text(encoding="utf-8"))
    return [
        release
        for page in pages
        for release in page
        if release["draft"] and release["tag_name"] == RELEASE_TAG
    ]


def _snapshot_assets(release):
    """Return the snapshot assets of a release sorted from oldest to newest."""
    assets = release["assets"]
    snapshots = [asset for asset in assets if ASSET_PATTERN.fullmatch(asset["name"])]
    if len(snapshots) != len(assets):
        raise ValueError("Draft benchmark release contains an unknown asset")
    if any(asset["state"] != "uploaded" for asset in snapshots):
        raise ValueError("Found an incomplete benchmark snapshot asset")
    return sorted(snapshots, key=lambda asset: (asset["created_at"], asset["id"]))


def select_asset(releases_path):
    """Describe the draft benchmark release and its newest snapshot asset."""
    matches = _benchmark_releases(releases_path)
    if len(matches) > 1:
        raise ValueError("Found multiple draft benchmark-data releases")
    snapshots = _snapshot_assets(matches[0]) if matches else []
    return {
        "release_exists": bool(matches),
        "asset_name": snapshots[-1]["name"] if snapshots else "",
        "asset_count": len(snapshots),
    }


def old_assets(releases_path):
    """Return the names of the snapshot assets that are no longer retained."""
    matches = _benchmark_releases(releases_path)
    if len(matches) != 1:
        raise ValueError("Expected exactly one draft benchmark-data release")
    return [asset["name"] for asset in _snapshot_assets(matches[0])[:-RETAINED_SNAPSHOTS]]


def load_state(path):
    """Read and validate a benchmark history file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Benchmark snapshot is missing {path.name}")
    return validate_state(json.loads(path.read_text(encoding="utf-8")))


def build_snapshot(state_path, dashboard_page, snapshot_path, work_directory):
    """Write the dashboard data and pack the history and dashboard into a snapshot archive."""
    state = load_state(state_path)
    work_directory = Path(work_directory)
    dashboard = work_directory / DASHBOARD_DIRECTORY
    dashboard.mkdir(parents=True, exist_ok=True)
    (dashboard / "index.html").write_bytes(Path(dashboard_page).read_bytes())
    (dashboard / "data.js").write_text(
        "window.BENCHMARK_DATA = " + json.dumps(state, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (work_directory / "state.json").write_text(json.dumps(state), encoding="utf-8")
    with tarfile.open(snapshot_path, "w:gz") as archive:
        for name in ("state.json", "dev/test-resources/index.html", "dev/test-resources/data.js"):
            archive.add(work_directory / name, arcname=name)


def verify_snapshot(snapshot_path, destination):
    """Extract a snapshot archive and check that it is complete."""
    destination = Path(destination)
    extract_archive(Path(snapshot_path), destination)
    required = [
        destination / "state.json",
        destination / DASHBOARD_DIRECTORY / "index.html",
        destination / DASHBOARD_DIRECTORY / "data.js",
    ]
    if not all(path.is_file() for path in required):
        raise ValueError("Benchmark snapshot is incomplete")
    load_state(destination / "state.json")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="print the newest snapshot asset as JSON")
    select.add_argument("releases", type=Path)
    prune = commands.add_parser("old-assets", help="list snapshot assets to delete")
    prune.add_argument("releases", type=Path)
    init = commands.add_parser("init-state", help="write an empty benchmark history")
    init.add_argument("state", type=Path)
    check = commands.add_parser("validate-state", help="validate a benchmark history")
    check.add_argument("state", type=Path)
    build = commands.add_parser("build", help="pack a snapshot archive")
    build.add_argument("--state", type=Path, required=True)
    build.add_argument("--dashboard", type=Path, required=True)
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--work-directory", type=Path, required=True)
    verify = commands.add_parser("verify", help="extract and check a snapshot archive")
    verify.add_argument("snapshot", type=Path)
    verify.add_argument("destination", type=Path)
    return parser


def main():
    """Run one snapshot management command."""
    arguments = _parser().parse_args()
    try:
        if arguments.command == "select":
            print(json.dumps(select_asset(arguments.releases)))
        elif arguments.command == "old-assets":
            print("\n".join(old_assets(arguments.releases)))
        elif arguments.command == "init-state":
            arguments.state.write_text(json.dumps(empty_state()) + "\n", encoding="utf-8")
        elif arguments.command == "validate-state":
            load_state(arguments.state)
        elif arguments.command == "build":
            build_snapshot(
                arguments.state, arguments.dashboard, arguments.snapshot, arguments.work_directory
            )
        else:
            verify_snapshot(arguments.snapshot, arguments.destination)
    except ValueError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
