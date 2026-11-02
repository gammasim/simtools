"""Assemble a complete Pages site from the selected immutable release archives."""

import argparse
import importlib
import json
import re
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

VERSION = re.compile(r"v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?\Z")
SWITCHER_NAME = re.compile(r"[A-Za-z0-9._-]+\Z")
MAX_SITE_BYTES = 1_000_000_000
MAX_ARCHIVE_MEMBERS = 200_000


def _normalize_entry(entry):
    """Validate one switcher entry and return a copy with a stable name."""
    if not isinstance(entry, dict) or not isinstance(entry.get("version"), str):
        raise ValueError("Each switcher entry must contain a version")
    version = entry["version"]
    if version != "latest" and not VERSION.fullmatch(version):
        raise ValueError(f"Invalid switcher version: {version}")
    if not isinstance(entry.get("url"), str) or not entry["url"].startswith("https://"):
        raise ValueError(f"Switcher entry {version} must contain an https url")
    normalized = dict(entry)
    if "name" not in normalized:
        normalized["name"] = "Latest" if version == "latest" else version.removeprefix("v")
    if not isinstance(normalized["name"], str) or not SWITCHER_NAME.fullmatch(normalized["name"]):
        raise ValueError(f"Invalid switcher name: {normalized['name']}")
    return normalized


def read_manifest(path):
    """Read a switcher list and return entries with stable names."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Switcher must be a nonempty list")
    normalized = [_normalize_entry(entry) for entry in entries]
    versions = [entry["version"] for entry in normalized]
    if len(set(versions)) != len(versions):
        raise ValueError("Duplicate switcher version in switcher")
    if "latest" not in versions:
        raise ValueError("Switcher must include latest")
    try:
        importlib.import_module("simtools.reporting.docs_pages").read_version_switcher(path)
    except ValueError as exc:
        raise ValueError(f"Invalid version switcher: {exc}") from exc
    return normalized


def _member_path(member):
    """Return a normalized archive path after rejecting unsafe names."""
    name = member.name
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or any(part in {"..", ""} for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive member: {name}")
    if path.parts and ":" in path.parts[0]:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def _validate_member(member, relative, seen):
    if relative == PurePosixPath():
        if member.isdir():
            return False
        raise ValueError(f"Unsafe archive member: {member.name}")
    if relative in seen:
        raise ValueError(f"Duplicate archive member: {member.name}")
    seen.add(relative)
    if member.isdir():
        return False
    if member.isfile():
        return True
    raise ValueError(f"Unsafe archive member: {member.name}")


def _write_member(archive, member, target, total_size):
    if member.size < 0 or total_size + member.size > MAX_SITE_BYTES:
        raise ValueError("Assembled documentation exceeds the 1 GB Pages limit")
    for parent in target.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError(f"Archive member conflicts with a file: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"Could not read archive member: {member.name}")
    with source, target.open("xb") as output:
        shutil.copyfileobj(source, output)
    return total_size + member.size


def _extract_members(archive, members, destination, current_size):
    total_size = current_size
    seen = set()
    for member in members:
        relative = _member_path(member)
        is_file = _validate_member(member, relative, seen)
        if not relative.parts:
            continue
        target = destination.joinpath(*relative.parts)
        if is_file:
            total_size = _write_member(archive, member, target, total_size)
        else:
            target.mkdir(parents=True, exist_ok=True)
    return total_size


def extract_archive(archive_path, destination, *, current_size=0):
    """Extract regular files and directories while enforcing archive safety."""
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError(f"Archive has too many members: {archive_path}")
            return _extract_members(archive, members, destination, current_size)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Could not safely read archive {archive_path}: {exc}") from exc


def _validate_inputs(site, manifest, archives_dir, latest_archive):
    entries = read_manifest(manifest)
    selected = [entry["version"] for entry in entries if entry["version"] != "latest"]
    archives = {version: archives_dir / f"{version}.tar.gz" for version in selected}
    missing = [str(path) for path in archives.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing archive for selected version: {', '.join(missing)}")
    if not latest_archive.is_file():
        raise ValueError(f"Latest documentation archive does not exist: {latest_archive}")
    if site.exists() and not site.is_dir():
        raise ValueError(f"Site output must be a directory: {site}")
    return entries, archives


def _preserve_dashboard(site, staging):
    dashboard = site / "dev/test-resources"
    if dashboard.exists():
        if not dashboard.is_dir() or any(path.is_symlink() for path in (site / "dev", dashboard)):
            raise ValueError(f"Benchmark dashboard must be a directory: {dashboard}")
        for path in dashboard.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Benchmark dashboard contains a symbolic link: {path}")
        shutil.copytree(dashboard, staging / "dev/test-resources", symlinks=False)


def _validate_site_tree(site):
    total_size = 0
    for path in site.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Assembled site contains a symbolic link: {path}")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ValueError(f"Assembled site contains a hard link: {path}")
            total_size += metadata.st_size
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Assembled site contains a special file: {path}")
    if total_size > MAX_SITE_BYTES:
        raise ValueError("Assembled documentation exceeds the 1 GB Pages limit")


def _install_staging(site, staging, temporary_root):
    backup = temporary_root / "previous-site"
    moved_old_site = False
    if site.exists():
        site.replace(backup)
        moved_old_site = True
    try:
        staging.replace(site)
    except OSError:
        if moved_old_site:
            backup.replace(site)
        raise
    if moved_old_site:
        shutil.rmtree(backup)


def assemble(site, manifest, archives_dir, latest_archive):
    """Rebuild the site from the manifest and archives without retaining old versions."""
    entries, archives = _validate_inputs(site, manifest, archives_dir, latest_archive)
    site.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pages-site-", dir=site.parent) as temporary:
        temporary_root = Path(temporary)
        staging = temporary_root / "assembled-site"
        staging.mkdir()
        _preserve_dashboard(site, staging)
        total_size = extract_archive(latest_archive, staging)
        if not (staging / "index.html").is_file():
            raise ValueError("The latest archive must contain index.html at its root")
        for version, archive_path in archives.items():
            total_size = extract_archive(archive_path, staging / version, current_size=total_size)
            if not (staging / version / "index.html").is_file():
                raise ValueError(f"The {version} archive must contain index.html at its root")
        static_directory = staging / "_static"
        static_directory.mkdir(exist_ok=True)
        (static_directory / "switcher.json").write_text(
            json.dumps(entries, indent=2) + "\n", encoding="utf-8"
        )
        (staging / ".nojekyll").touch()
        _validate_site_tree(staging)
        _install_staging(site, staging, temporary_root)


def _write_versions(manifest):
    versions = [
        entry["version"] for entry in read_manifest(manifest) if entry["version"] != "latest"
    ]
    if versions:
        print("\n".join(versions))


def main():
    """Validate manifests, safely extract archives, or assemble the Pages site."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    versions_parser = commands.add_parser("versions", help="list validated release versions")
    versions_parser.add_argument("manifest", type=Path)

    extract_parser = commands.add_parser("extract", help="safely extract a site archive")
    extract_parser.add_argument("archive", type=Path)
    extract_parser.add_argument("destination", type=Path)

    assemble_parser = commands.add_parser("assemble", help="build the complete Pages site")
    assemble_parser.add_argument("--site", type=Path, required=True)
    assemble_parser.add_argument("--manifest", type=Path, required=True)
    assemble_parser.add_argument("--archives-dir", type=Path, required=True)
    assemble_parser.add_argument("--latest", type=Path, required=True)

    arguments = parser.parse_args()
    if arguments.command == "versions":
        _write_versions(arguments.manifest)
    elif arguments.command == "extract":
        extract_archive(arguments.archive, arguments.destination)
    else:
        assemble(arguments.site, arguments.manifest, arguments.archives_dir, arguments.latest)


if __name__ == "__main__":
    main()
