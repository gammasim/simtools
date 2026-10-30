"""Assemble a multi-version documentation site from released documentation builds.

A version switcher file (as used by the pydata-sphinx-theme) lists the released versions to
be published. Each version is downloaded as a tarball of its built HTML site and extracted to
``<output_path>/<version>/``. The newest version is additionally copied to the site root.
Identical files are replaced by relative symlinks to one copy, so that unchanged pages of
different versions are stored only once.
"""

import hashlib
import io
import json
import logging
import os
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

VERSION_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")
LATEST_VERSION = "latest"
SWITCHER_FILE_PATH = Path("_static") / "switcher.json"
DOWNLOAD_TIMEOUT_SECONDS = 60


def version_key(version):
    """Return a sort key for version strings of the form vX.Y.Z."""
    return tuple(int(part) for part in version.lstrip("v").split("."))


def _site_url(entries, errors):
    """Return the site URL given by the 'latest' entry (served at the site root)."""
    urls = [entry.get("url", "") for entry in entries if entry.get("version") == LATEST_VERSION]
    if len(urls) != 1:
        errors.append(f"Exactly one '{LATEST_VERSION}' entry for the site root is required.")
        return None
    if not urls[0].startswith("https://") or not urls[0].endswith("/"):
        errors.append(f"Site URL {urls[0]!r} must start with 'https://' and end with '/'.")
    return urls[0]


def _release_versions(entries, site_url, errors):
    """Return the listed release versions; each must be served at <site_url>vX.Y.Z/."""
    versions = []
    for entry in entries:
        version = entry.get("version")
        if version == LATEST_VERSION:
            continue
        if not VERSION_PATTERN.match(str(version)):
            errors.append(f"Version {version!r} does not match vX.Y.Z.")
        elif site_url is not None and entry.get("url") != f"{site_url}{version}/":
            errors.append(f"URL of {version} must be {site_url}{version}/.")
        else:
            versions.append(version)
    return versions


def read_version_switcher(switcher_file):
    """
    Read and validate a version switcher file.

    The 'latest' entry gives the site URL; each released version vX.Y.Z must be served
    at ``<site URL>vX.Y.Z/``.

    Parameters
    ----------
    switcher_file : Path or str
        Version switcher JSON file.

    Returns
    -------
    list of str
        Listed release versions, newest first.

    Raises
    ------
    ValueError
        If the file content is not a valid version switcher.
    """
    entries = json.loads(Path(switcher_file).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{switcher_file}: expected a non-empty list of versions.")
    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{switcher_file}: each version entry must be an object.")

    errors = []
    versions = _release_versions(entries, _site_url(entries, errors), errors)
    if sum(1 for entry in entries if entry.get("preferred")) != 1:
        errors.append("Exactly one entry must be marked as 'preferred'.")
    if len(set(versions)) != len(versions):
        errors.append("Duplicate versions listed.")
    if not versions:
        errors.append("No released version listed.")
    if errors:
        raise ValueError(f"Invalid version switcher file {switcher_file}: " + " ".join(errors))
    return sorted(versions, key=version_key, reverse=True)


def download_version(version, package_url, output_path, token=None):
    """
    Download and extract the documentation tarball of a released version.

    Parameters
    ----------
    version : str
        Release version (vX.Y.Z).
    package_url : str
        URL template of the documentation tarball with a ``{version}`` placeholder.
    output_path : Path
        Site directory; the version is extracted to ``output_path / version``.
    token : str, optional
        GitLab CI job token sent as ``JOB-TOKEN`` header.
    """
    url = package_url.format(version=version)
    request = urllib.request.Request(url)
    if token:
        request.add_header("JOB-TOKEN", token)
    logger.info(f"Downloading documentation {version} from {url}")
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        data = response.read()
    target = Path(output_path) / version
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(target, filter="data")


def copy_version_to_root(output_path, version):
    """Copy a version to the site root (served as latest version)."""
    output_path = Path(output_path)
    for item in (output_path / version).iterdir():
        target = output_path / item.name
        if item.is_dir():
            shutil.copytree(item, target, symlinks=True)
        else:
            shutil.copy2(item, target, follow_symlinks=False)


def remove_source_maps(output_path):
    """Remove JavaScript and CSS source maps, which are not needed for serving."""
    for path in Path(output_path).rglob("*.map"):
        path.unlink()


def deduplicate(output_path):
    """
    Replace identical files by relative symlinks to one canonical copy.

    Files at the site root are preferred as canonical copies, as the root is always
    deployed while individual versions may be removed.

    Parameters
    ----------
    output_path : Path
        Site directory.
    """
    output_path = Path(output_path)

    def in_version_directory(path):
        return bool(VERSION_PATTERN.match(path.relative_to(output_path).parts[0]))

    canonical = {}
    files = [path for path in output_path.rglob("*") if path.is_file() and not path.is_symlink()]
    for path in sorted(files, key=lambda path: (in_version_directory(path), str(path))):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in canonical:
            canonical[digest] = path
            continue
        target = os.path.relpath(canonical[digest], path.parent)
        path.unlink()
        path.symlink_to(target)


def deployed_size(path):
    """Return the size in bytes of a directory; symlinks count with the length of their target."""
    return sum(
        item.lstat().st_size
        for item in Path(path).rglob("*")
        if item.is_symlink() or item.is_file()
    )


def check_size(output_path, versions, size_budget_mb):
    """
    Log the deployed size per version and check it against a size budget.

    Raises
    ------
    RuntimeError
        If the deployed size exceeds the size budget.
    """
    output_path = Path(output_path)
    for version in versions:
        logger.info(f"Deployed size {version}: {deployed_size(output_path / version) / 1e6:.1f} MB")
    total = deployed_size(output_path)
    logger.info(f"Total deployed size: {total / 1e6:.1f} MB (budget {size_budget_mb} MB)")
    if total > size_budget_mb * 1e6:
        raise RuntimeError(
            f"Deployed size {total / 1e6:.1f} MB exceeds the budget of {size_budget_mb} MB; "
            "remove versions from the version switcher file."
        )


def assemble_pages(switcher_file, output_path, package_url, size_budget_mb, token=None):
    """
    Assemble the multi-version documentation site.

    Parameters
    ----------
    switcher_file : Path or str
        Version switcher JSON file listing the versions to publish.
    output_path : Path or str
        Site directory (must be empty or not exist).
    package_url : str
        URL template of the documentation tarballs with a ``{version}`` placeholder.
    size_budget_mb : float
        Maximum deployed size in MB.
    token : str, optional
        GitLab CI job token for downloading the tarballs.
    """
    versions = read_version_switcher(switcher_file)
    logger.info(f"Versions to publish: {', '.join(versions)} (latest: {versions[0]})")

    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Output directory {output_path} is not empty.")
    output_path.mkdir(parents=True, exist_ok=True)
    for version in versions:
        download_version(version, package_url, output_path, token)

    copy_version_to_root(output_path, versions[0])
    (output_path / SWITCHER_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(switcher_file, output_path / SWITCHER_FILE_PATH)
    remove_source_maps(output_path)
    deduplicate(output_path)
    check_size(output_path, versions, size_budget_mb)
