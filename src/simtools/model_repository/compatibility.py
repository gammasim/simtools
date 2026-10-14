"""Resolve external model identities and enforce release compatibility contracts."""

import hashlib
import io
import json
import logging
import re
import subprocess
import tarfile
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)
MODEL_KEY = "model-repository"
LEGACY_MODEL_KEY = "model-database"
RANGE_KEY = "compatible-versions"
DEFAULT_KEY = "default"
ALLOWED_KEY = "allowed"
POLICY_KEY = "policy"
VERSION_KEY = "version"
COMMIT_KEY = "commit"
DIGEST_KEY = "content_sha256"
SOURCE_KEY = "source"
STATUS_KEY = "status"
MODELS_DIRECTORY = "simulation-models"
RELEASE_MANIFEST = "simulation-models-release.json"
LEGACY_POLICY = "legacy exact-default fallback"
CATALOG_POLICY = "target catalog range"
DEVELOPMENT_STATUS = "development; not production-qualified"


def _contract(catalog):
    """Read a model range, including catalogs bundled with historical images."""
    model = catalog.get(MODEL_KEY) or catalog.get(LEGACY_MODEL_KEY)
    if not model:
        raise ValueError("Target catalog has no model dependency declaration.")
    default = next(
        (model[key] for key in ("default-ref", "default-tag", "default-version") if model.get(key)),
        None,
    )
    try:
        version = Version(str(default))
        constraint = model.get(RANGE_KEY, f"=={version}")
        allowed = SpecifierSet(constraint)
    except (InvalidVersion, InvalidSpecifier, TypeError) as exc:
        raise ValueError(f"Invalid model release default or compatible-versions: {model}") from exc
    if not str(allowed):
        raise ValueError("An explicit nonempty model compatibility constraint is required.")
    if (
        version.is_prerelease
        or version.is_devrelease
        or not allowed.contains(version, prereleases=False)
    ):
        raise ValueError(f"Model default {default} is not a supported stable release in {allowed}.")
    return {
        DEFAULT_KEY: default,
        ALLOWED_KEY: str(allowed),
        POLICY_KEY: CATALOG_POLICY if RANGE_KEY in model else LEGACY_POLICY,
        "catalog_sha256": hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest(),
    }


def _git(path, *arguments):
    """Read Git metadata without fetching or changing the working tree."""
    return subprocess.run(
        ["git", "-C", str(path), *arguments], check=True, capture_output=True
    ).stdout


def _release_tag(tags):
    """Require an unambiguous stable release exactly at the selected commit."""
    versions = set()
    for tag in tags.splitlines():
        if not re.fullmatch(r"v?\d+\.\d+\.\d+", tag):
            continue
        versions.add(str(Version(tag)))
    if len(versions) != 1:
        raise ValueError(
            "Model identity unknown: select one exact, unambiguous stable release tag."
        )
    return versions.pop()


def _tree_digest(path):
    """Hash model file names and contents, rejecting links outside the snapshot."""
    root = Path(path) / MODELS_DIRECTORY
    if not root.is_dir():
        raise ValueError(f"Model data directory is missing: {root}")
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    for item in paths:
        if item.is_symlink():
            raise ValueError(f"Production model snapshots cannot contain symbolic links: {item}")
        if item.is_file():
            _add_file_digest(digest, item.relative_to(root.parent).as_posix(), item.read_bytes())
    return digest.hexdigest()


def _add_file_digest(digest, name, content):
    digest.update(name.encode() + b"\0" + hashlib.sha256(content).digest())


def _archive_digest(path, commit):
    """Hash the model tree in a fixed Git commit without exporting it."""
    archive = _git(path, "archive", commit, MODELS_DIRECTORY)
    digest = hashlib.sha256()
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        for member in sorted(handle.getmembers(), key=lambda entry: entry.name):
            if member.issym() or member.islnk():
                raise ValueError(f"Production model snapshot contains a link: {member.name}")
            if member.isfile():
                _add_file_digest(digest, member.name, handle.extractfile(member).read())
    return digest.hexdigest()


def _identity(path, revision=None):
    """Identify a clean checkout, a fixed Git source, or a verified model export."""
    root = Path(path).expanduser().resolve()
    if revision is not None or (root / ".git").exists():
        commit = _git(root, "rev-parse", f"{revision or 'HEAD'}^{{commit}}").decode().strip()
        version = _release_tag(_git(root, "tag", "--points-at", commit).decode())
        if (
            revision is None
            and _git(
                root, STATUS_KEY, "--porcelain", "--untracked-files=all", "--", MODELS_DIRECTORY
            ).strip()
        ):
            raise ValueError(f"Production models have local modifications: {root}")
        digest = _archive_digest(root, commit)
        if revision is None and _tree_digest(root) != digest:
            raise ValueError(f"Production model contents differ from release {version}: {root}")
    else:
        manifest_path = root / RELEASE_MANIFEST
        if not manifest_path.is_file():
            raise ValueError(
                f"Model identity unknown: {root} needs Git metadata or {RELEASE_MANIFEST}."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = _release_tag(str(manifest.get(VERSION_KEY, "")))
        commit = manifest.get(COMMIT_KEY, "")
        digest = _tree_digest(root)
        if not re.fullmatch(r"[0-9a-f]{40}", commit) or digest != manifest.get(DIGEST_KEY):
            raise ValueError(
                f"Model release manifest identity or content digest is invalid: {root}"
            )
    return {VERSION_KEY: version, COMMIT_KEY: commit, DIGEST_KEY: digest, SOURCE_KEY: str(root)}


def _validate(path, catalog, revision=None, development=False):
    """Validate the selected source without silently replacing it."""
    contract = _contract(catalog)
    try:
        identity = _identity(path, revision)
    except (ValueError, subprocess.CalledProcessError) as exc:
        if not development:
            raise ValueError(f"Dependency compatibility check failed: {exc}") from exc
        logger.warning("Model identity unknown: %s; %s", exc, DEVELOPMENT_STATUS)
        return {**contract, SOURCE_KEY: str(path), STATUS_KEY: DEVELOPMENT_STATUS}
    compatible = SpecifierSet(contract[ALLOWED_KEY]).contains(
        identity[VERSION_KEY], prereleases=False
    )
    if not compatible and not development:
        raise ValueError(
            f"Dependency compatibility check failed: simulation-models {identity[VERSION_KEY]} "
            f"at {identity[SOURCE_KEY]}; allowed {contract[ALLOWED_KEY]} "
            f"({contract[POLICY_KEY]}), default {contract[DEFAULT_KEY]}. "
            "Select a compatible filesystem checkout. No production jobs were submitted."
        )
    status = "compatible" if compatible and not development else DEVELOPMENT_STATUS
    result = {**contract, **identity, STATUS_KEY: status}
    logger.info(
        "Models: %s at %s; allowed %s (%s); %s",
        identity[VERSION_KEY],
        identity[SOURCE_KEY],
        contract[ALLOWED_KEY],
        contract[POLICY_KEY],
        status,
    )
    return result
