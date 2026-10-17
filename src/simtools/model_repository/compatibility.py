"""Check model release versions against the simtools dependency catalog."""

import logging
import re
import subprocess
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
SOURCE_KEY = "source"
STATUS_KEY = "status"
LEGACY_POLICY = "legacy exact-default fallback"
CATALOG_POLICY = "target catalog range"
DEVELOPMENT_STATUS = "development; not production-qualified"
TAG_SELECTION_ERROR = "Tags at selected revision: {tags}. Required: exactly one stable release tag."


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
    }


def _validate(path, catalog, revision=None, development=False):
    """Check an exact release tag without changing the selected model checkout."""
    contract = _contract(catalog)
    source = str(Path(path).expanduser().resolve())
    result = {**contract, SOURCE_KEY: source, STATUS_KEY: DEVELOPMENT_STATUS}
    try:
        tags = subprocess.run(
            ["git", "-C", source, "tag", "--points-at", revision or "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        versions = {str(Version(tag)) for tag in tags if re.fullmatch(r"v?\d+\.\d+\.\d+", tag)}
        if len(versions) != 1:
            found_tags = ", ".join(tags) if tags else "none"
            raise ValueError(TAG_SELECTION_ERROR.format(tags=found_tags))
        result[VERSION_KEY] = versions.pop()
        if not SpecifierSet(contract[ALLOWED_KEY]).contains(result[VERSION_KEY], prereleases=False):
            raise ValueError(
                f"simulation-models {result[VERSION_KEY]} is outside the allowed range."
            )
        if not development:
            result[STATUS_KEY] = "compatible"
    except (ValueError, subprocess.CalledProcessError) as exc:
        if not development:
            raise ValueError(
                f"Model compatibility failed for {source}: {exc} "
                f"Allowed {contract[ALLOWED_KEY]} ({contract[POLICY_KEY]}). "
                f"Select a compatible checkout, for example {contract[DEFAULT_KEY]}."
            ) from exc
    log = logger.warning if development else logger.info
    log(
        "Models %s; version %s at %s; allowed %s (%s)",
        result[STATUS_KEY],
        result.get(VERSION_KEY, "unknown"),
        source,
        contract[ALLOWED_KEY],
        contract[POLICY_KEY],
    )
    return result
