"""Read, validate, and export the simtools dependency version catalog."""

import json
import os
import re
import sys
import tomllib
from copy import deepcopy
from functools import cache
from pathlib import Path

import yaml

from simtools import version as versioning

DEPENDENCY_VERSIONS_FILENAME = "dependency_versions.yml"
PYPROJECT_FILENAME = "pyproject.toml"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ARCHIVE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SIMTOOLS_TESTS_REPOSITORY_PATTERN = re.compile(r"^[^/]+/[^/]+$")
CORSIKA_TAG_PATTERN = re.compile(r"^v\d+\.\d+$")
SOURCE_REF_PATTERN = re.compile(
    r"^(?![-/.])(?!HEAD$)(?!.*[\x00-\x20\x7f~^:?*\[\\])(?!.*\.\.)(?!.*//)"
    r"(?!.*@\{)(?!.*(?:/\.|\.lock(?:/|$)))(?!.*[./]$).+$"
)
IMAGE_TAG_PATTERN = re.compile(r"^\w[\w.-]{0,127}$", re.ASCII)
CORSIKA_INTERACTION_TABLES_LABEL = "CORSIKA interaction tables"
READABLE_REF_SCHEMAS = {"0.5.0", "0.6.0", "0.7.0"}
SOURCE_SNAPSHOT_REGISTRY = "ghcr.io/gammasim"


def _source_snapshot_image(package, digest, revisions):
    """Return an immutable private source-snapshot image reference."""
    if not digest or not all(revisions):
        return ""
    return f"{SOURCE_SNAPSHOT_REGISTRY}/{package}@{digest}"


def update_dependency_source_revisions(catalog_path, updates):
    """Update resolved source revisions and snapshot digests in a dependency catalog.

    Parameters
    ----------
    catalog_path : pathlib.Path
        Dependency catalog to update.
    updates : dict
        Resolved source revisions and immutable snapshot digests keyed by source reference.
    """
    catalog_path = Path(catalog_path)
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    for section in ("corsika", "sim-telarray"):
        if not catalog.get(section):
            raise ValueError(f"Missing dependency catalog section: {section}")
        components = {component["source-ref"]: component for component in catalog[section]}
        for source_ref, values in updates[section].items():
            components[source_ref].update(values)
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")


def _corsika_tag(component):
    """Return the external CORSIKA source tag from either catalog schema."""
    return component.get("tag", component.get("source-ref"))


def _corsika_build_id(component):
    """Return the CORSIKA build identifier from either catalog schema."""
    if "build-id" in component:
        build_id = component["build-id"]
        if not isinstance(build_id, str) or re.fullmatch(r"\d+", build_id, re.ASCII) is None:
            raise ValueError(f"CORSIKA build ID must contain only digits: {build_id!r}")
        return build_id
    if "version" in component:
        return component["version"]
    tag = _corsika_tag(component)
    if CORSIKA_TAG_PATTERN.fullmatch(tag or "") is None:
        raise ValueError(
            "CORSIKA tag must have the form v<major>.<minor> to derive its legacy build ID."
        )
    return tag.removeprefix("v").replace(".", "")


def _corsika_reference(component):
    """Return the production-combination reference for a CORSIKA record."""
    return _corsika_build_id(component) if "version" in component else _corsika_tag(component)


def corsika_source_tag_for_build_id(build_id, catalog=None):
    """Return the CORSIKA source tag matching a legacy build identifier.

    Parameters
    ----------
    build_id : str or int
        Legacy CORSIKA build identifier, for example ``78010``.
    catalog : dict, optional
        Validated dependency catalog. The installed catalog is loaded when omitted.

    Returns
    -------
    str or None
        Matching CORSIKA source tag, or ``None`` when the catalog has no match.

    Raises
    ------
    ValueError
        If more than one catalog entry matches the build identifier.
    """
    if catalog is None:
        catalog = load_dependency_catalog()
    matches = [
        _corsika_tag(component)
        for component in catalog.get("corsika", [])
        if str(_corsika_build_id(component)) == str(build_id)
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple CORSIKA source tags match build ID {build_id!r}: {matches}")
    return matches[0] if matches else None


def _simtel_tag(component):
    """Return a sim_telarray tag from either catalog schema."""
    return component.get("tag", component.get("source-ref", component.get("version")))


def _safe_build_id(source_ref, build_id=None):
    """Return an OCI-safe identifier while preserving readable ref fragments."""
    if build_id is None:
        build_id = re.sub(r"[^\w.-]+", "-", source_ref or "", flags=re.ASCII).strip("-")
    if not isinstance(build_id, str) or IMAGE_TAG_PATTERN.fullmatch(build_id) is None:
        raise ValueError(
            "sim_telarray build-id must be a valid OCI image tag when its source ref "
            f"is not safe for image names: {build_id!r}"
        )
    return build_id


def _simtel_build_id(component):
    """Return the safe image and artifact identifier for a sim_telarray record."""
    source_ref = _simtel_tag(component)
    build_id = component.get("build-id")
    if build_id is None and IMAGE_TAG_PATTERN.fullmatch(source_ref or ""):
        build_id = source_ref
    if build_id is None:
        raise ValueError(
            "sim_telarray source refs that are not valid OCI image tags require build-id: "
            f"{source_ref!r}"
        )
    return _safe_build_id(source_ref, build_id)


def _model_revision(catalog):
    """Return the configured simulation-model repository ref or revision."""
    model = catalog["model-repository"]
    return model.get("git-revision") or model.get(
        "default-ref", model.get("default-tag", model.get("default-version"))
    )


def _tests_ref(test_resources):
    """Return the human-readable simtools-tests Git ref."""
    return test_resources.get("ref")


def _tests_resource_version(test_resources):
    """Return the versioned simtools-tests resource directory name."""
    return test_resources.get("resource-version")


def _tests_tag(test_resources):
    """Return the simtools-tests tag from pre-0.5 catalogs."""
    return test_resources.get("tag", test_resources.get("version", ""))


def _dependency_tag(component, new_key, *old_keys):
    """Read a renamed dependency field while retaining old catalogs."""
    for key in (new_key, *old_keys):
        if key in component:
            return component[key]
    return None


def find_dependency_versions(start_path=None):
    """Find the nearest root-level dependency version catalog.

    Parameters
    ----------
    start_path : str or Path, optional
        Directory from which to start searching.

    Returns
    -------
    pathlib.Path
        Path to the matching ``dependency_versions.yml`` file.

    Raises
    ------
    FileNotFoundError
        If no matching project file can be found.
    """
    configured_path = os.getenv("SIMTOOLS_DEPENDENCY_VERSIONS")
    candidates = [Path(configured_path)] if configured_path else []
    start = Path(start_path or Path.cwd()).resolve()
    candidates.extend(parent / DEPENDENCY_VERSIONS_FILENAME for parent in (start, *start.parents))
    candidates.append(Path(__file__).resolve().parents[2] / DEPENDENCY_VERSIONS_FILENAME)
    candidates.append(Path(sys.prefix) / "simtools" / DEPENDENCY_VERSIONS_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {DEPENDENCY_VERSIONS_FILENAME}.")


def find_pyproject(start_path=None):
    """Find the nearest ``pyproject.toml`` file."""
    configured_path = os.getenv("SIMTOOLS_PYPROJECT")
    candidates = [Path(configured_path)] if configured_path else []
    start = Path(start_path or Path.cwd()).resolve()
    candidates.extend(parent / PYPROJECT_FILENAME for parent in (start, *start.parents))
    candidates.append(Path(__file__).resolve().parents[2] / PYPROJECT_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {PYPROJECT_FILENAME}.")


@cache
def _load_dependency_catalog_from_file(catalog_file, validate):
    """Load and optionally validate one catalog file for the process lifetime."""
    catalog = yaml.safe_load(Path(catalog_file).read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError(f"Dependency catalog must contain a mapping: {catalog_file}")
    return validate_dependency_catalog(catalog) if validate else catalog


def load_dependency_catalog(catalog_path=None, validate=True):
    """Load the dependency version catalog from YAML.

    Parameters
    ----------
    catalog_path : str or Path, optional
        Explicit catalog file. The repository is searched when omitted.
    validate : bool, optional
        Validate the catalog structure when True.

    Returns
    -------
    dict
        Dependency version catalog.
    """
    catalog_file = (Path(catalog_path) if catalog_path else find_dependency_versions()).resolve()
    return deepcopy(_load_dependency_catalog_from_file(str(catalog_file), validate))


def validate_dependency_catalog(catalog):
    """Validate required dependency catalog values.

    Parameters
    ----------
    catalog : dict
        Dependency version catalog.

    Returns
    -------
    dict
        The validated catalog.

    Raises
    ------
    ValueError
        If a required value is missing or invalid.
    """
    required = {
        "schema_version",
        "python",
        "apptainer",
        "cpu-variants",
        "base-image",
        "corsika-interaction-tables",
        "archives",
        "model-repository",
        "production-combinations",
        "corsika",
        "sim-telarray",
    }
    missing = sorted(required - catalog.keys())
    if missing:
        raise ValueError(f"Missing dependency catalog keys: {', '.join(missing)}")
    schema_version = catalog["schema_version"]
    if schema_version not in {"0.1.0", "0.2.0", "0.3.0", "0.4.0", *READABLE_REF_SCHEMAS}:
        raise ValueError(f"Unsupported dependency catalog schema version: {schema_version}")
    if schema_version != "0.1.0" and "simtools-tests" not in catalog:
        raise ValueError("Missing dependency catalog keys: simtools-tests")
    _validate_optional_digest(catalog["base-image"].get("runtime-digest"), "runtime base image")
    _validate_optional_digest(catalog["base-image"].get("build-digest"), "build base image")
    _validate_archive_checksums(catalog["archives"])
    _validate_components(catalog, schema_version)
    _validate_production_combinations(catalog)
    return catalog


def _validate_components(catalog, schema_version):
    """Validate CORSIKA and sim_telarray component records."""
    readable_refs = schema_version in READABLE_REF_SCHEMAS
    ref_validator = _validate_source_ref if readable_refs else _validate_release_tag
    interaction_tables = catalog["corsika-interaction-tables"]
    interaction_ref = (
        interaction_tables["ref"]
        if readable_refs
        else interaction_tables.get("tag", interaction_tables.get("version"))
    )
    ref_validator(interaction_ref, CORSIKA_INTERACTION_TABLES_LABEL)
    require_revisions = schema_version in {"0.4.0", "0.5.0"}
    if readable_refs:
        _validate_source_url(interaction_tables.get("source-url"), CORSIKA_INTERACTION_TABLES_LABEL)
        revision_validator = (
            _validate_revision if require_revisions else _validate_optional_revision
        )
        revision_validator(interaction_tables.get("revision"), CORSIKA_INTERACTION_TABLES_LABEL)
    _validate_corsika_components(
        catalog["corsika"], require_revisions, ref_validator, readable_refs
    )
    _validate_simtel_components(
        catalog["sim-telarray"], require_revisions, ref_validator, readable_refs
    )
    _validate_model_and_test_components(catalog, schema_version)
    _validate_archive_versions(catalog["archives"])


def _validate_corsika_components(
    components, require_revisions=False, ref_validator=None, current_schema=False
):
    """Validate CORSIKA tags, legacy IDs, revisions, and image digests."""
    ref_validator = ref_validator or _validate_release_tag
    for component in components:
        source_tag = _corsika_tag(component)
        ref_validator(source_tag, "CORSIKA source")
        config_ref = (
            component["config-ref"]
            if current_schema
            else _dependency_tag(component, "config-tag", "config-version")
        )
        opt_patch_ref = (
            component["opt-patch-ref"]
            if current_schema
            else _dependency_tag(component, "opt-patch-tag", "opt-patch-version")
        )
        ref_validator(config_ref, "CORSIKA configuration")
        ref_validator(opt_patch_ref, "CORSIKA optimization patch")
        try:
            _corsika_build_id(component)
        except ValueError as exc:
            raise ValueError(f"Invalid CORSIKA build ID mapping for {source_tag!r}: {exc}") from exc
        revision_validator = (
            _validate_revision if require_revisions else _validate_optional_revision
        )
        revision_validator(component.get("source-revision"), "CORSIKA source")
        revision_validator(component.get("config-revision"), "CORSIKA configuration")
        revision_validator(component.get("opt-patch-revision"), "CORSIKA optimization patch")
        _validate_optional_digest(
            component.get("source-snapshot-digest"), "CORSIKA source snapshot"
        )
        if component.get("source-snapshot-digest") and not all(
            component.get(key)
            for key in ("source-revision", "config-revision", "opt-patch-revision")
        ):
            raise ValueError("CORSIKA source snapshot requires all source revisions")
        for variant, digest in component.get("image-digests", {}).items():
            _validate_optional_digest(digest, f"CORSIKA {source_tag} {variant}")


def _validate_simtel_components(
    components, require_revisions=False, ref_validator=None, current_schema=False
):
    """Validate sim_telarray component revisions and image digests."""
    ref_validator = ref_validator or _validate_release_tag
    for component in components:
        ref_validator(_simtel_tag(component), "sim_telarray")
        _simtel_build_id(component)
        hessio_ref = (
            component["hessio-ref"]
            if current_schema
            else _dependency_tag(component, "hessio-tag", "hessio-version")
        )
        stdtools_ref = (
            component["stdtools-ref"]
            if current_schema
            else _dependency_tag(component, "stdtools-tag", "stdtools-version")
        )
        ref_validator(hessio_ref, "hessio")
        ref_validator(stdtools_ref, "stdtools")
        _safe_build_id(hessio_ref)
        _safe_build_id(stdtools_ref)
        revision_validator = (
            _validate_revision if require_revisions else _validate_optional_revision
        )
        for key in ("revision", "hessio-revision", "stdtools-revision"):
            revision_validator(component.get(key), key)
        _validate_optional_digest(
            component.get("source-snapshot-digest"), "sim_telarray source snapshot"
        )
        if component.get("source-snapshot-digest") and not all(
            component.get(key) for key in ("revision", "hessio-revision", "stdtools-revision")
        ):
            raise ValueError("sim_telarray source snapshot requires all source revisions")
        _validate_optional_digest(component.get("image-digest"), "sim_telarray image")


def _validate_model_and_test_components(catalog, schema_version):
    """Validate model-repository and simtools-tests catalog values."""
    model = catalog["model-repository"]
    readable_refs = schema_version in READABLE_REF_SCHEMAS
    model_version = (
        model["default-ref"]
        if readable_refs
        else model.get("default-tag", model.get("default-version"))
    )
    if readable_refs:
        valid_model_version = _is_valid_source_ref(model_version)
    elif schema_version in {"0.2.0", "0.3.0", "0.4.0"}:
        valid_model_version = versioning.is_valid_release_tag(model_version)
    else:
        valid_model_version = isinstance(model_version, str) and versioning.is_valid_model_version(
            model_version
        )
    if not valid_model_version:
        message = "Invalid simulation-model repository revision."
        if readable_refs:
            message = "Invalid simulation-model source ref."
        elif schema_version in {"0.2.0", "0.3.0", "0.4.0"}:
            message = "Invalid simulation-model release tags."
        raise ValueError(message)
    if model.get("repository-url") is not None:
        _validate_source_url(model["repository-url"], "simulation-model repository")
    if model.get("git-revision") is not None:
        _validate_revision(model["git-revision"], "simulation-model repository")
    if schema_version != "0.1.0":
        _validate_simtools_tests(
            catalog["simtools-tests"], schema_version == "0.5.0", readable_refs
        )


def _validate_simtools_tests(test_resources, require_revision=False, readable_refs=False):
    """Validate the simtools-tests repository configuration."""
    repository = test_resources.get("repository")
    if not isinstance(repository, str) or not SIMTOOLS_TESTS_REPOSITORY_PATTERN.fullmatch(
        repository
    ):
        raise ValueError("simtools-tests repository must use the owner/name format.")
    _validate_source_url(test_resources.get("source-url"), "simtools-tests")
    source_ref = _tests_ref(test_resources) if readable_refs else _tests_tag(test_resources)
    if readable_refs:
        _validate_source_ref(source_ref, "simtools-tests")
        revision_validator = _validate_revision if require_revision else _validate_optional_revision
        revision_validator(test_resources.get("revision"), "simtools-tests")
        resource_version = _tests_resource_version(test_resources)
        if not versioning.is_valid_release_tag(resource_version):
            raise ValueError(
                "simtools-tests resource version must be a release tag starting with 'v'."
            )
    elif not versioning.is_valid_release_tag(source_ref):
        raise ValueError("simtools-tests tag must be a release tag starting with 'v'.")


def _validate_production_combinations(catalog):
    """Validate every production combination against catalogued components."""
    corsika_versions = {_corsika_reference(component) for component in catalog["corsika"]}
    simtel_versions = {_simtel_tag(component) for component in catalog["sim-telarray"]}
    cpu_variants = set(catalog["cpu-variants"])
    for combination in catalog["production-combinations"]:
        if combination["corsika"] not in corsika_versions:
            raise ValueError("Unknown CORSIKA production combination.")
        if combination["sim-telarray"] not in simtel_versions:
            raise ValueError("Unknown sim_telarray production combination.")
        invalid_variants = set(combination.get("cpu-variants", cpu_variants)) - cpu_variants
        if invalid_variants:
            raise ValueError("Unknown CPU variant in production combination.")


def _validate_digest(value, label):
    """Validate an OCI SHA-256 digest."""
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 digest for {label}: {value}")


def _validate_revision(value, label):
    """Validate a Git commit revision."""
    if not versioning.is_valid_revision(value):
        raise ValueError(f"Invalid Git revision for {label}: {value}")


def _validate_optional_digest(value, label):
    """Validate an OCI SHA-256 digest when one is declared."""
    if value is not None:
        _validate_digest(value, label)


def _validate_archive_checksums(archives):
    """Validate every optional archive SHA-256 checksum."""
    for archive_name, archive in archives.items():
        value = archive.get("sha256")
        if value is not None and (
            not isinstance(value, str) or not ARCHIVE_SHA256_PATTERN.fullmatch(value)
        ):
            raise ValueError(f"Invalid SHA-256 checksum for {archive_name}: {value}")


def _validate_optional_revision(value, label):
    """Validate a Git commit revision when one is declared."""
    if value is not None:
        _validate_revision(value, label)


def _validate_release_tag(value, label):
    """Validate a catalog-managed release tag."""
    if not versioning.is_valid_release_tag(value):
        raise ValueError(f"Invalid release tag for {label}: {value!r}")


def _is_valid_source_ref(value):
    """Return whether a Git branch or tag name is safe to use as a source ref."""
    return isinstance(value, str) and SOURCE_REF_PATTERN.fullmatch(value) is not None


def _validate_source_ref(value, label):
    """Validate a human-readable Git tag or branch name."""
    if not _is_valid_source_ref(value):
        raise ValueError(f"Invalid source ref for {label}: {value!r}")


def _validate_source_url(value, label):
    """Validate a source URL that may be cloned by a build workflow."""
    if not isinstance(value, str) or not value.startswith("https://"):
        raise ValueError(f"{label} source URL must use HTTPS.")


def _validate_archive_versions(archives):
    """Validate package versions recorded for archived dependencies."""
    for archive_name, archive in archives.items():
        if not versioning.is_valid_package_version(archive.get("version")):
            raise ValueError(
                f"Invalid package version for {archive_name}: {archive.get('version')!r}"
            )


def validate_env_template(template_path):
    """Validate non-secret runtime defaults against the dependency catalog.

    Parameters
    ----------
    template_path : str or Path
        Environment template to validate.

    Raises
    ------
    ValueError
        If catalog-managed versions are duplicated in the template.
    """
    values = {}
    for line in Path(template_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key] = value
    version_keys = {"SIMTOOLS_TESTS_RESOURCE_VERSION"}
    configured_versions = sorted(version_keys & values.keys())
    if configured_versions:
        raise ValueError(
            ".env_template must not define catalog-managed versions: "
            + ", ".join(configured_versions)
        )


def build_workflow_matrices(catalog):
    """Build GitHub Actions matrices from the dependency catalog."""
    variants = catalog["cpu-variants"]
    platform_matrix = [
        {"platform": "linux/amd64", "arch": "amd64", "runner": "ubuntu-24.04"},
        {"platform": "linux/arm64/v8", "arch": "arm64", "runner": "ubuntu-24.04-arm"},
    ]
    corsika_components = {_corsika_reference(item): item for item in catalog["corsika"]}
    simtel_components = {_simtel_tag(item): item for item in catalog["sim-telarray"]}
    corsika_matrix = [
        {
            "corsika_tag": _corsika_tag(corsika),
            "corsika_build_id": _corsika_build_id(corsika),
            "corsika_source_url": corsika["source-url"],
            "corsika_source_revision": corsika.get("source-revision", ""),
            "corsika_config_tag": _dependency_tag(
                corsika, "config-ref", "config-tag", "config-version"
            ),
            "corsika_config_source_url": corsika["config-source-url"],
            "corsika_config_revision": corsika.get("config-revision", ""),
            "corsika_opt_patch_tag": _dependency_tag(
                corsika, "opt-patch-ref", "opt-patch-tag", "opt-patch-version"
            ),
            "corsika_opt_patch_source_url": corsika["opt-patch-source-url"],
            "corsika_opt_patch_revision": corsika.get("opt-patch-revision", ""),
            "corsika_source_snapshot": _source_snapshot_image(
                "corsika7-build-inputs",
                corsika.get("source-snapshot-digest", ""),
                (
                    corsika.get("source-revision", ""),
                    corsika.get("config-revision", ""),
                    corsika.get("opt-patch-revision", ""),
                ),
            ),
            "avx_flag": variant,
        }
        for corsika in catalog["corsika"]
        for variant in variants
    ]
    production_matrix = [
        _production_matrix_entry(corsika_components, simtel_components, combination, variant)
        for combination in catalog["production-combinations"]
        for variant in combination.get("cpu-variants", variants)
    ]
    simtel_matrix = [
        {
            "simtel_tag": _simtel_tag(component),
            "simtel_build_id": _simtel_build_id(component),
            "simtel_source_url": component["source-url"],
            "simtel_revision": component.get("revision", ""),
            "hessio_tag": _dependency_tag(component, "hessio-ref", "hessio-tag", "hessio-version"),
            "hessio_build_id": _safe_build_id(
                _dependency_tag(component, "hessio-ref", "hessio-tag", "hessio-version")
            ),
            "hessio_source_url": component["hessio-source-url"],
            "hessio_revision": component.get("hessio-revision", ""),
            "stdtools_tag": _dependency_tag(
                component, "stdtools-ref", "stdtools-tag", "stdtools-version"
            ),
            "stdtools_build_id": _safe_build_id(
                _dependency_tag(component, "stdtools-ref", "stdtools-tag", "stdtools-version")
            ),
            "stdtools_source_url": component["stdtools-source-url"],
            "stdtools_revision": component.get("stdtools-revision", ""),
            "simtel_source_snapshot": _source_snapshot_image(
                "simtel-array-build-inputs",
                component.get("source-snapshot-digest", ""),
                (
                    component.get("revision", ""),
                    component.get("hessio-revision", ""),
                    component.get("stdtools-revision", ""),
                ),
            ),
        }
        for component in catalog["sim-telarray"]
    ]
    corsika_source_matrix = [
        {
            "corsika_tag": _corsika_tag(component),
            "corsika_source_url": component["source-url"],
            "corsika_source_revision": component.get("source-revision", ""),
            "corsika_config_tag": _dependency_tag(
                component, "config-ref", "config-tag", "config-version"
            ),
            "corsika_config_source_url": component["config-source-url"],
            "corsika_config_revision": component.get("config-revision", ""),
            "corsika_opt_patch_tag": _dependency_tag(
                component, "opt-patch-ref", "opt-patch-tag", "opt-patch-version"
            ),
            "corsika_opt_patch_source_url": component["opt-patch-source-url"],
            "corsika_opt_patch_revision": component.get("opt-patch-revision", ""),
            "corsika_source_snapshot": _source_snapshot_image(
                "corsika7-build-inputs",
                component.get("source-snapshot-digest", ""),
                (
                    component.get("source-revision", ""),
                    component.get("config-revision", ""),
                    component.get("opt-patch-revision", ""),
                ),
            ),
        }
        for component in catalog["corsika"]
    ]
    return {
        "corsika_matrix": corsika_matrix,
        "corsika_build_matrix": [
            {**item, **platform}
            for item in corsika_matrix
            for platform in platform_matrix
            if item["avx_flag"] == "generic" or platform["arch"] == "amd64"
        ],
        "corsika_source_matrix": [
            {"corsika_build_id": _corsika_build_id(component), **source}
            for component, source in zip(catalog["corsika"], corsika_source_matrix)
        ],
        "simtel_matrix": simtel_matrix,
        "simtel_build_matrix": [
            {**item, **platform} for item in simtel_matrix for platform in platform_matrix
        ],
        "production_matrix": production_matrix,
    }


def _production_matrix_entry(corsika_components, simtel_components, combination, variant):
    """Build one production image matrix entry."""
    corsika = corsika_components[combination["corsika"]]
    simtel = simtel_components[combination["sim-telarray"]]
    production_corsika = _corsika_tag(corsika) or f"v{_corsika_build_id(corsika)}"
    return {
        "corsika_tag": production_corsika,
        "corsika_build_id": _corsika_build_id(corsika),
        "corsika_image": _image_reference(
            "ghcr.io/gammasim/corsika7",
            f"v{_corsika_build_id(corsika)}-{variant}",
            corsika.get("image-digests", {}).get(variant),
        ),
        "simtel_tag": _simtel_tag(simtel),
        "simtel_build_id": _simtel_build_id(simtel),
        "simtel_image": _image_reference(
            "ghcr.io/gammasim/sim_telarray",
            _simtel_build_id(simtel),
            simtel.get("image-digest"),
        ),
        "avx_flag": variant,
    }


def _image_reference(name, tag, digest=None):
    """Return a digest reference when declared, otherwise a version tag."""
    return f"{name}@{digest}" if digest else f"{name}:{tag}"


def dependency_catalog_summary(catalog):
    """Return stable scalar build values used by Docker workflows."""
    base = catalog["base-image"]
    default_corsika = catalog["corsika"][0]
    default_simtel = catalog["sim-telarray"][0]
    test_resources = catalog.get("simtools-tests", {})
    return {
        "python_version": catalog["python"],
        "apptainer_version": catalog["apptainer"],
        "base_image": _image_reference(
            base["name"], base["runtime-version"], base.get("runtime-digest")
        ),
        "build_base_image": _image_reference(
            base["name"], base["build-version"], base.get("build-digest")
        ),
        "almalinux_version": base["runtime-version"].removesuffix("-minimal"),
        "autoconf_version": catalog["archives"]["autoconf"]["version"],
        "autoconf_sha256": catalog["archives"]["autoconf"].get("sha256", ""),
        "default_corsika_source_snapshot": _source_snapshot_image(
            "corsika7-build-inputs",
            default_corsika.get("source-snapshot-digest", ""),
            (
                default_corsika.get("source-revision", ""),
                default_corsika.get("config-revision", ""),
                default_corsika.get("opt-patch-revision", ""),
            ),
        ),
        "gsl_version": catalog["archives"]["gsl"]["version"],
        "gsl_sha256": catalog["archives"]["gsl"].get("sha256", ""),
        "default_simtel_source_snapshot": _source_snapshot_image(
            "simtel-array-build-inputs",
            default_simtel.get("source-snapshot-digest", ""),
            (
                default_simtel.get("revision", ""),
                default_simtel.get("hessio-revision", ""),
                default_simtel.get("stdtools-revision", ""),
            ),
        ),
        "corsika_tables_ref": _dependency_tag(
            catalog["corsika-interaction-tables"], "ref", "tag", "version"
        ),
        "corsika_tables_revision": catalog["corsika-interaction-tables"].get("revision", ""),
        "model_repository": catalog["model-repository"].get("repository-url", ""),
        "model_repository_revision": _model_revision(catalog),
        "simtools_tests_repository": test_resources.get("repository", ""),
        "simtools_tests_url": test_resources.get("source-url", ""),
        "simtools_tests_ref": _tests_ref(test_resources) or _tests_tag(test_resources),
        "simtools_tests_resource_version": _tests_resource_version(test_resources),
        "simtools_tests_revision": test_resources.get("revision", ""),
        "dev_corsika_image": _image_reference(
            "ghcr.io/gammasim/corsika7",
            f"v{_corsika_build_id(default_corsika)}-generic",
            default_corsika.get("image-digests", {}).get("generic"),
        ),
        "dev_simtel_image": _image_reference(
            "ghcr.io/gammasim/sim_telarray",
            _simtel_build_id(default_simtel),
            default_simtel.get("image-digest"),
        ),
    }


def dependency_catalog_environment(catalog):
    """Return catalog-managed runtime values as environment assignments.

    Parameters
    ----------
    catalog : dict
        Validated dependency version catalog.

    Returns
    -------
    dict
        Environment variable names and values for model repository and test-resource
        configuration. Local paths and credentials are intentionally omitted.
    """
    environment = {
        "SIMTOOLS_SIMULATION_MODELS_GIT_REVISION": _model_revision(catalog),
    }
    if "simtools-tests" in catalog:
        if catalog["schema_version"] in READABLE_REF_SCHEMAS:
            test_environment = {
                "SIMTOOLS_TESTS_REF": _tests_ref(catalog["simtools-tests"]),
                "SIMTOOLS_TESTS_RESOURCE_VERSION": _tests_resource_version(
                    catalog["simtools-tests"]
                ),
            }
        else:
            if catalog["schema_version"] in {"0.3.0", "0.4.0"}:
                tag_key = "SIMTOOLS_TESTS_TAG"
            else:
                tag_key = "SIMTOOLS_TESTS_VERSION"
            test_environment = {tag_key: _tests_tag(catalog["simtools-tests"])}
        environment.update(
            {
                **test_environment,
                "SIMTOOLS_TESTS_REPOSITORY": catalog["simtools-tests"]["repository"],
                "SIMTOOLS_TESTS_URL": catalog["simtools-tests"]["source-url"],
            }
        )
        revision = catalog["simtools-tests"].get("revision")
        if revision:
            environment["SIMTOOLS_TESTS_REVISION"] = revision
    return environment


def project_requirements(pyproject_path, extras):
    """Return project requirements, optionally including named extras."""
    with Path(pyproject_path).open("rb") as file:
        project = tomllib.load(file)["project"]
    requirements = list(project["dependencies"])
    optional = project.get("optional-dependencies", {})
    for extra in extras:
        try:
            requirements.extend(optional[extra])
        except KeyError as exc:
            available = ", ".join(sorted(optional)) or "none"
            raise ValueError(
                f"Unknown optional-dependency group: {extra}. Available groups: {available}"
            ) from exc
    return requirements


def export_dependency_configuration(pyproject_path=None, output_format="catalog", extras=None):
    """Return dependency configuration in a selected export format.

    Parameters
    ----------
    pyproject_path : str or Path, optional
        Explicit project file, required for ``python-requirements`` output.
        The repository is searched when omitted for that output format.
    output_format : str, optional
        One of ``catalog``, ``env``, ``github-output``, ``python-requirements``, or ``summary``.
    extras : list of str, optional
        Optional dependency groups included in ``python-requirements`` output.

    Returns
    -------
    str
        Serialized dependency configuration, including a trailing newline.
    """
    project_file = Path(pyproject_path) if pyproject_path else None
    catalog_file = (
        project_file.with_name(DEPENDENCY_VERSIONS_FILENAME)
        if project_file
        else find_dependency_versions()
    )
    catalog = load_dependency_catalog(catalog_file)
    env_template = catalog_file.parent / ".env_template"
    if env_template.is_file():
        validate_env_template(env_template)
    extras = extras or []
    if output_format == "python-requirements":
        project_file = project_file or find_pyproject()
        return "\n".join(project_requirements(project_file, extras)) + "\n"
    if output_format == "catalog":
        return json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    if output_format == "summary":
        return json.dumps(dependency_catalog_summary(catalog), sort_keys=True) + "\n"
    if output_format == "env":
        return "".join(
            f"{key}={value}\n" for key, value in dependency_catalog_environment(catalog).items()
        )
    if output_format == "github-output":
        output = {**dependency_catalog_summary(catalog), **build_workflow_matrices(catalog)}
        return "".join(f"{key}={_github_output_value(value)}\n" for key, value in output.items())
    raise ValueError(f"Unsupported dependency export format: {output_format}")


def _github_output_value(value):
    """Serialize a GitHub Actions output value."""
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return value
