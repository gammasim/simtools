"""Read, validate, and export the simtools dependency version catalog."""

import json
import os
import re
import sys
import tomllib
from pathlib import Path

import yaml

from simtools import version as versioning

DEPENDENCY_VERSIONS_FILENAME = "dependency_versions.yml"
PYPROJECT_FILENAME = "pyproject.toml"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ARCHIVE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
LEGACY_MODEL_VERSION_PATTERN = re.compile(r"^\d+\.\d+(?:\.\d+)?$")
SIMTOOLS_TESTS_REPOSITORY_PATTERN = re.compile(r"^[^/]+/[^/]+$")
CORSIKA_TAG_PATTERN = re.compile(r"^v\d+\.\d+$")


def _corsika_tag(component):
    """Return the external CORSIKA source tag from either catalog schema."""
    return component.get("tag", component.get("source-ref"))


def _corsika_build_id(component):
    """Return the legacy CORSIKA build identifier.

    New catalog records derive this legacy identifier from the source tag.
    """
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
    return _corsika_tag(component) if "tag" in component else _corsika_build_id(component)


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
    return component.get("tag", component.get("version"))


def _model_tag(catalog):
    """Return the model-database tag from either catalog schema."""
    model = catalog["model-database"]
    return model.get("default-tag", model.get("default-version"))


def _tests_tag(test_resources):
    """Return the simtools-tests tag from either catalog schema."""
    return test_resources.get("tag", test_resources.get("version", ""))


def _dependency_tag(component, new_key, old_key):
    """Read a renamed dependency field while retaining old catalogs."""
    return component.get(new_key, component.get(old_key))


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
    catalog_file = Path(catalog_path) if catalog_path else find_dependency_versions()
    catalog = yaml.safe_load(catalog_file.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise ValueError(f"Dependency catalog must contain a mapping: {catalog_file}")
    return validate_dependency_catalog(catalog) if validate else catalog


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
        "model-database",
        "production-combinations",
        "corsika",
        "sim-telarray",
    }
    missing = sorted(required - catalog.keys())
    if missing:
        raise ValueError(f"Missing dependency catalog keys: {', '.join(missing)}")
    schema_version = catalog["schema_version"]
    if schema_version not in {"0.1.0", "0.2.0", "0.3.0"}:
        raise ValueError(f"Unsupported dependency catalog schema version: {schema_version}")
    if schema_version in {"0.2.0", "0.3.0"} and "simtools-tests" not in catalog:
        raise ValueError("Missing dependency catalog keys: simtools-tests")
    _validate_optional_digest(catalog["base-image"].get("runtime-digest"), "runtime base image")
    _validate_optional_digest(catalog["base-image"].get("build-digest"), "build base image")
    _validate_archive_checksums(catalog["archives"])
    _validate_components(catalog, schema_version)
    _validate_production_combinations(catalog)
    return catalog


def _validate_components(catalog, schema_version):
    """Validate CORSIKA and sim_telarray component records."""
    _validate_corsika_components(catalog["corsika"])
    _validate_simtel_components(catalog["sim-telarray"])
    _validate_model_and_test_components(catalog, schema_version)


def _validate_corsika_components(components):
    """Validate CORSIKA tags, legacy IDs, revisions, and image digests."""
    for component in components:
        source_tag = _corsika_tag(component)
        if source_tag in {"latest", "master", "main"}:
            raise ValueError("CORSIKA tag must identify a release.")
        try:
            _corsika_build_id(component)
        except ValueError as exc:
            raise ValueError(f"Invalid CORSIKA build ID mapping for {source_tag!r}: {exc}") from exc
        _validate_optional_revision(component.get("config-revision"), "CORSIKA configuration")
        _validate_optional_revision(
            component.get("opt-patch-revision"), "CORSIKA optimization patch"
        )
        for variant, digest in component.get("image-digests", {}).items():
            _validate_optional_digest(digest, f"CORSIKA {source_tag} {variant}")


def _validate_simtel_components(components):
    """Validate sim_telarray component revisions and image digests."""
    for component in components:
        for key in ("revision", "hessio-revision", "stdtools-revision"):
            _validate_optional_revision(component.get(key), key)
        _validate_optional_digest(component.get("image-digest"), "sim_telarray image")


def _validate_model_and_test_components(catalog, schema_version):
    """Validate model-database and simtools-tests catalog values."""
    model_version = _model_tag(catalog)
    valid_model_version = (
        versioning.is_valid_release_tag(model_version)
        if schema_version in {"0.2.0", "0.3.0"}
        else isinstance(model_version, str)
        and bool(LEGACY_MODEL_VERSION_PATTERN.fullmatch(model_version))
    )
    if not valid_model_version:
        message = (
            "Model database values must be release tags starting with 'v'."
            if schema_version in {"0.2.0", "0.3.0"}
            else "Invalid model database version."
        )
        raise ValueError(message)
    if schema_version in {"0.2.0", "0.3.0"}:
        _validate_simtools_tests(catalog["simtools-tests"])


def _validate_simtools_tests(test_resources):
    """Validate the simtools-tests repository configuration."""
    repository = test_resources.get("repository")
    if not isinstance(repository, str) or not SIMTOOLS_TESTS_REPOSITORY_PATTERN.fullmatch(
        repository
    ):
        raise ValueError("simtools-tests repository must use the owner/name format.")
    source_url = test_resources.get("source-url")
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        raise ValueError("simtools-tests source URL must use HTTPS.")
    tag = test_resources.get("tag", test_resources.get("version"))
    if not versioning.is_valid_release_tag(tag):
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
    if not isinstance(value, str) or not REVISION_PATTERN.fullmatch(value):
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


def validate_env_template(catalog, template_path):
    """Validate non-secret runtime defaults against the dependency catalog.

    Parameters
    ----------
    catalog : dict
        Validated dependency catalog.
    template_path : str or Path
        Environment template to validate.

    Raises
    ------
    ValueError
        If catalog-managed versions are duplicated in a schema 0.2.0 template
        or if the model database defaults disagree with the catalog.
    """
    values = {}
    for line in Path(template_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key] = value
    version_keys = {"SIMTOOLS_TESTS_VERSION", "SIMTOOLS_TESTS_TAG"}
    if catalog["schema_version"] in {"0.2.0", "0.3.0"}:
        version_keys.update(
            {"SIMTOOLS_DB_SIMULATION_MODEL_VERSION", "SIMTOOLS_DB_SIMULATION_MODEL_TAG"}
        )
    configured_versions = sorted(version_keys & values.keys())
    if configured_versions:
        raise ValueError(
            ".env_template must not define catalog-managed versions: "
            + ", ".join(configured_versions)
        )
    model = catalog["model-database"]
    expected = {"SIMTOOLS_DB_SIMULATION_MODEL": model["name"]}
    if catalog["schema_version"] == "0.1.0":
        expected["SIMTOOLS_DB_SIMULATION_MODEL_VERSION"] = model["default-version"]
    mismatches = {
        key: (values.get(key), value) for key, value in expected.items() if values.get(key) != value
    }
    if mismatches:
        raise ValueError(f".env_template defaults disagree with dependency catalog: {mismatches}")


def build_workflow_matrices(catalog):
    """Build GitHub Actions matrices from the dependency catalog."""
    variants = catalog["cpu-variants"]
    platform_matrix = [
        {"platform": "linux/amd64", "arch": "amd64", "runner": "ubuntu-latest"},
        {"platform": "linux/arm64/v8", "arch": "arm64", "runner": "ubuntu-24.04-arm"},
    ]
    corsika_components = {_corsika_reference(item): item for item in catalog["corsika"]}
    simtel_components = {_simtel_tag(item): item for item in catalog["sim-telarray"]}
    corsika_matrix = [
        {
            "corsika_tag": _corsika_tag(corsika),
            "corsika_build_id": _corsika_build_id(corsika),
            "corsika_source_url": corsika["source-url"],
            "corsika_config_tag": _dependency_tag(corsika, "config-tag", "config-version"),
            "corsika_config_source_url": corsika["config-source-url"],
            "corsika_config_revision": corsika.get("config-revision", ""),
            "corsika_opt_patch_tag": _dependency_tag(corsika, "opt-patch-tag", "opt-patch-version"),
            "corsika_opt_patch_source_url": corsika["opt-patch-source-url"],
            "corsika_opt_patch_revision": corsika.get("opt-patch-revision", ""),
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
            "simtel_source_url": component["source-url"],
            "simtel_revision": component.get("revision", ""),
            "hessio_tag": _dependency_tag(component, "hessio-tag", "hessio-version"),
            "hessio_source_url": component["hessio-source-url"],
            "hessio_revision": component.get("hessio-revision", ""),
            "stdtools_tag": _dependency_tag(component, "stdtools-tag", "stdtools-version"),
            "stdtools_source_url": component["stdtools-source-url"],
            "stdtools_revision": component.get("stdtools-revision", ""),
        }
        for component in catalog["sim-telarray"]
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
            {
                "corsika_tag": _corsika_tag(component),
                "corsika_build_id": _corsika_build_id(component),
                "corsika_source_url": component["source-url"],
            }
            for component in catalog["corsika"]
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
    production_corsika = (
        _corsika_tag(corsika) if "tag" in corsika else f"v{_corsika_build_id(corsika)}"
    )
    return {
        "corsika_tag": production_corsika,
        "corsika_build_id": _corsika_build_id(corsika),
        "corsika_image": _image_reference(
            "ghcr.io/gammasim/corsika7",
            f"v{_corsika_build_id(corsika)}-{variant}",
            corsika.get("image-digests", {}).get(variant),
        ),
        "simtel_tag": _simtel_tag(simtel),
        "simtel_image": _image_reference(
            "ghcr.io/gammasim/sim_telarray",
            _simtel_tag(simtel),
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
        "gsl_version": catalog["archives"]["gsl"]["version"],
        "gsl_sha256": catalog["archives"]["gsl"].get("sha256", ""),
        "corsika_tables_tag": _dependency_tag(
            catalog["corsika-interaction-tables"], "tag", "version"
        ),
        "model_database": catalog["model-database"]["name"],
        "model_database_tag": _model_tag(catalog),
        "simtools_tests_repository": test_resources.get("repository", ""),
        "simtools_tests_url": test_resources.get("source-url", ""),
        "simtools_tests_tag": _tests_tag(test_resources),
        "dev_corsika_image": _image_reference(
            "ghcr.io/gammasim/corsika7",
            f"v{_corsika_build_id(default_corsika)}-generic",
            default_corsika.get("image-digests", {}).get("generic"),
        ),
        "dev_simtel_image": _image_reference(
            "ghcr.io/gammasim/sim_telarray",
            _simtel_tag(default_simtel),
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
        Environment variable names and values for database and test-resource
        configuration. Local paths and credentials are intentionally omitted.
    """
    model_tag = _model_tag(catalog)
    if catalog["schema_version"] == "0.3.0":
        environment = {
            "SIMTOOLS_DB_SIMULATION_MODEL": catalog["model-database"]["name"],
            "SIMTOOLS_DB_SIMULATION_MODEL_TAG": model_tag,
        }
    else:
        environment = {
            "SIMTOOLS_DB_SIMULATION_MODEL": catalog["model-database"]["name"],
            "SIMTOOLS_DB_SIMULATION_MODEL_VERSION": model_tag,
        }
    if "simtools-tests" in catalog:
        test_tag = _tests_tag(catalog["simtools-tests"])
        environment.update(
            {
                "SIMTOOLS_TESTS_REPOSITORY": catalog["simtools-tests"]["repository"],
                "SIMTOOLS_TESTS_URL": catalog["simtools-tests"]["source-url"],
            }
        )
        tag_key = (
            "SIMTOOLS_TESTS_TAG"
            if catalog["schema_version"] == "0.3.0"
            else "SIMTOOLS_TESTS_VERSION"
        )
        environment[tag_key] = test_tag
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
        validate_env_template(catalog, env_template)
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
