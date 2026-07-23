"""Read, validate, and export the simtools dependency version catalog."""

import json
import os
import re
import tomllib
from pathlib import Path

CATALOG_KEYS = ("tool", "gammasimtools", "dependency-versions")
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ARCHIVE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def find_pyproject(start_path=None):
    """Find the nearest pyproject.toml containing the dependency catalog.

    Parameters
    ----------
    start_path : str or Path, optional
        Directory from which to start searching.

    Returns
    -------
    pathlib.Path
        Path to the matching ``pyproject.toml`` file.

    Raises
    ------
    FileNotFoundError
        If no matching project file can be found.
    """
    configured_path = os.getenv("SIMTOOLS_PYPROJECT")
    candidates = [Path(configured_path)] if configured_path else []
    start = Path(start_path or Path.cwd()).resolve()
    candidates.extend(parent / "pyproject.toml" for parent in (start, *start.parents))
    candidates.append(Path(__file__).resolve().parents[2] / "pyproject.toml")
    for candidate in candidates:
        if candidate.is_file() and _contains_catalog(candidate):
            return candidate
    raise FileNotFoundError("Could not find pyproject.toml with simtools dependency versions.")


def _contains_catalog(pyproject_path):
    """Return whether a project file contains the simtools catalog."""
    try:
        with pyproject_path.open("rb") as file:
            data = tomllib.load(file)
        return _nested_value(data, CATALOG_KEYS) is not None
    except (OSError, tomllib.TOMLDecodeError):  # fmt: skip
        return False


def _nested_value(data, keys):
    """Return a nested mapping value or None."""
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def load_dependency_catalog(pyproject_path=None, validate=True):
    """Load the dependency version catalog from pyproject.toml.

    Parameters
    ----------
    pyproject_path : str or Path, optional
        Explicit project file. The repository is searched when omitted.
    validate : bool, optional
        Validate the catalog structure when True.

    Returns
    -------
    dict
        Dependency version catalog.
    """
    project_file = Path(pyproject_path) if pyproject_path else find_pyproject()
    with project_file.open("rb") as file:
        project_data = tomllib.load(file)
    catalog = _nested_value(project_data, CATALOG_KEYS)
    if catalog is None:
        raise KeyError("Missing [tool.gammasimtools.dependency-versions] table.")
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
        "base-image",
        "archives",
        "production-combinations",
        "corsika",
        "sim-telarray",
    }
    missing = sorted(required - catalog.keys())
    if missing:
        raise ValueError(f"Missing dependency catalog keys: {', '.join(missing)}")
    _validate_optional_digest(catalog["base-image"].get("runtime-digest"), "runtime base image")
    _validate_optional_digest(catalog["base-image"].get("build-digest"), "build base image")
    _validate_archive_checksums(catalog["archives"])
    _validate_components(catalog)
    _validate_production_combinations(catalog)
    return catalog


def _validate_components(catalog):
    """Validate CORSIKA and sim_telarray component records."""
    for component in catalog["corsika"]:
        if component.get("source-ref") in {"latest", "master", "main"}:
            raise ValueError("CORSIKA source-ref must identify a release.")
        _validate_optional_revision(component.get("config-revision"), "CORSIKA configuration")
        _validate_optional_revision(
            component.get("opt-patch-revision"), "CORSIKA optimization patch"
        )
        for variant, digest in component.get("image-digests", {}).items():
            _validate_optional_digest(digest, f"CORSIKA {component.get('version')} {variant}")
    for component in catalog["sim-telarray"]:
        for key in ("revision", "hessio-revision", "stdtools-revision"):
            _validate_optional_revision(component.get(key), key)
        _validate_optional_digest(component.get("image-digest"), "sim_telarray image")
    model_version = catalog.get("model-database", {}).get("default-version", "")
    if model_version.startswith("v"):
        raise ValueError("Model database versions must not start with 'v'.")


def _validate_production_combinations(catalog):
    """Validate every production combination against catalogued components."""
    corsika_versions = {component["version"] for component in catalog["corsika"]}
    simtel_versions = {component["version"] for component in catalog["sim-telarray"]}
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
        If the model database defaults disagree with the catalog.
    """
    values = {}
    for line in Path(template_path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key] = value
    model = catalog["model-database"]
    expected = {
        "SIMTOOLS_DB_SIMULATION_MODEL": model["name"],
        "SIMTOOLS_DB_SIMULATION_MODEL_VERSION": model["default-version"],
    }
    mismatches = {
        key: (values.get(key), value) for key, value in expected.items() if values.get(key) != value
    }
    if mismatches:
        raise ValueError(f".env_template defaults disagree with dependency catalog: {mismatches}")


def build_workflow_matrices(catalog):
    """Build GitHub Actions matrices from the dependency catalog."""
    variants = catalog["cpu-variants"]
    corsika_components = {item["version"]: item for item in catalog["corsika"]}
    simtel_components = {item["version"]: item for item in catalog["sim-telarray"]}
    corsika_matrix = [
        {
            "corsika": corsika["version"],
            "corsika_source_ref": corsika["source-ref"],
            "corsika_source_url": corsika["source-url"],
            "corsika_config": corsika["config-version"],
            "corsika_config_source_url": corsika["config-source-url"],
            "corsika_config_revision": corsika.get("config-revision", ""),
            "corsika_opt_patch": corsika["opt-patch-version"],
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
            "simtel_version": component["version"],
            "simtel_source_url": component["source-url"],
            "simtel_revision": component.get("revision", ""),
            "hessio_version": component["hessio-version"],
            "hessio_source_url": component["hessio-source-url"],
            "hessio_revision": component.get("hessio-revision", ""),
            "stdtools_version": component["stdtools-version"],
            "stdtools_source_url": component["stdtools-source-url"],
            "stdtools_revision": component.get("stdtools-revision", ""),
        }
        for component in catalog["sim-telarray"]
    ]
    return {
        "corsika_matrix": corsika_matrix,
        "simtel_matrix": simtel_matrix,
        "production_matrix": production_matrix,
    }


def _production_matrix_entry(corsika_components, simtel_components, combination, variant):
    """Build one production image matrix entry."""
    corsika = corsika_components[combination["corsika"]]
    simtel = simtel_components[combination["sim-telarray"]]
    return {
        "corsika": f"v{corsika['version']}",
        "corsika_image": _image_reference(
            "ghcr.io/gammasim/corsika7",
            f"v{corsika['version']}-{variant}",
            corsika.get("image-digests", {}).get(variant),
        ),
        "sim_telarray": simtel["version"],
        "simtel_image": _image_reference(
            "ghcr.io/gammasim/sim_telarray",
            simtel["version"],
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
        "model_database": catalog["model-database"]["name"],
        "model_version": catalog["model-database"]["default-version"],
        "dev_corsika_image": _image_reference(
            "ghcr.io/gammasim/corsika7",
            f"v{default_corsika['version']}-generic",
            default_corsika.get("image-digests", {}).get("generic"),
        ),
        "dev_simtel_image": _image_reference(
            "ghcr.io/gammasim/sim_telarray",
            default_simtel["version"],
            default_simtel.get("image-digest"),
        ),
    }


def project_requirements(pyproject_path, extras):
    """Return project requirements, optionally including named extras."""
    with Path(pyproject_path).open("rb") as file:
        project = tomllib.load(file)["project"]
    requirements = list(project["dependencies"])
    for extra in extras:
        requirements.extend(project["optional-dependencies"][extra])
    return requirements


def export_dependency_configuration(pyproject_path=None, output_format="catalog", extras=None):
    """Return dependency configuration in a selected export format.

    Parameters
    ----------
    pyproject_path : str or Path, optional
        Explicit project file. The repository is searched when omitted.
    output_format : str, optional
        One of ``catalog``, ``github-output``, ``python-requirements``, or ``summary``.
    extras : list of str, optional
        Optional dependency groups included in ``python-requirements`` output.

    Returns
    -------
    str
        Serialized dependency configuration, including a trailing newline.
    """
    project_file = Path(pyproject_path) if pyproject_path else find_pyproject()
    catalog = load_dependency_catalog(project_file)
    env_template = project_file.parent / ".env_template"
    if env_template.is_file():
        validate_env_template(catalog, env_template)
    extras = extras or []
    if output_format == "python-requirements":
        return "\n".join(project_requirements(project_file, extras)) + "\n"
    if output_format == "catalog":
        return json.dumps(catalog, indent=2, sort_keys=True) + "\n"
    if output_format == "summary":
        return json.dumps(dependency_catalog_summary(catalog), sort_keys=True) + "\n"
    if output_format == "github-output":
        output = {**dependency_catalog_summary(catalog), **build_workflow_matrices(catalog)}
        return "".join(f"{key}={_github_output_value(value)}\n" for key, value in output.items())
    raise ValueError(f"Unsupported dependency export format: {output_format}")


def _github_output_value(value):
    """Serialize a GitHub Actions output value."""
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return value
