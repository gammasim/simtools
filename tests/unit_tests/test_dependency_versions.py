"""Tests for the dependency version catalog helpers."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from simtools import dependency_versions


def _load_catalog(simtools_root_path):
    return dependency_versions.load_dependency_catalog(
        simtools_root_path / "dependency_versions.yml"
    )


def _model_revision(catalog):
    return catalog["model-repository"].get(
        "default-ref",
        catalog["model-repository"].get(
            "default-tag", catalog["model-repository"].get("default-version")
        ),
    )


def _legacy_catalog(schema_version="0.2.0"):
    """Return a minimal catalog using the pre-tag dependency fields."""
    catalog = {
        "schema_version": schema_version,
        "python": "3.14",
        "apptainer": "1.5.0",
        "cpu-variants": ["generic"],
        "base-image": {"name": "almalinux", "runtime-version": "9-minimal", "build-version": "9"},
        "corsika-interaction-tables": {"version": "v1.0.0"},
        "archives": {"autoconf": {"version": "2.71"}, "gsl": {"version": "2.8"}},
        "model-repository": {"name": "CTAO-Simulation-Model", "default-version": "v0.1.0"},
        "production-combinations": [{"corsika": "78010", "sim-telarray": "v1.0.0"}],
        "corsika": [
            {
                "version": "78010",
                "source-ref": "v7.8010",
                "source-url": "https://example.test/c.git",
                "config-version": "v0.1.0",
                "config-source-url": "https://example.test/cc.git",
                "opt-patch-version": "v0.1.0",
                "opt-patch-source-url": "https://example.test/op.git",
            }
        ],
        "sim-telarray": [
            {
                "version": "v1.0.0",
                "source-url": "https://example.test/s.git",
                "hessio-version": "v1.0.0",
                "hessio-source-url": "https://example.test/h.git",
                "stdtools-version": "v1.0.0",
                "stdtools-source-url": "https://example.test/st.git",
            }
        ],
    }
    if schema_version == "0.1.0":
        catalog["model-repository"]["default-version"] = "0.16.0"
    else:
        catalog["simtools-tests"] = {
            "repository": "owner/tests",
            "source-url": "https://example.test/tests.git",
            "version": "v0.1.0",
        }
    return catalog


def _readable_catalog(schema_version="0.6.0"):
    """Return a small catalog with readable refs and no Git pins."""
    catalog = _legacy_catalog(schema_version)
    catalog["corsika-interaction-tables"] = {
        "ref": "v1.0.0",
        "source-url": "https://example.test/tables.git",
    }
    catalog["model-repository"] = {
        "name": "CTAO-Simulation-Model",
        "default-ref": "v0.1.0",
        "repository-url": "https://example.test/models.git",
    }
    tests = catalog["simtools-tests"]
    tests["ref"] = "main"
    tests["resource-version"] = tests.pop("version")
    corsika = catalog["corsika"][0]
    corsika.pop("version")
    corsika["config-ref"] = corsika.pop("config-version")
    corsika["opt-patch-ref"] = corsika.pop("opt-patch-version")
    simtel = catalog["sim-telarray"][0]
    simtel["source-ref"] = simtel.pop("version")
    simtel["hessio-ref"] = simtel.pop("hessio-version")
    simtel["stdtools-ref"] = simtel.pop("stdtools-version")
    catalog["production-combinations"][0]["corsika"] = "v7.8010"
    return catalog


def test_catalog_derives_corsika_build_id_from_tag(simtools_root_path):
    """Use source tags for selection and derive the legacy build ID."""
    catalog = _load_catalog(simtools_root_path)
    combination = catalog["production-combinations"][0]
    corsika = next(
        item for item in catalog["corsika"] if item["source-ref"] == combination["corsika"]
    )
    build_id = corsika["source-ref"].removeprefix("v").replace(".", "")
    variant = combination.get("cpu-variants", catalog["cpu-variants"])[0]
    matrices = dependency_versions.build_workflow_matrices(catalog)
    production = next(
        item
        for item in matrices["production_matrix"]
        if item["corsika_tag"] == corsika["source-ref"] and item["avx_flag"] == variant
    )

    assert "build-id" not in corsika
    assert production["corsika_tag"] == corsika["source-ref"]
    assert production["corsika_build_id"] == build_id
    assert production["corsika_image"].endswith(f":v{build_id}-{variant}")


def test_corsika_build_id_is_derived_without_a_fixed_length():
    """Derive the legacy build ID directly from the source tag."""
    assert dependency_versions._corsika_build_id({"tag": "v8.10000"}) == "810000"  # pylint: disable=protected-access


@pytest.mark.parametrize("build_id", ["latest", "\u0661\u0662", "12\u0663"])
def test_corsika_build_id_rejects_non_numeric_explicit_value(build_id):
    """Reject explicit CORSIKA build IDs that cannot be used in image names."""
    with pytest.raises(ValueError, match="must contain only digits"):
        dependency_versions._corsika_build_id({"build-id": build_id})  # pylint: disable=protected-access


@pytest.mark.parametrize("build_id", ["v2025-11-30-rc", "_build.01", "a" * 128])
def test_simtel_build_id_accepts_ascii_image_tags(build_id):
    """Allow ASCII letters, digits, underscores, dots, and hyphens in OCI tags."""
    assert dependency_versions._simtel_build_id({"source-ref": build_id}) == build_id  # pylint: disable=protected-access


@pytest.mark.parametrize("build_id", ["\u00e9build", "build\u00e9", "\u0661", "a" * 129])
def test_simtel_build_id_rejects_invalid_explicit_image_tags(build_id):
    """Reject Unicode characters and overlong OCI image identifiers."""
    with pytest.raises(ValueError, match="valid OCI image tag"):
        dependency_versions._simtel_build_id(  # pylint: disable=protected-access
            {"source-ref": "main", "build-id": build_id}
        )


def test_safe_build_id_replaces_unicode_characters():
    """Sanitize source refs into ASCII image identifiers."""
    assert dependency_versions._safe_build_id("release/\u00e9build") == "release-build"  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "source_ref", ["topic/.hidden", "topic/main.lock/next", "topic/\x01next", "HEAD"]
)
def test_source_ref_rejects_invalid_git_components(source_ref):
    """Reject invalid Git ref components and control characters."""
    with pytest.raises(ValueError, match="Invalid source ref"):
        dependency_versions._validate_source_ref(source_ref, "source")  # pylint: disable=protected-access


@pytest.mark.parametrize("component", ["hessio", "stdtools"])
def test_catalog_rejects_refs_without_a_usable_artifact_identifier(component):
    """Fail catalog validation when a valid Git ref cannot produce an ASCII artifact ID."""
    catalog = _readable_catalog()
    catalog["sim-telarray"][0][f"{component}-ref"] = "\u00e9"

    with pytest.raises(ValueError, match="valid OCI image tag"):
        dependency_versions.validate_dependency_catalog(catalog)


def test_simtel_branch_ref_requires_a_safe_build_id():
    """Require a separate image identifier for a branch ref containing a slash."""
    with pytest.raises(ValueError, match="require build-id"):
        dependency_versions._simtel_build_id(  # pylint: disable=protected-access
            {"source-ref": "release/2025"}
        )


def test_corsika_source_tag_for_build_id_handles_missing_and_ambiguous_values():
    """Resolve CORSIKA source tags only when the catalog mapping is unambiguous."""
    catalog = {"corsika": [{"tag": "v7.8010"}]}
    assert dependency_versions.corsika_source_tag_for_build_id("78050", catalog) is None

    with pytest.raises(ValueError, match="Multiple CORSIKA source tags"):
        dependency_versions.corsika_source_tag_for_build_id(
            "78010",
            {
                "corsika": [
                    {"tag": "v7.8010"},
                    {"tag": "v7.8010"},
                ]
            },
        )


def test_catalog_reads_legacy_corsika_fields():
    """Keep schema 0.2 CORSIKA records readable during migration."""
    catalog = _legacy_catalog()
    assert dependency_versions.validate_dependency_catalog(catalog) == catalog


def test_legacy_catalog_fields_are_preserved_in_workflow_and_summary():
    """Use legacy names when producing matrices and summary exports."""
    catalog = _legacy_catalog()

    matrices = dependency_versions.build_workflow_matrices(catalog)
    summary = dependency_versions.dependency_catalog_summary(catalog)

    assert matrices["corsika_matrix"][0]["corsika_config_tag"] == "v0.1.0"
    assert matrices["simtel_matrix"][0]["hessio_tag"] == "v1.0.0"
    assert summary["corsika_tables_ref"] == "v1.0.0"
    assert summary["simtools_tests_ref"] == "v0.1.0"


def test_legacy_catalog_rejects_invalid_model_release_tag():
    """Report the release-tag validation error for pre-0.5 catalogs."""
    catalog = _legacy_catalog()
    catalog["model-repository"]["default-version"] = "not-a-release-tag"

    with pytest.raises(ValueError, match="Invalid simulation-model release tags"):
        dependency_versions.validate_dependency_catalog(catalog)


def test_schema_0_5_requires_source_revisions():
    """Keep required pins for catalogs declaring schema 0.5."""
    catalog = _readable_catalog("0.5.0")
    catalog["corsika-interaction-tables"]["revision"] = "a" * 40

    with pytest.raises(ValueError, match="Invalid Git revision"):
        dependency_versions.validate_dependency_catalog(catalog)


def test_schema_0_6_accepts_branch_refs_without_revisions():
    """Use branch names for source checkouts and safe identifiers for images."""
    catalog = _readable_catalog()
    corsika = catalog["corsika"][0]
    corsika["source-ref"] = "release/7.8"
    corsika["build-id"] = "78010"
    catalog["production-combinations"][0]["corsika"] = "release/7.8"
    catalog["sim-telarray"][0]["source-ref"] = "release/2025"
    catalog["sim-telarray"][0]["build-id"] = "release-2025"
    for combination in catalog["production-combinations"]:
        combination["sim-telarray"] = "release/2025"
    catalog["model-repository"]["default-ref"] = "main"
    catalog["simtools-tests"]["ref"] = "release/3"

    assert dependency_versions.validate_dependency_catalog(catalog) is catalog
    matrix = dependency_versions.build_workflow_matrices(catalog)
    assert matrix["simtel_matrix"][0]["simtel_tag"] == "release/2025"
    assert matrix["simtel_matrix"][0]["simtel_build_id"] == "release-2025"
    assert matrix["production_matrix"][0]["simtel_image"].endswith(":release-2025")
    assert dependency_versions.dependency_catalog_summary(catalog)["dev_simtel_image"].endswith(
        ":release-2025"
    )


def test_load_dependency_catalog_and_build_matrices(simtools_root_path, monkeypatch):
    """Test catalog loading and matrix construction."""
    monkeypatch.chdir(simtools_root_path)
    catalog = dependency_versions.load_dependency_catalog()
    matrices = dependency_versions.build_workflow_matrices(catalog)

    variants = catalog["cpu-variants"]
    assert len(matrices["corsika_matrix"]) == len(catalog["corsika"]) * len(variants)
    assert len(matrices["corsika_build_matrix"]) == len(catalog["corsika"]) * sum(
        2 if variant == "generic" else 1 for variant in variants
    )
    assert len(matrices["corsika_source_matrix"]) == len(catalog["corsika"])
    assert len(matrices["simtel_matrix"]) == len(catalog["sim-telarray"])
    assert len(matrices["simtel_build_matrix"]) == 2 * len(catalog["sim-telarray"])
    assert len(matrices["production_matrix"]) == sum(
        len(combination.get("cpu-variants", variants))
        for combination in catalog["production-combinations"]
    )
    assert {item["avx_flag"] for item in matrices["corsika_build_matrix"]} == set(variants)
    assert {item["arch"] for item in matrices["simtel_build_matrix"]} == {"amd64", "arm64"}
    for matrix_name in ("corsika_build_matrix", "simtel_build_matrix"):
        matrix = matrices[matrix_name]
        assert {item["runner"] for item in matrix if item["arch"] == "amd64"} == {"ubuntu-24.04"}
        assert {item["runner"] for item in matrix if item["arch"] == "arm64"} == {
            "ubuntu-24.04-arm"
        }
    first_corsika = catalog["corsika"][0]
    assert matrices["corsika_source_matrix"][0]["corsika_config_tag"] == first_corsika["config-ref"]
    assert (
        matrices["corsika_source_matrix"][0]["corsika_opt_patch_tag"]
        == first_corsika["opt-patch-ref"]
    )
    assert matrices["corsika_source_matrix"][0]["corsika_source_revision"] == ""
    assert matrices["corsika_build_matrix"][0]["corsika_source_revision"] == ""
    assert all(
        item["corsika_image"].startswith("ghcr.io/gammasim/corsika7:v")
        for item in matrices["production_matrix"]
    )


def test_catalog_summary_uses_version_tags_without_digests(simtools_root_path):
    """Test optional digests do not affect the current catalog references."""
    catalog = _load_catalog(simtools_root_path)
    summary = dependency_versions.dependency_catalog_summary(catalog)

    base = catalog["base-image"]
    corsika = catalog["corsika"][0]

    assert summary["base_image"] == f"{base['name']}:{base['runtime-version']}"
    assert summary["corsika_tables_ref"] == catalog["corsika-interaction-tables"]["ref"]
    assert summary["corsika_tables_revision"] == ""
    build_id = corsika["source-ref"].removeprefix("v").replace(".", "")
    assert summary["dev_corsika_image"] == f"ghcr.io/gammasim/corsika7:v{build_id}-generic"
    assert summary["model_repository_revision"] == dependency_versions._model_revision(catalog)
    assert summary["simtools_tests_repository"] == catalog["simtools-tests"]["repository"]
    assert summary["simtools_tests_ref"] == catalog["simtools-tests"]["ref"]
    assert (
        summary["simtools_tests_resource_version"] == catalog["simtools-tests"]["resource-version"]
    )
    assert summary["simtools_tests_revision"] == ""
    assert summary["simtools_tests_url"] == catalog["simtools-tests"]["source-url"]


def test_env_template_matches_catalog(simtools_root_path):
    """Test the documented environment defaults match the catalog."""
    assert dependency_versions.validate_env_template(simtools_root_path / ".env_template") is None


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda data: data.pop("python"), "Missing dependency catalog keys"),
        (
            lambda data: data.update({"schema_version": "9.9.9"}),
            "Unsupported dependency catalog schema version",
        ),
        (
            lambda data: (data.update({"schema_version": "0.2.0"}), data.pop("simtools-tests")),
            "Missing dependency catalog keys: simtools-tests",
        ),
        (
            lambda data: data["base-image"].update({"runtime-digest": "latest"}),
            "Invalid SHA-256 digest",
        ),
        (
            lambda data: data["archives"]["gsl"].update({"sha256": "invalid"}),
            "Invalid SHA-256 checksum",
        ),
        (
            lambda data: data["corsika"][0].update({"source-ref": "bad ref"}),
            "Invalid source ref",
        ),
        (
            lambda data: data["sim-telarray"][0].update({"source-ref": "bad ref"}),
            "Invalid source ref",
        ),
        (
            lambda data: data["corsika-interaction-tables"].update({"ref": "bad ref"}),
            "Invalid source ref",
        ),
        (
            lambda data: data["sim-telarray"][0].update({"revision": "short"}),
            "Invalid Git revision",
        ),
        (
            lambda data: data["corsika"][0].update({"source-revision": "short"}),
            "Invalid Git revision",
        ),
        (
            lambda data: data["model-repository"].update({"default-ref": "bad ref"}),
            "source ref",
        ),
        (
            lambda data: data["production-combinations"][0].update({"cpu-variants": ["unknown"]}),
            "Unknown CPU variant",
        ),
        (
            lambda data: data["simtools-tests"].pop("repository"),
            "owner/name",
        ),
        (
            lambda data: data["simtools-tests"].pop("source-url"),
            "HTTPS",
        ),
        (
            lambda data: data["simtools-tests"].pop("ref"),
            "Invalid source ref",
        ),
        (
            lambda data: data["simtools-tests"].pop("resource-version"),
            "resource version",
        ),
        (
            lambda data: data["simtools-tests"].update({"repository": "foo"}),
            "owner/name",
        ),
        (
            lambda data: data["simtools-tests"].update({"source-url": "ftp://example.com"}),
            "HTTPS",
        ),
        (
            lambda data: data["simtools-tests"].update({"ref": "bad ref"}),
            "Invalid source ref",
        ),
    ],
)
def test_validate_dependency_catalog_rejects_invalid_values(simtools_root_path, mutator, error):
    """Test catalog validation rejects invalid optional and required values."""
    catalog = _load_catalog(simtools_root_path)
    invalid = copy.deepcopy(catalog)
    mutator(invalid)

    with pytest.raises(ValueError, match=error):
        dependency_versions.validate_dependency_catalog(invalid)


def test_load_dependency_catalog_rejects_non_mapping(tmp_test_directory):
    """Test a catalog without a top-level mapping fails clearly."""
    project_file = tmp_test_directory / "dependency_versions.yml"
    project_file.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        dependency_versions.load_dependency_catalog(project_file)


def test_load_dependency_catalog_caches_file_parsing(tmp_test_directory, mocker):
    """Repeated catalog reads reuse the parsed catalog without sharing mutations."""
    catalog_file = tmp_test_directory / "dependency_versions.yml"
    catalog_file.write_text(yaml.safe_dump(_legacy_catalog()), encoding="utf-8")
    safe_load = mocker.spy(dependency_versions.yaml, "safe_load")

    first = dependency_versions.load_dependency_catalog(catalog_file)
    first["python"] = "changed"
    second = dependency_versions.load_dependency_catalog(catalog_file)

    assert second["python"] == "3.14"
    assert safe_load.call_count == 1


def test_find_pyproject_from_environment(monkeypatch, simtools_root_path):
    project_file = simtools_root_path / "pyproject.toml"
    monkeypatch.setenv("SIMTOOLS_PYPROJECT", str(project_file))

    assert dependency_versions.find_pyproject("/") == project_file


def test_find_dependency_versions_from_environment(monkeypatch, tmp_test_directory):
    """Test an explicit catalog-file environment setting wins."""
    catalog_file = tmp_test_directory / "dependency_versions.yml"
    catalog_file.write_text("schema_version: 0.1.0\n", encoding="utf-8")
    monkeypatch.setenv("SIMTOOLS_DEPENDENCY_VERSIONS", str(catalog_file))

    assert dependency_versions.find_dependency_versions("/") == catalog_file


def test_find_dependency_versions_raises_when_missing(mocker, tmp_test_directory):
    """Test catalog discovery reports a clear error when no file is available."""
    mocker.patch("simtools.dependency_versions.Path.is_file", return_value=False)

    with pytest.raises(FileNotFoundError, match="Could not find"):
        dependency_versions.find_dependency_versions(tmp_test_directory)


def test_find_dependency_versions_falls_back_to_installed_catalog(monkeypatch, tmp_test_directory):
    """Test installed applications can use the root catalog installed as data."""
    installed_catalog = Path(str(tmp_test_directory)) / "simtools" / "dependency_versions.yml"
    installed_catalog.parent.mkdir()
    installed_catalog.write_text("schema_version: 0.1.0\n", encoding="utf-8")
    monkeypatch.delenv("SIMTOOLS_DEPENDENCY_VERSIONS", raising=False)
    monkeypatch.setattr(
        dependency_versions,
        "__file__",
        str(tmp_test_directory / "src" / "simtools" / "dependency_versions.py"),
    )
    monkeypatch.setattr(dependency_versions.sys, "prefix", str(tmp_test_directory))

    assert dependency_versions.find_dependency_versions(tmp_test_directory) == installed_catalog


def test_validate_dependency_catalog_preserves_schema_0_1_contract(simtools_root_path):
    """Test catalogs using the original schema remain accepted."""
    catalog = _legacy_catalog("0.1.0")

    assert dependency_versions.validate_dependency_catalog(catalog) is catalog


def test_validate_dependency_catalog_accepts_valid_revisions(simtools_root_path):
    """Test valid component revisions pass catalog validation."""
    catalog = _load_catalog(simtools_root_path)
    revision = "a" * 40
    catalog["corsika"][0]["source-revision"] = revision
    catalog["corsika"][0]["config-revision"] = revision
    catalog["corsika"][0]["opt-patch-revision"] = revision
    catalog["sim-telarray"][0].update(
        {"revision": revision, "hessio-revision": revision, "stdtools-revision": revision}
    )

    assert dependency_versions.validate_dependency_catalog(catalog) is catalog


def test_build_workflow_matrices_uses_optional_image_digests(simtools_root_path):
    """Test optional immutable image references are propagated to production matrices."""
    catalog = _load_catalog(simtools_root_path)
    digest = "sha256:" + "a" * 64
    catalog["corsika"][0]["image-digests"] = {"generic": digest}
    catalog["sim-telarray"][0]["image-digest"] = digest

    dependency_versions.validate_dependency_catalog(catalog)
    matrix = dependency_versions.build_workflow_matrices(catalog)["production_matrix"]

    assert matrix[0]["corsika_image"] == f"ghcr.io/gammasim/corsika7@{digest}"
    assert matrix[0]["simtel_image"] == f"ghcr.io/gammasim/sim_telarray@{digest}"


def test_build_workflow_matrices_selects_private_source_snapshots(simtools_root_path):
    """Use a deterministic private snapshot only when every source is pinned."""
    catalog = _load_catalog(simtools_root_path)
    revision = "a" * 40
    catalog["corsika"][0].update(
        {
            "source-revision": revision,
            "config-revision": revision,
            "opt-patch-revision": revision,
        }
    )
    catalog["sim-telarray"][0].update(
        {"revision": revision, "hessio-revision": revision, "stdtools-revision": revision}
    )

    matrices = dependency_versions.build_workflow_matrices(catalog)

    assert matrices["corsika_source_matrix"][0]["corsika_source_snapshot"].startswith(
        "ghcr.io/gammasim/corsika7-build-inputs:78010-"
    )
    assert matrices["simtel_matrix"][0]["simtel_source_snapshot"].startswith(
        "ghcr.io/gammasim/simtel-array-build-inputs:v2025-11-30-rc-"
    )
    assert (
        dependency_versions._source_snapshot_image(  # pylint: disable=protected-access
            "corsika7-build-inputs", "v78010", (revision, "", revision)
        )
        == ""
    )
    summary = dependency_versions.dependency_catalog_summary(catalog)
    assert (
        summary["default_corsika_source_snapshot"]
        == matrices["corsika_source_matrix"][0]["corsika_source_snapshot"]
    )
    assert (
        summary["default_simtel_source_snapshot"]
        == matrices["simtel_matrix"][0]["simtel_source_snapshot"]
    )


def test_production_matrix_uses_global_cpu_variants_by_default(simtools_root_path):
    """Test production combinations inherit the catalog CPU variants."""
    catalog = _load_catalog(simtools_root_path)
    expected = dependency_versions.build_workflow_matrices(catalog)["production_matrix"]
    catalog["production-combinations"][0].pop("cpu-variants", None)

    matrix = dependency_versions.build_workflow_matrices(catalog)["production_matrix"]

    assert matrix == expected


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("corsika", "unknown", "Unknown CORSIKA"),
        ("sim-telarray", "unknown", "Unknown sim_telarray"),
    ],
)
def test_validate_dependency_catalog_rejects_unknown_production_components(
    simtools_root_path, field, value, error
):
    """Test production combinations must use catalogued component versions."""
    catalog = _load_catalog(simtools_root_path)
    catalog["production-combinations"][0][field] = value

    with pytest.raises(ValueError, match=error):
        dependency_versions.validate_dependency_catalog(catalog)


def test_export_dependency_configuration_returns_github_outputs(simtools_root_path):
    output = dependency_versions.export_dependency_configuration(
        simtools_root_path / "pyproject.toml", "github-output"
    )

    catalog = _load_catalog(simtools_root_path)

    assert "production_matrix=" in output
    assert f"python_version={catalog['python']}" in output


def test_export_dependency_configuration_returns_environment_values(simtools_root_path):
    """Test env output contains catalog-managed runtime values only."""
    output = dependency_versions.export_dependency_configuration(output_format="env")
    catalog = _load_catalog(simtools_root_path)
    test_resources = catalog["simtools-tests"]
    expected = [
        f"SIMTOOLS_SIMULATION_MODELS_GIT_REVISION={dependency_versions._model_revision(catalog)}",
        f"SIMTOOLS_TESTS_REF={test_resources['ref']}",
        f"SIMTOOLS_TESTS_RESOURCE_VERSION={test_resources['resource-version']}",
        f"SIMTOOLS_TESTS_REPOSITORY={test_resources['repository']}",
        f"SIMTOOLS_TESTS_URL={test_resources['source-url']}",
    ]

    assert output.splitlines() == expected


def test_dependency_catalog_environment_supports_schema_0_1(simtools_root_path):
    """Test the legacy catalog environment excludes simtools-tests settings."""
    catalog = _legacy_catalog("0.1.0")

    environment = dependency_versions.dependency_catalog_environment(catalog)

    assert environment == {
        "SIMTOOLS_SIMULATION_MODELS_GIT_REVISION": _model_revision(catalog),
    }


def test_export_dependency_configuration_returns_python_requirements(simtools_root_path):
    requirements = dependency_versions.export_dependency_configuration(
        simtools_root_path / "pyproject.toml", "python-requirements", ["tests"]
    )

    assert "astropy" in requirements.splitlines()
    assert "pytest" in requirements.splitlines()


def test_project_requirements_rejects_unknown_extra(tmp_test_directory):
    """Test unknown optional dependency groups produce an actionable error."""
    project_file = tmp_test_directory / "pyproject.toml"
    project_file.write_text(
        '[project]\ndependencies = []\n[project.optional-dependencies]\ntests = ["pytest"]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Available groups: tests"):
        dependency_versions.project_requirements(project_file, ["missing"])


@pytest.mark.parametrize("output_format", ["catalog", "summary"])
def test_export_dependency_configuration_returns_json(simtools_root_path, output_format):
    """Test JSON export formats return parseable serialized data."""
    output = dependency_versions.export_dependency_configuration(output_format=output_format)

    assert json.loads(output)


def test_export_dependency_configuration_rejects_unknown_format(simtools_root_path):
    """Test unsupported exports are rejected clearly."""
    with pytest.raises(ValueError, match="Unsupported"):
        dependency_versions.export_dependency_configuration(
            simtools_root_path / "pyproject.toml", "unknown"
        )


def test_catalog_matches_yaml_schema(simtools_root_path):
    """Test the YAML catalog conforms to the project schema."""
    import jsonschema

    catalog = _load_catalog(simtools_root_path)
    schema_path = simtools_root_path / "src/simtools/schemas/dependency_versions.schema.yml"
    schemas = list(yaml.safe_load_all(schema_path.read_text(encoding="utf-8")))
    schemas_by_version = {item["schema_version"]: item for item in schemas}
    for schema_version in dependency_versions.READABLE_REF_SCHEMAS:
        source_ref_pattern = schemas_by_version[schema_version]["definitions"]["source-ref"][
            "pattern"
        ]
        assert source_ref_pattern == dependency_versions.SOURCE_REF_PATTERN.pattern
    schema = schemas_by_version[catalog["schema_version"]]

    jsonschema.validate(catalog, schema)
    assert sorted(item["schema_version"] for item in schemas) == [
        "0.1.0",
        "0.2.0",
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "0.6.0",
    ]
    assert "simtools-tests" not in schemas_by_version["0.1.0"]["required"]
    assert "simtools-tests" in schemas_by_version["0.2.0"]["required"]
    assert catalog["schema_version"] in schemas_by_version
    legacy_schema = next(schema for schema in schemas if "simtools-tests" not in schema["required"])
    tagged_schema = next(schema for schema in schemas if "simtools-tests" in schema["required"])
    assert "simtools-tests" not in legacy_schema["required"]
    assert "simtools-tests" in tagged_schema["required"]
    assert "default-ref" in schema["properties"]["model-repository"]["required"]
    assert "resource-version" in schema["properties"]["simtools-tests"]["required"]
    assert "source-revision" not in schema["definitions"]["corsika"]["required"]
    assert "revision" not in schema["definitions"]["simtel"]["required"]
    assert "source-revision" in schemas_by_version["0.5.0"]["definitions"]["corsika"]["required"]


def test_catalog_refs_drive_unpinned_builds():
    """Build matrices and runtime settings use readable refs without Git pins."""
    catalog = _readable_catalog()
    assert dependency_versions.validate_dependency_catalog(catalog) is catalog
    matrix = dependency_versions.build_workflow_matrices(catalog)
    assert matrix["corsika_source_matrix"][0]["corsika_tag"] == "v7.8010"
    assert matrix["corsika_source_matrix"][0]["corsika_source_revision"] == ""
    assert matrix["simtel_matrix"][0]["simtel_tag"] == "v1.0.0"
    assert matrix["simtel_matrix"][0]["simtel_revision"] == ""
    environment = dependency_versions.dependency_catalog_environment(catalog)
    assert environment["SIMTOOLS_SIMULATION_MODELS_GIT_REVISION"] == "v0.1.0"
    assert environment["SIMTOOLS_TESTS_REF"] == "main"
    assert "SIMTOOLS_TESTS_REVISION" not in environment


@pytest.mark.parametrize("revision", ["a" * 40, "short"])
def test_catalog_optional_test_revision(revision):
    """Validate optional test pins and export them only when supplied."""
    catalog = _readable_catalog()
    catalog["simtools-tests"]["revision"] = revision
    if revision == "short":
        with pytest.raises(ValueError, match="Invalid Git revision"):
            dependency_versions.validate_dependency_catalog(catalog)
    else:
        assert dependency_versions.validate_dependency_catalog(catalog) is catalog
        assert (
            dependency_versions.dependency_catalog_environment(catalog)["SIMTOOLS_TESTS_REVISION"]
            == revision
        )
