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


def test_load_dependency_catalog_and_build_matrices(simtools_root_path, monkeypatch):
    """Test catalog loading and matrix construction."""
    monkeypatch.chdir(simtools_root_path)
    catalog = dependency_versions.load_dependency_catalog()
    matrices = dependency_versions.build_workflow_matrices(catalog)

    assert catalog["python"] == "3.14"
    assert len(matrices["corsika_matrix"]) == 8
    assert len(matrices["corsika_build_matrix"]) == 10
    assert len(matrices["corsika_source_matrix"]) == 2
    assert len(matrices["simtel_matrix"]) == 1
    assert len(matrices["simtel_build_matrix"]) == 2
    assert len(matrices["production_matrix"]) == 8
    assert {(item["avx_flag"], item["arch"]) for item in matrices["corsika_build_matrix"]} == {
        ("generic", "amd64"),
        ("generic", "arm64"),
        ("avx2", "amd64"),
        ("avx512f", "amd64"),
        ("sse4", "amd64"),
    }
    assert {item["arch"] for item in matrices["simtel_build_matrix"]} == {"amd64", "arm64"}
    assert all(
        item["corsika_image"].startswith("ghcr.io/gammasim/corsika7:v")
        for item in matrices["production_matrix"]
    )


def test_catalog_summary_uses_version_tags_without_digests(simtools_root_path):
    """Test optional digests do not affect the current catalog references."""
    catalog = _load_catalog(simtools_root_path)
    summary = dependency_versions.dependency_catalog_summary(catalog)

    assert summary["base_image"] == "docker.io/library/almalinux:9.8-minimal"
    assert summary["corsika_tables_version"] == "v1.0.0"
    assert summary["dev_corsika_image"] == "ghcr.io/gammasim/corsika7:v78010-generic"
    assert summary["model_version"] == catalog["model-database"]["default-version"]
    assert summary["simtools_tests_repository"] == "gammasim/simtools-tests"
    assert summary["simtools_tests_url"].endswith("/simtools-tests.git")
    assert summary["simtools_tests_version"] == "v0.36.0"


def test_env_template_matches_catalog(simtools_root_path):
    """Test the documented environment defaults match the catalog."""
    catalog = _load_catalog(simtools_root_path)

    assert (
        dependency_versions.validate_env_template(catalog, simtools_root_path / ".env_template")
        is None
    )


def test_env_template_rejects_catalog_managed_versions(tmp_test_directory, simtools_root_path):
    """Test catalog-managed versions are not duplicated in the environment template."""
    catalog = _load_catalog(simtools_root_path)
    template = tmp_test_directory / ".env_template"
    template.write_text(
        "SIMTOOLS_DB_SIMULATION_MODEL=CTAO-Simulation-Model\n"
        "SIMTOOLS_DB_SIMULATION_MODEL_VERSION=v0.16.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="catalog-managed"):
        dependency_versions.validate_env_template(catalog, template)


def test_env_template_rejects_mismatched_model_name(tmp_test_directory, simtools_root_path):
    """Test the non-version catalog default is still validated."""
    catalog = _load_catalog(simtools_root_path)
    template = tmp_test_directory / ".env_template"
    template.write_text(
        "SIMTOOLS_DB_SIMULATION_MODEL=wrong-model\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="defaults disagree"):
        dependency_versions.validate_env_template(catalog, template)


def test_env_template_matches_legacy_catalog(tmp_test_directory, simtools_root_path):
    """Test legacy catalogs still validate their model-version template default."""
    catalog = _load_catalog(simtools_root_path)
    catalog.pop("simtools-tests")
    catalog["schema_version"] = "0.1.0"
    catalog["model-database"]["default-version"] = "0.16.0"
    template = tmp_test_directory / ".env_template"
    template.write_text(
        "SIMTOOLS_DB_SIMULATION_MODEL=CTAO-Simulation-Model\n"
        "SIMTOOLS_DB_SIMULATION_MODEL_VERSION=0.16.0\n",
        encoding="utf-8",
    )

    assert dependency_versions.validate_env_template(catalog, template) is None


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
            lambda data: data["corsika"][0].update({"source-ref": "master"}),
            "must identify a release",
        ),
        (
            lambda data: data["sim-telarray"][0].update({"revision": "short"}),
            "Invalid Git revision",
        ),
        (
            lambda data: data["model-database"].update({"default-version": "0.16.0"}),
            "release tags",
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
            lambda data: data["simtools-tests"].pop("version"),
            "release tag",
        ),
        (
            lambda data: data["simtools-tests"].update({"repository": "foo"}),
            "owner/name",
        ),
        (
            lambda data: data["simtools-tests"].update({"source-url": "http://example.com"}),
            "HTTPS",
        ),
        (
            lambda data: data["simtools-tests"].update({"version": "latest"}),
            "release tag",
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
    catalog = _load_catalog(simtools_root_path)
    catalog.pop("simtools-tests")
    catalog["schema_version"] = "0.1.0"
    catalog["model-database"]["default-version"] = "0.16.0"

    assert dependency_versions.validate_dependency_catalog(catalog) is catalog


def test_validate_dependency_catalog_accepts_valid_revisions(simtools_root_path):
    """Test valid component revisions pass catalog validation."""
    catalog = _load_catalog(simtools_root_path)
    revision = "a" * 40
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

    assert "production_matrix=" in output
    assert "python_version=3.14" in output


def test_export_dependency_configuration_returns_environment_values(simtools_root_path):
    """Test env output contains catalog-managed runtime values only."""
    output = dependency_versions.export_dependency_configuration(output_format="env")
    model_version = _load_catalog(simtools_root_path)["model-database"]["default-version"]

    assert output.splitlines() == [
        "SIMTOOLS_DB_SIMULATION_MODEL=CTAO-Simulation-Model",
        f"SIMTOOLS_DB_SIMULATION_MODEL_VERSION={model_version}",
        "SIMTOOLS_TESTS_VERSION=v0.36.0",
        "SIMTOOLS_TESTS_REPOSITORY=gammasim/simtools-tests",
        "SIMTOOLS_TESTS_URL=https://github.com/gammasim/simtools-tests.git",
    ]


def test_dependency_catalog_environment_supports_schema_0_1(simtools_root_path):
    """Test the legacy catalog environment excludes simtools-tests settings."""
    catalog = _load_catalog(simtools_root_path)
    catalog.pop("simtools-tests")
    catalog["schema_version"] = "0.1.0"
    catalog["model-database"]["default-version"] = "0.16.0"

    environment = dependency_versions.dependency_catalog_environment(catalog)

    assert environment == {
        "SIMTOOLS_DB_SIMULATION_MODEL": "CTAO-Simulation-Model",
        "SIMTOOLS_DB_SIMULATION_MODEL_VERSION": "0.16.0",
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
    schema = next(item for item in schemas if item["schema_version"] == catalog["schema_version"])

    jsonschema.validate(catalog, schema)
    assert [item["schema_version"] for item in schemas] == ["0.1.0", "0.2.0"]
    assert "simtools-tests" not in schemas[0]["required"]
    assert "simtools-tests" in schemas[1]["required"]
