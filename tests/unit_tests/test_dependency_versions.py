"""Tests for the dependency version catalog helpers."""

import copy
import json

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
    assert summary["model_version"] == "0.16.0"


def test_env_template_matches_catalog(simtools_root_path):
    """Test the documented environment defaults match the catalog."""
    catalog = _load_catalog(simtools_root_path)

    dependency_versions.validate_env_template(catalog, simtools_root_path / ".env_template")


def test_env_template_rejects_mismatched_model_version(tmp_test_directory, simtools_root_path):
    """Test invalid model-version defaults are rejected."""
    catalog = _load_catalog(simtools_root_path)
    template = tmp_test_directory / ".env_template"
    template.write_text(
        "SIMTOOLS_DB_SIMULATION_MODEL=CTAO-Simulation-Model\n"
        "SIMTOOLS_DB_SIMULATION_MODEL_VERSION=v0.16.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disagree"):
        dependency_versions.validate_env_template(catalog, template)


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda data: data.pop("python"), "Missing dependency catalog keys"),
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
            lambda data: data["model-database"].update({"default-version": "v0.16.0"}),
            "must not start",
        ),
        (
            lambda data: data["production-combinations"][0].update({"cpu-variants": ["unknown"]}),
            "Unknown CPU variant",
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
    """Test an explicit project-file environment setting wins."""
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
    """Test the catalog library returns GitHub Actions outputs."""
    output = dependency_versions.export_dependency_configuration(
        simtools_root_path / "pyproject.toml", "github-output"
    )

    assert "production_matrix=" in output
    assert "python_version=3.14" in output


def test_export_dependency_configuration_returns_python_requirements(simtools_root_path):
    """Test the catalog library returns optional Python dependencies."""
    requirements = dependency_versions.export_dependency_configuration(
        simtools_root_path / "pyproject.toml", "python-requirements", ["tests"]
    )

    assert "astropy" in requirements.splitlines()
    assert "pytest" in requirements.splitlines()


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
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    jsonschema.validate(catalog, schema)
