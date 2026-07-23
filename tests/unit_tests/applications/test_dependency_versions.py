import copy

import pytest
import yaml

from simtools.applications import dependency_versions


def test_application_definition_is_configured():
    """Test the dependency-version exporter uses the standard application definition."""
    assert dependency_versions.APPLICATION is not None
    assert dependency_versions.APPLICATION.setup_io_handler is False
    assert dependency_versions.APPLICATION.resolve_sim_software_executables is False


def test_load_dependency_catalog_and_build_matrices(simtools_root_path):
    catalog = dependency_versions.load_dependency_catalog(simtools_root_path / "pyproject.toml")
    matrices = dependency_versions.build_workflow_matrices(catalog)

    assert catalog["python"] == "3.14"
    assert len(matrices["corsika_matrix"]) == 8
    assert len(matrices["simtel_matrix"]) == 1
    assert len(matrices["production_matrix"]) == 8
    assert all("@sha256:" in item["corsika_image"] for item in matrices["production_matrix"])


def test_catalog_summary_uses_immutable_images(simtools_root_path):
    catalog = dependency_versions.load_dependency_catalog(simtools_root_path / "pyproject.toml")
    summary = dependency_versions.dependency_catalog_summary(catalog)

    assert summary["base_image"].startswith("docker.io/library/almalinux@sha256:")
    assert summary["dev_corsika_image"].startswith("ghcr.io/gammasim/corsika7@sha256:")
    assert summary["model_version"] == "0.16.0"


def test_env_template_matches_catalog(simtools_root_path):
    catalog = dependency_versions.load_dependency_catalog(simtools_root_path / "pyproject.toml")

    dependency_versions.validate_env_template(catalog, simtools_root_path / ".env_template")


def test_env_template_rejects_mismatched_model_version(tmp_test_directory, simtools_root_path):
    catalog = dependency_versions.load_dependency_catalog(simtools_root_path / "pyproject.toml")
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
    ],
)
def test_validate_dependency_catalog_rejects_invalid_values(simtools_root_path, mutator, error):
    catalog = dependency_versions.load_dependency_catalog(simtools_root_path / "pyproject.toml")
    invalid = copy.deepcopy(catalog)
    mutator(invalid)

    with pytest.raises(ValueError, match=error):
        dependency_versions.validate_dependency_catalog(invalid)


def test_load_dependency_catalog_missing_table(tmp_test_directory):
    project_file = tmp_test_directory / "pyproject.toml"
    project_file.write_text('[project]\nname = "example"\n', encoding="utf-8")

    with pytest.raises(KeyError, match="Missing"):
        dependency_versions.load_dependency_catalog(project_file)


def test_find_pyproject_from_environment(monkeypatch, simtools_root_path):
    project_file = simtools_root_path / "pyproject.toml"
    monkeypatch.setenv("SIMTOOLS_PYPROJECT", str(project_file))

    assert dependency_versions.find_pyproject("/") == project_file


def test_main_exports_github_outputs(capsys, simtools_root_path):
    dependency_versions.main(
        ["--pyproject", str(simtools_root_path / "pyproject.toml"), "--format", "github-output"]
    )

    output = capsys.readouterr().out
    assert "production_matrix=" in output
    assert "python_version=3.14" in output


def test_main_exports_python_requirements(capsys, simtools_root_path):
    dependency_versions.main(
        [
            "--pyproject",
            str(simtools_root_path / "pyproject.toml"),
            "--format",
            "python-requirements",
            "--extras",
            "tests",
        ]
    )

    requirements = capsys.readouterr().out.splitlines()
    assert "astropy" in requirements
    assert "pytest" in requirements


def test_catalog_matches_json_schema(simtools_root_path):
    import jsonschema

    catalog = dependency_versions.load_dependency_catalog(simtools_root_path / "pyproject.toml")
    schema_path = simtools_root_path / "src/simtools/schemas/dependency_versions.schema.yml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    jsonschema.validate(catalog, schema)
