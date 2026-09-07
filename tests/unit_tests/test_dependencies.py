import json
import subprocess
from pathlib import Path

import pytest
import yaml
from astropy import units as u

from simtools.dependencies import (
    _collect_dependency_error,
    _get_build_options_from_file,
    _get_package_path,
    _is_git_lfs_pointer,
    _manifest_entries,
    _validate_corsika_interaction_tables,
    _validate_table_entry,
    _validate_table_file,
    _validate_table_manifest_structure,
    build_dependency_manifest,
    canonical_manifest_bytes,
    export_build_info,
    get_corsika_version,
    get_database_tag_or_name,
    get_database_version_or_name,
    get_dependency_manifest,
    get_dependency_manifest_digest,
    get_sim_telarray_version,
    get_software_version,
    get_version_string,
    validate_simulation_dependencies,
    write_dependency_manifest,
    write_development_dependency_manifest,
)
from simtools.version import __version__


def _write_interaction_table_manifest(table_path):
    """Create a compact manifest fixture for dependency validation tests."""
    table_path = Path(str(table_path))
    groups = {
        "common": [{"path": "common.dat", "size": 5}],
        "electromagnetic": {"egs4": [{"path": "egs.dat", "size": 3}]},
        "low_energy": {"urqmd": [{"path": "urqmd.dat", "size": 4}]},
        "high_energy": {
            "qgs3": [
                {"path": "qgsdat-III", "size": 6},
                {"path": "sectnu-III", "size": 7},
            ]
        },
    }
    manifest = {"schema_version": "1.0.0", "files": groups}
    for entries in (
        groups["common"],
        groups["electromagnetic"]["egs4"],
        groups["low_energy"]["urqmd"],
        groups["high_energy"]["qgs3"],
    ):
        for entry in entries:
            (table_path / entry["path"]).write_bytes(b"x" * entry["size"])
    (table_path / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def _mock_corsika_config(mocker, table_path):
    """Patch settings with a valid CORSIKA dependency fixture."""
    table_path = Path(str(table_path))
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.args = {}
    mock_config.corsika_exe = table_path / "corsika"
    mock_config.corsika_exe_curved = table_path / "corsika_curved"
    mock_config.corsika_interaction_table_path = table_path
    mock_config.corsika_interaction_models = ("qgs3", "urqmd")
    return mock_config


def test_validate_simulation_dependencies_accepts_manifest(tmp_test_directory, mocker):
    """A hydrated manifest and all selected files satisfy CORSIKA validation."""
    _write_interaction_table_manifest(tmp_test_directory)
    _mock_corsika_config(mocker, tmp_test_directory)

    validate_simulation_dependencies("corsika")


def test_validate_simulation_dependencies_uses_curved_corsika_executable(
    tmp_test_directory, mocker
):
    """Curved simulations validate the executable selected by their zenith angle."""
    _write_interaction_table_manifest(tmp_test_directory)
    mock_config = _mock_corsika_config(mocker, tmp_test_directory)
    mock_config.args = {
        "zenith_angle": 70 * u.deg,
        "curved_atmosphere_min_zenith_angle": 65 * u.deg,
    }
    mock_config.corsika_exe_curved = None

    with pytest.raises(ValueError, match=r"CORSIKA \(curved\): not configured"):
        validate_simulation_dependencies("corsika")


def test_validate_simulation_dependencies_uses_flat_corsika_executable(tmp_test_directory, mocker):
    """Flat simulations do not require the curved CORSIKA executable."""
    _write_interaction_table_manifest(tmp_test_directory)
    mock_config = _mock_corsika_config(mocker, tmp_test_directory)
    mock_config.args = {
        "zenith_angle": 20 * u.deg,
        "curved_atmosphere_min_zenith_angle": 65 * u.deg,
    }
    mock_config.corsika_exe_curved = None

    validate_simulation_dependencies("corsika")


def test_validate_simulation_dependencies_rejects_missing_and_unhydrated_tables(
    tmp_test_directory, mocker
):
    """Validation reports missing files, size mismatches, and LFS pointers."""
    _write_interaction_table_manifest(tmp_test_directory)
    table_path = Path(str(tmp_test_directory))
    _mock_corsika_config(mocker, table_path)
    (table_path / "common.dat").unlink()
    (table_path / "egs.dat").write_bytes(b"x")
    (table_path / "qgsdat-III").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")

    with pytest.raises(ValueError, match=r"missing file.*common.dat") as error:
        validate_simulation_dependencies("corsika")

    message = str(error.value)
    assert "size mismatch" in message
    assert "Git LFS pointer" in message


def test_validate_simulation_dependencies_rejects_unknown_model_group(tmp_test_directory, mocker):
    """A model absent from the manifest fails before any table is used."""
    _write_interaction_table_manifest(tmp_test_directory)
    mock_config = _mock_corsika_config(mocker, tmp_test_directory)
    mock_config.corsika_interaction_models = ("qgs2", "urqmd")

    with pytest.raises(ValueError, match=r"files\.high_energy\.qgs2"):
        validate_simulation_dependencies("corsika")


def test_validate_simulation_dependencies_only_checks_selected_software(mocker):
    """sim_telarray-only validation does not access CORSIKA settings."""
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_exe = "sim_telarray"

    validate_simulation_dependencies("sim_telarray")


def test_validate_simulation_dependencies_requires_both_for_combined_mode(mocker):
    """Combined simulations report an unavailable sim_telarray executable."""
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_exe = None
    mock_config.corsika_exe = None
    mock_config.corsika_interaction_table_path = None
    mock_config.corsika_interaction_models = ("qgs3", "urqmd")

    with pytest.raises(ValueError, match="sim_telarray: not configured"):
        validate_simulation_dependencies("corsika_sim_telarray")


def test_validate_simulation_dependencies_rejects_unknown_software():
    """Unknown simulation software selections are rejected explicitly."""
    with pytest.raises(ValueError, match="Unknown simulation software: unknown"):
        validate_simulation_dependencies("unknown")


@pytest.mark.parametrize("error_type", [FileNotFoundError, PermissionError, TypeError, ValueError])
def test_collect_dependency_error_reports_access_errors(error_type):
    """Dependency access errors are collected with their dependency name."""
    errors = []

    def raise_error():
        raise error_type("bad")

    _collect_dependency_error(errors, "test", raise_error)

    assert errors == ["test: bad"]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ([], "manifest root is not a mapping"),
        ({"schema_version": "invalid"}, "invalid schema_version"),
        ({"schema_version": "2.0", "files": {}}, "unsupported schema_version"),
        ({"schema_version": "1.0", "files": []}, "manifest.files is not a mapping"),
        ({"schema_version": "1.0", "files": {}}, "missing category group"),
    ],
)
def test_validate_table_manifest_structure_rejects_invalid_manifests(
    manifest, message, tmp_test_directory
):
    """Malformed manifests produce actionable structure errors."""
    with pytest.raises(ValueError, match=message):
        _validate_table_manifest_structure(manifest, Path(tmp_test_directory) / "manifest.yaml")


@pytest.mark.parametrize(
    ("manifest", "category", "model", "message"),
    [
        ({"files": {"common": {}}}, "common", None, "manifest group is not a list"),
        (
            {"files": {"high_energy": {}}},
            "high_energy",
            "qgs3",
            "manifest model group is missing",
        ),
        (
            {"files": {"high_energy": {"qgs3": {}}}},
            "high_energy",
            "qgs3",
            "manifest group is not a list",
        ),
    ],
)
def test_manifest_entries_rejects_invalid_groups(manifest, category, model, message):
    """Selected manifest groups must exist and contain lists."""
    with pytest.raises(ValueError, match=message):
        _manifest_entries(manifest, category, model)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (None, "invalid manifest table entry"),
        ({"path": "nested/file", "size": 1}, "invalid manifest table path"),
        ({"path": "file", "size": -1}, "invalid manifest table size"),
        ({"path": "file", "size": "1"}, "invalid manifest table size"),
    ],
)
def test_validate_table_entry_rejects_invalid_entries(entry, message, tmp_test_directory):
    """Manifest entries must contain safe filenames and non-negative sizes."""
    assert message in _validate_table_entry(entry, Path(tmp_test_directory))


def test_validate_table_file_reports_unreadable_file(tmp_test_directory, mocker):
    """Unreadable table files are reported before content validation."""
    table_file = Path(tmp_test_directory) / "table.dat"
    table_file.write_bytes(b"x")
    mocker.patch("simtools.dependencies.os.access", return_value=False)

    assert "file is not readable" in _validate_table_file(table_file, 1)


def test_is_git_lfs_pointer_handles_read_errors(mocker):
    """An unreadable file is not mistaken for an LFS pointer."""
    path = mocker.Mock()
    path.open.side_effect = OSError("cannot open")

    assert _is_git_lfs_pointer(path) is False


def test_validate_corsika_interaction_tables_reports_manifest_read_error(tmp_test_directory):
    """A missing manifest is reported as a table validation error."""
    with pytest.raises(ValueError, match="cannot read manifest"):
        _validate_corsika_interaction_tables(Path(tmp_test_directory) / "missing")


def test_get_version_string(mocker):
    mocker.patch("simtools.dependencies.get_sim_telarray_version", return_value="2024.271.0")
    mocker.patch("simtools.dependencies.get_corsika_version", return_value="7.7550")
    mocker.patch(
        "simtools.dependencies.get_build_options",
        return_value={"simtel_tag": "master", "corsika_build_id": "78010"},
    )
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {
        "db_simulation_model": "test_db",
        "db_simulation_model_tag": "1.2.3",
    }
    result = get_version_string(run_time=["docker"])
    assert "Database name: test_db" in result
    assert "Database release tag: 1.2.3" in result
    assert "sim_telarray version: 2024.271.0" in result
    assert "CORSIKA version: 7.7550" in result
    assert "Build options: {'simtel_tag': 'master', 'corsika_build_id': '78010'}" in result
    assert "Runtime environment: ['docker']" in result


def test_database_tag_accessors_keep_the_legacy_interface(mocker):
    """Expose canonical tag terminology while retaining the old accessor."""
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {
        "db_simulation_model": "test_db",
        "db_simulation_model_tag": "v1.2.3",
    }

    assert get_database_tag_or_name() == "v1.2.3"
    assert get_database_tag_or_name(tag=False) == "test_db"
    assert get_database_version_or_name() == "v1.2.3"
    assert get_database_version_or_name(version=False) == "test_db"


def test_direct_dependency_manifest_includes_mongodb_only_when_installed(mocker):
    """Optional MongoDB is reported when its extra has installed the distribution."""
    import simtools.dependencies as dependencies

    mocker.patch.object(
        dependencies.metadata,
        "requires",
        return_value=["astropy", 'pymongo; extra == "mongodb"', 'pytest; extra == "tests"'],
    )
    mocker.patch.object(
        dependencies,
        "_distribution_version",
        side_effect=lambda name: {"astropy": "8.0.0", "pymongo": "4.15.0"}.get(name),
    )

    assert dependencies.get_direct_python_dependency_versions() == {
        "astropy": "8.0.0",
        "pymongo": "4.15.0",
    }


def test_get_version_string_without_software_versions(mocker):
    mock_simtel = mocker.patch("simtools.dependencies.get_sim_telarray_version")
    mock_corsika = mocker.patch("simtools.dependencies.get_corsika_version")
    mock_build_options = mocker.patch(
        "simtools.dependencies.get_build_options",
        return_value={"simtel_tag": "master", "corsika_build_id": "78010"},
    )
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {
        "db_simulation_model": "test_db",
        "db_simulation_model_tag": "1.2.3",
    }
    mock_config.sim_telarray_exe = None
    mock_config.corsika_exe = None

    result = get_version_string(run_time=["docker"], include_software_versions=False)

    mock_simtel.assert_not_called()
    mock_corsika.assert_not_called()
    mock_build_options.assert_not_called()
    assert "sim_telarray version: None" in result
    assert "CORSIKA version: None" in result
    assert "Build options: None" in result


def test_get_software_version_simtools():
    assert get_software_version("simtools") == __version__


def test_get_software_version_unknown():
    with pytest.raises(ValueError, match="Unknown software: unknown_package"):
        get_software_version("unknown_package")


def test_get_sim_telarray_version_simple(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_exe = "sim_telarray"
    mock_run = mocker.patch("simtools.dependencies.subprocess.run")
    mock_run.return_value.stdout = "Release: 2024.271.0 from 2024-09-27\n"
    version = get_sim_telarray_version()
    assert version == "2024.271.0"


def test_get_corsika_version_simple(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")

    class PathMock:
        def __truediv__(self, other):
            return f"/mocked/path/{other}"

        def __fspath__(self):
            return "/mocked/path"

    mock_config.corsika_path = PathMock()
    mock_config.corsika_exe = "corsika"
    mock_popen = mocker.patch("simtools.dependencies.subprocess.Popen")
    process_mock = mocker.Mock()
    process_mock.stdout = ["NUMBER OF VERSION :  7.7550\n"]
    mock_popen.return_value = process_mock
    process_mock.terminate = mocker.Mock()
    version = get_corsika_version()
    assert version == "7.7550"


def test_get_sim_telarray_version_no_release(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_exe = "sim_telarray"
    mock_run = mocker.patch("simtools.dependencies.subprocess.run")
    mock_run.return_value.stdout = "No version info here"
    with pytest.raises(ValueError, match="sim_telarray release not found"):
        get_sim_telarray_version()


def test_get_corsika_version_typeerror(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_exe = None
    version = get_corsika_version()
    assert version is None


def test_get_corsika_version_no_version_but_build_opts(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")

    class PathMock:
        def __truediv__(self, other):
            return f"/mocked/path/{other}"

        def __fspath__(self):
            return "/mocked/path"

    mock_config.corsika_path = PathMock()
    mock_config.corsika_exe = "corsika"
    mock_popen = mocker.patch("simtools.dependencies.subprocess.Popen")
    process_mock = mocker.Mock()
    process_mock.stdout = ["DATA CARDS FOR RUN STEERING ARE EXPECTED FROM STANDARD INPUT\n"]
    mock_popen.return_value = process_mock
    process_mock.terminate = mocker.Mock()
    mocker.patch(
        "simtools.dependencies.get_build_options", return_value={"corsika_build_id": "99999"}
    )
    version = get_corsika_version()
    assert version == "99999"


def test_get_corsika_version_no_build_opts(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")

    class PathMock:
        def __truediv__(self, other):
            return f"/mocked/path/{other}"

        def __fspath__(self):
            return "/mocked/path"

    mock_config.corsika_path = PathMock()
    mock_config.corsika_exe = "corsika"
    mock_popen = mocker.patch("simtools.dependencies.subprocess.Popen")
    process_mock = mocker.Mock()
    process_mock.stdout = ["DATA CARDS FOR RUN STEERING ARE EXPECTED FROM STANDARD INPUT\n"]
    mock_popen.return_value = process_mock
    process_mock.terminate = mocker.Mock()
    mocker.patch("simtools.dependencies.get_build_options", side_effect=FileNotFoundError)
    version = get_corsika_version()
    assert version is None


def test__get_build_options_from_file_yaml_error(mocker):
    mocker.patch("simtools.dependencies.yaml.safe_load", side_effect=yaml.YAMLError("bad yaml"))
    mock_run = mocker.patch("simtools.dependencies.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "bad: yaml"
    with pytest.raises(ValueError, match=r"Error parsing build_opts.yml from container"):
        _get_build_options_from_file("/mocked/path/build_opts.yml", run_time=["docker"])


def test__get_build_options_from_file_subprocess_error(mocker):
    mock_run = mocker.patch("simtools.dependencies.subprocess.run")
    mock_run.return_value.returncode = 1
    mock_run.return_value.stderr = "file not found"
    with pytest.raises(FileNotFoundError, match="No build option file found in container"):
        _get_build_options_from_file("/mocked/path/build_opts.yml", run_time=["docker"])


def test_get_sim_telarray_version_with_run_time(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_exe = "sim_telarray"
    mock_run = mocker.patch("simtools.dependencies.subprocess.run")
    mock_run.return_value.stdout = "Release: 2024.271.0 from 2024-09-27\n"
    mock_run.return_value.stderr = ""
    run_time = ["docker"]
    version = get_sim_telarray_version(run_time=run_time)
    assert version == "2024.271.0"
    mock_run.assert_called_once_with(
        ["docker", "sim_telarray", "--version"], capture_output=True, text=True, check=False
    )


def test_get_corsika_version_with_run_time(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")

    class PathMock:
        def __truediv__(self, other):
            return f"/mocked/path/{other}"

    mock_config.corsika_exe = PathMock() / "corsika"
    mock_popen = mocker.patch("simtools.dependencies.subprocess.Popen")
    process_mock = mocker.Mock()
    process_mock.stdout = ["NUMBER OF VERSION :  7.7550\n"]
    process_mock.terminate = mocker.Mock()
    mock_popen.return_value = process_mock
    run_time = ["docker"]
    version = get_corsika_version(run_time=run_time)
    assert version == "7.7550"
    mock_popen.assert_called_once_with(
        ["docker", "/mocked/path/corsika"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
    )


def test_export_build_info(mocker, tmp_test_directory):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = None
    mock_config.sim_telarray_path = None
    mock_write = mocker.patch("simtools.dependencies.ascii_handler.write_data_to_file")
    mocker.patch(
        "simtools.dependencies.get_build_options", return_value={"corsika_build_id": "78010"}
    )
    mocker.patch(
        "simtools.dependencies.get_database_tag_or_name", side_effect=["test_db", "v1.2.3"]
    )
    mocker.patch(
        "simtools.dependencies.get_dependency_manifest",
        return_value={"schema_version": "0.1.0"},
    )

    output_file = Path(str(tmp_test_directory)) / "build_info.yml"
    export_build_info(output_file, run_time=None)

    mock_write.assert_called_once()
    call_args = mock_write.call_args
    assert call_args[1]["data"]["corsika_build_id"] == "78010"
    assert call_args[1]["data"]["simtools"] == __version__
    assert call_args[1]["data"]["database_name"] == "test_db"
    assert call_args[1]["data"]["database_tag"] == "v1.2.3"


def test_get_package_path_from_environment(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = None
    mock_load_env = mocker.patch("simtools.dependencies.gen.load_environment_variables")
    mock_load_env.return_value = {"corsika_path": "/env/corsika"}
    result = _get_package_path("corsika")
    assert result == Path("/env/corsika")


def test_get_package_path_not_found(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = None
    mock_load_env = mocker.patch("simtools.dependencies.gen.load_environment_variables")
    mock_load_env.return_value = {}
    result = _get_package_path("corsika")
    assert result is None


def test_get_dependency_manifest_reads_configured_file(monkeypatch, tmp_test_directory):
    manifest_path = tmp_test_directory / "manifest.json"
    manifest_path.write_text('{"schema_version": "0.1.0"}', encoding="utf-8")
    monkeypatch.setenv("SIMTOOLS_DEPENDENCY_MANIFEST", str(manifest_path))

    assert get_dependency_manifest() == {"schema_version": "0.1.0"}


def test_get_dependency_manifest_invalid_json(monkeypatch, tmp_test_directory):
    manifest_path = tmp_test_directory / "manifest.json"
    manifest_path.write_text("not-json", encoding="utf-8")
    monkeypatch.setenv("SIMTOOLS_DEPENDENCY_MANIFEST", str(manifest_path))

    with pytest.raises(ValueError, match="Invalid dependency manifest"):
        get_dependency_manifest()


def test_get_dependency_manifest_from_container(mocker):
    run = mocker.patch("simtools.dependencies.subprocess.run")
    run.return_value.returncode = 0
    run.return_value.stdout = '{"schema_version": "0.1.0"}'

    manifest = get_dependency_manifest(["apptainer", "exec", "image.sif"])

    assert manifest["schema_version"] == "0.1.0"
    assert run.call_args.args[0][-2:] == [
        "cat",
        "/opt/simtools/provenance/dependency-manifest.json",
    ]


def test_get_dependency_manifest_container_missing(mocker):
    run = mocker.patch("simtools.dependencies.subprocess.run")
    run.return_value.returncode = 1
    run.return_value.stderr = "missing"

    with pytest.raises(FileNotFoundError, match="not found in container"):
        get_dependency_manifest(["docker", "run"])


def test_build_dependency_manifest(mocker, monkeypatch, simtools_root_path):
    mocker.patch(
        "simtools.dependencies.get_build_options", return_value={"corsika_build_id": "78010"}
    )
    mocker.patch(
        "simtools.dependencies.get_direct_python_dependency_versions",
        return_value={"astropy": "8.0.0"},
    )
    mocker.patch("simtools.dependencies._distribution_version", return_value="26.1.2")
    monkeypatch.setenv("SIMTOOLS_CONTAINER_BUILD", "1")
    monkeypatch.setenv("SIMTOOLS_GIT_REVISION", "b" * 40)

    manifest = build_dependency_manifest()

    assert manifest["source"] == "container-build"
    assert manifest["simtools"]["revision"] == "b" * 40
    assert manifest["runtime"]["direct_python_dependencies"] == {"astropy": "8.0.0"}
    assert "model_source" not in manifest["runtime"]

    import jsonschema

    schema_path = simtools_root_path / "src/simtools/schemas/dependency_manifest.schema.yml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)


def test_manifest_digest_is_independent_of_dictionary_order(mocker):
    first = {"b": 2, "a": 1}
    second = {"a": 1, "b": 2}
    mocker.patch("simtools.dependencies.get_dependency_manifest", return_value=first)
    first_digest = get_dependency_manifest_digest()
    mocker.patch("simtools.dependencies.get_dependency_manifest", return_value=second)

    assert canonical_manifest_bytes(first) == canonical_manifest_bytes(second)
    assert get_dependency_manifest_digest() == first_digest


def test_write_dependency_manifest(mocker, tmp_test_directory):
    manifest = {"schema_version": "0.1.0", "value": "test"}
    mocker.patch("simtools.dependencies.build_dependency_manifest", return_value=manifest)
    output = Path(str(tmp_test_directory)) / "dependency-manifest.json"

    write_dependency_manifest(output)

    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert Path(str(output)).with_suffix(".json.sha256").is_file()


def test_write_development_dependency_manifest(mocker, monkeypatch, tmp_test_directory):
    project_file = tmp_test_directory / "pyproject.toml"
    project_file.write_text('[project]\ndependencies = ["astropy>=7", "numpy"]\n', encoding="utf-8")
    corsika_options = tmp_test_directory / "corsika_build_opts.yml"
    corsika_options.write_text("corsika_version: '78010'\nbuild_date: ignored\n", encoding="utf-8")
    simtel_options = tmp_test_directory / "simtel_build_opts.yml"
    simtel_options.write_text("simtel_version: v2025-11-30-rc\n", encoding="utf-8")
    mocker.patch(
        "simtools.dependencies._distribution_version",
        side_effect=lambda package: {"pip": "26.1.2", "astropy": "8.0.0", "numpy": "2.5.0"}.get(
            package
        ),
    )
    monkeypatch.setenv("SIMTOOLS_GIT_REVISION", "a" * 40)
    monkeypatch.setenv("SIMTOOLS_BASE_IMAGE", "alma:9")
    output = tmp_test_directory / "dependency-manifest.json"

    write_development_dependency_manifest(output, project_file, [corsika_options, simtel_options])

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["simtools"] == {"revision": "a" * 40, "version": "not-installed"}
    assert manifest["runtime"]["direct_python_dependencies"] == {
        "astropy": "8.0.0",
        "numpy": "2.5.0",
    }
    assert manifest["build_options"] == {
        "corsika_build_id": "78010",
        "simtel_tag": "v2025-11-30-rc",
    }
    assert manifest["container"] == {"base_image": "alma:9"}
    assert Path(str(output)).with_suffix(".json.sha256").is_file()
