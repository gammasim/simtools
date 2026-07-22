import json
import logging
import subprocess
from pathlib import Path

import pytest
import yaml

from simtools.dependencies import (
    _get_build_options_from_file,
    _get_package_path,
    build_dependency_manifest,
    canonical_manifest_bytes,
    export_build_info,
    get_build_options,
    get_corsika_version,
    get_database_version_or_name,
    get_dependency_manifest,
    get_dependency_manifest_digest,
    get_dependency_metadata,
    get_direct_python_dependency_versions,
    get_sim_telarray_version,
    get_software_version,
    get_version_string,
    write_dependency_manifest,
)
from simtools.version import __version__


def test_get_version_string(mocker):
    mocker.patch("simtools.dependencies.get_sim_telarray_version", return_value="2024.271.0")
    mocker.patch("simtools.dependencies.get_corsika_version", return_value="7.7550")
    mocker.patch(
        "simtools.dependencies.get_build_options",
        return_value={"simtel_version": "master", "corsika_version": "78010"},
    )
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {
        "db_simulation_model": "test_db",
        "db_simulation_model_version": "1.2.3",
    }
    result = get_version_string(run_time=["docker"])
    assert "Database name: test_db" in result
    assert "Database version: 1.2.3" in result
    assert "sim_telarray version: 2024.271.0" in result
    assert "CORSIKA version: 7.7550" in result
    assert "Build options: {'simtel_version': 'master', 'corsika_version': '78010'}" in result
    assert "Runtime environment: ['docker']" in result


def test_get_version_string_without_software_versions(mocker):
    mock_simtel = mocker.patch("simtools.dependencies.get_sim_telarray_version")
    mock_corsika = mocker.patch("simtools.dependencies.get_corsika_version")
    mock_build_options = mocker.patch(
        "simtools.dependencies.get_build_options",
        return_value={"simtel_version": "master", "corsika_version": "78010"},
    )
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {
        "db_simulation_model": "test_db",
        "db_simulation_model_version": "1.2.3",
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


def test_get_version_string_without_software_versions_skips_executable_access(mocker):
    class ConfigWithoutExecutables:
        db_config = {
            "db_simulation_model": "test_db",
            "db_simulation_model_version": "1.2.3",
        }

        @property
        def sim_telarray_exe(self):
            raise FileNotFoundError("sim_telarray path not found")

        @property
        def corsika_exe(self):
            raise FileNotFoundError("corsika path not found")

    mocker.patch("simtools.dependencies.settings.config", new=ConfigWithoutExecutables())
    mock_build_options = mocker.patch(
        "simtools.dependencies.get_build_options",
        return_value={"simtel_version": "master", "corsika_version": "78010"},
    )
    mock_simtel = mocker.patch("simtools.dependencies.get_sim_telarray_version")
    mock_corsika = mocker.patch("simtools.dependencies.get_corsika_version")

    result = get_version_string(run_time=["docker"], include_software_versions=False)

    mock_simtel.assert_not_called()
    mock_corsika.assert_not_called()
    mock_build_options.assert_not_called()
    assert "sim_telarray exe: None" in result
    assert "CORSIKA exe: None" in result
    assert "Build options: None" in result


def test_get_software_version_simtools():
    assert get_software_version("simtools") == __version__


def test_get_software_version_unknown():
    with pytest.raises(ValueError, match="Unknown software: unknown_package"):
        get_software_version("unknown_package")


def test_get_database_version_or_name_version_true(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {
        "db_simulation_model": "test_db",
        "db_simulation_model_version": "1.2.3",
    }
    result = get_database_version_or_name(version=True)
    assert result == "1.2.3"


def test_get_database_version_or_name_version_false(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {
        "db_simulation_model": "test_db",
        "db_simulation_model_version": "1.2.3",
    }
    result = get_database_version_or_name(version=False)
    assert result == "test_db"


def test_get_database_version_or_name_none_config(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {"db_simulation_model": "test_db", "db_simulation_model_version": None}
    result = get_database_version_or_name(version=True)
    assert result is None


def test_get_database_version_or_name_missing_keys(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.db_config = {}
    result_version = get_database_version_or_name(version=True)
    result_name = get_database_version_or_name(version=False)
    assert result_version == {}
    assert result_name == {}


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


def test_get_build_options_corsika_and_simtelarray(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")

    class PathMock:
        def __truediv__(self, other):
            return f"/mocked/path/{other}"

        @property
        def parent(self):
            return self

        def __fspath__(self):
            return "/mocked/path"

    mock_config.corsika_path = PathMock()
    mock_config.sim_telarray_path = PathMock()
    mock_ascii_handler = mocker.patch("simtools.dependencies.ascii_handler.collect_data_from_file")
    # Simulate CORSIKA build_opts.yml
    corsika_opts = {
        "corsika_version": "78010",
        "corsika_opt_patch_version": "v1.1.0",
        "variant": [
            {"executable": "corsika_epos_urqmd_curved", "config": "config_epos_urqmd_curved"},
            {"executable": "corsika_epos_urqmd_flat", "config": "config_epos_urqmd_flat"},
        ],
    }
    # Simulate sim_telarray build_opts.yml
    simtel_opts = {
        "simtel_version": "master",
        "components": [
            {"name": "sim_telarray", "version": "master", "executables": ["sim_telarray"]}
        ],
    }
    mock_ascii_handler.side_effect = [corsika_opts, simtel_opts]
    opts = get_build_options()
    assert opts["corsika_version"] == "78010"
    assert opts["simtel_version"] == "master"
    assert "variant" in opts
    assert "components" in opts


def test_get_build_options_legacy(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_path = "/mocked/path"
    mock_ascii_handler = mocker.patch("simtools.dependencies.ascii_handler.collect_data_from_file")
    legacy_opts = {
        "build_opt": "prod6-baseline",
        "corsika_version": "78010",
        "bernlohr_version": "1.70",
    }
    # Simulate the call sequence: corsika new-style fails, sim_telarray new-style fails
    # (both raising FileNotFoundError), then sim_telarray legacy succeeds and returns legacy_opts.
    mock_ascii_handler.side_effect = [FileNotFoundError, FileNotFoundError, legacy_opts]
    opts = get_build_options()
    assert opts["build_opt"] == "prod6-baseline"
    assert opts["corsika_version"] == "78010"
    assert opts["bernlohr_version"] == "1.70"


def test_get_software_version_keyerror():
    # Should raise ValueError for unknown software
    with pytest.raises(ValueError, match="Unknown software: not_a_real_package"):
        get_software_version("not_a_real_package")


def test_get_sim_telarray_version_no_release(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_exe = "sim_telarray"
    mock_run = mocker.patch("simtools.dependencies.subprocess.run")
    mock_run.return_value.stdout = "No version info here"
    with pytest.raises(ValueError, match="sim_telarray release not found"):
        get_sim_telarray_version()


def test_get_sim_telarray_version_empty_output(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_exe = "sim_telarray"
    mock_run = mocker.patch("simtools.dependencies.subprocess.run")
    mock_run.return_value.stdout = ""
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
        "simtools.dependencies.get_build_options", return_value={"corsika_version": "99999"}
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


def test_get_build_options_file_not_found(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")

    class PathMock:
        def __truediv__(self, other):
            return f"/mocked/path/{other}"

        @property
        def parent(self):
            return self

        def __fspath__(self):
            return "/mocked/path"

    mock_config.corsika_path = PathMock()
    mock_config.sim_telarray_path = PathMock()
    mocker.patch(
        "simtools.dependencies.ascii_handler.collect_data_from_file", side_effect=FileNotFoundError
    )
    with pytest.raises(FileNotFoundError, match="No build option file found"):
        get_build_options()


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


def test_get_build_options_debug_logging_on_exception(mocker, caplog):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = "/mocked/corsika"
    mock_config.sim_telarray_path = "/mocked/simtel"
    mock_ascii_handler = mocker.patch("simtools.dependencies.ascii_handler.collect_data_from_file")
    # First call raises FileNotFoundError, legacy fallback also raises
    # Ensure all attempted reads fail:
    # corsika new-style, sim_telarray new-style, sim_telarray legacy.
    mock_ascii_handler.side_effect = [FileNotFoundError, FileNotFoundError, FileNotFoundError]
    caplog.set_level(logging.DEBUG)
    with pytest.raises(FileNotFoundError):
        get_build_options()
    assert any("No build options found for sim_telarray." in m for m in caplog.text.splitlines())


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
        "simtools.dependencies.get_build_options", return_value={"corsika_version": "78010"}
    )
    mocker.patch(
        "simtools.dependencies.get_database_version_or_name", side_effect=["test_db", "1.2.3"]
    )
    mocker.patch(
        "simtools.dependencies.get_dependency_manifest",
        return_value={"schema_version": "0.1.0"},
    )

    output_file = Path(str(tmp_test_directory)) / "build_info.yml"
    export_build_info(output_file, run_time=None)

    mock_write.assert_called_once()
    call_args = mock_write.call_args
    assert call_args[1]["data"]["corsika_version"] == "78010"
    assert call_args[1]["data"]["simtools"] == __version__
    assert call_args[1]["data"]["database_name"] == "test_db"
    assert call_args[1]["data"]["database_version"] == "1.2.3"


def test_export_build_info_with_run_time(mocker, tmp_test_directory):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = None
    mock_config.sim_telarray_path = None
    mock_write = mocker.patch("simtools.dependencies.ascii_handler.write_data_to_file")
    mock_get_build = mocker.patch(
        "simtools.dependencies.get_build_options", return_value={"simtel_version": "master"}
    )
    mocker.patch(
        "simtools.dependencies.get_database_version_or_name", side_effect=["prod_db", "2.0.0"]
    )
    mocker.patch(
        "simtools.dependencies.get_dependency_manifest",
        return_value={"schema_version": "0.1.0"},
    )

    output_file = Path(str(tmp_test_directory)) / "build_info.yml"
    run_time = ["docker"]
    export_build_info(output_file, run_time=run_time)

    mock_get_build.assert_called_once_with(run_time)
    mock_write.assert_called_once()
    call_args = mock_write.call_args
    assert call_args[1]["data"]["simtel_version"] == "master"
    assert call_args[1]["data"]["database_name"] == "prod_db"
    assert call_args[1]["data"]["database_version"] == "2.0.0"


def test_get_package_path_from_config(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = Path("/mocked/corsika")
    result = _get_package_path("corsika")
    assert result == Path("/mocked/corsika")


def test_get_package_path_from_environment(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = None
    mock_load_env = mocker.patch("simtools.dependencies.gen.load_environment_variables")
    mock_load_env.return_value = {"corsika_path": "/env/corsika"}
    result = _get_package_path("corsika")
    assert result == Path("/env/corsika")


def test_get_package_path_sim_telarray_from_config(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_path = Path("/mocked/simtel")
    result = _get_package_path("sim_telarray")
    assert result == Path("/mocked/simtel")


def test_get_package_path_sim_telarray_from_environment(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.sim_telarray_path = None
    mock_load_env = mocker.patch("simtools.dependencies.gen.load_environment_variables")
    mock_load_env.return_value = {"sim_telarray_path": "/env/simtel"}
    result = _get_package_path("sim_telarray")
    assert result == Path("/env/simtel")


def test_get_package_path_not_found(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = None
    mock_load_env = mocker.patch("simtools.dependencies.gen.load_environment_variables")
    mock_load_env.return_value = {}
    result = _get_package_path("corsika")
    assert result is None


def test_get_package_path_config_takes_precedence(mocker):
    mock_config = mocker.patch("simtools.dependencies.settings.config")
    mock_config.corsika_path = Path("/config/corsika")
    mock_load_env = mocker.patch("simtools.dependencies.gen.load_environment_variables")
    mock_load_env.return_value = {"corsika_path": "/env/corsika"}
    result = _get_package_path("corsika")
    assert result == Path("/config/corsika")
    mock_load_env.assert_not_called()


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


def test_get_direct_python_dependency_versions(mocker):
    mocker.patch(
        "simtools.dependencies.metadata.requires",
        return_value=["astropy>=7", "pytest; extra == 'tests'", "numpy"],
    )
    versions = {"astropy": "8.0.0", "numpy": "2.5.0"}
    mocker.patch("simtools.dependencies.metadata.version", side_effect=versions.get)

    assert get_direct_python_dependency_versions() == versions


def test_build_dependency_manifest(mocker, monkeypatch, simtools_root_path):
    mocker.patch(
        "simtools.dependencies.get_build_options", return_value={"corsika_version": "78010"}
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
    assert output.with_suffix(".json.sha256").is_file()


def test_get_dependency_metadata(mocker):
    manifest = {
        "simtools": {"revision": "a" * 40},
        "runtime": {"python_version": "3.14.6"},
    }
    mocker.patch("simtools.dependencies.get_dependency_manifest", return_value=manifest)

    result = get_dependency_metadata()

    assert result["simtools_git_revision"] == "a" * 40
    assert result["simtools_python_version"] == "3.14.6"
    assert len(result["simtools_dependency_manifest_sha256"]) == 64
