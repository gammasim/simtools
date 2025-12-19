import logging
import subprocess
from pathlib import Path

import pytest
import yaml

from simtools.dependencies import (
    _get_build_options_from_file,
    _get_package_path,
    export_build_info,
    get_build_options,
    get_corsika_version,
    get_database_version_or_name,
    get_sim_telarray_version,
    get_software_version,
    get_version_string,
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
    # Calls order in legacy mode:
    # 1) corsika new-style -> FileNotFoundError
    # 2) sim_telarray new-style -> FileNotFoundError
    # 3) sim_telarray legacy fallback -> returns legacy_opts
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
    # Ensure all attempted reads fail: corsika new-style, sim_telarray new-style, sim_telarray legacy
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


def test_export_build_info(mocker, tmp_path):
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

    output_file = tmp_path / "build_info.yml"
    export_build_info(output_file, run_time=None)

    mock_write.assert_called_once()
    call_args = mock_write.call_args
    assert call_args[1]["data"]["corsika_version"] == "78010"
    assert call_args[1]["data"]["simtools"] == __version__
    assert call_args[1]["data"]["database_name"] == "test_db"
    assert call_args[1]["data"]["database_version"] == "1.2.3"


def test_export_build_info_with_run_time(mocker, tmp_path):
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

    output_file = tmp_path / "build_info.yml"
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
