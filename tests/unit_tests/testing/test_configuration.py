#!/usr/bin/python3

from pathlib import Path

import pytest
import yaml

import simtools.testing.configuration as configuration


@pytest.fixture
def integration_test_config_files():
    config_path = Path("tests/integration_tests/config")
    return sorted(config_path.glob("*.yml"))


@pytest.fixture
def tmp_config_string():
    return "tmp_config.yml"


@pytest.fixture
def mocker_pytest_skip(mocker):
    return mocker.patch("pytest.skip")


def test_get_list_of_test_configurations_test_names(integration_test_config_files):
    _, test_names = configuration.get_list_of_test_configurations(integration_test_config_files)

    for test_name in test_names:
        assert isinstance(test_name, str)
        assert "simtools" in test_name
    assert "simtools-calculate-trigger-rate_file_list" in test_names


def test_get_list_of_test_configurations(integration_test_config_files):
    test_configs, test_names = configuration.get_list_of_test_configurations(
        integration_test_config_files
    )

    list_test_with_help = []
    list_test_with_version = []
    list_test_without_config = []

    for test_config in test_configs:
        assert isinstance(test_config, dict)
        assert "APPLICATION" in test_config
        assert "TEST_NAME" in test_config
        if "HELP" in test_config.get("CONFIGURATION", {}):
            list_test_with_help.append(test_config)
        if "VERSION" in test_config.get("CONFIGURATION", {}):
            list_test_with_version.append(test_config)
        if "no_config" in test_config["TEST_NAME"]:
            list_test_without_config.append(test_config)

    assert len(test_names) == len(test_configs)
    assert len(list_test_with_help) == len(list_test_with_version)
    assert len(list_test_without_config) == len(list_test_with_help)


def test_read_configs_from_files(integration_test_config_files):
    config_files = integration_test_config_files

    configs = configuration._read_configs_from_files(config_files)
    assert len(configs) == len(config_files)


def test_create_tmp_output_path(tmp_test_directory):
    config = {"APPLICATION": "test_app", "TEST_NAME": "test_name"}
    tmp_output_path = configuration.create_tmp_output_path(tmp_test_directory, config)
    expected_path = tmp_test_directory / "test_app-test_name"

    assert tmp_output_path == expected_path
    assert tmp_output_path.exists()
    assert tmp_output_path.is_dir()

    with pytest.raises(
        KeyError, match="No application defined in configuration {'TEST_NAME': 'test_name'}."
    ):
        configuration.create_tmp_output_path(tmp_test_directory, {"TEST_NAME": "test_name"})


def test_get_application_command_with_config_file():
    app = "test_app"
    config_file = "test_config.yml"
    expected_command = "python simtools/applications/test_app.py --config test_config.yml"

    command = configuration.get_application_command(app, config_file=config_file)

    assert command == expected_command


def test_get_application_command_with_config_string():
    app = "test_app"
    config_string = "--version"
    expected_command = "python simtools/applications/test_app.py --version"

    command = configuration.get_application_command(app, config_string=config_string)

    assert command == expected_command


def test_get_application_command_with_simtools_app():
    app = "simtools-test_app"
    config_file = "test_config.yml"
    expected_command = "simtools-test_app --config test_config.yml"

    command = configuration.get_application_command(app, config_file=config_file)

    assert command == expected_command


def test_get_application_command_with_no_config():
    app = "test_app"
    expected_command = "python simtools/applications/test_app.py"

    command = configuration.get_application_command(app)

    assert command == expected_command


def test_prepare_test_options_with_single_boolean_option(tmp_test_directory):
    config = {"VERSION": True}
    model_version = None

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file is None
    assert config_string == "--version"
    assert config_file_model_version is None


def test_prepare_test_options_with_model_version(tmp_test_directory, tmp_config_string):
    config = {"MODEL_VERSION": "v1.0"}
    model_version = "v2.0"

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file == tmp_test_directory / tmp_config_string
    assert config_string is None
    assert config_file_model_version == "v1.0"

    with open(config_file, encoding="utf-8") as file:
        written_config = yaml.safe_load(file)
    assert written_config["MODEL_VERSION"] == "v2.0"


def test_prepare_test_options_with_output_path(tmp_test_directory, tmp_config_string):
    config = {"OUTPUT_PATH": "results"}
    model_version = None

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file == tmp_test_directory / tmp_config_string
    assert config_string is None
    assert config_file_model_version is None

    with open(config_file, encoding="utf-8") as file:
        written_config = yaml.safe_load(file)
    assert written_config["OUTPUT_PATH"] == str(tmp_test_directory / "results")
    assert written_config["USE_PLAIN_OUTPUT_PATH"] is True


def test_prepare_test_options_with_data_directory(tmp_test_directory, tmp_config_string):
    config = {"DATA_DIRECTORY": "data"}
    model_version = None

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file == tmp_test_directory / tmp_config_string
    assert config_string is None
    assert config_file_model_version is None

    with open(config_file, encoding="utf-8") as file:
        written_config = yaml.safe_load(file)
    assert written_config["DATA_DIRECTORY"] == str(tmp_test_directory / "data")


def test_prepare_test_options_with_full_config(tmp_test_directory, tmp_config_string):
    config = {"MODEL_VERSION": "v1.0", "OUTPUT_PATH": "results", "DATA_DIRECTORY": "data"}
    model_version = "v2.0"

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file == tmp_test_directory / tmp_config_string
    assert config_string is None
    assert config_file_model_version == "v1.0"

    with open(config_file, encoding="utf-8") as file:
        written_config = yaml.safe_load(file)
    assert written_config["MODEL_VERSION"] == "v2.0"
    assert written_config["OUTPUT_PATH"] == str(tmp_test_directory / "results")
    assert written_config["USE_PLAIN_OUTPUT_PATH"] is True
    assert written_config["DATA_DIRECTORY"] == str(tmp_test_directory / "data")


def test_configure_with_model_version_use_current(tmp_test_directory, mocker, tmp_config_string):
    config = {
        "APPLICATION": "test_app",
        "TEST_NAME": "test_name",
        "CONFIGURATION": {"MODEL_VERSION": "v1.0", "MODEL_VERSION_USE_CURRENT": True},
    }
    request = mocker.Mock()
    request.config.getoption.return_value = "v1.0"

    cmd, config_file_model_version = configuration.configure(config, tmp_test_directory, request)

    expected_cmd = "python simtools/applications/test_app.py --config " + str(
        tmp_test_directory / "test_app-test_name" / tmp_config_string
    )
    assert cmd == expected_cmd
    assert config_file_model_version == "v1.0"


def test_configure_without_configuration(tmp_test_directory, mocker):
    config = {"APPLICATION": "test_app", "TEST_NAME": "test_name"}
    request = mocker.Mock()
    request.config.getoption.return_value = None

    cmd, config_file_model_version = configuration.configure(config, tmp_test_directory, request)

    expected_cmd = "python simtools/applications/test_app.py"
    assert cmd == expected_cmd
    assert config_file_model_version is None


def test_configure_with_configuration(tmp_test_directory, mocker, tmp_config_string):
    config = {
        "APPLICATION": "test_app",
        "TEST_NAME": "test_name",
        "CONFIGURATION": {"OUTPUT_PATH": "results", "DATA_DIRECTORY": "data"},
    }
    request = mocker.Mock()
    request.config.getoption.return_value = None

    cmd, config_file_model_version = configuration.configure(config, tmp_test_directory, request)

    expected_cmd = "python simtools/applications/test_app.py --config " + str(
        tmp_test_directory / "test_app-test_name" / tmp_config_string
    )
    assert cmd == expected_cmd
    assert config_file_model_version is None

    with open(
        tmp_test_directory / "test_app-test_name" / tmp_config_string, encoding="utf-8"
    ) as file:
        written_config = yaml.safe_load(file)
    assert written_config["OUTPUT_PATH"] == str(
        tmp_test_directory / "test_app-test_name" / "results"
    )
    assert written_config["DATA_DIRECTORY"] == str(
        tmp_test_directory / "test_app-test_name" / "data"
    )


def test_skip_test_for_model_version_no_model_version_use_current(mocker_pytest_skip):
    config = {"CONFIGURATION": {"MODEL_VERSION": "v1.0"}}
    model_version_requested = "v1.0"
    configuration._skip_test_for_model_version(config, model_version_requested)
    pytest.skip.assert_not_called()


def test_skip_test_for_model_version_no_model_version_requested(mocker_pytest_skip):
    config = {"CONFIGURATION": {"MODEL_VERSION": "v1.0"}, "MODEL_VERSION_USE_CURRENT": True}
    model_version_requested = None
    configuration._skip_test_for_model_version(config, model_version_requested)
    pytest.skip.assert_not_called()


def test_skip_test_for_model_version_skip():
    config = {"CONFIGURATION": {"MODEL_VERSION": "v1.0"}, "MODEL_VERSION_USE_CURRENT": True}
    model_version_requested = "v2.0"
    with pytest.raises(
        configuration.VersionError, match="Model version requested v2.0 not supported for this test"
    ):
        configuration._skip_test_for_model_version(config, model_version_requested)

    config = {"CONFIGURATION": {"MODEL_VERSION": "v1.0"}, "MODEL_VERSION_USE_CURRENT": True}
    model_version_requested = "v1.0"
    configuration._skip_test_for_model_version(config, model_version_requested)
