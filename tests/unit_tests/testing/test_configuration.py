#!/usr/bin/python3

import sys
from pathlib import Path

import pytest
import yaml

import simtools.testing.configuration as configuration

PYTHON_APP_PREFIX = f"{sys.executable} src/simtools/applications"


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
    assert "simtools-simulate-prod_gamma_20_deg_multiple_model_versions" in test_names


def test_get_list_of_test_configurations(integration_test_config_files):
    test_configs, test_names = configuration.get_list_of_test_configurations(
        integration_test_config_files
    )

    list_test_with_help = []
    list_test_with_version = []
    list_test_without_config = []

    for test_config in test_configs:
        assert isinstance(test_config, dict)
        assert "application" in test_config
        assert "test_name" in test_config
        if "help" in test_config.get("configuration", {}):
            list_test_with_help.append(test_config)
        if "version" in test_config.get("configuration", {}):
            list_test_with_version.append(test_config)
        if "no_config" in test_config["test_name"]:
            list_test_without_config.append(test_config)

    assert len(test_names) == len(test_configs)
    assert len(list_test_with_help) == len(list_test_with_version)
    assert len(list_test_without_config) == len(list_test_with_help)


def test_get_resource_benchmark_configurations():
    configs = [
        {"application": "simtools-run", "test_name": "included"},
        {
            "application": "simtools-run",
            "test_name": "excluded",
            "exclude_from_resource_benchmark": "requires external service",
        },
    ]

    included, excluded = configuration.get_resource_benchmark_configurations(configs)

    assert included == [configs[0]]
    assert excluded == [{"id": "simtools-run_excluded", "reason": "requires external service"}]


def test_get_resource_benchmark_test_ids_preserves_duplicate_suffixes():
    first = {"application": "simtools-example", "test_name": "run"}
    second = {"application": "simtools-example", "test_name": "run"}

    assert configuration.get_resource_benchmark_test_ids([first, second]) == [
        "simtools-example_run0",
        "simtools-example_run1",
    ]


def test_create_tmp_output_path(tmp_test_directory):
    config = {"application": "test_app", "test_name": "test_name"}
    tmp_output_path = configuration.create_tmp_output_path(tmp_test_directory, config)
    expected_path = tmp_test_directory / "test_app-test_name"

    assert tmp_output_path == expected_path
    assert tmp_output_path.exists()
    assert tmp_output_path.is_dir()

    with pytest.raises(
        KeyError, match=r"No application defined in configuration {'test_name': 'test_name'}."
    ):
        configuration.create_tmp_output_path(tmp_test_directory, {"test_name": "test_name"})


def test_get_application_command_with_config_file():
    app = "test_app"
    config_file = "test_config.yml"
    expected_command = f"{PYTHON_APP_PREFIX}/test_app.py --config test_config.yml"

    command = configuration.get_application_command(app, config_file=config_file)

    assert command == expected_command


def test_get_application_command_with_config_string():
    app = "test_app"
    config_string = "--version"
    expected_command = f"{PYTHON_APP_PREFIX}/test_app.py --version"

    command = configuration.get_application_command(app, config_string=config_string)

    assert command == expected_command


def test_get_application_command_with_simtools_app(mocker):
    app = "simtools-test_app"
    config_file = "test_config.yml"
    expected_command = "simtools-test_app --config test_config.yml"
    mocker.patch(
        "simtools.testing.configuration.shutil.which", return_value="/usr/bin/simtools-test_app"
    )

    command = configuration.get_application_command(app, config_file=config_file)

    assert command == expected_command


def test_get_application_command_with_no_config():
    app = "test_app"
    expected_command = f"{PYTHON_APP_PREFIX}/test_app.py"

    command = configuration.get_application_command(app)

    assert command == expected_command


def test_prepare_test_options_with_single_boolean_option(tmp_test_directory):
    config = {"version": True}
    model_version = None

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file is None
    assert config_string == "--version"
    assert config_file_model_version is None


def test_prepare_test_options_copies_resolved_resource_config_files(tmp_test_directory):
    resources_path = Path(tmp_test_directory) / "versioned-resources"
    static_path = resources_path / "static"
    static_path.mkdir(parents=True)
    plot_config = static_path / "plot.yml"
    plot_config.write_text(
        """
plot:
  tables:
  - file_name: tests/resources/generated/table.ecsv
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = {
        "plot_config": str(plot_config),
        "output_file": "plot",
        "output_path": "simtools-output",
    }

    configuration._prepare_test_options(
        config,
        output_path=tmp_test_directory,
        test_resources_path=resources_path,
    )

    resolved_plot_config = Path(config["plot_config"])
    assert resolved_plot_config.parent == tmp_test_directory / "resolved-resource-configs"
    assert yaml.safe_load(resolved_plot_config.read_text(encoding="utf-8")) == {
        "plot": {"tables": [{"file_name": str(resources_path / "generated/table.ecsv")}]}
    }


def test_prepare_test_options_with_model_version(tmp_test_directory, tmp_config_string):
    config = {"model_version": "v1.0"}
    model_version = "v2.0"

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file == tmp_test_directory / tmp_config_string
    assert config_string is None
    assert config_file_model_version == "v1.0"

    with open(config_file, encoding="utf-8") as file:
        written_config = yaml.safe_load(file)
    assert written_config["model_version"] == "v2.0"


def test_prepare_test_options_with_model_version_list(tmp_test_directory, tmp_config_string):
    config = {"model_version": ["v1.0", "v1.1"]}
    model_version = "v2.0"

    config_file, config_string, config_file_model_version = configuration._prepare_test_options(
        config, tmp_test_directory, model_version
    )

    assert config_file == tmp_test_directory / tmp_config_string
    assert config_string is None
    assert config_file_model_version == ["v1.0", "v1.1"]

    with open(config_file, encoding="utf-8") as file:
        written_config = yaml.safe_load(file)
    assert written_config["model_version"] == "v2.0"


def test_configure_with_model_version_use_current(tmp_test_directory, mocker, tmp_config_string):
    config = {
        "application": "test_app",
        "test_name": "test_name",
        "configuration": {"model_version": "v1.0", "model_version_use_current": True},
    }
    request = mocker.Mock()
    request.config.getoption.return_value = "v1.0"

    cmd, config_file_model_version = configuration.configure(config, tmp_test_directory, request)

    expected_cmd = f"{PYTHON_APP_PREFIX}/test_app.py --config " + str(
        tmp_test_directory / "test_app-test_name" / tmp_config_string
    )
    assert cmd == expected_cmd
    assert config_file_model_version == "v1.0"


def test_configure_without_configuration(tmp_test_directory, mocker):
    config = {"application": "test_app", "test_name": "test_name"}
    request = mocker.Mock()
    request.config.getoption.return_value = None

    cmd, config_file_model_version = configuration.configure(config, tmp_test_directory, request)

    expected_cmd = f"{PYTHON_APP_PREFIX}/test_app.py"
    assert cmd == expected_cmd
    assert config_file_model_version is None


def test_skip_test_for_model_version_no_model_version_requested(mocker_pytest_skip):
    config = {"configuration": {"model_version": "v1.0"}, "model_version_use_current": True}
    model_version_requested = None
    configuration._skip_test_for_model_version(config, model_version_requested)
    pytest.skip.assert_not_called()


def test_skip_test_for_model_version_skip():
    config = {"configuration": {"model_version": "v1.0"}, "model_version_use_current": True}
    model_version_requested = "v2.0"
    with pytest.raises(
        configuration.VersionError,
        match=r"Model version requested v2.0 not supported for this test",
    ):
        configuration._skip_test_for_model_version(config, model_version_requested)

    config = {"configuration": {"model_version": "v1.0"}, "model_version_use_current": True}
    model_version_requested = "v1.0"
    configuration._skip_test_for_model_version(config, model_version_requested)


def test_skip_test_for_production_db_no_db_server(monkeypatch):
    config = {"skip_for_production_db": True}
    monkeypatch.delenv("SIMTOOLS_DB_SERVER", raising=False)
    assert configuration._skip_test_for_production_db(config) is None


def test_skip_test_for_production_db_skip(monkeypatch):
    config = {"skip_for_production_db": True}
    monkeypatch.setenv("SIMTOOLS_DB_SERVER", "db.zeuthen.desy.de")
    with pytest.raises(
        configuration.ProductionDBError, match="Production database used for this test"
    ):
        configuration._skip_test_for_production_db(config)


def test_skip_test_for_production_db_skip_for_user(monkeypatch):
    config = {"skip_for_production_db": True}
    monkeypatch.setenv("SIMTOOLS_DB_API_USER", "simpipe")
    with pytest.raises(
        configuration.ProductionDBError, match="Production database used for this test"
    ):
        configuration._skip_test_for_production_db(config)
