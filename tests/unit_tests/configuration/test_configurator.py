#!/usr/bin/python3

import logging
import sys
from copy import copy
from unittest.mock import MagicMock

import astropy.units as u
import pytest
import yaml

from simtools import constants
from simtools.configuration.arguments import (
    ARRAY_LAYOUT_NAME,
    DB_SIMULATION_MODEL_VERSION,
    OUTPUT_ARGUMENTS,
    OUTPUT_PATH_ARGUMENTS,
    STANDARD_ARGUMENTS,
)
from simtools.configuration.configurator import Configurator

logger = logging.getLogger()


@pytest.fixture
def configurator(tmp_test_directory, _mock_settings_env_vars):
    config = Configurator()
    config.parser.add_argument_definitions((*OUTPUT_PATH_ARGUMENTS, *STANDARD_ARGUMENTS))
    config.config = vars(config.parser.parse_args(("--output_path", str(tmp_test_directory))))
    return config


def test_command_line_precedence_over_config_file(tmp_test_directory, monkeypatch):
    # Create a config file with label='config_label' and log_level='debug'
    _config_dict = {
        "label": "config_label",
        "log_level": "debug",
    }
    _config_file = tmp_test_directory / "configuration-precedence-test.yml"
    with open(_config_file, "w") as output:
        yaml.safe_dump(_config_dict, output, sort_keys=False)

    # Initialize configurator with command-line args that differ from config file
    configurator = Configurator()
    configurator.parser.add_argument_definitions(
        (*OUTPUT_PATH_ARGUMENTS, *OUTPUT_ARGUMENTS, *STANDARD_ARGUMENTS)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_configurator.py",
            "--config",
            str(_config_file),
            "--label",
            "cli_label",
            "--log_level",
            "info",
        ],
    )
    config, _ = configurator.configure(initialize_output=True)

    # Command-line values should take precedence
    assert config["label"] == "cli_label"
    assert config["log_level"] == "info"


def test_config_from_file_preserves_selected_by_version_keys(tmp_test_directory):
    config_dict = {
        "applications": [
            {
                "application": "simtools-simulate-prod",
                "configuration": {
                    "model_version": ["6.3.0", "7.0.0"],
                    "array_layout_name": {
                        "by_version": {
                            "<7.0.0": "alpha",
                            ">=7.0.0": "CTAO-North-Alpha",
                        }
                    },
                },
            }
        ]
    }
    config_file = tmp_test_directory / "configuration-preserve-by-version.yml"
    with open(config_file, "w", encoding="utf-8") as output:
        yaml.safe_dump(config_dict, output, sort_keys=False)

    config_builder = Configurator()
    config_builder.parser.add_argument_definitions((ARRAY_LAYOUT_NAME,))
    loaded_config = config_builder._config_from_file(config_file)

    assert loaded_config["model_version"] == ["6.3.0", "7.0.0"]
    assert loaded_config["array_layout_name"] == {
        "by_version": {
            "<7.0.0": "alpha",
            ">=7.0.0": "CTAO-North-Alpha",
        }
    }


def test_config_from_file_rejects_inconsistent_unpreserved_by_version_key(
    tmp_test_directory,
):
    config_file = tmp_test_directory / "configuration-reject-by-version.yml"
    with open(config_file, "w", encoding="utf-8") as output:
        yaml.safe_dump(
            {
                "model_version": ["6.3.0", "7.0.0"],
                "site": {
                    "by_version": {
                        "<7.0.0": "North",
                        ">=7.0.0": "South",
                    }
                },
            },
            output,
            sort_keys=False,
        )

    config_builder = Configurator()

    with pytest.raises(ValueError, match="Inconsistent by_version resolution for key 'site'"):
        config_builder._config_from_file(config_file)


def test_config_from_file_resolves_test_resource_paths(tmp_test_directory):
    config_dict = {
        "applications": [
            {
                "application": "simtools-production-derive-monte-carlo-statistics",
                "configuration": {
                    "model_version": "7.0.0",
                    "trigger_histogram_file": (
                        "${generated:gamma_diffuse_run000010.trigger_histograms.hdf5}"
                    ),
                    "plot_config": "${static:plot_config.yml}",
                    "table_data_path": "${downloaded:table_data}",
                },
            }
        ]
    }
    config_file = tmp_test_directory / "configuration-resource-macros.yml"
    with open(config_file, "w", encoding="utf-8") as output:
        yaml.safe_dump(config_dict, output, sort_keys=False)

    config_builder = Configurator()
    loaded_config = config_builder._config_from_file(config_file)

    resources_path = constants.TEST_RESOURCES_ROOT.resolve()
    assert loaded_config["trigger_histogram_file"] == str(
        resources_path / "generated/gamma_diffuse_run000010.trigger_histograms.hdf5"
    )
    assert loaded_config["plot_config"] == str(resources_path / "static/plot_config.yml")
    assert loaded_config["table_data_path"] == str(resources_path / "downloaded/table_data")


def test_arglist_from_config():
    _tmp_dict = {"a": 1.0, "b": None, "c": True, "d": ["d1", "d2", "d3"], "e": 5.0 * u.m}

    assert [
        "--a",
        "1.0",
        "--c",
        "--d",
        "d1",
        "d2",
        "d3",
        "--e",
        "5.0 m",
    ] == Configurator._arglist_from_config(_tmp_dict)

    assert [] == Configurator._arglist_from_config({})

    assert [] == Configurator._arglist_from_config(None)
    assert [] == Configurator._arglist_from_config(5.0)

    assert ["--a", "1.0", "--b", "None", "--c"] == Configurator._arglist_from_config(
        ["--a", "1.0", "--b", None, "--c"]
    )


def test_arglist_from_config_splits_scalar_for_fixed_nargs():
    configurator = Configurator()
    configurator.parser.add_argument("--showers_per_run_power_law", nargs=3, type=str)

    assert [
        "--showers_per_run_power_law",
        "0.0",
        "1",
        "TeV",
    ] == Configurator._arglist_from_config(
        {"showers_per_run_power_law": "0.0 1 TeV"},
        parser=configurator.parser,
    )


def test_convert_string_none_to_none():
    assert {} == Configurator._convert_string_none_to_none({})

    _tmp_dict = {
        "a": 1.0,
        "b": None,
        "c": True,
        "d": "None",
    }
    _tmp_none = copy(_tmp_dict)
    _tmp_none["d"] = None

    assert _tmp_none == Configurator._convert_string_none_to_none(_tmp_dict)


def test_initialize_output(configurator):
    configurator.config.update(output_file=None, output_file_format="ecsv")

    # output file for testing
    configurator.config["test"] = True
    configurator._initialize_output()
    assert configurator.config["output_file"] == "TEST.ecsv"

    # output is not configured (and not activity_id)
    configurator.config["test"] = False
    configurator.config["output_file"] = None
    configurator.config.pop("activity_id", None)
    with pytest.raises(KeyError):
        configurator._initialize_output()

    # output is not configured (but activity_id)
    configurator.config["activity_id"] = "A-ID"
    configurator.config["label"] = "test_label"
    configurator._initialize_output()
    assert configurator.config["output_file"] == "A-ID-test_label.ecsv"

    # output file is configured
    configurator.config["test"] = False
    configurator.config["output_file"] = "unit_test.txt"
    configurator._initialize_output()
    assert configurator.config["output_file"] == "unit_test.txt"


def test_required_argument_can_be_supplied_by_constructor_configuration(monkeypatch):
    configurator = Configurator(config={"arg": "configured"})
    configurator.parser.add_argument("--arg", required=True)
    configurator.parser.add_argument("--label")
    monkeypatch.setattr("sys.argv", ["application", "--label", "test"])

    config, _ = configurator.configure()

    assert config["arg"] == "configured"
    assert next(action for action in configurator.parser._actions if action.dest == "arg").required


def test_set_model_versions(configurator):
    assert "model_version" not in configurator.config

    model_version_1 = "5.0.0"
    model_version_2 = "6.0.0"
    configurator.config["model_version"] = None
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] is None

    configurator.config["model_version"] = [model_version_1]
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] == model_version_1

    configurator.config["model_version"] = [model_version_1, model_version_2]
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] == [model_version_1, model_version_2]

    configurator.config["model_version"] = model_version_1
    configurator._initialize_model_versions()
    assert configurator.config["model_version"] == model_version_1


def test_dependency_database_defaults_are_read_from_catalog():
    """Test database name and version defaults come from the dependency catalog."""
    defaults = Configurator._dependency_defaults(  # pylint: disable=protected-access
        {"db_simulation_model": None, "db_simulation_model_version": None}
    )

    assert defaults == {
        "db_simulation_model": "CTAO-Simulation-Model",
        "db_simulation_model_version": "v0.17.0",
    }
    assert (
        Configurator._dependency_defaults(  # pylint: disable=protected-access
            {"label": None}
        )
        == {}
    )
    assert Configurator._dependency_defaults(  # pylint: disable=protected-access
        {"db_simulation_model": None}
    ) == {"db_simulation_model": "CTAO-Simulation-Model"}
    assert Configurator._dependency_defaults(  # pylint: disable=protected-access
        {"db_simulation_model_version": None}
    ) == {"db_simulation_model_version": "v0.17.0"}


def test_environment_database_version_overrides_catalog(tmp_test_directory, monkeypatch):
    """Test an explicit .env database version overrides the catalog default."""
    env_file = tmp_test_directory / ".env"
    env_file.write_text("SIMTOOLS_DB_SIMULATION_MODEL_VERSION=v0.17.0\n", encoding="utf-8")
    monkeypatch.delenv("SIMTOOLS_DB_SIMULATION_MODEL_VERSION", raising=False)

    configurator = Configurator()
    configurator.parser.add_argument_definitions((DB_SIMULATION_MODEL_VERSION,))

    assert configurator._config_from_env(env_file)["db_simulation_model_version"] == "v0.17.0"


def test_initialize(configurator):
    configurator._get_cli_arglist = MagicMock(return_value=[])
    configurator._config_from_env = MagicMock(return_value={})
    configurator._config_from_file = MagicMock(return_value={})
    configurator._initialize_model_versions = MagicMock()
    configurator._initialize_io_handler = MagicMock()
    configurator._initialize_output = MagicMock()
    configurator._get_db_parameters = MagicMock(return_value={"db_param": "test"})

    config, db_dict = configurator.configure()

    # Assert that the methods were called
    configurator._get_cli_arglist.assert_called_once_with()
    configurator._config_from_env.assert_called_once_with(".env")
    configurator._config_from_file.assert_called_once_with(None)
    configurator._initialize_model_versions.assert_called_once()
    configurator._initialize_io_handler.assert_called_once()
    configurator._initialize_output.assert_not_called()
    configurator._get_db_parameters.assert_called_once()
    configurator._get_db_parameters.reset_mock()

    # Assert that activity_id and label are set
    assert "activity_id" in config
    assert config["label"] == configurator.label
    assert db_dict == {"db_param": "test"}

    configurator.configure(initialize_output=True)

    # Assert that the methods were called with the correct parameters
    configurator._get_cli_arglist.assert_called()
    configurator._config_from_env.assert_called()
    configurator._config_from_file.assert_called()
    configurator._initialize_model_versions.assert_called()
    configurator._initialize_io_handler.assert_called()
    configurator._initialize_output.assert_called_once()
    configurator._get_db_parameters.assert_called_once()

    # test activity_id and label
    configurator.config_class_init = {"activity_id": "test_activity_id", "label": "test_label"}
    configurator.configure()
    assert configurator.config["activity_id"] == "test_activity_id"
    assert configurator.config["label"] == "test_label"
