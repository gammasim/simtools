#!/usr/bin/python3

import logging
import sys
from unittest.mock import MagicMock

import pytest
import yaml

from simtools import constants
from simtools.configuration.arguments import (
    ARRAY_LAYOUT_NAME,
    OUTPUT_ARGUMENTS,
    OUTPUT_PATH_ARGUMENTS,
    SITE,
    STANDARD_ARGUMENTS,
    TELESCOPE,
    TELESCOPES,
    ArgumentDefinition,
)
from simtools.configuration.configurator import Configurator

logger = logging.getLogger()
LSTN_DESIGN = "LSTN-design"
LSFS_DESIGN = "LSFS-design"
SHARED_LIGHT_SOURCE = "MSFx-FlashCam"
LIGHT_SOURCE = "light_source"
NORTH = "North"
SOUTH = "South"
SITE_OPTION = "--site"
LIGHT_SOURCE_OPTION = "--light_source"
LAYOUT_KEY = "array_layout_name"
NORTH_ALPHA_LAYOUT = "CTAO-North-Alpha"
ALPHA_LAYOUT = "alpha"


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
    config = configurator.configure(initialize_output=True)

    # Command-line values should take precedence
    assert config["label"] == "cli_label"
    assert config["log_level"] == "info"


@pytest.mark.parametrize(
    ("telescope_argument", "telescope_name", "site"),
    [
        (TELESCOPE, LSTN_DESIGN, SOUTH),
        (TELESCOPE, LSFS_DESIGN, NORTH),
        (TELESCOPES, "SSTS-25", NORTH),
        (ARRAY_LAYOUT_NAME, "LSTS-02", NORTH),
    ],
)
def test_configure_rejects_telescope_from_different_site(
    telescope_argument, telescope_name, site, monkeypatch, capsys
):
    config_builder = Configurator(label="test")
    config_builder.parser.add_argument_definitions((SITE, telescope_argument))
    option = f"--{telescope_argument.name}"
    monkeypatch.setattr(
        sys,
        "argv",
        ["test", SITE_OPTION, site, option, telescope_name],
    )

    with pytest.raises(SystemExit, match="2"):
        config_builder.configure()
    element_site = NORTH if site == SOUTH else SOUTH
    assert (
        f"array element {telescope_name} belongs to site {element_site}, "
        f"but site {site} was selected"
    ) in capsys.readouterr().err


def test_configure_accepts_matching_telescope_and_site(monkeypatch):
    config_builder = Configurator(label="test")
    config_builder.parser.add_argument_definitions((*STANDARD_ARGUMENTS, SITE, TELESCOPE))
    monkeypatch.setattr(
        sys,
        "argv",
        ["test", SITE_OPTION, NORTH, "--telescope", LSTN_DESIGN],
    )

    assert config_builder.configure()["telescope"] == LSTN_DESIGN


def test_configure_rejects_light_source_from_different_site(monkeypatch, capsys):
    light_source = ArgumentDefinition(LIGHT_SOURCE, type=str)
    config_builder = Configurator(label="test")
    config_builder.parser.add_argument_definitions((*STANDARD_ARGUMENTS, SITE, light_source))
    monkeypatch.setattr(
        sys,
        "argv",
        ["test", SITE_OPTION, NORTH, LIGHT_SOURCE_OPTION, LSFS_DESIGN],
    )

    with pytest.raises(SystemExit, match="2"):
        config_builder.configure()
    assert (
        f"array element {LSFS_DESIGN} belongs to site {SOUTH}, but site {NORTH} was selected"
    ) in capsys.readouterr().err


@pytest.mark.parametrize("site", [NORTH, SOUTH])
def test_configure_accepts_shared_light_source_at_either_site(monkeypatch, site):
    light_source = ArgumentDefinition(LIGHT_SOURCE, type=str)
    config_builder = Configurator(label="test")
    config_builder.parser.add_argument_definitions((*STANDARD_ARGUMENTS, SITE, light_source))
    monkeypatch.setattr(
        sys,
        "argv",
        ["test", SITE_OPTION, site, LIGHT_SOURCE_OPTION, SHARED_LIGHT_SOURCE],
    )

    assert config_builder.configure()[LIGHT_SOURCE] == SHARED_LIGHT_SOURCE


@pytest.mark.parametrize("layout", [LSTN_DESIGN, NORTH_ALPHA_LAYOUT, ALPHA_LAYOUT, "LSTN-subarray"])
def test_configure_accepts_matching_element_or_named_layout(monkeypatch, layout):
    config_builder = Configurator(label="test")
    config_builder.parser.add_argument_definitions((*STANDARD_ARGUMENTS, SITE, ARRAY_LAYOUT_NAME))
    monkeypatch.setattr(sys, "argv", ["test", SITE_OPTION, NORTH, f"--{LAYOUT_KEY}", layout])

    assert config_builder.configure()[LAYOUT_KEY] == [layout]


def test_configure_rejects_invalid_light_source(monkeypatch, capsys):
    config_builder = Configurator(label="test")
    config_builder.parser.add_argument_definitions(
        (SITE, ArgumentDefinition(LIGHT_SOURCE, type=str))
    )
    monkeypatch.setattr(sys, "argv", ["test", SITE_OPTION, NORTH, LIGHT_SOURCE_OPTION, "bad"])

    with pytest.raises(SystemExit, match="2"):
        config_builder.configure()

    assert "error: Invalid name bad" in capsys.readouterr().err


def test_config_from_file_preserves_selected_by_version_keys(tmp_test_directory):
    config_dict = {
        "applications": [
            {
                "application": "simtools-simulate-prod",
                "configuration": {
                    "model_version": ["6.3.0", "7.0.0"],
                    LAYOUT_KEY: {
                        "by_version": {
                            "<7.0.0": ALPHA_LAYOUT,
                            ">=7.0.0": NORTH_ALPHA_LAYOUT,
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
    assert loaded_config[LAYOUT_KEY] == {
        "by_version": {
            "<7.0.0": ALPHA_LAYOUT,
            ">=7.0.0": NORTH_ALPHA_LAYOUT,
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
                        "<7.0.0": NORTH,
                        ">=7.0.0": SOUTH,
                    }
                },
            },
            output,
            sort_keys=False,
        )

    config_builder = Configurator()

    with pytest.raises(ValueError, match="Inconsistent by_version resolution for key 'site'"):
        config_builder._config_from_file(config_file)


def test_config_from_file_resolves_test_resource_paths_from_environment(
    tmp_test_directory, monkeypatch
):
    monkeypatch.setattr(constants, "TEST_RESOURCES_ROOT", constants._DEFAULT_TEST_RESOURCES_ROOT)
    monkeypatch.setenv("SIMTOOLS_TESTS_PATH", str(tmp_test_directory / "simtools-tests"))
    monkeypatch.setenv("SIMTOOLS_TESTS_RESOURCE_VERSION", "v0.38.0")
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

    resource_path = tmp_test_directory / "simtools-tests" / "v0.38.0" / "integration_tests"
    assert loaded_config["trigger_histogram_file"] == str(
        resource_path / "generated/gamma_diffuse_run000010.trigger_histograms.hdf5"
    )
    assert loaded_config["plot_config"] == str(resource_path / "static/plot_config.yml")
    assert loaded_config["table_data_path"] == str(resource_path / "downloaded/table_data")


def test_config_from_file_does_not_resolve_test_resource_paths_without_environment(
    tmp_test_directory, monkeypatch
):
    monkeypatch.setattr(constants, "TEST_RESOURCES_ROOT", constants._DEFAULT_TEST_RESOURCES_ROOT)
    monkeypatch.delenv("SIMTOOLS_TESTS_PATH", raising=False)
    monkeypatch.delenv("SIMTOOLS_TESTS_RESOURCE_VERSION", raising=False)
    config_file = tmp_test_directory / "configuration-resource-macros.yml"
    config_file.write_text("input: ${generated:input.ecsv}\n", encoding="utf-8")

    config_builder = Configurator()
    loaded_config = config_builder._config_from_file(config_file)

    assert loaded_config["input"] == "${generated:input.ecsv}"


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


def test_required_argument_must_be_supplied_by_supported_source(monkeypatch):
    configurator = Configurator()
    configurator.parser.add_argument("--arg", required=True)
    monkeypatch.setattr("sys.argv", ["application", "--label", "test"])

    with pytest.raises(SystemExit):
        configurator.configure()


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


def test_configure_can_disable_dependency_defaults(configurator):
    """Test applications can disable catalog defaults."""
    configurator.use_dependency_defaults = False
    configurator._get_cli_arglist = MagicMock(return_value=[])
    configurator._config_from_env = MagicMock(return_value={})
    configurator._config_from_file = MagicMock(return_value={})
    configurator._initialize_model_versions = MagicMock()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ({"default-ref": "v0.17.2"}, "v0.17.2"),
        ({"default-ref": "main", "git-revision": "a" * 40}, "a" * 40),
    ],
)
def test_dependency_defaults_select_model_ref(model, expected, mocker):
    """The parser uses readable model refs and accepts an explicit legacy pin."""
    mocker.patch(
        "simtools.configuration.configurator.dependency_versions.load_dependency_catalog",
        return_value={"model-repository": model},
    )

    assert Configurator._dependency_defaults({"simulation_models_git_revision": None}) == {
        "simulation_models_git_revision": expected
    }


def test_initialize(configurator):
    configurator._get_cli_arglist = MagicMock(return_value=[])
    configurator._config_from_env = MagicMock(return_value={})
    configurator._config_from_file = MagicMock(return_value={})
    configurator._initialize_model_versions = MagicMock()
    configurator._initialize_output = MagicMock()
    config = configurator.configure()

    # Assert that the methods were called
    configurator._get_cli_arglist.assert_called_once_with()
    configurator._config_from_env.assert_called_once_with(".env")
    configurator._config_from_file.assert_called_once_with(None)
    configurator._initialize_model_versions.assert_called_once()
    configurator._initialize_output.assert_not_called()

    # Assert that activity_id and label are set
    assert "activity_id" in config
    assert config["label"] == configurator.label

    configurator.configure(initialize_output=True)

    # Assert that the methods were called with the correct parameters
    configurator._get_cli_arglist.assert_called()
    configurator._config_from_env.assert_called()
    configurator._config_from_file.assert_called()
    configurator._initialize_model_versions.assert_called()
    configurator._initialize_output.assert_called_once()
