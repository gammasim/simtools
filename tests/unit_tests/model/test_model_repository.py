"""Unit tests for model_repository module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simtools.model import model_repository

TEST_PRODUCTION_FILE = "test_production.json"
TEST_MODIFICATIONS_FILE = "modifications.json"
PATH_PATCH = "simtools.model.model_repository.get_model_parameter_file_path"


def test_verify_simulation_model_production_tables_success(tmp_test_directory):
    productions_path = tmp_test_directory / "simulation-models" / "productions"
    productions_path.ensure(dir=True)

    production_data = {
        "parameters": {"telescope": {"camera_config": "1.0.0", "optics_config": "2.1.0"}}
    }
    production_file = productions_path / TEST_PRODUCTION_FILE

    production_file.write_text(json.dumps(production_data), encoding="utf-8")

    with patch(
        "simtools.model.model_repository._verify_model_parameters_for_production"
    ) as mock_verify:
        mock_verify.return_value = ([], 2)

        result = model_repository.verify_simulation_model_production_tables(str(tmp_test_directory))

        assert result is True
        mock_verify.assert_called_once()


def test_verify_simulation_model_production_tables_missing_files(tmp_test_directory):
    productions_path = tmp_test_directory / "simulation-models" / "productions"
    productions_path.ensure(dir=True)

    production_file = productions_path / TEST_PRODUCTION_FILE
    production_file.write_text('{"parameters": {}}', encoding="utf-8")

    with patch(
        "simtools.model.model_repository._verify_model_parameters_for_production"
    ) as mock_verify:
        mock_verify.return_value = (["/missing/file.json"], 1)

        result = model_repository.verify_simulation_model_production_tables(str(tmp_test_directory))

        assert result is False


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production_with_missing_files(
    mock_collect_data, tmp_test_directory
):

    production_data = {
        "parameters": {"telescope": {"camera_config": "1.0.0", "mirror_config": "2.0.0"}}
    }
    mock_collect_data.return_value = production_data

    production_file = Path(TEST_PRODUCTION_FILE)

    with patch(PATH_PATCH) as mock_get_path:
        mock_file = Mock()
        mock_file.exists.return_value = False
        mock_get_path.return_value = mock_file

        missing_files, total_checked = model_repository._verify_model_parameters_for_production(
            str(tmp_test_directory), production_file
        )

        assert total_checked == 2
        assert len(missing_files) == 2


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production_all_files_exist(
    mock_collect_data, tmp_test_directory
):
    production_data = {"parameters": {"telescope": {"camera_config": "1.0.0"}}}
    mock_collect_data.return_value = production_data

    production_file = Path(TEST_PRODUCTION_FILE)

    with patch(PATH_PATCH) as mock_get_path:
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_get_path.return_value = mock_file

        missing_files, total_checked = model_repository._verify_model_parameters_for_production(
            str(tmp_test_directory), production_file
        )

        assert total_checked == 1
        assert len(missing_files) == 0


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production_non_dict_parameters(
    mock_collect_data, tmp_test_directory
):
    production_data = {"parameters": {"telescope": "not_a_dict", "array": {"valid_param": "1.0.0"}}}
    mock_collect_data.return_value = production_data

    production_file = Path(TEST_PRODUCTION_FILE)

    with patch(PATH_PATCH) as mock_get_path:
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_get_path.return_value = mock_file

        _, total_checked = model_repository._verify_model_parameters_for_production(
            str(tmp_test_directory), production_file
        )

        assert total_checked == 1


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_global_corsika_parameters_uses_global_scope(mock_collect_data, tmp_test_directory):
    mock_collect_data.return_value = {
        "production_table_name": "configuration_corsika",
        "parameters": {"global": {"corsika_iact_io_buffer": "1.0.0"}},
    }
    production_file = Path(TEST_PRODUCTION_FILE)

    with patch(PATH_PATCH) as mock_get_path:
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_get_path.return_value = mock_file

        _, total_checked = model_repository._verify_model_parameters_for_production(
            str(tmp_test_directory), production_file
        )

    assert total_checked == 1
    mock_get_path.assert_called_once_with(
        str(tmp_test_directory), "global", "corsika_iact_io_buffer", "1.0.0"
    )


def test_get_model_parameter_file_path_regular_collection(tmp_test_directory):
    result = model_repository.get_model_parameter_file_path(
        str(tmp_test_directory), "telescope", "camera_config", "1.0.0"
    )

    expected = (
        tmp_test_directory
        / "simulation-models"
        / "model_parameters"
        / "telescope"
        / "camera_config"
        / "camera_config-1.0.0.json"
    )
    assert result == expected


def test_get_model_parameter_file_path_configuration_sim_telarray(tmp_test_directory):
    result = model_repository.get_model_parameter_file_path(
        str(tmp_test_directory), "telescope", "sim_telarray_config", "1.0.0"
    )

    expected = (
        tmp_test_directory
        / "simulation-models"
        / "model_parameters"
        / "telescope"
        / "sim_telarray_config"
        / "sim_telarray_config-1.0.0.json"
    )
    assert result == expected


def test_get_model_parameter_file_path_configuration_corsika(tmp_test_directory):
    result = model_repository.get_model_parameter_file_path(
        str(tmp_test_directory), "telescope", "corsika_config", "1.0.0"
    )

    expected = (
        tmp_test_directory
        / "simulation-models"
        / "model_parameters"
        / "telescope"
        / "corsika_config"
        / "corsika_config-1.0.0.json"
    )
    assert result == expected


def test_get_model_parameter_file_path_global_scope(tmp_test_directory):
    result = model_repository.get_model_parameter_file_path(
        str(tmp_test_directory), None, "corsika_config", "1.0.0"
    )

    expected = (
        tmp_test_directory
        / "simulation-models"
        / "model_parameters"
        / "global"
        / "corsika_config"
        / "corsika_config-1.0.0.json"
    )
    assert result == expected


@pytest.mark.parametrize(
    ("telescope", "expected_scope"),
    [("configuration_corsika", "global"), ("LSTN-design", "LSTN-design")],
)
def test_get_model_parameter_scope(telescope, expected_scope):
    assert model_repository._get_model_parameter_scope(telescope) == expected_scope


def test_check_for_major_version_jump_no_major_jump():
    json_data = {
        "schema_version": "0.3.0",
        "parameter_version": "4.0.1",
    }
    param_data = {"version": "4.1.0", "value": 62.5}
    param = "dsum_threshold"
    telescope = "MSTx-FlashCam"

    result = model_repository._check_for_major_version_jump(json_data, param_data, param, telescope)

    assert result == "4.1.0"


def test_check_for_major_version_jump_major_jump():
    json_data = {
        "schema_version": "0.3.0",
        "parameter_version": "4.0.1",
    }
    param_data = {"version": "6.0.0", "value": 62.5}
    param = "dsum_threshold"
    telescope = "MSTx-FlashCam"

    result = model_repository._check_for_major_version_jump(json_data, param_data, param, telescope)

    assert result == "6.0.0"


@patch("simtools.model.model_repository.Path")
def test_get_latest_model_parameter_file_no_files(mock_path):
    mock_directory = Mock()
    mock_path.return_value = mock_directory

    mock_directory.glob.return_value = []

    with pytest.raises(FileNotFoundError, match="No JSON files found for parameter 'parameter'"):
        model_repository._get_latest_model_parameter_file("mock_directory", "parameter", "1.0.0")


@patch("simtools.model.model_repository.Path")
def test_get_latest_model_parameter_file_unsorted_versions(mock_path):
    mock_directory = Mock()
    mock_path.return_value = mock_directory

    mock_file_1 = Mock()
    mock_file_1.stem = "parameter-1.0.0"
    mock_file_2 = Mock()
    mock_file_2.stem = "parameter-3.0.0"
    mock_file_3 = Mock()
    mock_file_3.stem = "parameter-2.0.0"
    mock_directory.glob.return_value = [mock_file_1, mock_file_3, mock_file_2]

    result = model_repository._get_latest_model_parameter_file(
        "mock_directory", "parameter", "10.0.0"
    )

    assert result == str(mock_file_2)


@patch("simtools.model.model_repository.Path")
def test_get_latest_model_parameter_file_no_files_within_max_version(mock_path):
    mock_directory = Mock()
    mock_path.return_value = mock_directory

    mock_file = Mock()
    mock_file.stem = "parameter-2.0.0"
    mock_directory.glob.return_value = [mock_file]

    with pytest.raises(FileNotFoundError, match=r"with version <= 1\.0\.0"):
        model_repository._get_latest_model_parameter_file("mock_directory", "parameter", "1.0.0")


def test_update_parameters_dict_new_function():
    existing_params = {"dsum_threshold": "3.0.0"}
    changes = {
        "MSTx-FlashCam": {
            "dsum_threshold": {"version": "4.0.0", "value": 62.5},
            "dsum_clipping": {"version": "1.0.0", "deprecated": True},
        },
        "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31.9}},
    }
    table_name = "MSTx-FlashCam"

    parameters, deprecated = model_repository._update_parameters_dict(
        existing_params, changes, table_name
    )

    assert "MSTx-FlashCam" in parameters
    assert parameters["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"
    assert "dsum_clipping" not in parameters["MSTx-FlashCam"]  # Should be removed
    assert "dsum_clipping" in deprecated  # Should be in deprecated list


def test_get_production_table_key_configuration_corsika():
    assert model_repository._get_production_table_key("configuration_corsika") == "global"


def test_apply_changes_to_production_table_update_model_version():
    data = {
        "model_version": "6.0.0",
        "production_table_name": "SSTS-design",
        "parameters": {
            "SSTS-39": {
                "array_element_position_ground": "2.0.0",
                "array_element_position_utm": "2.0.0",
            }
        },
        "design_model": {"SSTS-39": "SSTS-design"},
    }
    changes = {
        "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}},
        "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31.9}},
        "SSTS-design": {"discriminator_threshold": {"version": "4.0.0", "value": 8.92}},
    }
    model_version = "6.5.0"

    model_repository._apply_changes_to_production_table(
        data["production_table_name"], data, changes, model_version, False
    )

    assert data["model_version"] == "6.5.0"


def test_apply_changes_to_production_table_configuration_corsika_full_update():
    data = {
        "production_table_name": "configuration_corsika",
        "parameters": {
            "global": {
                "corsika_starting_grammage": "1.0.0",
                "corsika_first_interaction_height": "1.0.0",
            },
        },
    }
    changes = {
        "configuration_corsika": {
            "corsika_starting_grammage": {"version": "2.0.0", "value": 12.0},
            "corsika_first_interaction_height": {"version": "1.0.0", "deprecated": True},
        }
    }
    model_version = "6.5.0"

    model_repository._apply_changes_to_production_table(
        data["production_table_name"], data, changes, model_version, False
    )

    assert data["model_version"] == "6.5.0"
    assert "configuration_corsika" not in data["parameters"]
    assert data["parameters"]["global"]["corsika_starting_grammage"] == "2.0.0"
    assert "corsika_first_interaction_height" not in data["parameters"]["global"]


def test_apply_changes_to_production_table_configuration_corsika_patch_update():
    data = {
        "production_table_name": "configuration_corsika",
        "parameters": {
            "global": {
                "corsika_starting_grammage": "1.0.0",
                "corsika_first_interaction_height": "1.2.0",
            },
        },
    }
    changes = {
        "configuration_corsika": {
            "corsika_starting_grammage": {"version": "2.0.0", "value": 12.0},
            "corsika_first_interaction_height": {"version": "1.2.0", "deprecated": True},
        }
    }
    model_version = "6.5.0"

    model_repository._apply_changes_to_production_table(
        data["production_table_name"], data, changes, model_version, True
    )

    assert data["model_version"] == "6.5.0"
    assert "configuration_corsika" not in data["parameters"]
    assert data["parameters"]["global"]["corsika_starting_grammage"] == "2.0.0"
    assert "corsika_first_interaction_height" not in data["parameters"]["global"]
    assert data["deprecated_parameters"] == ["corsika_first_interaction_height"]


def test_apply_changes_to_production_tables(tmp_test_directory):
    # Create source directory with sample files
    source_prod_table_path = tmp_test_directory / "simulation-models/productions" / "6.0.0"
    source_prod_table_path.ensure(dir=True)
    target_prod_table_path = tmp_test_directory / "simulation-models/productions" / "6.5.0"

    # Create sample production table files in source
    prod_table_data = {
        "production_table_name": "MSTx-FlashCam",
        "model_version": "6.0.0",
        "parameters": {
            "MSTx-FlashCam": {"dsum_threshold": "3.0.0"},
        },
    }
    config_table_data = {
        "production_table_name": "configuration_sim_telarray",
        "model_version": "6.0.0",
        "parameters": {
            "MSTx-FlashCam": {"dsum_threshold": "3.0.0"},
            "MSTx-NectarCam": {"discriminator_threshold": "3.0.0"},
        },
    }

    prod_table_file = source_prod_table_path / "MSTx-FlashCam.json"
    prod_table_file.write_text(json.dumps(prod_table_data), encoding="utf-8")
    config_file = source_prod_table_path / "configuration_sim_telarray.json"
    config_file.write_text(json.dumps(config_table_data), encoding="utf-8")

    # Mock changes to be applied
    changes = {
        "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}},
    }

    # Apply changes from source to target
    model_repository._apply_changes_to_production_tables(
        changes, "6.0.0", "6.5.0", "full_update", tmp_test_directory
    )

    # Verify the production table file is updated with changes
    updated_prod_file = target_prod_table_path / "MSTx-FlashCam.json"
    assert updated_prod_file.exists()
    updated_data = json.loads(updated_prod_file.read_text(encoding="utf-8"))
    assert updated_data["model_version"] == "6.5.0"
    assert updated_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"

    # Verify configuration table file model_version is updated but parameters unchanged
    config_target_file = target_prod_table_path / "configuration_sim_telarray.json"
    assert config_target_file.exists()
    config_data = json.loads(config_target_file.read_text(encoding="utf-8"))
    assert config_data["model_version"] == "6.5.0"
    # Parameters unchanged since production_table_name doesn't match changes
    assert config_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "3.0.0"
    assert config_data["parameters"]["MSTx-NectarCam"]["discriminator_threshold"] == "3.0.0"


def test_apply_changes_to_production_tables_no_parameters(tmp_test_directory):
    # Create source directory with sample files
    source_prod_table_path = tmp_test_directory / "simulation-models/productions" / "6.0.0"
    source_prod_table_path.ensure(dir=True)
    target_prod_table_path = tmp_test_directory / "simulation-models/productions" / "6.5.0"

    # Create a sample production table file in source
    prod_table_data = {
        "model_version": "6.0.0",
        "production_table_name": "MSTx-FlashCam",
        "parameters": {},
    }
    prod_table_file = source_prod_table_path / "MSTx-FlashCam.json"
    prod_table_file.write_text(json.dumps(prod_table_data), encoding="utf-8")

    # Mock changes to be applied
    changes = {
        "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}},
    }

    # Call the function
    model_repository._apply_changes_to_production_tables(
        changes, "6.0.0", "6.5.0", "full_update", tmp_test_directory
    )

    # Verify the production table file is updated in target
    target_file = target_prod_table_path / "MSTx-FlashCam.json"
    assert target_file.exists()
    updated_data = json.loads(target_file.read_text(encoding="utf-8"))
    assert updated_data["model_version"] == "6.5.0"
    # Parameters should be created with the matching telescope parameters
    assert updated_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"


def test_apply_changes_to_production_tables_invalid_data_type(tmp_test_directory):
    # Create source directory with a malformed JSON file
    source_prod_table_path = tmp_test_directory / "simulation-models/productions" / "6.0.0"
    source_prod_table_path.ensure(dir=True)

    # Create a JSON file with list data instead of dict
    malformed_file = source_prod_table_path / "malformed.json"
    malformed_file.write_text('["not", "a", "dict"]', encoding="utf-8")

    changes = {"test_table": {"param1": {"version": "4.0.0", "value": 42}}}

    with pytest.raises(TypeError, match=r"Unsupported data type .* in .*malformed.json"):
        model_repository._apply_changes_to_production_tables(
            changes, "6.0.0", "6.5.0", "full_update", tmp_test_directory
        )


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository._apply_changes_to_production_tables")
@patch("simtools.model.model_repository._apply_changes_to_model_parameters")
def test_generate_new_production_empty_version_history(
    mock_apply_model_changes, mock_apply_table_changes, mock_collect_data, tmp_test_directory
):
    mock_collect_data.return_value = {
        "model_version": "6.5.0",
        "model_version_history": [],
        "setting_workflows_git_tag": "v0.3.0",
        "setting_workflows_git_repository": "https://example.org/workflows.git",
        "changes": {},
    }

    model_repository.generate_new_production("fake_modifications.yml", str(tmp_test_directory))

    mock_apply_table_changes.assert_called_once_with(
        {}, "6.5.0", "6.5.0", "full_update", str(tmp_test_directory)
    )
    mock_apply_model_changes.assert_called_once_with(
        {}, str(tmp_test_directory), "v0.3.0", "https://example.org/workflows.git"
    )


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository._get_changes_to_production")
@patch("simtools.model.model_repository._apply_changes_to_production_tables")
@patch("simtools.model.model_repository._apply_changes_to_model_parameters")
def test_generate_new_production_separates_recursive_table_changes_from_parameters(
    mock_apply_model_changes,
    mock_apply_table_changes,
    mock_get_changes_to_production,
    mock_collect_data,
    tmp_test_directory,
):
    direct_changes = {
        "MSTx-FlashCam": {
            "new_parameter": {"version": "2.0.0", "value": 42.0},
        }
    }
    recursive_changes = {
        **direct_changes,
        "LSTN-design": {
            "inherited_parameter": {"version": "1.0.0", "value": 10.0},
        },
    }
    mock_collect_data.return_value = {
        "model_version": "7.0.0",
        "model_version_history": ["6.3.0"],
        "changes": direct_changes,
    }
    mock_get_changes_to_production.return_value = recursive_changes, "6.0.0"

    model_repository.generate_new_production("7.0.0", str(tmp_test_directory))

    mock_apply_table_changes.assert_called_once_with(
        recursive_changes, "6.0.0", "7.0.0", "full_update", str(tmp_test_directory)
    )
    mock_apply_model_changes.assert_called_once_with(
        direct_changes,
        str(tmp_test_directory),
        "main",
        "https://gitlab.cta-observatory.org/cta-science/simulations/"
        "simulation-model/simulation-model-parameter-setting.git",
    )


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository._apply_changes_to_production_tables")
@patch("simtools.model.model_repository._apply_changes_to_model_parameters")
def test_generate_new_production_setting_workflows_git_tag_override(
    mock_apply_model_changes, mock_apply_table_changes, mock_collect_data, tmp_test_directory
):
    mock_collect_data.return_value = {
        "model_version": "6.5.0",
        "model_version_history": [],
        "setting_workflows_git_tag": "from-info-file",
        "setting_workflows_git_repository": "https://example.org/workflows.git",
        "changes": {},
    }

    model_repository.generate_new_production(
        "fake_modifications.yml",
        str(tmp_test_directory),
        setting_workflows_git_tag="from-cli",
    )

    mock_apply_table_changes.assert_called_once_with(
        {}, "6.5.0", "6.5.0", "full_update", str(tmp_test_directory)
    )
    mock_apply_model_changes.assert_called_once_with(
        {}, str(tmp_test_directory), "from-cli", "https://example.org/workflows.git"
    )


def test_apply_changes_to_production_table_patch_update():
    data = {
        "model_version": "6.0.0",
        "production_table_name": "test_table",
        "parameters": {"test_table": {"dsum_threshold": "1.0.0"}},
    }
    changes = {"test_table": {"dsum_threshold": {"version": "2.0.0", "value": 42}}}
    model_version = "6.5.0"

    result = model_repository._apply_changes_to_production_table(
        data["production_table_name"], data, changes, model_version, True
    )

    assert result is True  # Should return True when changes match
    assert data["model_version"] == "6.5.0"

    # Test case where patch_update is True but no changes apply to this table
    data_no_changes = {
        "model_version": "6.0.0",
        "production_table_name": "other_table",
        "parameters": {"other_table": {"dsum_threshold": "1.0.0"}},
    }

    result_no_changes = model_repository._apply_changes_to_production_table(
        data_no_changes["production_table_name"], data_no_changes, changes, model_version, True
    )

    assert result_no_changes is False  # Should return False when no changes apply


def test_apply_changes_to_production_table_patch_update_only_sim_telarray_changes():
    data = {
        "model_version": "6.0.0",
        "production_table_name": "LSTN-design",
        "parameters": {"LSTN-design": {"transit_time_random": "1.0.0"}},
    }
    changes = {"LSTN-design": {"min_photons": {"version": "2.0.0", "value": 0}}}

    result = model_repository._apply_changes_to_production_table(
        data["production_table_name"], data, changes, "6.5.0", True
    )

    assert result is False
    assert "parameters" in data
    assert data["parameters"]["LSTN-design"]["transit_time_random"] == "1.0.0"


@patch("simtools.model.model_repository._create_new_model_parameter_entry")
def test_apply_changes_to_model_parameters_simple(mock_create_entry, tmp_test_directory):
    model_parameters_dir = tmp_test_directory / "model_parameters"
    changes = {
        "MSTx-FlashCam": {
            "dsum_threshold": {"version": "4.0.0", "value": 62.5},
            "param_without_value": {"version": "1.0.0"},  # Should be skipped
        },
        "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31.9}},
    }

    model_repository._apply_changes_to_model_parameters(changes, model_parameters_dir)

    # Should only call _create_new_model_parameter_entry for parameters with values
    assert mock_create_entry.call_count == 2
    mock_create_entry.assert_any_call(
        "MSTx-FlashCam", "dsum_threshold", {"version": "4.0.0", "value": 62.5}, model_parameters_dir
    )
    mock_create_entry.assert_any_call(
        "MSTx-NectarCam",
        "discriminator_threshold",
        {"version": "4.0.0", "value": 31.9},
        model_parameters_dir,
    )


@patch("simtools.model.model_repository._download_model_parameter_from_workflow")
@patch("simtools.model.model_repository._create_new_model_parameter_entry")
def test_apply_changes_to_model_parameters_with_activity_id(
    mock_create_entry, mock_download_workflow, tmp_test_directory
):
    model_parameters_dir = tmp_test_directory / "model_parameters"
    changes = {
        "LSTN-design": {
            "pm_photoelectron_spectrum": {
                "version": "3.0.0",
                "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8",
            },
            "value_only_param": {"version": "2.0.0", "value": 42},
        }
    }

    model_repository._apply_changes_to_model_parameters(
        changes, model_parameters_dir, setting_workflows_git_tag="release-v1"
    )

    assert mock_download_workflow.call_count == 1
    mock_download_workflow.assert_any_call(
        "LSTN-design",
        "pm_photoelectron_spectrum",
        {
            "version": "3.0.0",
            "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8",
        },
        model_parameters_dir,
        "release-v1",
        "https://gitlab.cta-observatory.org/cta-science/simulations/"
        "simulation-model/simulation-model-parameter-setting.git",
    )
    mock_create_entry.assert_called_once_with(
        "LSTN-design", "value_only_param", {"version": "2.0.0", "value": 42}, model_parameters_dir
    )


@patch("simtools.model.model_repository._download_model_parameter_from_workflow")
@patch("simtools.model.model_repository._create_new_model_parameter_entry")
def test_apply_changes_to_model_parameters_with_both_value_and_activity_id_raises(
    mock_create_entry, mock_download_workflow, tmp_test_directory
):
    changes = {
        "LSTN-design": {
            "param_with_both": {
                "version": "1.0.0",
                "activity_id": "workflow-123",
                "value": 10,
            }
        }
    }

    with pytest.raises(ValueError, match="Both activity_id and value are set"):
        model_repository._apply_changes_to_model_parameters(changes, tmp_test_directory)

    mock_download_workflow.assert_not_called()
    mock_create_entry.assert_not_called()


@patch("simtools.model.model_repository._create_new_model_parameter_entry")
def test_apply_changes_to_model_parameters_adds_info_entry_to_error(
    mock_create_entry, tmp_test_directory
):
    mock_create_entry.side_effect = TypeError("Error validating dictionary")
    changes = {
        "OBS-South": {
            "array_layouts": {
                "version": "4.0.0",
                "value": "019fd316-c3f8-7796-81dc-eab12f221a1c",
            }
        }
    }

    with pytest.raises(TypeError) as exc_info:
        model_repository._apply_changes_to_model_parameters(changes, tmp_test_directory)

    assert str(exc_info.value) == (
        "Failed to process info.yml entry 'OBS-South -> array_layouts' "
        "(version='4.0.0', value='019fd316-c3f8-7796-81dc-eab12f221a1c'): "
        "Error validating dictionary"
    )


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_git")
@patch("simtools.model.model_repository.writer.ModelDataWriter.write_model_parameter_json")
def test_download_model_parameter_from_workflow(
    mock_write_json, mock_collect_data, tmp_test_directory
):
    telescope = "LSTN-design"
    param = "pm_photoelectron_spectrum"
    param_data = {
        "version": "3.0.0",
        "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8",
    }
    mock_collect_data.return_value = {"parameter_version": "3.0.0", "value": [1, 2, 3]}

    model_repository._download_model_parameter_from_workflow(
        telescope=telescope,
        param=param,
        param_data=param_data,
        simulation_models_path=tmp_test_directory,
        setting_workflows_git_tag="v2.1.0",
        setting_workflows_git_repository="https://example.org/workflows.git",
    )

    mock_collect_data.assert_called_once_with(
        file_name=(
            "output/LSTN-design/pm_photoelectron_spectrum/"
            "019d85b6-1f98-715b-b92b-bfbcd06d7cd8/pm_photoelectron_spectrum/"
            "pm_photoelectron_spectrum-3.0.0.json"
        ),
        git_repository="https://example.org/workflows.git",
        git_branch="v2.1.0",
    )
    mock_write_json.assert_called_once_with(
        {"parameter_version": "3.0.0", "value": [1, 2, 3]},
        tmp_test_directory
        / "simulation-models"
        / "model_parameters"
        / "LSTN-design"
        / "pm_photoelectron_spectrum"
        / "pm_photoelectron_spectrum-3.0.0.json",
    )


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_git")
@patch("simtools.model.model_repository.writer.ModelDataWriter.write_model_parameter_json")
def test_download_model_parameter_from_workflow_raises_on_version_mismatch(
    mock_write_json, mock_collect_data, tmp_test_directory
):
    telescope = "LSTN-design"
    param = "pm_photoelectron_spectrum"
    param_data = {
        "version": "3.0.0",
        "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8",
    }
    mock_collect_data.return_value = {"parameter_version": "2.9.9", "value": [1, 2, 3]}

    with pytest.raises(ValueError, match="Version mismatch"):
        model_repository._download_model_parameter_from_workflow(
            telescope=telescope,
            param=param,
            param_data=param_data,
            simulation_models_path=tmp_test_directory,
            setting_workflows_git_tag="v2.1.0",
        )

    mock_write_json.assert_not_called()


@patch("simtools.model.model_repository._get_latest_model_parameter_file")
@patch("simtools.model.model_repository.writer.ModelDataWriter.write_model_parameter")
def test_create_new_model_parameter_entry_simple(mock_dump, mock_get_latest, tmp_test_directory):
    telescope = "MSTx-FlashCam"
    param = "dsum_threshold"
    param_data = {"version": "1.0.0", "value": 42.5}
    model_parameters_dir = Path(tmp_test_directory / "simulation-models/model_parameters")
    telescope_dir = model_parameters_dir / telescope
    telescope_dir.mkdir(parents=True)

    # Mock no existing file
    mock_get_latest.side_effect = FileNotFoundError("No files found")

    model_repository._create_new_model_parameter_entry(
        telescope, param, param_data, Path(tmp_test_directory)
    )

    # Verify write_model_parameter was called with correct arguments
    mock_dump.assert_called_once_with(
        parameter_name=param,
        value=param_data["value"],
        instrument=telescope,
        parameter_version=param_data["version"],
        output_file=f"{param}-{param_data['version']}.json",
        output_path=model_parameters_dir / telescope / param,
        unit=None,
        model_parameter_schema_version=None,
    )


def test_create_new_model_parameter_entry_telescope_dir_not_exists(tmp_test_directory):
    telescope = "NonExistentTelescope"
    param = "some_param"
    param_data = {"version": "1.0.0", "value": 42.5}

    # Don't create the telescope directory - the function will create it but fail on schema
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        model_repository._create_new_model_parameter_entry(
            telescope, param, param_data, Path(tmp_test_directory)
        )


@patch("simtools.model.model_repository.get_model_parameter_file_path")
@patch("simtools.model.model_repository._check_for_major_version_jump")
@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository._get_latest_model_parameter_file")
@patch("simtools.model.model_repository.writer.ModelDataWriter.write_model_parameter")
def test_create_new_model_parameter_entry_with_existing_file(
    mock_dump,
    mock_get_latest,
    mock_collect_data,
    mock_check_version,
    mock_get_parameter_file_path,
    tmp_test_directory,
):
    telescope = "MSTx-FlashCam"
    param = "dsum_threshold"
    param_data = {"version": "1.0.0", "value": 42.5, "unit": "count"}
    simulation_models_path = Path(tmp_test_directory)
    model_parameters_dir = simulation_models_path / "simulation-models/model_parameters"
    telescope_dir = model_parameters_dir / telescope
    telescope_dir.mkdir(parents=True)
    mock_get_parameter_file_path.return_value = (
        model_parameters_dir / telescope / param / f"{param}-{param_data['version']}.json"
    )

    mock_get_latest.return_value = "/path/to/existing/file.json"
    mock_collect_data.return_value = {
        "value": [30.0, 31.0, 32.0],  # List value to trigger the conversion
    }
    mock_check_version.return_value = "2.0.0"

    model_repository._create_new_model_parameter_entry(
        telescope, param, param_data, simulation_models_path
    )

    # Verify that existing file data was processed
    mock_collect_data.assert_called_once_with("/path/to/existing/file.json")
    mock_check_version.assert_called_once()

    # Verify that param_data was updated with existing file info
    assert param_data["version"] == "2.0.0"
    assert param_data["value"] == [42.5, 42.5, 42.5]  # Single value converted to list


@patch("simtools.model.model_repository.get_model_parameter_file_path")
@patch("simtools.model.model_repository._get_latest_model_parameter_file")
@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository.writer.ModelDataWriter.write_model_parameter")
def test_create_new_model_parameter_entry_reuses_matching_existing_file(
    mock_dump,
    mock_collect_data,
    mock_get_latest,
    mock_get_parameter_file_path,
    tmp_test_directory,
):
    telescope = "MSTx-FlashCam"
    param = "dsum_threshold"
    param_data = {"version": "1.0.0", "value": [42.5, 43.5], "unit": ["count", "count"]}
    target_file = (
        Path(tmp_test_directory)
        / "simulation-models/model_parameters"
        / telescope
        / param
        / f"{param}-{param_data['version']}.json"
    )
    target_file.parent.mkdir(parents=True)
    target_file.touch()
    mock_get_parameter_file_path.return_value = target_file
    mock_get_latest.return_value = target_file
    mock_collect_data.return_value = {
        "parameter": param,
        "instrument": telescope,
        "parameter_version": param_data["version"],
        "value": param_data["value"],
        "unit": "count",
    }

    model_repository._create_new_model_parameter_entry(
        telescope, param, param_data, Path(tmp_test_directory)
    )

    mock_dump.assert_not_called()


@patch("simtools.model.model_repository.get_model_parameter_file_path")
@patch("simtools.model.model_repository._get_latest_model_parameter_file")
@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository.writer.ModelDataWriter.write_model_parameter")
def test_create_new_model_parameter_entry_rejects_mismatching_existing_file(
    mock_dump,
    mock_collect_data,
    mock_get_latest,
    mock_get_parameter_file_path,
    tmp_test_directory,
):
    telescope = "MSTx-FlashCam"
    param = "dsum_threshold"
    param_data = {"version": "1.0.0", "value": 42.5, "unit": "count"}
    target_file = (
        Path(tmp_test_directory)
        / "simulation-models/model_parameters"
        / telescope
        / param
        / f"{param}-{param_data['version']}.json"
    )
    target_file.parent.mkdir(parents=True)
    target_file.touch()
    mock_get_parameter_file_path.return_value = target_file
    mock_get_latest.return_value = target_file
    mock_collect_data.return_value = {
        "parameter": param,
        "instrument": telescope,
        "parameter_version": param_data["version"],
        "value": 41.5,
        "unit": param_data["unit"],
    }

    with pytest.raises(ValueError, match="does not match the requested value"):
        model_repository._create_new_model_parameter_entry(
            telescope, param, param_data, Path(tmp_test_directory)
        )

    mock_dump.assert_not_called()


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        (["cm", "cm"], "cm"),
        (["null", "null"], None),
        (["null", "cm"], [None, "cm"]),
        ("count", "ct"),
    ],
)
def test_normalize_units_for_comparison(unit, expected):
    assert model_repository._normalize_units_for_comparison(unit) == expected


def test_get_changes_to_production_path_update(tmp_test_directory):
    modifications_data = {
        "model_version": "6.5.0",
        "model_version_history": ["6.0.0"],
        "changes": {
            "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62}},
            "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31}},
        },
    }

    with patch("simtools.io.ascii_handler.collect_data_from_file") as mock_collect:
        mock_collect.return_value = modifications_data

        changes, base_model_version = model_repository._get_changes_to_production(
            modifications_data, tmp_test_directory, "patch_update"
        )

        assert base_model_version == "6.0.0"
        assert "MSTx-FlashCam" in changes
        assert changes["MSTx-FlashCam"]["dsum_threshold"]["value"] == 62
        assert "MSTx-NectarCam" in changes
        assert changes["MSTx-NectarCam"]["discriminator_threshold"]["value"] == 31


def test_update_two_levels_in_changes_dict():
    # Test basic update with nested dictionaries
    d = {"LSTN-design": {"param1": {"version": "1.0.0"}}}
    u = {"LSTN-design": {"param2": {"version": "2.0.0"}}}

    result = model_repository._update_two_levels_in_changes_dict(d, u)

    assert "LSTN-design" in result
    assert "param1" in result["LSTN-design"]
    assert "param2" in result["LSTN-design"]
    assert result["LSTN-design"]["param1"]["version"] == "1.0.0"
    assert result["LSTN-design"]["param2"]["version"] == "2.0.0"


def test_update_two_levels_in_changes_dict_new_telescope():
    d = {"LSTN-design": {"param1": {"version": "1.0.0"}}}
    u = {"MST-design": {"param2": {"version": "2.0.0"}}}

    result = model_repository._update_two_levels_in_changes_dict(d, u)

    assert "LSTN-design" in result
    assert "MST-design" in result
    assert result["LSTN-design"]["param1"]["version"] == "1.0.0"
    assert result["MST-design"]["param2"]["version"] == "2.0.0"


def test_update_two_levels_in_changes_dict_non_dict_value():
    d = {"LSTN-design": {"param1": {"version": "1.0.0"}}}
    u = {"LSTN-design": "non_dict_value"}

    result = model_repository._update_two_levels_in_changes_dict(d, u)

    assert result["LSTN-design"] == "non_dict_value"


@patch("simtools.model.model_repository._get_changes_dict")
def test_get_changes_to_production_full_update(mock_get_changes_dict, tmp_test_directory):
    # Mock data for version 7.0.0 (full_update with history pointing to 6.0.2)
    modification_dict_7_0_0 = {
        "model_version": "7.0.0",
        "model_update": "full_update",
        "model_version_history": ["6.0.2"],
        "changes": {
            "LSTN-design": {
                "transit_time_random": {"version": "1.0.0", "value": 0.36, "unit": "ns"}
            },
            "LSTS-design": {
                "transit_time_random": {"version": "1.0.0", "value": 0.36, "unit": "ns"}
            },
            "MSTx-FlashCam": {
                "transit_time_random": {"version": "1.0.0", "value": 0.0, "unit": "ns"}
            },
        },
    }

    # Mock data for version 6.0.2 (patch_update with history [6.0.1, 6.0.0])
    info_6_0_2 = {
        "model_version": "6.0.2",
        "model_update": "patch_update",
        "model_version_history": ["6.0.1", "6.0.0"],
        "changes": {
            "LSTN-design": {"pedestal_events": {"deprecated": True}},
            "MSTx-FlashCam": {
                "calibration_devices": {
                    "version": "1.0.0",
                    "value": {"flat_fielding": "MSFx-FlashCam"},
                }
            },
        },
    }

    # Mock data for version 6.0.1 (patch_update with history [6.0.0])
    info_6_0_1 = {
        "model_version": "6.0.1",
        "model_update": "patch_update",
        "model_version_history": ["6.0.0"],
        "changes": {"LSTN-design": {"some_parameter": {"version": "1.0.0", "value": 100}}},
    }

    # Mock data for version 6.0.0 (base version with no history, marked as full_update)
    info_6_0_0 = {
        "model_version": "6.0.0",
        "model_update": "full_update",  # This triggers the break statement
        "model_version_history": [],
        "changes": {},
    }

    # Setup mock to return appropriate info.yml data based on version
    def get_changes_side_effect(version, path):
        version_data = {
            "6.0.2": info_6_0_2,
            "6.0.1": info_6_0_1,
            "6.0.0": info_6_0_0,
        }
        return version_data.get(version, {})

    mock_get_changes_dict.side_effect = get_changes_side_effect

    changes, base_version = model_repository._get_changes_to_production(
        modification_dict_7_0_0, tmp_test_directory, update_type="full_update"
    )

    # Verify the base version is correctly identified as the oldest
    assert base_version == "6.0.0"

    # Verify changes from 7.0.0 are present
    assert "LSTN-design" in changes
    assert "transit_time_random" in changes["LSTN-design"]
    assert changes["LSTN-design"]["transit_time_random"]["value"] == pytest.approx(0.36)

    assert "LSTS-design" in changes
    assert changes["LSTS-design"]["transit_time_random"]["value"] == pytest.approx(0.36)

    assert "MSTx-FlashCam" in changes
    assert changes["MSTx-FlashCam"]["transit_time_random"]["value"] == pytest.approx(0.0)

    # Verify changes from 6.0.2 are merged
    assert "pedestal_events" in changes["LSTN-design"]
    assert changes["LSTN-design"]["pedestal_events"]["deprecated"] is True
    assert "calibration_devices" in changes["MSTx-FlashCam"]

    # Verify that 6.0.1 changes are NOT included because the loop breaks at 6.0.0 (full_update)
    assert "some_parameter" not in changes["LSTN-design"]

    # Verify mock was called for versions up to the full_update
    # Should call 6.0.2 and 6.0.0, but NOT 6.0.1 due to the break
    mock_get_changes_dict.assert_any_call("6.0.2", tmp_test_directory)
    mock_get_changes_dict.assert_any_call("6.0.0", tmp_test_directory)


@patch("simtools.utils.names.get_collection_name_from_parameter_name")
def test_apply_changes_to_sim_telarray_production_table_new_params(mock_get_collection):

    def collection_side_effect(param):
        return "configuration_sim_telarray" if param == "min_photons" else "telescopes"

    mock_get_collection.side_effect = collection_side_effect

    data = {
        "model_version": "6.0.0",
        "parameters": {"MSTN-design": {"other_param": "1.0.0"}},
    }
    changes = {
        "LSTN-design": {
            "min_photons": {"version": "2.0.0", "value": 0},
            "transit_time_random": {"version": "1.0.0", "value": 0.36},
        },
        "configuration_corsika": {"corsika_param": {"version": "1.0.2"}},
    }

    has_cst_changes = model_repository._apply_changes_to_sim_telarray_production_table(
        data, changes, "7.0.0", False
    )

    assert has_cst_changes is True
    assert data["model_version"] == "7.0.0"
    assert data["parameters"]["LSTN-design"]["min_photons"] == "2.0.0"
    assert data["parameters"]["MSTN-design"]["other_param"] == "1.0.0"
    assert "transit_time_random" not in data["parameters"]["LSTN-design"]


@patch("simtools.utils.names.get_collection_name_from_parameter_name")
def test_apply_changes_to_sim_telarray_production_table_existing_telescope(mock_get_collection):
    mock_get_collection.side_effect = lambda param: "configuration_sim_telarray"

    data = {
        "model_version": "6.0.0",
        "parameters": {"LSTN-design": {"min_photons": "1.0.0", "other_cst_param": "1.5.0"}},
    }
    changes = {"LSTN-design": {"min_photons": {"version": "2.0.0", "value": 0}}}

    has_cst_changes = model_repository._apply_changes_to_sim_telarray_production_table(
        data, changes, "7.0.0", False
    )

    assert has_cst_changes is True
    assert data["parameters"]["LSTN-design"]["min_photons"] == "2.0.0"
    assert data["parameters"]["LSTN-design"]["other_cst_param"] == "1.5.0"


@patch("simtools.utils.names.get_collection_name_from_parameter_name")
def test_apply_changes_to_sim_telarray_production_table_deprecated_patch_update(
    mock_get_collection,
):
    mock_get_collection.side_effect = lambda param: "configuration_sim_telarray"

    data = {
        "model_version": "6.0.0",
        "parameters": {"LSTN-design": {"min_photons": "1.0.0"}},
    }
    changes = {"LSTN-design": {"min_photons": {"version": "1.0.0", "deprecated": True}}}

    has_cst_changes = model_repository._apply_changes_to_sim_telarray_production_table(
        data, changes, "7.0.0", True
    )

    assert has_cst_changes is True
    assert "min_photons" not in data["parameters"]["LSTN-design"]
    assert "min_photons" in data["deprecated_parameters"]


@patch("simtools.utils.names.get_collection_name_from_parameter_name")
def test_apply_changes_to_production_tables_routes_cst_to_correct_table(
    mock_get_collection, tmp_test_directory
):

    def collection_side_effect(param):
        return "configuration_sim_telarray" if param == "min_photons" else "telescopes"

    mock_get_collection.side_effect = collection_side_effect

    source_path = tmp_test_directory / "simulation-models/productions" / "6.0.0"
    source_path.ensure(dir=True)

    telescope_table = {
        "production_table_name": "LSTN-design",
        "model_version": "6.0.0",
        "parameters": {"LSTN-design": {"transit_time_random": "0.9.0"}},
    }
    cst_table = {
        "production_table_name": "configuration_sim_telarray",
        "model_version": "6.0.0",
        "parameters": {"LSTN-design": {"min_photons": "1.0.0"}},
    }
    (source_path / "LSTN-design.json").write_text(json.dumps(telescope_table), encoding="utf-8")
    (source_path / "configuration_sim_telarray.json").write_text(
        json.dumps(cst_table), encoding="utf-8"
    )

    changes = {
        "LSTN-design": {
            "transit_time_random": {"version": "1.0.0", "value": 0.36},
            "min_photons": {"version": "2.0.0", "value": 0},
        }
    }

    model_repository._apply_changes_to_production_tables(
        changes, "6.0.0", "7.0.0", "patch_update", tmp_test_directory
    )

    target_path = tmp_test_directory / "simulation-models/productions" / "7.0.0"

    telescope_result = json.loads((target_path / "LSTN-design.json").read_text(encoding="utf-8"))
    assert telescope_result["parameters"]["LSTN-design"]["transit_time_random"] == "1.0.0"
    assert "min_photons" not in telescope_result["parameters"]["LSTN-design"]

    cst_result = json.loads(
        (target_path / "configuration_sim_telarray.json").read_text(encoding="utf-8")
    )
    assert cst_result["parameters"]["LSTN-design"]["min_photons"] == "2.0.0"
