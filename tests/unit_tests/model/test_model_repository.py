"""Unit tests for model_repository module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simtools.model import model_repository

TEST_PRODUCTION_FILE = "test_production.json"
TEST_MODIFICATIONS_FILE = "modifications.json"
PATH_PATCH = "simtools.model.model_repository._get_model_parameter_file_path"


def test_verify_simulation_model_production_tables_success(tmp_path):
    """Test successful verification of production tables."""
    productions_path = tmp_path / "simulation-models" / "productions"
    productions_path.mkdir(parents=True)

    production_data = {
        "parameters": {"telescope": {"camera_config": "1.0.0", "optics_config": "2.1.0"}}
    }
    production_file = productions_path / TEST_PRODUCTION_FILE

    production_file.write_text(json.dumps(production_data))

    with patch(
        "simtools.model.model_repository._verify_model_parameters_for_production"
    ) as mock_verify:
        mock_verify.return_value = ([], 2)

        result = model_repository.verify_simulation_model_production_tables(str(tmp_path))

        assert result is True
        mock_verify.assert_called_once()


def test_verify_simulation_model_production_tables_missing_files(tmp_path):
    """Test verification with missing parameter files."""
    productions_path = tmp_path / "simulation-models" / "productions"
    productions_path.mkdir(parents=True)

    production_file = productions_path / TEST_PRODUCTION_FILE
    production_file.write_text('{"parameters": {}}')

    with patch(
        "simtools.model.model_repository._verify_model_parameters_for_production"
    ) as mock_verify:
        mock_verify.return_value = (["/missing/file.json"], 1)

        result = model_repository.verify_simulation_model_production_tables(str(tmp_path))

        assert result is False


def test_verify_simulation_model_production_tables_no_production_files(tmp_path):
    """Test verification with no production files."""
    productions_path = tmp_path / "simulation-models" / "productions"
    productions_path.mkdir(parents=True)

    result = model_repository.verify_simulation_model_production_tables(str(tmp_path))

    assert result is True


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production_with_missing_files(mock_collect_data, tmp_path):
    """Test verification of model parameters with missing files."""

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
            str(tmp_path), production_file
        )

        assert total_checked == 2
        assert len(missing_files) == 2


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production_all_files_exist(mock_collect_data, tmp_path):
    """Test verification when all parameter files exist."""
    production_data = {"parameters": {"telescope": {"camera_config": "1.0.0"}}}
    mock_collect_data.return_value = production_data

    production_file = Path(TEST_PRODUCTION_FILE)

    with patch(PATH_PATCH) as mock_get_path:
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_get_path.return_value = mock_file

        missing_files, total_checked = model_repository._verify_model_parameters_for_production(
            str(tmp_path), production_file
        )

        assert total_checked == 1
        assert len(missing_files) == 0


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production_no_parameters(mock_collect_data, tmp_path):
    """Test verification with no parameters in production file."""
    production_data = {}
    mock_collect_data.return_value = production_data

    production_file = Path(TEST_PRODUCTION_FILE)

    missing_files, total_checked = model_repository._verify_model_parameters_for_production(
        str(tmp_path), production_file
    )

    assert total_checked == 0
    assert len(missing_files) == 0


@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production_non_dict_parameters(mock_collect_data, tmp_path):
    """Test verification with non-dict parameter values."""
    production_data = {"parameters": {"telescope": "not_a_dict", "array": {"valid_param": "1.0.0"}}}
    mock_collect_data.return_value = production_data

    production_file = Path(TEST_PRODUCTION_FILE)

    with patch(PATH_PATCH) as mock_get_path:
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_get_path.return_value = mock_file

        _, total_checked = model_repository._verify_model_parameters_for_production(
            str(tmp_path), production_file
        )

        assert total_checked == 1


@patch("simtools.utils.names.get_collection_name_from_parameter_name")
def test_get_model_parameter_file_path_regular_collection(mock_get_collection, tmp_path):
    """Test getting file path for regular collection."""
    mock_get_collection.return_value = "camera"

    result = model_repository._get_model_parameter_file_path(
        str(tmp_path), "telescope", "camera_config", "1.0.0"
    )

    expected = (
        tmp_path
        / "simulation-models"
        / "model_parameters"
        / "telescope"
        / "camera_config"
        / "camera_config-1.0.0.json"
    )
    assert result == expected


@patch("simtools.utils.names.get_collection_name_from_parameter_name")
def test_get_model_parameter_file_path_configuration_sim_telarray(mock_get_collection, tmp_path):
    """Test getting file path for configuration_sim_telarray collection."""
    mock_get_collection.return_value = "configuration_sim_telarray"

    result = model_repository._get_model_parameter_file_path(
        str(tmp_path), "telescope", "sim_telarray_config", "1.0.0"
    )

    expected = (
        tmp_path
        / "simulation-models"
        / "model_parameters"
        / "configuration_sim_telarray"
        / "telescope"
        / "sim_telarray_config"
        / "sim_telarray_config-1.0.0.json"
    )
    assert result == expected


@patch("simtools.utils.names.get_collection_name_from_parameter_name")
def test_get_model_parameter_file_path_configuration_corsika(mock_get_collection, tmp_path):
    """Test getting file path for configuration_corsika collection."""
    mock_get_collection.return_value = "configuration_corsika"

    result = model_repository._get_model_parameter_file_path(
        str(tmp_path), "telescope", "corsika_config", "1.0.0"
    )

    expected = (
        tmp_path
        / "simulation-models"
        / "model_parameters"
        / "configuration_corsika"
        / "corsika_config"
        / "corsika_config-1.0.0.json"
    )
    assert result == expected


def test_check_for_major_version_jump_no_major_jump():
    """Test version update without a major version jump."""
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
    """Test version update with a major version jump."""
    json_data = {
        "schema_version": "0.3.0",
        "parameter_version": "4.0.1",
    }
    param_data = {"version": "6.0.0", "value": 62.5}
    param = "dsum_threshold"
    telescope = "MSTx-FlashCam"

    result = model_repository._check_for_major_version_jump(json_data, param_data, param, telescope)

    assert result == "6.0.0"


def test_check_for_major_version_jump_no_previous_version():
    """Test version update when no previous version exists."""
    json_data = {
        "schema_version": "0.3.0",
    }
    param_data = {"version": "1.0.0", "value": 62.5}
    param = "dsum_threshold"
    telescope = "MSTx-FlashCam"

    result = model_repository._check_for_major_version_jump(json_data, param_data, param, telescope)

    assert result == "1.0.0"


@patch("simtools.model.model_repository.Path")
def test_get_latest_model_parameter_file_no_files(mock_path):
    """Test retrieving the latest model parameter file when no files exist."""
    mock_directory = Mock()
    mock_path.return_value = mock_directory

    mock_directory.glob.return_value = []

    with pytest.raises(FileNotFoundError, match="No JSON files found for parameter 'parameter'"):
        model_repository._get_latest_model_parameter_file("mock_directory", "parameter")


@patch("simtools.model.model_repository.Path")
def test_get_latest_model_parameter_file_unsorted_versions(mock_path):
    """Test retrieving the latest model parameter file with unsorted versions."""
    mock_directory = Mock()
    mock_path.return_value = mock_directory

    mock_file_1 = Mock()
    mock_file_1.stem = "parameter-1.0.0"
    mock_file_2 = Mock()
    mock_file_2.stem = "parameter-3.0.0"
    mock_file_3 = Mock()
    mock_file_3.stem = "parameter-2.0.0"
    mock_directory.glob.return_value = [mock_file_1, mock_file_3, mock_file_2]

    result = model_repository._get_latest_model_parameter_file("mock_directory", "parameter")

    assert result == str(mock_file_2)


def test_update_parameters_dict_new_function():
    """Test the new _update_parameters_dict function."""
    existing_params = {"dsum_threshold": "3.0.0"}
    changes = {
        "MSTx-FlashCam": {
            "dsum_threshold": {"version": "4.0.0", "value": 62.5},
            "param_to_deprecate": {"version": "1.0.0", "deprecated": True},
        },
        "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31.9}},
    }
    table_name = "MSTx-FlashCam"

    parameters, deprecated = model_repository._update_parameters_dict(
        existing_params, changes, table_name
    )

    assert "MSTx-FlashCam" in parameters
    assert parameters["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"
    assert "param_to_deprecate" not in parameters["MSTx-FlashCam"]  # Should be removed
    assert "param_to_deprecate" in deprecated  # Should be in deprecated list


def test_apply_changes_to_production_table_update_model_version():
    """Test updating the model version in the production table."""
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


def test_apply_changes_to_production_table_update_parameters_dict():
    """Test updating parameters in the production table."""
    data = {
        "production_table_name": "MSTx-FlashCam",
        "parameters": {
            "MSTx-FlashCam": {"dsum_threshold": "3.0.0"},
            "MSTx-NectarCam": {"discriminator_threshold": "3.0.0"},
        },
    }
    changes = {
        "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}},
        "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31.9}},
    }
    model_version = "6.5.0"

    model_repository._apply_changes_to_production_table(
        data["production_table_name"], data, changes, model_version, False
    )

    # Only parameters for the matching production_table_name should be included
    assert data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"
    assert "MSTx-NectarCam" not in data["parameters"]


def test_apply_changes_to_production_table_no_parameters():
    """Test applying changes when no parameters exist in the production table."""
    data = {"model_version": "6.0.0", "production_table_name": "MSTx-FlashCam", "parameters": {}}
    changes = {"MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}}}
    model_version = "6.5.0"

    model_repository._apply_changes_to_production_table(
        data["production_table_name"], data, changes, model_version, False
    )

    assert data["model_version"] == "6.5.0"
    # Parameters should now be created with only the matching telescope parameters
    assert data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"


def test_apply_changes_to_production_table_with_list_data():
    """Test applying changes when the production table contains a list."""
    data = [
        {
            "model_version": "6.0.0",
            "production_table_name": "MSTx-FlashCam",
            "parameters": {"MSTx-FlashCam": {"dsum_threshold": "3.0.0"}},
        }
    ]
    changes = {"MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}}}
    model_version = "6.5.0"

    with pytest.raises(TypeError, match="list indices must be integers or slices, not str"):
        model_repository._apply_changes_to_production_table(
            "MSTx-FlashCam", data, changes, model_version, False
        )


def test_apply_changes_to_production_tables(tmp_path):
    """Test applying changes to production tables."""
    # Create source directory with sample files
    source_prod_table_path = tmp_path / "productions" / "6.0.0"
    source_prod_table_path.mkdir(parents=True)
    target_prod_table_path = tmp_path / "productions" / "6.5.0"

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
    prod_table_file.write_text(json.dumps(prod_table_data))
    config_file = source_prod_table_path / "configuration_sim_telarray.json"
    config_file.write_text(json.dumps(config_table_data))

    # Mock changes to be applied
    changes = {
        "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}},
    }

    # Apply changes from source to target
    model_repository._apply_changes_to_production_tables(
        changes, "6.0.0", "6.5.0", "full_update", str(tmp_path)
    )

    # Verify the production table file is updated with changes
    updated_prod_file = target_prod_table_path / "MSTx-FlashCam.json"
    assert updated_prod_file.exists()
    updated_data = json.loads(updated_prod_file.read_text())
    assert updated_data["model_version"] == "6.5.0"
    assert updated_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"

    # Verify configuration table file model_version is updated but parameters unchanged
    config_target_file = target_prod_table_path / "configuration_sim_telarray.json"
    assert config_target_file.exists()
    config_data = json.loads(config_target_file.read_text())
    assert config_data["model_version"] == "6.5.0"
    # Parameters unchanged since production_table_name doesn't match changes
    assert config_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "3.0.0"
    assert config_data["parameters"]["MSTx-NectarCam"]["discriminator_threshold"] == "3.0.0"


def test_apply_changes_to_production_tables_no_parameters(tmp_path):
    """Test applying changes to production tables with no parameters."""
    # Create source directory with sample files
    source_prod_table_path = tmp_path / "productions" / "6.0.0"
    source_prod_table_path.mkdir(parents=True)
    target_prod_table_path = tmp_path / "productions" / "6.5.0"

    # Create a sample production table file in source
    prod_table_data = {
        "model_version": "6.0.0",
        "production_table_name": "MSTx-FlashCam",
        "parameters": {},
    }
    prod_table_file = source_prod_table_path / "MSTx-FlashCam.json"
    prod_table_file.write_text(json.dumps(prod_table_data))

    # Mock changes to be applied
    changes = {
        "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}},
    }

    # Call the function
    model_repository._apply_changes_to_production_tables(
        changes, "6.0.0", "6.5.0", "full_update", str(tmp_path)
    )

    # Verify the production table file is updated in target
    target_file = target_prod_table_path / "MSTx-FlashCam.json"
    assert target_file.exists()
    updated_data = json.loads(target_file.read_text())
    assert updated_data["model_version"] == "6.5.0"
    # Parameters should be created with the matching telescope parameters
    assert updated_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"


def test_apply_changes_to_production_tables_simple(tmp_path):
    """Test applying changes to production tables."""
    # Create source directory with sample files
    source_prod_table_path = tmp_path / "productions" / "6.0.0"
    source_prod_table_path.mkdir(parents=True)
    target_prod_table_path = tmp_path / "productions" / "6.5.0"

    # Create a sample production table file in source
    prod_table_data = {
        "model_version": "6.0.0",
        "production_table_name": "MSTx-FlashCam",
        "parameters": {"MSTx-FlashCam": {"dsum_threshold": "3.0.0"}},
    }
    prod_table_file = source_prod_table_path / "MSTx-FlashCam.json"
    prod_table_file.write_text(json.dumps(prod_table_data))

    # Changes to be applied
    changes = {"MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}}}

    # Call the function
    model_repository._apply_changes_to_production_tables(
        changes, "6.0.0", "6.5.0", "full_update", tmp_path
    )

    # Verify the production table file is updated in target
    target_file = target_prod_table_path / "MSTx-FlashCam.json"
    assert target_file.exists()
    updated_data = json.loads(target_file.read_text())
    assert updated_data["model_version"] == "6.5.0"
    assert updated_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"


def test_apply_changes_to_production_tables_multiple_files(tmp_path):
    """Test applying changes to multiple production table files."""
    # Create source directory with sample files
    source_prod_table_path = tmp_path / "productions" / "6.0.0"
    source_prod_table_path.mkdir(parents=True)
    target_prod_table_path = tmp_path / "productions" / "6.5.0"

    # Create multiple sample production table files in source
    prod_table_data_1 = {
        "model_version": "6.0.0",
        "production_table_name": "MSTx-FlashCam",
        "parameters": {"MSTx-FlashCam": {"dsum_threshold": "3.0.0"}},
    }
    prod_table_data_2 = {
        "model_version": "6.0.0",
        "production_table_name": "MSTx-NectarCam",
        "parameters": {"MSTx-NectarCam": {"discriminator_threshold": "3.0.0"}},
    }
    prod_table_file_1 = source_prod_table_path / "MSTx-FlashCam.json"
    prod_table_file_2 = source_prod_table_path / "MSTx-NectarCam.json"
    prod_table_file_1.write_text(json.dumps(prod_table_data_1))
    prod_table_file_2.write_text(json.dumps(prod_table_data_2))

    # Mock changes to be applied
    changes = {
        "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}},
        "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31.9}},
    }

    # Call the function
    model_repository._apply_changes_to_production_tables(
        changes, "6.0.0", "6.5.0", "full_update", tmp_path
    )

    # Verify the production table files are updated in target
    target_file_1 = target_prod_table_path / "MSTx-FlashCam.json"
    target_file_2 = target_prod_table_path / "MSTx-NectarCam.json"
    assert target_file_1.exists()
    assert target_file_2.exists()
    updated_data_1 = json.loads(target_file_1.read_text())
    updated_data_2 = json.loads(target_file_2.read_text())
    assert updated_data_1["model_version"] == "6.5.0"
    assert updated_data_1["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"
    assert updated_data_2["model_version"] == "6.5.0"
    assert updated_data_2["parameters"]["MSTx-NectarCam"]["discriminator_threshold"] == "4.0.0"


def test_apply_changes_to_production_tables_invalid_data_type(tmp_path):
    """Test error handling when JSON file contains non-dict data."""
    # Create source directory with a malformed JSON file
    source_prod_table_path = tmp_path / "productions" / "6.0.0"
    source_prod_table_path.mkdir(parents=True)

    # Create a JSON file with list data instead of dict
    malformed_file = source_prod_table_path / "malformed.json"
    malformed_file.write_text('["not", "a", "dict"]')

    changes = {"test_table": {"param1": {"version": "4.0.0", "value": 42}}}

    with pytest.raises(TypeError, match="Unsupported data type .* in .*malformed.json"):
        model_repository._apply_changes_to_production_tables(
            changes, "6.0.0", "6.5.0", "full_update", tmp_path
        )


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository._apply_changes_to_production_tables")
@patch("simtools.model.model_repository._apply_changes_to_model_parameters")
def test_generate_new_production_success(
    mock_apply_model_changes, mock_apply_table_changes, mock_collect_data, tmp_path
):
    """Test successful execution of generate_new_production."""
    args_dict = {
        "simulation_models_path": str(tmp_path),
        "base_model_version": "source",
        "modifications": TEST_MODIFICATIONS_FILE,
    }
    mock_collect_data.return_value = {
        "model_version": "6.5.0",
        "model_version_history": ["6.0.0"],
        "changes": {"telescope": {"param": {"version": "1.0.0", "value": 42}}},
    }

    model_repository.generate_new_production(
        args_dict["modifications"], args_dict["simulation_models_path"]
    )

    # Verify the new functions were called correctly
    mock_apply_table_changes.assert_called_once_with(
        {"telescope": {"param": {"version": "1.0.0", "value": 42}}},
        "6.0.0",
        "6.5.0",
        "full_update",
        str(tmp_path),
    )
    mock_apply_model_changes.assert_called_once()


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository._apply_changes_to_production_tables")
@patch("simtools.model.model_repository._apply_changes_to_model_parameters")
def test_generate_new_production_no_changes(
    mock_apply_model_changes, mock_apply_table_changes, mock_collect_data, tmp_path
):
    """Test execution with no changes in modifications."""
    args_dict = {
        "simulation_models_path": str(tmp_path),
        "base_model_version": "source",
        "modifications": TEST_MODIFICATIONS_FILE,
    }
    mock_collect_data.return_value = {"model_version": "6.5.0", "model_version_history": ["6.0.0"]}

    model_repository.generate_new_production(
        args_dict["modifications"], args_dict["simulation_models_path"]
    )

    mock_apply_table_changes.assert_called_once()
    # Since there are no changes, _apply_changes_to_model_parameters should still be called
    mock_apply_model_changes.assert_called_once_with({}, args_dict["simulation_models_path"])


@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
def test_generate_new_production_empty_version_history(mock_collect_data, tmp_path):
    """Test error handling when model_version_history is empty."""
    mock_collect_data.return_value = {
        "model_version": "6.5.0",
        "model_version_history": [],  # Empty list should trigger IndexError
        "changes": {},
    }

    with pytest.raises(IndexError, match="Base model version not found in"):
        model_repository.generate_new_production("fake_modifications.yml", str(tmp_path))


def test_apply_changes_to_production_table_patch_update():
    """Test patch update behavior with matching changes."""
    data = {
        "model_version": "6.0.0",
        "production_table_name": "test_table",
        "parameters": {"test_table": {"param1": "1.0.0"}},
    }
    changes = {"test_table": {"param1": {"version": "2.0.0", "value": 42}}}
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
        "parameters": {"other_table": {"param1": "1.0.0"}},
    }

    result_no_changes = model_repository._apply_changes_to_production_table(
        data_no_changes["production_table_name"], data_no_changes, changes, model_version, True
    )

    assert result_no_changes is False  # Should return False when no changes apply


def test_apply_changes_to_production_table_with_deprecated_parameters():
    """Test that deprecated_parameters are set when there are deprecated changes."""
    data = {
        "model_version": "6.0.0",
        "production_table_name": "test_table",
        "parameters": {"test_table": {"param1": "1.0.0"}},
    }
    changes = {
        "test_table": {
            "param1": {"version": "2.0.0", "value": 42},
            "param_to_remove": {"version": "1.0.0", "deprecated": True},
        }
    }
    model_version = "6.5.0"

    result = model_repository._apply_changes_to_production_table(
        "test_table", data, changes, model_version, False
    )

    assert result is True
    assert data["model_version"] == "6.5.0"
    assert "deprecated_parameters" in data
    assert "param_to_remove" in data["deprecated_parameters"]


@patch("simtools.model.model_repository._create_new_model_parameter_entry")
def test_apply_changes_to_model_parameters_simple(mock_create_entry, tmp_path):
    """Test applying changes to model parameters."""
    model_parameters_dir = tmp_path / "model_parameters"
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


@patch("simtools.model.model_repository._get_latest_model_parameter_file")
@patch("simtools.model.model_repository.writer.ModelDataWriter.dump_model_parameter")
def test_create_new_model_parameter_entry_simple(mock_dump, mock_get_latest, tmp_test_directory):
    """Test creating a new model parameter entry."""
    telescope = "MSTx-FlashCam"
    param = "dsum_threshold"
    param_data = {"version": "1.0.0", "value": 42.5}
    model_parameters_dir = Path(tmp_test_directory / "model_parameters")
    telescope_dir = model_parameters_dir / telescope
    telescope_dir.mkdir(parents=True)

    # Mock no existing file
    mock_get_latest.side_effect = FileNotFoundError("No files found")

    model_repository._create_new_model_parameter_entry(
        telescope, param, param_data, Path(tmp_test_directory)
    )

    # Verify dump_model_parameter was called with correct arguments
    mock_dump.assert_called_once_with(
        parameter_name=param,
        value=param_data["value"],
        instrument=telescope,
        parameter_version=param_data["version"],
        output_file=f"{param}-{param_data['version']}.json",
        output_path=model_parameters_dir / telescope / param,
        use_plain_output_path=True,
        unit=None,
        meta_parameter=False,
    )


def test_create_new_model_parameter_entry_telescope_dir_not_exists(tmp_test_directory):
    """Test that FileNotFoundError is raised when telescope directory doesn't exist."""
    telescope = "NonExistentTelescope"
    param = "some_param"
    param_data = {"version": "1.0.0", "value": 42.5}

    # Don't create the telescope directory - the function will create it but fail on schema
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        model_repository._create_new_model_parameter_entry(
            telescope, param, param_data, Path(tmp_test_directory)
        )


@patch("simtools.model.model_repository._check_for_major_version_jump")
@patch("simtools.model.model_repository.ascii_handler.collect_data_from_file")
@patch("simtools.model.model_repository._get_latest_model_parameter_file")
@patch("simtools.model.model_repository.writer.ModelDataWriter.dump_model_parameter")
def test_create_new_model_parameter_entry_with_existing_file(
    mock_dump, mock_get_latest, mock_collect_data, mock_check_version, tmp_test_directory
):
    """Test creating a new model parameter entry when existing file exists."""
    telescope = "MSTx-FlashCam"
    param = "dsum_threshold"
    param_data = {"version": "1.0.0", "value": 42.5, "unit": "count"}
    model_parameters_dir = Path(tmp_test_directory / "model_parameters")
    telescope_dir = model_parameters_dir / telescope
    telescope_dir.mkdir(parents=True)

    # Mock existing file found
    mock_get_latest.return_value = "/path/to/existing/file.json"
    mock_collect_data.return_value = {
        "value": [30.0, 31.0, 32.0],  # List value to trigger the conversion
        "meta_parameter": True,
    }
    mock_check_version.return_value = "2.0.0"

    model_repository._create_new_model_parameter_entry(
        telescope, param, param_data, model_parameters_dir
    )

    # Verify that existing file data was processed
    mock_collect_data.assert_called_once_with("/path/to/existing/file.json")
    mock_check_version.assert_called_once()

    # Verify that param_data was updated with existing file info
    assert param_data["version"] == "2.0.0"
    assert param_data["meta_parameter"] is True
    assert param_data["value"] == [42.5, 42.5, 42.5]  # Single value converted to list
