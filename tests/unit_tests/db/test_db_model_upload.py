#!/usr/bin/python3

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import simtools.db.db_model_upload as db_model_upload


@patch("simtools.db.db_model_upload.ascii_handler.collect_data_from_file")
def test_add_values_from_json_to_db(mock_collect_data_from_file):
    mock_collect_data_from_file.return_value = {
        "parameter": "test_param",
        "parameter_version": "1.0",
    }
    mock_db = Mock()
    file = "test_file.json"
    collection = "test_collection"
    file_prefix = "test_prefix"

    db_model_upload.add_values_from_json_to_db(file, collection, mock_db, file_prefix)

    mock_collect_data_from_file.assert_called_once_with(file_name=file)
    mock_db.add_new_parameter.assert_called_once_with(
        par_dict={"parameter": "test_param", "parameter_version": "1.0"},
        collection_name=collection,
        file_prefix=file_prefix,
    )


@patch("simtools.db.db_model_upload.ascii_handler.collect_data_from_file")
def test_read_production_table(mock_collect_data_from_file):
    mock_collect_data_from_file.return_value = {
        "parameters": {"LSTN-design": {"param1": "param_value_1"}},
        "design_model": {"LSTN-design": "design_value_1"},
    }
    model_dict = {}
    file = Mock()
    file.stem = "LSTN-design"
    model_name = "test_model"

    db_model_upload._read_production_table(model_dict, file, model_name)

    assert "LSTN-design" in model_dict["telescopes"]["parameters"]
    assert model_dict["telescopes"]["parameters"]["LSTN-design"]["param1"] == "param_value_1"
    assert model_dict["telescopes"]["design_model"]["LSTN-design"] == "design_value_1"

    file.stem = "MSTx-NectarCam"
    with pytest.raises(KeyError, match="MSTx-NectarCam"):
        db_model_upload._read_production_table(model_dict, file, model_name)

    file.stem = "configuration_corsika"
    mock_collect_data_from_file.return_value = {"parameters": "config_param_value"}

    db_model_upload._read_production_table(model_dict, file, model_name)

    assert model_dict["configuration_corsika"]["parameters"] == "config_param_value"


@patch("simtools.db.db_model_upload._read_production_table")
def test_add_production_tables_to_db(mock_read_production_table, tmp_test_directory, caplog):
    mock_db = Mock()
    input_path = Path(tmp_test_directory)
    model_dir = input_path / "1.0.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "file1.json").touch()
    (model_dir / "MSTS-02.json").touch()

    mock_read_production_table.side_effect = lambda model_dict, file, _: model_dict.update(
        {
            "telescopes": {
                "parameters": {"MSTS-02": "param_value"},
                "design_model": {"MSTS-02": "MSTx-FlashCam"},
            }
        }
    )

    with patch("simtools.db.db_model_upload.Path.iterdir", return_value=[model_dir]):
        with patch("simtools.db.db_model_upload.Path.is_dir", return_value=True):
            db_model_upload.add_production_tables_to_db(input_path, mock_db)

    assert mock_read_production_table.call_count == 2
    mock_db.add_production_table.assert_called_once_with(
        production_table={
            "parameters": {"MSTS-02": "param_value"},
            "design_model": {"MSTS-02": "MSTx-FlashCam"},
            "model_version": "1.0.0",
        },
    )

    mock_read_production_table.side_effect = lambda model_dict, file, _: model_dict.update(
        {"telescopes": {"parameters": {}, "design_model": {}}}
    )
    with patch("simtools.db.db_model_upload.Path.iterdir", return_value=[model_dir]):
        with patch("simtools.db.db_model_upload.Path.is_dir", return_value=True):
            with caplog.at_level("INFO"):
                db_model_upload.add_production_tables_to_db(input_path, mock_db)
    assert "No production table for telescopes in model version 1.0.0" in caplog.text

    # Test with info.yml file containing model_version_history
    caplog.clear()
    info_content = {"model_version_history": ["0.9.0", "0.8.0"]}
    info_file = model_dir / "info.yml"
    with open(info_file, "w") as f:
        f.write('model_version_history: ["0.9.0", "0.8.0"]\n')

    with patch(
        "simtools.db.db_model_upload.ascii_handler.collect_data_from_file",
        return_value=info_content,
    ):
        with patch("simtools.db.db_model_upload.Path.iterdir", return_value=[model_dir]):
            with patch("simtools.db.db_model_upload.Path.is_dir", return_value=True):
                with caplog.at_level("INFO"):
                    db_model_upload.add_production_tables_to_db(input_path, mock_db)
    assert "model_version_history" in info_content
    assert "Reading production tables from repository" in caplog.text


@patch("simtools.db.db_model_upload.add_values_from_json_to_db")
def test_add_model_parameters_to_db(mock_add_values_from_json_to_db, tmp_test_directory):
    mock_db = Mock()
    input_path = Path(tmp_test_directory)
    array_element_dir = input_path / "LSTS-01"
    array_element_dir.mkdir(parents=True, exist_ok=True)
    (array_element_dir / "num_gains-0.1.0.json").touch()
    (array_element_dir / "mirror_list-0.2.1.json").touch()

    with patch("simtools.db.db_model_upload.Path.iterdir", return_value=[array_element_dir]):
        with patch("simtools.db.db_model_upload.Path.is_dir", return_value=True):
            db_model_upload.add_model_parameters_to_db(input_path, mock_db)

    mock_add_values_from_json_to_db.assert_any_call(
        file=array_element_dir / "num_gains-0.1.0.json",
        collection="telescopes",
        db=mock_db,
        file_prefix=input_path / "Files",
    )
    mock_add_values_from_json_to_db.assert_any_call(
        file=array_element_dir / "mirror_list-0.2.1.json",
        collection="telescopes",
        db=mock_db,
        file_prefix=input_path / "Files",
    )
    assert mock_add_values_from_json_to_db.call_count == 2


@patch("simtools.db.db_model_upload.add_values_from_json_to_db")
def test_add_model_parameters_to_db_skip_files_collection(
    mock_add_values_from_json_to_db, tmp_test_directory
):
    mock_db = Mock()
    input_path = Path(tmp_test_directory)
    files_dir = input_path / "Files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (files_dir / "file1.json").touch()

    with patch("simtools.db.db_model_upload.Path.iterdir", return_value=[files_dir]):
        with patch("simtools.db.db_model_upload.Path.is_dir", return_value=True):
            db_model_upload.add_model_parameters_to_db(input_path, mock_db)

    mock_add_values_from_json_to_db.assert_not_called()


def test_remove_deprecated_model_parameters():
    model_dict = {
        "telescopes": {
            "collection": "telescopes",
            "model_version": "1.0.0",
            "parameters": {
                "LSTN-01": {
                    "param1": "value1",
                    "deprecated_param": "deprecated_value",
                    "param2": "value2",
                },
                "MSTS-01": {"param3": "value3", "deprecated_param": "another_deprecated_value"},
            },
            "design_model": {"LSTN-01": "LST"},
            "deprecated_parameters": ["deprecated_param"],
        },
        "sites": {
            "collection": "sites",
            "model_version": "1.0.0",
            "parameters": {"North": {"altitude": "2147m", "old_param": "old_value"}},
            "deprecated_parameters": ["old_param"],
        },
        "configuration": {
            "collection": "configuration",
            "model_version": "1.0.0",
            "parameters": {"config_param": "config_value"},
        },
    }

    db_model_upload._remove_deprecated_model_parameters(model_dict)

    assert "deprecated_param" not in model_dict["telescopes"]["parameters"]["LSTN-01"]
    assert "deprecated_param" not in model_dict["telescopes"]["parameters"]["MSTS-01"]
    assert "param1" in model_dict["telescopes"]["parameters"]["LSTN-01"]
    assert "param2" in model_dict["telescopes"]["parameters"]["LSTN-01"]
    assert "param3" in model_dict["telescopes"]["parameters"]["MSTS-01"]

    assert "old_param" not in model_dict["sites"]["parameters"]["North"]
    assert "altitude" in model_dict["sites"]["parameters"]["North"]

    assert "config_param" in model_dict["configuration"]["parameters"]


@patch("simtools.db.db_model_upload.ascii_handler.collect_data_from_file")
@patch("simtools.db.db_model_upload._read_production_table")
def test_read_production_tables_basic(
    mock_read_production_table, mock_collect_data, tmp_test_directory
):
    """Test basic functionality of _read_production_tables."""
    model_path = Path(tmp_test_directory) / "1.0.0"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "LSTN-01.json").touch()
    (model_path / "MSTS-02.json").touch()

    mock_read_production_table.side_effect = lambda md, file, model: md.update(
        {
            "telescopes": {
                "collection": "telescopes",
                "model_version": model,
                "parameters": {"LSTN-01": {"param1": "value1"}},
                "design_model": {},
                "deprecated_parameters": [],
            }
        }
    )

    result = db_model_upload._read_production_tables(model_path)

    assert mock_read_production_table.call_count == 2
    assert "telescopes" in result
    assert result["telescopes"]["model_version"] == "1.0.0"


@patch("simtools.db.db_model_upload.ascii_handler.collect_data_from_file")
def test_read_production_tables_with_info_yml(mock_collect_data, tmp_test_directory):
    """Test _read_production_tables with model version history from info.yml."""
    model_path = Path(tmp_test_directory) / "2.0.0"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "info.yml").touch()
    (model_path / "LSTN-01.json").touch()

    # Create parent directory with older version
    old_model_path = model_path.parent / "1.0.0"
    old_model_path.mkdir(parents=True, exist_ok=True)
    (old_model_path / "LSTN-01.json").touch()

    # Mock info.yml content
    mock_collect_data.return_value = {"model_version_history": ["1.0.0"]}

    with patch("simtools.db.db_model_upload._read_production_table") as mock_read_table:
        mock_read_table.side_effect = lambda md, file, model: md.update(
            {
                "telescopes": {
                    "collection": "telescopes",
                    "model_version": model,
                    "parameters": {},
                    "design_model": {},
                    "deprecated_parameters": [],
                }
            }
        )

        result = db_model_upload._read_production_tables(model_path)

        # Should read from both 1.0.0 and 2.0.0 directories
        assert mock_read_table.call_count == 2
        assert result["telescopes"]["model_version"] == "2.0.0"


@patch("simtools.db.db_model_upload.ascii_handler.collect_data_from_file")
def test_read_production_tables_missing_info_yml(mock_collect_data, tmp_test_directory):
    """Test _read_production_tables when info.yml doesn't exist."""
    model_path = Path(tmp_test_directory) / "1.0.0"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "LSTN-01.json").touch()

    with patch("simtools.db.db_model_upload._read_production_table") as mock_read_table:
        mock_read_table.side_effect = lambda md, file, model: md.update(
            {
                "telescopes": {
                    "collection": "telescopes",
                    "model_version": model,
                    "parameters": {},
                    "design_model": {},
                    "deprecated_parameters": [],
                }
            }
        )

        result = db_model_upload._read_production_tables(model_path)

        # Should only process files from current model directory
        assert mock_read_table.call_count == 1
        assert result["telescopes"]["model_version"] == "1.0.0"


def test_read_production_tables_invalid_model_path(tmp_test_directory):
    """Test _read_production_tables when model path has invalid version format."""
    model_path = Path(tmp_test_directory) / "nonexistent"

    # The function will fail on version parsing for invalid model names
    from packaging.version import InvalidVersion

    with pytest.raises(InvalidVersion, match="Invalid version: 'nonexistent'"):
        db_model_upload._read_production_tables(model_path)


def test_read_production_tables_empty_directory(tmp_test_directory):
    """Test _read_production_tables with empty directory (no JSON files)."""
    model_path = Path(tmp_test_directory) / "1.0.0"
    model_path.mkdir(parents=True, exist_ok=True)

    result = db_model_upload._read_production_tables(model_path)

    # Should return empty dict when no JSON files found
    assert result == {}


@patch("simtools.db.db_model_upload.ascii_handler.collect_data_from_file")
def test_read_production_tables_invalid_version_in_history(mock_collect_data, tmp_test_directory):
    """Test _read_production_tables with invalid version strings in model_version_history."""
    model_path = Path(tmp_test_directory) / "1.0.0"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "info.yml").touch()

    # Mock info.yml with invalid version in history
    mock_collect_data.return_value = {"model_version_history": ["not.a.version"]}

    from packaging.version import InvalidVersion

    with pytest.raises(InvalidVersion, match="Invalid version: 'not.a.version'"):
        db_model_upload._read_production_tables(model_path)


@patch("simtools.db.db_model_upload.ascii_handler.collect_data_from_file")
def test_read_production_tables_ascii_handler_error(mock_collect_data, tmp_test_directory):
    """Test _read_production_tables when ascii_handler fails."""
    model_path = Path(tmp_test_directory) / "1.0.0"
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / "info.yml").touch()

    # Mock ascii_handler to raise an exception
    mock_collect_data.side_effect = ValueError("Invalid file format")

    with pytest.raises(ValueError, match="Invalid file format"):
        db_model_upload._read_production_tables(model_path)
