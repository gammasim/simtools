#!/usr/bin/python3

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import simtools.db.db_model_upload as db_model_upload

pytestmark = pytest.mark.db_unit_test


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


@patch("simtools.db.db_model_upload.read_production_tables")
def test_add_production_tables_to_db(mock_read_production_tables, tmp_test_directory, caplog):
    mock_db = Mock()
    input_path = Path(tmp_test_directory) / "productions"
    model_dir = input_path / "1.0.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    mock_read_production_tables.return_value = {
        "telescopes": {
            "model_version": "1.0.0",
            "parameters": {"MSTS-02": "param_value"},
            "design_model": {"MSTS-02": "MSTx-FlashCam"},
        }
    }

    db_model_upload.add_production_tables_to_db(input_path, mock_db)

    mock_db.add_production_table.assert_called_once_with(
        production_table={
            "parameters": {"MSTS-02": "param_value"},
            "design_model": {"MSTS-02": "MSTx-FlashCam"},
            "model_version": "1.0.0",
        },
    )

    mock_read_production_tables.return_value = {"telescopes": {"parameters": {}}}
    with caplog.at_level("INFO"):
        db_model_upload.add_production_tables_to_db(input_path, mock_db)
    assert "No production table for telescopes in model version 1.0.0" in caplog.text


@patch("simtools.db.db_model_upload.add_values_from_json_to_db")
def test_add_model_parameters_to_db(mock_add_values_from_json_to_db, tmp_test_directory):
    mock_db = Mock()
    input_path = Path(tmp_test_directory)
    array_element_dir = input_path / "LSTS-01"
    (array_element_dir / "num_gains").mkdir(parents=True, exist_ok=True)
    (array_element_dir / "mirror_list").mkdir(parents=True, exist_ok=True)
    (array_element_dir / "num_gains" / "num_gains-0.1.0.json").touch()
    (array_element_dir / "mirror_list" / "mirror_list-0.2.1.json").touch()

    with patch("simtools.db.db_model_upload.Path.iterdir", return_value=[array_element_dir]):
        with patch("simtools.db.db_model_upload.Path.is_dir", return_value=True):
            db_model_upload.add_model_parameters_to_db(input_path, mock_db)

    mock_add_values_from_json_to_db.assert_any_call(
        file=array_element_dir / "num_gains" / "num_gains-0.1.0.json",
        collection="telescopes",
        db=mock_db,
        file_prefix=array_element_dir / "num_gains",
    )
    mock_add_values_from_json_to_db.assert_any_call(
        file=array_element_dir / "mirror_list" / "mirror_list-0.2.1.json",
        collection="telescopes",
        db=mock_db,
        file_prefix=array_element_dir / "mirror_list",
    )
    assert mock_add_values_from_json_to_db.call_count == 2


@patch("simtools.db.db_model_upload.add_values_from_json_to_db")
def test_add_model_parameters_to_db_uses_parameter_schema_collection(
    mock_add_values_from_json_to_db, tmp_test_directory
):
    mock_db = Mock()
    input_path = Path(tmp_test_directory)
    simtel_parameter = input_path / "LSTN-design" / "min_photons" / "min_photons-1.0.0.json"
    corsika_parameter = (
        input_path / "global" / "corsika_iact_io_buffer" / "corsika_iact_io_buffer-1.0.0.json"
    )
    calibration_parameter = (
        input_path
        / "ILLN-01"
        / "array_element_position_ground"
        / "array_element_position_ground-2.0.0.json"
    )
    simtel_parameter.parent.mkdir(parents=True, exist_ok=True)
    corsika_parameter.parent.mkdir(parents=True, exist_ok=True)
    calibration_parameter.parent.mkdir(parents=True, exist_ok=True)
    simtel_parameter.touch()
    corsika_parameter.touch()
    calibration_parameter.touch()

    db_model_upload.add_model_parameters_to_db(input_path, mock_db)

    mock_add_values_from_json_to_db.assert_any_call(
        file=simtel_parameter,
        collection="configuration_sim_telarray",
        db=mock_db,
        file_prefix=simtel_parameter.parent,
    )
    mock_add_values_from_json_to_db.assert_any_call(
        file=corsika_parameter,
        collection="configuration_corsika",
        db=mock_db,
        file_prefix=corsika_parameter.parent,
    )
    mock_add_values_from_json_to_db.assert_any_call(
        file=calibration_parameter,
        collection="calibration_devices",
        db=mock_db,
        file_prefix=calibration_parameter.parent,
    )


@patch("simtools.db.db_model_upload.add_values_from_json_to_db")
def test_add_model_parameters_to_db_processes_all_parameter_directories(
    mock_add_values_from_json_to_db, tmp_test_directory
):
    mock_db = Mock()
    input_path = Path(tmp_test_directory)
    parameter_dir = input_path / "LSTN-design" / "num_gains"
    parameter_dir.mkdir(parents=True, exist_ok=True)
    (parameter_dir / "file1.json").touch()

    with patch("simtools.db.db_model_upload.Path.iterdir", return_value=[parameter_dir]):
        with patch("simtools.db.db_model_upload.Path.is_dir", return_value=True):
            db_model_upload.add_model_parameters_to_db(input_path, mock_db)

    mock_add_values_from_json_to_db.assert_called_once_with(
        file=parameter_dir / "file1.json",
        collection="telescopes",
        db=mock_db,
        file_prefix=parameter_dir,
    )


@patch("builtins.input")
def test_confirm_remote_database_upload_local_db(mock_input):
    mock_db = Mock()
    mock_db.is_remote_database.return_value = False

    result = db_model_upload._confirm_remote_database_upload(mock_db)

    assert result is True
    mock_input.assert_not_called()


@patch("builtins.input")
def test_confirm_remote_database_upload_remote_db_confirmed(mock_input):
    mock_db = Mock()
    mock_db.is_remote_database.return_value = True
    mock_db.db_config = {"db_server": "test-server"}
    mock_input.side_effect = ["yes", "yes"]

    result = db_model_upload._confirm_remote_database_upload(mock_db)

    assert result is True
    assert mock_input.call_count == 2


@patch("builtins.input")
def test_confirm_remote_database_upload_remote_db_first_prompt_denied(mock_input, caplog):
    mock_db = Mock()
    mock_db.is_remote_database.return_value = True
    mock_db.db_config = {"db_server": "test-server"}
    mock_input.return_value = "no"

    with caplog.at_level("INFO"):
        result = db_model_upload._confirm_remote_database_upload(mock_db)

    assert result is False
    assert mock_input.call_count == 1
    assert "Operation aborted." in caplog.text


@patch("builtins.input")
def test_confirm_remote_database_upload_remote_db_second_prompt_denied(mock_input, caplog):
    mock_db = Mock()
    mock_db.is_remote_database.return_value = True
    mock_db.db_config = {"db_server": "test-server"}
    mock_input.side_effect = ["yes", "no"]

    with caplog.at_level("INFO"):
        result = db_model_upload._confirm_remote_database_upload(mock_db)

    assert result is False
    assert mock_input.call_count == 2
    assert "Operation aborted." in caplog.text


@patch("builtins.input")
def test_confirm_remote_database_upload_keyboard_interrupt(mock_input, caplog):
    mock_db = Mock()
    mock_db.is_remote_database.return_value = True
    mock_db.db_config = {"db_server": "test-server"}
    mock_input.side_effect = KeyboardInterrupt()

    with caplog.at_level("INFO"):
        result = db_model_upload._confirm_remote_database_upload(mock_db)

    assert result is False
    assert "Operation aborted." in caplog.text


@patch("builtins.input")
def test_confirm_remote_database_upload_no_db_config(mock_input):
    mock_db = Mock()
    mock_db.is_remote_database.return_value = True
    mock_db.db_config = None
    mock_input.side_effect = ["yes", "yes"]

    result = db_model_upload._confirm_remote_database_upload(mock_db)

    assert result is True
    assert "unknown server" in mock_input.call_args_list[0][0][0]


@patch("simtools.db.db_model_upload.retry_command")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_clone_simulation_model_repository_with_branch(
    mock_rmtree, mock_retry_command, tmp_test_directory
):
    target_dir = Path(tmp_test_directory) / "repo"
    repository_url = "https://github.com/test/repo.git"
    repository_branch = "test-branch"
    db_simulation_model_version = "1.0.0"

    with patch("simtools.db.db_model_upload.Path.exists", return_value=True):
        result = db_model_upload.clone_simulation_model_repository(
            target_dir, repository_url, db_simulation_model_version, repository_branch
        )

    mock_rmtree.assert_called_once_with(target_dir)
    expected_command = f'git clone --depth=1 -b "test-branch" "{repository_url}" "{target_dir}"'
    mock_retry_command.assert_called_once_with(expected_command, max_attempts=3, delay=30)
    assert result == target_dir


@patch("simtools.db.db_model_upload.retry_command")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_clone_simulation_model_repository_with_version_tag(
    mock_rmtree, mock_retry_command, tmp_test_directory
):
    target_dir = Path(tmp_test_directory) / "repo"
    repository_url = "https://github.com/test/repo.git"
    repository_branch = None
    db_simulation_model_version = "2.0.0"

    with patch("simtools.db.db_model_upload.Path.exists", return_value=True):
        result = db_model_upload.clone_simulation_model_repository(
            target_dir, repository_url, db_simulation_model_version, repository_branch
        )

    mock_rmtree.assert_called_once_with(target_dir)
    expected_command = f'git clone --branch "2.0.0" --depth 1 "{repository_url}" "{target_dir}"'
    mock_retry_command.assert_called_once_with(expected_command, max_attempts=3, delay=30)
    assert result == target_dir


@patch("simtools.db.db_model_upload.retry_command")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_clone_simulation_model_repository_directory_not_exists(
    mock_rmtree, mock_retry_command, tmp_test_directory
):
    target_dir = Path(tmp_test_directory) / "repo"
    repository_url = "https://github.com/test/repo.git"
    repository_branch = None
    db_simulation_model_version = "1.0.0"

    with patch("simtools.db.db_model_upload.Path.exists", return_value=False):
        result = db_model_upload.clone_simulation_model_repository(
            target_dir, repository_url, db_simulation_model_version, repository_branch
        )

    mock_rmtree.assert_not_called()
    expected_command = f'git clone --branch "1.0.0" --depth 1 "{repository_url}" "{target_dir}"'
    mock_retry_command.assert_called_once_with(expected_command, max_attempts=3, delay=30)
    assert result == target_dir


@patch("simtools.db.db_model_upload.retry_command")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_clone_simulation_model_repository_relative_path(mock_rmtree, mock_retry_command):
    target_dir = "repo"
    repository_url = "https://github.com/test/repo.git"
    repository_branch = None
    db_simulation_model_version = "1.0.0"
    expected_absolute_path = Path.cwd() / target_dir

    with patch("simtools.db.db_model_upload.Path.exists", return_value=False):
        result = db_model_upload.clone_simulation_model_repository(
            target_dir, repository_url, db_simulation_model_version, repository_branch
        )

    expected_command = (
        f'git clone --branch "1.0.0" --depth 1 "{repository_url}" "{expected_absolute_path}"'
    )
    mock_retry_command.assert_called_once_with(expected_command, max_attempts=3, delay=30)
    assert result == expected_absolute_path


@patch("simtools.db.db_model_upload.retry_command")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_clone_simulation_model_repository_invalid_max_attempts(
    mock_rmtree, mock_retry_command, tmp_test_directory
):
    target_dir = Path(tmp_test_directory) / "repo"
    repository_url = "https://github.com/test/repo.git"

    with patch("simtools.db.db_model_upload.Path.exists", return_value=False):
        with pytest.raises(ValueError, match="Max attempts must be a positive integer"):
            db_model_upload.clone_simulation_model_repository(
                target_dir,
                repository_url,
                db_simulation_model_tag="1.0.0",
                repository_branch=None,
                max_attempts=0,
            )

    mock_rmtree.assert_not_called()
    mock_retry_command.assert_not_called()


@patch("simtools.db.db_model_upload.clone_simulation_model_repository")
@patch("simtools.db.db_model_upload.add_model_parameters_to_db")
@patch("simtools.db.db_model_upload.add_production_tables_to_db")
@patch("simtools.db.db_model_upload._confirm_remote_database_upload")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_add_complete_model_success(
    mock_rmtree,
    mock_confirm,
    mock_add_production_tables,
    mock_add_model_parameters,
    mock_clone_repo,
    tmp_test_directory,
    caplog,
):
    mock_db = Mock()
    tmp_dir = Path(tmp_test_directory) / "tmp"
    repository_dir = tmp_dir / "repo"
    db_simulation_model = "test_model"
    db_simulation_model_version = "1.0.0"
    repository_url = "https://github.com/test/repo.git"

    mock_confirm.return_value = True
    mock_clone_repo.return_value = repository_dir
    repository_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level("INFO"):
        db_model_upload.add_complete_model(
            tmp_dir, mock_db, db_simulation_model, db_simulation_model_version, repository_url
        )

    mock_confirm.assert_called_once_with(mock_db)
    mock_clone_repo.assert_called_once_with(
        tmp_dir,
        repository_url,
        db_simulation_model_tag=db_simulation_model_version,
        repository_branch=None,
        max_attempts=3,
    )
    mock_add_model_parameters.assert_called_once_with(
        input_path=repository_dir / "simulation-models" / "model_parameters", db=mock_db
    )
    mock_add_production_tables.assert_called_once_with(
        input_path=repository_dir / "simulation-models" / "productions", db=mock_db
    )
    mock_db.generate_compound_indexes_for_databases.assert_called_once_with(
        db_name=None,
        db_simulation_model=db_simulation_model,
        db_simulation_model_tag=db_simulation_model_version,
    )
    mock_rmtree.assert_called_once_with(repository_dir)
    assert "Upload of simulation model completed successfully" in caplog.text


@patch("simtools.db.db_model_upload.clone_simulation_model_repository")
@patch("simtools.db.db_model_upload._confirm_remote_database_upload")
def test_add_complete_model_confirmation_denied(mock_confirm, mock_clone_repo, tmp_test_directory):
    mock_db = Mock()
    tmp_dir = Path(tmp_test_directory) / "tmp"
    db_simulation_model = "test_model"
    db_simulation_model_version = "1.0.0"
    repository_url = "https://github.com/test/repo.git"

    mock_confirm.return_value = False

    db_model_upload.add_complete_model(
        tmp_dir, mock_db, db_simulation_model, db_simulation_model_version, repository_url
    )

    mock_confirm.assert_called_once_with(mock_db)
    mock_clone_repo.assert_not_called()
    mock_db.generate_compound_indexes_for_databases.assert_not_called()


@patch("simtools.db.db_model_upload.clone_simulation_model_repository")
@patch("simtools.db.db_model_upload._confirm_remote_database_upload")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_add_complete_model_clone_fails(
    mock_rmtree, mock_confirm, mock_clone_repo, tmp_test_directory
):
    mock_db = Mock()
    tmp_dir = Path(tmp_test_directory) / "tmp"
    db_simulation_model = "test_model"
    db_simulation_model_version = "1.0.0"
    repository_url = "https://github.com/test/repo.git"

    mock_confirm.return_value = True
    mock_clone_repo.side_effect = RuntimeError("Clone failed")

    with pytest.raises(RuntimeError, match="Upload of simulation model failed: Clone failed"):
        db_model_upload.add_complete_model(
            tmp_dir, mock_db, db_simulation_model, db_simulation_model_version, repository_url
        )

    mock_rmtree.assert_not_called()


@patch("simtools.db.db_model_upload.clone_simulation_model_repository")
@patch("simtools.db.db_model_upload.add_model_parameters_to_db")
@patch("simtools.db.db_model_upload._confirm_remote_database_upload")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_add_complete_model_parameters_upload_fails(
    mock_rmtree,
    mock_confirm,
    mock_add_model_parameters,
    mock_clone_repo,
    tmp_test_directory,
):
    mock_db = Mock()
    tmp_dir = Path(tmp_test_directory) / "tmp"
    repository_dir = tmp_dir / "repo"
    db_simulation_model = "test_model"
    db_simulation_model_version = "1.0.0"
    repository_url = "https://github.com/test/repo.git"

    mock_confirm.return_value = True
    mock_clone_repo.return_value = repository_dir
    mock_add_model_parameters.side_effect = ValueError("Parameters upload failed")
    repository_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        RuntimeError, match="Upload of simulation model failed: Parameters upload failed"
    ):
        db_model_upload.add_complete_model(
            tmp_dir, mock_db, db_simulation_model, db_simulation_model_version, repository_url
        )

    mock_rmtree.assert_called_once_with(repository_dir)


@patch("simtools.db.db_model_upload.clone_simulation_model_repository")
@patch("simtools.db.db_model_upload.add_model_parameters_to_db")
@patch("simtools.db.db_model_upload.add_production_tables_to_db")
@patch("simtools.db.db_model_upload._confirm_remote_database_upload")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_add_complete_model_repository_not_exists_after_clone(
    mock_rmtree,
    mock_confirm,
    mock_add_production_tables,
    mock_add_model_parameters,
    mock_clone_repo,
    tmp_test_directory,
):
    mock_db = Mock()
    tmp_dir = Path(tmp_test_directory) / "tmp"
    repository_dir = tmp_dir / "repo"
    db_simulation_model = "test_model"
    db_simulation_model_version = "1.0.0"
    repository_url = "https://github.com/test/repo.git"

    mock_confirm.return_value = True
    mock_clone_repo.return_value = repository_dir

    with patch("pathlib.Path.exists", return_value=False):
        db_model_upload.add_complete_model(
            tmp_dir, mock_db, db_simulation_model, db_simulation_model_version, repository_url
        )

    mock_rmtree.assert_not_called()


@patch("simtools.db.db_model_upload.clone_simulation_model_repository")
@patch("simtools.db.db_model_upload.add_model_parameters_to_db")
@patch("simtools.db.db_model_upload.add_production_tables_to_db")
@patch("simtools.db.db_model_upload._confirm_remote_database_upload")
@patch("simtools.db.db_model_upload.shutil.rmtree")
def test_add_complete_model_uses_repository_dir_without_clone(
    mock_rmtree,
    mock_confirm,
    mock_add_production_tables,
    mock_add_model_parameters,
    mock_clone_repo,
    tmp_test_directory,
):
    mock_db = Mock()
    repo_dir = Path(tmp_test_directory) / "models-repo"
    (repo_dir / "simulation-models" / "model_parameters").mkdir(parents=True, exist_ok=True)
    (repo_dir / "simulation-models" / "productions").mkdir(parents=True, exist_ok=True)

    mock_confirm.return_value = True

    db_model_upload.add_complete_model(
        tmp_dir=Path(tmp_test_directory) / "tmp",
        db=mock_db,
        db_simulation_model="test_model",
        db_simulation_model_tag="1.0.0",
        repository_url=None,
        repository_dir=str(repo_dir),
    )

    mock_clone_repo.assert_not_called()
    mock_add_model_parameters.assert_called_once_with(
        input_path=repo_dir / "simulation-models" / "model_parameters", db=mock_db
    )
    mock_add_production_tables.assert_called_once_with(
        input_path=repo_dir / "simulation-models" / "productions", db=mock_db
    )
    mock_rmtree.assert_not_called()


@patch("simtools.db.db_model_upload._confirm_remote_database_upload")
def test_add_complete_model_requires_repository_url_or_repository_dir(
    mock_confirm, tmp_test_directory
):
    mock_confirm.return_value = True

    with pytest.raises(
        RuntimeError,
        match="Upload of simulation model failed: Either repository_url or repository_dir must be provided",
    ):
        db_model_upload.add_complete_model(
            tmp_dir=Path(tmp_test_directory) / "tmp",
            db=Mock(),
            db_simulation_model="test_model",
            db_simulation_model_tag="1.0.0",
            repository_url=None,
            repository_dir=None,
        )


def test_validate_repository_directory_structure_missing_required_subdir(tmp_test_directory):
    repo_dir = Path(tmp_test_directory) / "models-repo"
    (repo_dir / "simulation-models" / "model_parameters").mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Expected directory not found"):
        db_model_upload._validate_repository_directory_structure(repo_dir)


def test_validate_repository_directory_structure_nonexistent_repository_dir(tmp_test_directory):
    repo_dir = Path(tmp_test_directory) / "models-repo-does-not-exist"

    with pytest.raises(FileNotFoundError, match="Repository directory does not exist"):
        db_model_upload._validate_repository_directory_structure(repo_dir)
