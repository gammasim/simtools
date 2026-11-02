"""Unit tests for model_repository.maintenance module."""

import json
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from simtools.io import ascii_handler
from simtools.model_repository import maintenance

TEST_PRODUCTION_FILE = "test_production.json"
TEST_MODIFICATIONS_FILE = "modifications.json"
PATH_PATCH = "simtools.model_repository.maintenance.get_model_parameter_file_path"
INITIALIZATION_BASE_VERSION = "7.0.0"
INITIALIZATION_TARGET_VERSION = "7.0.1"
INFO_FILE_NAME = "info.yml"
PATCH_UPDATE = "patch_update"
FULL_UPDATE = "full_update"
MODEL_UPDATE = "model_update"
MODEL_VERSION_HISTORY = "model_version_history"
UPDATE_TYPE = "update_type"
FILE_PARAMETER_NAME = "test_file_parameter"
FILE_PARAMETER_VERSION = "1.0.0"
FILE_PARAMETER_ACTIVITY_ID = "test-activity-id"
FILE_PARAMETER_ASSET = "test-file-parameter-1.0.0.ecsv"
FILE_PARAMETER_CONTENT = b"# test file asset\n"
WORKFLOW_REPOSITORY = "https://example.org/workflows.git"
WORKFLOW_BRANCH = "feature-branch"
DEFAULT_WORKFLOW_REPOSITORY = (
    "https://gitlab.cta-observatory.org/cta-science/simulations/"
    "simulation-model/simulation-model-parameter-setting.git"
)


def _write_production_file(directory, name, data):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(data), encoding="utf-8")


def test_verify_simulation_model_production_tables(tmp_test_directory):
    productions_path = Path(tmp_test_directory) / "simulation-models" / "productions"
    _write_production_file(
        productions_path,
        TEST_PRODUCTION_FILE,
        {"parameters": {"telescope": {"camera_config": "1.0.0", "optics_config": "2.1.0"}}},
    )

    with patch(
        "simtools.model_repository.maintenance._verify_model_parameters_for_production"
    ) as mock_verify:
        mock_verify.return_value = ([], 2)
        assert maintenance.verify_simulation_model_production_tables(str(tmp_test_directory))

        mock_verify.return_value = (["/missing/file.json"], 1)
        assert not maintenance.verify_simulation_model_production_tables(str(tmp_test_directory))


@pytest.mark.parametrize(UPDATE_TYPE, [PATCH_UPDATE, FULL_UPDATE])
@pytest.mark.parametrize("base_update_type", [PATCH_UPDATE, FULL_UPDATE])
def test_initialize_production_preserves_required_history(
    tmp_test_directory, update_type, base_update_type
):
    root = Path(tmp_test_directory)
    base_version = "7.0.2"
    target_version = "7.0.3"
    ancestors = [INITIALIZATION_TARGET_VERSION, INITIALIZATION_BASE_VERSION]
    base_directory = maintenance.get_production_directory(root, base_version)
    base_directory.mkdir(parents=True)
    ascii_handler.write_data_to_file(
        {MODEL_UPDATE: base_update_type, MODEL_VERSION_HISTORY: ancestors},
        base_directory / INFO_FILE_NAME,
    )

    maintenance.initialize_production(root, target_version, base_version, update_type)

    target_info = ascii_handler.collect_data_from_file(
        maintenance.get_production_directory(root, target_version) / INFO_FILE_NAME
    )
    expected_history = [base_version]
    if update_type == PATCH_UPDATE and base_update_type == PATCH_UPDATE:
        expected_history.extend(ancestors)
    assert target_info[MODEL_VERSION_HISTORY] == expected_history


@pytest.mark.parametrize(
    ("create_target", "update_type", "error", "match"),
    [
        (True, PATCH_UPDATE, FileExistsError, "target directory already exists"),
        (False, PATCH_UPDATE, FileNotFoundError, "base production info file does not exist"),
        (False, "invalid", ValueError, "Unsupported production update type: invalid"),
    ],
)
def test_initialize_production_errors(tmp_test_directory, create_target, update_type, error, match):
    if create_target:
        (
            Path(tmp_test_directory)
            / "simulation-models"
            / "productions"
            / INITIALIZATION_TARGET_VERSION
        ).mkdir(parents=True)

    with pytest.raises(error, match=match):
        maintenance.initialize_production(
            tmp_test_directory,
            INITIALIZATION_TARGET_VERSION,
            INITIALIZATION_BASE_VERSION,
            update_type=update_type,
        )


@pytest.mark.parametrize(
    ("data", "missing_flag", "expected_checked", "expected_missing"),
    [
        (
            {"parameters": {"telescope": {"camera_config": "1.0.0", "mirror_config": "2.0.0"}}},
            True,
            2,
            2,
        ),
        ({"parameters": {"telescope": {"camera_config": "1.0.0"}}}, False, 1, 0),
        (
            {"parameters": {"telescope": "not_a_dict", "array": {"valid_param": "1.0.0"}}},
            False,
            1,
            0,
        ),
    ],
)
@patch("simtools.io.ascii_handler.collect_data_from_file")
def test_verify_model_parameters_for_production(
    mock_collect_data,
    tmp_test_directory,
    data,
    missing_flag,
    expected_checked,
    expected_missing,
):
    mock_collect_data.return_value = data
    with patch(PATH_PATCH) as mock_get_path:
        mock_file = Mock()
        mock_file.exists.return_value = not missing_flag
        mock_get_path.return_value = mock_file

        missing_files, total_checked = maintenance._verify_model_parameters_for_production(
            str(tmp_test_directory), Path(TEST_PRODUCTION_FILE)
        )

    assert total_checked == expected_checked
    assert len(missing_files) == expected_missing


@pytest.mark.parametrize(
    ("telescope", "subdirectory", "expected_name"),
    [
        ("telescope", "camera_config", "camera_config"),
        ("telescope", "sim_telarray_config", "sim_telarray_config"),
        ("telescope", "corsika_config", "corsika_config"),
        (None, "corsika_config", "corsika_config"),
    ],
)
def test_get_model_parameter_file_path(tmp_test_directory, telescope, subdirectory, expected_name):
    result = maintenance.get_model_parameter_file_path(
        str(tmp_test_directory), telescope, subdirectory, "1.0.0"
    )

    scope = telescope or "global"
    expected = (
        Path(tmp_test_directory)
        / "simulation-models"
        / "model_parameters"
        / scope
        / subdirectory
        / f"{expected_name}-1.0.0.json"
    )
    assert result == expected


@pytest.mark.parametrize(("param_version", "expected"), [("4.1.0", "4.1.0"), ("6.0.0", "6.0.0")])
def test_check_for_major_version_jump(param_version, expected):
    json_data = {"schema_version": "0.3.0", "parameter_version": "4.0.1"}
    param_data = {"version": param_version, "value": 62.5}

    result = maintenance._check_for_major_version_jump(
        json_data, param_data, "dsum_threshold", "MSTx-FlashCam"
    )

    assert result == expected


@pytest.mark.parametrize("patch_update", [False, True])
def test_apply_changes_to_production_table_configuration_corsika(patch_update):
    data = {
        "production_table_name": "configuration_corsika",
        "parameters": {
            "global": {
                "corsika_starting_grammage": "1.0.0",
                "corsika_first_interaction_height": "1.0.0" if not patch_update else "1.2.0",
            },
        },
    }
    changes = {
        "configuration_corsika": {
            "corsika_starting_grammage": {"version": "2.0.0", "value": 12.0},
            "corsika_first_interaction_height": {
                "version": "1.0.0" if not patch_update else "1.2.0",
                "deprecated": True,
            },
        }
    }

    maintenance._apply_changes_to_production_table(
        data["production_table_name"], data, changes, "6.5.0", patch_update
    )

    assert data["model_version"] == "6.5.0"
    assert "configuration_corsika" not in data["parameters"]
    assert data["parameters"]["global"]["corsika_starting_grammage"] == "2.0.0"
    assert "corsika_first_interaction_height" not in data["parameters"]["global"]
    if patch_update:
        assert data["deprecated_parameters"] == ["corsika_first_interaction_height"]


def test_apply_changes_to_production_tables(tmp_test_directory):
    source_path = Path(tmp_test_directory) / "simulation-models/productions" / "6.0.0"
    target_path = Path(tmp_test_directory) / "simulation-models/productions" / "6.5.0"
    _write_production_file(
        source_path,
        "MSTx-FlashCam.json",
        {
            "production_table_name": "MSTx-FlashCam",
            "model_version": "6.0.0",
            "parameters": {"MSTx-FlashCam": {"dsum_threshold": "3.0.0"}},
        },
    )
    _write_production_file(
        source_path,
        "configuration_sim_telarray.json",
        {
            "production_table_name": "configuration_sim_telarray",
            "model_version": "6.0.0",
            "parameters": {
                "MSTx-FlashCam": {"dsum_threshold": "3.0.0"},
                "MSTx-NectarCam": {"discriminator_threshold": "3.0.0"},
            },
        },
    )
    changes = {"MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62.5}}}

    maintenance._apply_changes_to_production_tables(
        changes, "6.0.0", "6.5.0", "full_update", tmp_test_directory
    )

    updated_data = json.loads((target_path / "MSTx-FlashCam.json").read_text(encoding="utf-8"))
    assert updated_data["model_version"] == "6.5.0"
    assert updated_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "4.0.0"

    config_data = json.loads(
        (target_path / "configuration_sim_telarray.json").read_text(encoding="utf-8")
    )
    assert config_data["model_version"] == "6.5.0"
    assert config_data["parameters"]["MSTx-FlashCam"]["dsum_threshold"] == "3.0.0"
    assert config_data["parameters"]["MSTx-NectarCam"]["discriminator_threshold"] == "3.0.0"


def test_apply_changes_to_production_tables_invalid_data_type(tmp_test_directory):
    source_path = Path(tmp_test_directory) / "simulation-models/productions" / "6.0.0"
    source_path.mkdir(parents=True)
    (source_path / "malformed.json").write_text('["not", "a", "dict"]', encoding="utf-8")

    with pytest.raises(TypeError, match=r"Unsupported data type .* in .*malformed.json"):
        maintenance._apply_changes_to_production_tables(
            {"test_table": {"param1": {"version": "4.0.0", "value": 42}}},
            "6.0.0",
            "6.5.0",
            "full_update",
            tmp_test_directory,
        )


@patch("simtools.model_repository.maintenance.ascii_handler.collect_data_from_file")
@patch("simtools.model_repository.maintenance._apply_changes_to_production_tables")
@patch("simtools.model_repository.maintenance._apply_changes_to_model_parameters")
def test_generate_new_production_empty_version_history(
    mock_apply_model_changes, mock_apply_table_changes, mock_collect_data, tmp_test_directory
):
    mock_collect_data.return_value = {
        "model_version": "6.5.0",
        MODEL_VERSION_HISTORY: [],
        "setting_workflows_git_tag": "v0.3.0",
        "setting_workflows_git_repository": WORKFLOW_REPOSITORY,
        "changes": {},
    }

    maintenance.generate_new_production("fake_modifications.yml", str(tmp_test_directory))

    mock_apply_table_changes.assert_called_once_with(
        {}, "6.5.0", "6.5.0", "full_update", str(tmp_test_directory)
    )
    mock_apply_model_changes.assert_called_once_with(
        {}, str(tmp_test_directory), "v0.3.0", WORKFLOW_REPOSITORY, context=None
    )


@patch("simtools.model_repository.maintenance.ascii_handler.collect_data_from_file")
@patch("simtools.model_repository.maintenance._apply_changes_to_production_tables")
@patch("simtools.model_repository.maintenance._apply_changes_to_model_parameters")
def test_generate_new_production_setting_workflows_git_tag_override(
    mock_apply_model_changes, mock_apply_table_changes, mock_collect_data, tmp_test_directory
):
    mock_collect_data.return_value = {
        "model_version": "6.5.0",
        MODEL_VERSION_HISTORY: [],
        "setting_workflows_git_tag": "from-info-file",
        "setting_workflows_git_repository": WORKFLOW_REPOSITORY,
        "changes": {},
    }

    maintenance.generate_new_production(
        "fake_modifications.yml",
        str(tmp_test_directory),
        setting_workflows_git_tag="from-cli",
    )

    mock_apply_model_changes.assert_called_once_with(
        {}, str(tmp_test_directory), "from-cli", WORKFLOW_REPOSITORY, context=None
    )


def test_apply_changes_to_model_parameters_simple(tmp_test_directory):
    model_parameters_dir = Path(tmp_test_directory) / "model_parameters"
    changes = {
        "MSTx-FlashCam": {
            "dsum_threshold": {"version": "4.0.0", "value": 62.5},
            "param_without_value": {"version": "1.0.0"},
        },
        "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31.9}},
    }

    with patch(
        "simtools.model_repository.maintenance._create_new_model_parameter_entry"
    ) as mock_create_entry:
        maintenance._apply_changes_to_model_parameters(changes, model_parameters_dir)

    assert mock_create_entry.call_count == 2
    mock_create_entry.assert_any_call(
        "MSTx-FlashCam",
        "dsum_threshold",
        {"version": "4.0.0", "value": 62.5},
        model_parameters_dir,
        context=None,
    )


@patch("simtools.model_repository.maintenance._download_model_parameter_from_workflow")
@patch("simtools.model_repository.maintenance._create_new_model_parameter_entry")
def test_apply_changes_to_model_parameters_with_activity_id(
    mock_create_entry, mock_download_workflow, tmp_test_directory
):
    model_parameters_dir = Path(tmp_test_directory) / "model_parameters"
    changes = {
        "LSTN-design": {
            "pm_photoelectron_spectrum": {
                "version": "3.0.0",
                "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8",
            },
            "value_only_param": {"version": "2.0.0", "value": 42},
        }
    }

    maintenance._apply_changes_to_model_parameters(
        changes, model_parameters_dir, setting_workflows_git_tag="release-v1"
    )

    mock_download_workflow.assert_called_once_with(
        "LSTN-design",
        "pm_photoelectron_spectrum",
        {"version": "3.0.0", "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8"},
        model_parameters_dir,
        "release-v1",
        DEFAULT_WORKFLOW_REPOSITORY,
    )
    mock_create_entry.assert_called_once_with(
        "LSTN-design",
        "value_only_param",
        {"version": "2.0.0", "value": 42},
        model_parameters_dir,
        context=None,
    )


@patch("simtools.model_repository.maintenance._download_model_parameter_from_workflow")
@patch("simtools.model_repository.maintenance._create_new_model_parameter_entry")
def test_apply_changes_to_model_parameters_with_both_value_and_activity_id_raises(
    mock_create_entry, mock_download_workflow, tmp_test_directory
):
    changes = {
        "LSTN-design": {
            "param_with_both": {"version": "1.0.0", "activity_id": "workflow-123", "value": 10}
        }
    }

    with pytest.raises(ValueError, match="Both activity_id and value are set"):
        maintenance._apply_changes_to_model_parameters(changes, tmp_test_directory)

    mock_download_workflow.assert_not_called()
    mock_create_entry.assert_not_called()


@patch("simtools.model_repository.maintenance.ascii_handler.collect_data_from_git")
@patch("simtools.model_repository.maintenance.writer.ModelDataWriter.write_model_parameter_json")
def test_download_model_parameter_from_workflow(
    mock_write_json, mock_collect_data, tmp_test_directory
):
    param_data = {"version": "3.0.0", "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8"}
    mock_collect_data.return_value = {"parameter_version": "3.0.0", "value": [1, 2, 3]}

    maintenance._download_model_parameter_from_workflow(
        telescope="LSTN-design",
        param="pm_photoelectron_spectrum",
        param_data=param_data,
        simulation_models_path=tmp_test_directory,
        setting_workflows_git_tag="v2.1.0",
        setting_workflows_git_repository=WORKFLOW_REPOSITORY,
    )

    mock_collect_data.assert_called_once_with(
        file_name=(
            "output/LSTN-design/pm_photoelectron_spectrum/"
            "019d85b6-1f98-715b-b92b-bfbcd06d7cd8/pm_photoelectron_spectrum/"
            "pm_photoelectron_spectrum-3.0.0.json"
        ),
        git_repository=WORKFLOW_REPOSITORY,
        git_branch="v2.1.0",
    )
    mock_write_json.assert_called_once_with(
        {"parameter_version": "3.0.0", "value": [1, 2, 3]},
        Path(tmp_test_directory)
        / "simulation-models"
        / "model_parameters"
        / "LSTN-design"
        / "pm_photoelectron_spectrum"
        / "pm_photoelectron_spectrum-3.0.0.json",
    )


def test_copy_model_parameter_asset_from_workflow_preserves_existing_file_asset(
    tmp_test_directory,
):
    parameter_file = Path(tmp_test_directory) / "parameter.json"
    asset_file = parameter_file.parent / FILE_PARAMETER_ASSET
    asset_file.write_bytes(FILE_PARAMETER_CONTENT)

    maintenance._copy_model_parameter_asset_from_workflow(
        {"file": True, "value": FILE_PARAMETER_ASSET},
        parameter_file,
        "output/parameter.json",
        WORKFLOW_REPOSITORY,
        WORKFLOW_BRANCH,
    )

    assert asset_file.read_bytes() == FILE_PARAMETER_CONTENT


@pytest.mark.parametrize(
    ("parameter_data", "error", "match"),
    [
        ({"file": True, "value": None}, ValueError, "relative filename"),
        ({"file": True, "value": FILE_PARAMETER_ASSET}, IsADirectoryError, "not a file"),
    ],
)
def test_copy_model_parameter_asset_from_workflow_rejects_invalid_target(
    tmp_test_directory, parameter_data, error, match
):
    parameter_file = Path(tmp_test_directory) / "parameter.json"
    if error is IsADirectoryError:
        (parameter_file.parent / FILE_PARAMETER_ASSET).mkdir()
    with pytest.raises(error, match=match):
        maintenance._copy_model_parameter_asset_from_workflow(
            parameter_data,
            parameter_file,
            "output/parameter.json",
            WORKFLOW_REPOSITORY,
            WORKFLOW_BRANCH,
        )


def test_copy_model_parameter_asset_from_raw_repository_url(tmp_test_directory):
    parameter_file = Path(tmp_test_directory) / "parameter.json"
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = FILE_PARAMETER_CONTENT
    repository_url = "https://example.org/workflows/-/raw"

    with patch(
        "simtools.model_repository.maintenance.urllib.request.urlopen", return_value=response
    ) as mock_urlopen:
        maintenance._copy_model_parameter_asset_from_workflow(
            {"file": True, "value": FILE_PARAMETER_ASSET},
            parameter_file,
            "output/parameter.json",
            repository_url,
            WORKFLOW_BRANCH,
        )

    mock_urlopen.assert_called_once_with(
        f"{repository_url}/{WORKFLOW_BRANCH}/output/{FILE_PARAMETER_ASSET}", timeout=30
    )
    assert (parameter_file.parent / FILE_PARAMETER_ASSET).read_bytes() == FILE_PARAMETER_CONTENT


def test_copy_model_parameter_asset_from_workflow_reports_missing_asset(tmp_test_directory):
    http_error = urllib.error.HTTPError("https://example.org/missing", 404, "Not found", {}, None)
    with (
        patch(
            "simtools.model_repository.maintenance.urllib.request.urlopen",
            side_effect=http_error,
        ),
        pytest.raises(FileNotFoundError, match="Failed to download model parameter asset"),
    ):
        maintenance._copy_model_parameter_asset_from_workflow(
            {"file": True, "value": FILE_PARAMETER_ASSET},
            Path(tmp_test_directory) / "parameter.json",
            "output/parameter.json",
            WORKFLOW_REPOSITORY,
            WORKFLOW_BRANCH,
        )


@patch("simtools.model_repository.maintenance.ascii_handler.collect_data_from_git")
@patch("simtools.model_repository.maintenance.writer.ModelDataWriter.write_model_parameter_json")
def test_download_model_parameter_from_workflow_keeps_existing_file(
    mock_write_json, mock_collect_data, tmp_test_directory
):
    tmp_test_directory = Path(tmp_test_directory)
    target = (
        tmp_test_directory
        / "simulation-models"
        / "model_parameters"
        / "LSTN-design"
        / "fadc_pulse_shape"
        / "fadc_pulse_shape-2.0.1.json"
    )
    target.parent.mkdir(parents=True)
    target.write_text('{"type": "file"}', encoding="utf-8")

    maintenance._download_model_parameter_from_workflow(
        telescope="LSTN-design",
        param="fadc_pulse_shape",
        param_data={"version": "2.0.1", "activity_id": "activity-id"},
        simulation_models_path=tmp_test_directory,
    )

    mock_collect_data.assert_not_called()
    mock_write_json.assert_not_called()
    assert target.read_text(encoding="utf-8") == '{"type": "file"}'


@patch("simtools.model_repository.maintenance.ascii_handler.collect_data_from_git")
@patch("simtools.model_repository.maintenance.writer.ModelDataWriter.write_model_parameter_json")
def test_download_model_parameter_from_workflow_raises_on_version_mismatch(
    mock_write_json, mock_collect_data, tmp_test_directory
):
    mock_collect_data.return_value = {"parameter_version": "2.9.9", "value": [1, 2, 3]}

    with pytest.raises(ValueError, match="Version mismatch"):
        maintenance._download_model_parameter_from_workflow(
            telescope="LSTN-design",
            param="pm_photoelectron_spectrum",
            param_data={
                "version": "3.0.0",
                "activity_id": "019d85b6-1f98-715b-b92b-bfbcd06d7cd8",
            },
            simulation_models_path=tmp_test_directory,
            setting_workflows_git_tag="v2.1.0",
        )

    mock_write_json.assert_not_called()


def test_create_new_model_parameter_entry_telescope_dir_not_exists(tmp_test_directory, run_context):
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        maintenance._create_new_model_parameter_entry(
            "NonExistentTelescope",
            "some_param",
            {"version": "1.0.0", "value": 42.5},
            Path(tmp_test_directory),
            context=run_context,
        )


@patch("simtools.model_repository.maintenance.get_model_parameter_file_path")
@patch("simtools.model_repository.maintenance._check_for_major_version_jump")
@patch("simtools.model_repository.maintenance.ascii_handler.collect_data_from_file")
@patch("simtools.model_repository.maintenance._get_latest_model_parameter_file")
@patch("simtools.model_repository.maintenance.writer.ModelDataWriter.write_model_parameter")
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
    (model_parameters_dir / telescope).mkdir(parents=True)
    mock_get_parameter_file_path.return_value = (
        model_parameters_dir / telescope / param / f"{param}-{param_data['version']}.json"
    )
    mock_get_latest.return_value = "/path/to/existing/file.json"
    mock_collect_data.return_value = {"value": [30.0, 31.0, 32.0]}
    mock_check_version.return_value = "2.0.0"

    maintenance._create_new_model_parameter_entry(
        telescope, param, param_data, simulation_models_path
    )

    mock_collect_data.assert_called_once_with("/path/to/existing/file.json")
    mock_check_version.assert_called_once()
    assert param_data["version"] == "2.0.0"
    assert param_data["value"] == [42.5, 42.5, 42.5]


@pytest.mark.parametrize(
    ("existing_value", "expected_error"),
    [
        ([42.5, 43.5], None),
        (41.5, "does not match the requested value"),
    ],
)
@patch("simtools.model_repository.maintenance.get_model_parameter_file_path")
@patch("simtools.model_repository.maintenance._get_latest_model_parameter_file")
@patch("simtools.model_repository.maintenance.ascii_handler.collect_data_from_file")
@patch("simtools.model_repository.maintenance.writer.ModelDataWriter.write_model_parameter")
def test_create_new_model_parameter_entry_with_matching_existing_file(
    mock_dump,
    mock_collect_data,
    mock_get_latest,
    mock_get_parameter_file_path,
    tmp_test_directory,
    existing_value,
    expected_error,
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
        "value": existing_value,
        "unit": "count",
    }

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            maintenance._create_new_model_parameter_entry(
                telescope, param, param_data, Path(tmp_test_directory)
            )
        mock_dump.assert_not_called()
    else:
        maintenance._create_new_model_parameter_entry(
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
    assert maintenance._normalize_units_for_comparison(unit) == expected


def test_get_changes_to_production_path_update(tmp_test_directory):
    modifications_data = {
        "model_version": "6.5.0",
        MODEL_VERSION_HISTORY: ["6.0.0"],
        "changes": {
            "MSTx-FlashCam": {"dsum_threshold": {"version": "4.0.0", "value": 62}},
            "MSTx-NectarCam": {"discriminator_threshold": {"version": "4.0.0", "value": 31}},
        },
    }

    with patch("simtools.io.ascii_handler.collect_data_from_file") as mock_collect:
        mock_collect.return_value = modifications_data
        changes, base_model_version = maintenance._get_changes_to_production(
            modifications_data, tmp_test_directory, "patch_update"
        )

    assert base_model_version == "6.0.0"
    assert changes["MSTx-FlashCam"]["dsum_threshold"]["value"] == 62
    assert changes["MSTx-NectarCam"]["discriminator_threshold"]["value"] == 31


def test_update_two_levels_in_changes_dict():
    d = {"LSTN-design": {"param1": {"version": "1.0.0"}}}

    result = maintenance._update_two_levels_in_changes_dict(
        d, {"LSTN-design": {"param2": {"version": "2.0.0"}}, "MST-design": {"p": {"version": "3"}}}
    )
    assert result["LSTN-design"]["param1"]["version"] == "1.0.0"
    assert result["LSTN-design"]["param2"]["version"] == "2.0.0"
    assert result["MST-design"]["p"]["version"] == "3"

    result = maintenance._update_two_levels_in_changes_dict(d, {"LSTN-design": "non_dict_value"})
    assert result["LSTN-design"] == "non_dict_value"


@patch("simtools.data_model.parameter_catalog.get_collection_name_from_parameter_name")
def test_apply_changes_to_sim_telarray_production_table(mock_get_collection):
    mock_get_collection.side_effect = lambda param: (
        "configuration_sim_telarray" if param == "min_photons" else "telescopes"
    )

    data = {
        "model_version": "6.0.0",
        "parameters": {
            "MSTN-design": {"other_param": "1.0.0"},
            "LSTN-design": {"min_photons": "1.0.0", "other_cst_param": "1.5.0"},
        },
    }
    changes = {
        "LSTN-design": {
            "min_photons": {"version": "2.0.0", "value": 0},
            "transit_time_random": {"version": "1.0.0", "value": 0.36},
        },
        "configuration_corsika": {"corsika_param": {"version": "1.0.2"}},
    }

    assert maintenance._apply_changes_to_sim_telarray_production_table(
        data, changes, "7.0.0", False
    )
    assert data["model_version"] == "7.0.0"
    assert data["parameters"]["LSTN-design"]["min_photons"] == "2.0.0"
    assert data["parameters"]["LSTN-design"]["other_cst_param"] == "1.5.0"
    assert data["parameters"]["MSTN-design"]["other_param"] == "1.0.0"
    assert "transit_time_random" not in data["parameters"]["LSTN-design"]


@patch("simtools.data_model.parameter_catalog.get_collection_name_from_parameter_name")
def test_apply_changes_to_sim_telarray_production_table_global_parameters(mock_get_collection):
    mock_get_collection.side_effect = lambda param: "configuration_sim_telarray"
    data = {
        "model_version": "6.0.0",
        "parameters": {
            "global": {"iobuf_maximum": "1.0.0"},
            "LSTN-design": {"iobuf_maximum": "0.9.0", "min_photons": "1.0.0"},
        },
    }
    changes = {"LSTN-design": {"iobuf_maximum": {"version": "2.0.0", "value": 0}}}

    assert maintenance._apply_changes_to_sim_telarray_production_table(
        data, changes, "7.0.0", False
    )
    assert data["parameters"]["global"]["iobuf_maximum"] == "2.0.0"
    assert "iobuf_maximum" not in data["parameters"]["LSTN-design"]


@patch("simtools.data_model.parameter_catalog.get_collection_name_from_parameter_name")
def test_apply_changes_to_sim_telarray_production_table_deprecated_patch_update(
    mock_get_collection,
):
    mock_get_collection.side_effect = lambda param: "configuration_sim_telarray"
    data = {"model_version": "6.0.0", "parameters": {"LSTN-design": {"min_photons": "1.0.0"}}}
    changes = {"LSTN-design": {"min_photons": {"version": "1.0.0", "deprecated": True}}}

    assert maintenance._apply_changes_to_sim_telarray_production_table(data, changes, "7.0.0", True)
    assert "min_photons" not in data["parameters"]["LSTN-design"]
    assert "min_photons" in data["deprecated_parameters"]
