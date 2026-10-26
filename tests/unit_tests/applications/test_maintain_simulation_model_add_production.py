import pytest

from simtools.applications import maintain_simulation_model_add_production as add_production

MODEL_VERSION_KEY = "model_version"
SIMULATION_MODELS_PATH_KEY = "simulation_models_path"
SETTING_WORKFLOWS_GIT_TAG_KEY = "setting_workflows_git_tag"
INITIALIZE_FROM_KEY = "initialize_from"
INITIALIZE_UPDATE_TYPE_KEY = "initialize_update_type"
APPLICATION_ATTRIBUTE = "APPLICATION"
BASE_MODEL_VERSION = "7.0.0"
NEW_MODEL_VERSION = "7.0.1"
SECOND_NEW_MODEL_VERSION = "7.0.2"
PATCH_UPDATE = "patch_update"
FULL_UPDATE = "full_update"


@pytest.mark.parametrize("update_type", [PATCH_UPDATE, FULL_UPDATE])
def test_main_initializes_without_generating(mocker, tmp_test_directory, update_type):
    context = mocker.Mock()
    context.args = {
        MODEL_VERSION_KEY: [NEW_MODEL_VERSION],
        SIMULATION_MODELS_PATH_KEY: tmp_test_directory,
        SETTING_WORKFLOWS_GIT_TAG_KEY: None,
        INITIALIZE_FROM_KEY: BASE_MODEL_VERSION,
        INITIALIZE_UPDATE_TYPE_KEY: update_type,
    }
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(add_production, APPLICATION_ATTRIBUTE, application)
    initialize = mocker.patch.object(add_production.model_repository, "initialize_production")
    generate = mocker.patch.object(add_production.model_repository, "generate_new_production")

    add_production.main()

    simulation_models_path = add_production.Path(tmp_test_directory)
    initialize.assert_called_once_with(
        simulation_models_path, NEW_MODEL_VERSION, BASE_MODEL_VERSION, update_type
    )
    generate.assert_not_called()


def test_main_generates_production_without_initialization(mocker, tmp_test_directory):
    context = mocker.Mock()
    context.args = {
        MODEL_VERSION_KEY: [NEW_MODEL_VERSION],
        SIMULATION_MODELS_PATH_KEY: tmp_test_directory,
        SETTING_WORKFLOWS_GIT_TAG_KEY: None,
        INITIALIZE_FROM_KEY: None,
        INITIALIZE_UPDATE_TYPE_KEY: PATCH_UPDATE,
    }
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(add_production, APPLICATION_ATTRIBUTE, application)
    initialize = mocker.patch.object(add_production.model_repository, "initialize_production")
    generate = mocker.patch.object(add_production.model_repository, "generate_new_production")

    add_production.main()

    initialize.assert_not_called()
    generate.assert_called_once_with(
        model_version=[NEW_MODEL_VERSION],
        simulation_models_path=add_production.Path(tmp_test_directory),
        setting_workflows_git_tag=None,
    )


def test_main_rejects_multiple_versions_when_initializing(mocker, tmp_test_directory):
    context = mocker.Mock()
    context.args = {
        MODEL_VERSION_KEY: [NEW_MODEL_VERSION, SECOND_NEW_MODEL_VERSION],
        SIMULATION_MODELS_PATH_KEY: tmp_test_directory,
        SETTING_WORKFLOWS_GIT_TAG_KEY: None,
        INITIALIZE_FROM_KEY: BASE_MODEL_VERSION,
        INITIALIZE_UPDATE_TYPE_KEY: PATCH_UPDATE,
    }
    application = mocker.Mock()
    application.start.return_value = context
    mocker.patch.object(add_production, APPLICATION_ATTRIBUTE, application)

    with pytest.raises(ValueError, match="requires exactly one --model_version"):
        add_production.main()
