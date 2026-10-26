"""Generate a new simulation model production and update tables and model parameters."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model import model_repository

_MODEL_VERSION_KEY = "model_version"
_INITIALIZE_FROM_KEY = "initialize_from"
_INITIALIZE_UPDATE_TYPE_KEY = "initialize_update_type"
_SETTING_WORKFLOWS_GIT_TAG_KEY = "setting_workflows_git_tag"
_INITIALIZE_UPDATE_TYPES = ("patch_update", "full_update")

_ARGUMENTS = (
    cli.ArgumentDefinition(
        _SETTING_WORKFLOWS_GIT_TAG_KEY,
        help=(
            "Branch or tag used to download model parameters from the simulation workflow "
            "repository. Overrides setting_workflows_git_tag from info.yml."
        ),
        type=str,
        default=None,
    ),
    cli.ArgumentDefinition(
        _INITIALIZE_FROM_KEY,
        help=(
            "Initialize a new production version from this existing model version before "
            "generating its production tables."
        ),
        type=str,
        default=None,
    ),
    cli.ArgumentDefinition(
        _INITIALIZE_UPDATE_TYPE_KEY,
        help="Update type to write to the initialized production info file.",
        choices=_INITIALIZE_UPDATE_TYPES,
        default="patch_update",
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.SIMULATION_MODELS_PATH(required=True),
        cli.MODEL_VERSION,
        cli.IGNORE_EXISTING_PARAMETER_VERSION,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    model_version = app_context.args[_MODEL_VERSION_KEY]
    initialize_from = app_context.args[_INITIALIZE_FROM_KEY]
    initialize_update_type = app_context.args[_INITIALIZE_UPDATE_TYPE_KEY]
    simulation_models_path = Path(app_context.args["simulation_models_path"])

    if initialize_from is not None:
        if isinstance(model_version, list):
            if len(model_version) != 1:
                raise ValueError("--initialize_from requires exactly one --model_version value.")
            model_version = model_version[0]
        if model_version is None:
            raise ValueError("--initialize_from requires --model_version.")
        model_repository.initialize_production(
            simulation_models_path, model_version, initialize_from, initialize_update_type
        )
        return

    model_repository.generate_new_production(
        model_version=model_version,
        simulation_models_path=simulation_models_path,
        setting_workflows_git_tag=app_context.args[_SETTING_WORKFLOWS_GIT_TAG_KEY],
    )


if __name__ == "__main__":
    main()
