"""Generate a new simulation model production and update tables and model parameters."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model import model_repository

_ARGUMENTS = (
    cli.SIMULATION_MODELS_PATH(required=True),
    cli.ArgumentDefinition(
        "setting_workflows_git_tag",
        help=(
            "Branch or tag used to download model parameters from the simulation workflow "
            "repository. Overrides setting_workflows_git_tag from info.yml."
        ),
        type=str,
        default=None,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
    ),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    model_repository.generate_new_production(
        model_version=app_context.args["model_version"],
        simulation_models_path=Path(app_context.args["simulation_models_path"]),
        setting_workflows_git_tag=app_context.args["setting_workflows_git_tag"],
    )


if __name__ == "__main__":
    main()
