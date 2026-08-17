#!/usr/bin/python3

"""Verify simulation model production tables and model parameters for completeness."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model import model_repository

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "simulation_models_path",
        help="Path to the simulation models repository.",
        type=str,
        required=True,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS,),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    if not model_repository.verify_simulation_model_production_tables(
        simulation_models_path=app_context.args["simulation_models_path"]
    ):
        raise RuntimeError(
            "Verification failed: Some model parameters are missing in the repository."
        )


if __name__ == "__main__":
    main()
