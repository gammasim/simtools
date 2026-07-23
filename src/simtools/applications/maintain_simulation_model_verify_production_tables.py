#!/usr/bin/python3

r"""
Verify simulation model production tables and model parameters for completeness.

This application is a utility to be used in the CI pipeline of the SimulationModels
repository. It checks that all model parameters defined in the production tables
exist in the simulation models repository.

Example
-------

.. code-block:: console

    simtools-maintain-simulation-model-verify-production-tables \\
        --simulation_models_path /path/to/simulation/models/repository

"""

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
