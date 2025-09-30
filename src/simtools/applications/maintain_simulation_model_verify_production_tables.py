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

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.model import model_repository


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Verify simulation model production tables and model parameters for completeness. "
            "This application checks that all model parameters defined in the production tables "
            "exist in the simulation models repository."
        ),
    )
    config.parser.add_argument(
        "--simulation_models_path",
        help="Path to the simulation models repository.",
        type=str,
        required=True,
    )
    return config.initialize(db_config=False, output=False, paths=False)


def main():
    """Verify simulation model production tables."""
    args_dict, _, _, _ = startup_application(_parse)

    if not model_repository.verify_simulation_model_production_tables(
        simulation_models_path=args_dict["simulation_models_path"]
    ):
        raise RuntimeError(
            "Verification failed: Some model parameters are missing in the repository."
        )


if __name__ == "__main__":
    main()
