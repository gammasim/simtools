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

import logging

from simtools.configuration import configurator
from simtools.model import model_repository
from simtools.utils import general as gen


def _parse():
    """Parse command line arguments."""
    config = configurator.Configurator(
        description=(
            "Verify simulation model production tables and model parameters for completeness. "
            "This application checks that all model parameters defined in the production tables "
            "exist in the simulation models repository."
        )
    )
    config.parser.add_argument(
        "--simulation_models_path",
        help="Path to the simulation models repository.",
        type=str,
        required=True,
    )
    return config.initialize(db_config=False, output=False, paths=False)


def main():  # noqa: D103
    args_dict, _ = _parse()

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    if not model_repository.verify_simulation_model_production_tables(
        simulation_models_path=args_dict["simulation_models_path"]
    ):
        raise RuntimeError(
            "Verification failed: Some model parameters are missing in the repository."
        )


if __name__ == "__main__":
    main()
