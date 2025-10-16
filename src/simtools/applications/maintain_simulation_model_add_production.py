r"""
Generate a new simulation model production and update tables and model parameters.

This script is used to maintain the simulation model repository. It allows to create
new production tables by copying an existing base version and applies modifications
to production tables and model parameters as provided in a YAML file (see the example file below).

Two main use cases are covered by this script:

1. full update: Create a complete new set of production tables (e.g. for new major or minor
   versions of the simulation models). This will copy all production tables from the source
   directory and apply the modifications to the tables that are listed in the modifications file.

2. patch update: Create a set of new production tables including the changes defined in the
   modifications file. No unmodified tables are copied. For new production tables with patch
   modifications, the key-value pair 'base_model_version: <base_model version>' is added.

Both use cases will also apply the modifications to the model parameters as defined in the
modifications file.

Example
-------

The following example applies a patch update with changes defined in a YAML file.

.. code-block:: console

    simtools-maintain-simulation-model-add-new-production \\
        --model_path ../simulation-models-dev/simulation-models/ \\
        --model_version 6.0.2

"""

from pathlib import Path

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.model import model_repository


def _parse():
    """Parse command line arguments."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Generate a new simulation model production",
    )
    return config.initialize(db_config=False, output=False, simulation_model=["model_version"])


def main():
    """Generate a new simulation model production."""
    app_context = startup_application(_parse)

    model_repository.generate_new_production(
        model_version=app_context.args["model_version"],
        simulation_models_path=Path(app_context.args["model_path"]),
    )


if __name__ == "__main__":
    main()
