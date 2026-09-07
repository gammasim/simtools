# simtools-maintain-simulation-model-verify-production-tables

```{eval-rst}
.. automodule:: simtools.applications.maintain_simulation_model_verify_production_tables
   :members:
   :exclude-members: main
```

```{eval-rst}
This application is a utility to be used in the CI pipeline of the SimulationModels
repository. It checks that all model parameters defined in the production tables
exist in the simulation models repository.

**Example**

.. code-block:: console

    simtools-maintain-simulation-model-verify-production-tables \\
        --simulation_models_path /path/to/simulation/models/repository
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: maintain_simulation_model_verify_production_tables
   :no-heading:
```
