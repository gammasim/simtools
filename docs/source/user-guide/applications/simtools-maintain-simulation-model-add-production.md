# simtools-maintain-simulation-model-add-production

```{eval-rst}
.. automodule:: simtools.applications.maintain_simulation_model_add_production
   :members:
   :exclude-members: main
```

```{eval-rst}
This script is used to maintain the simulation model repository. It allows to create
new production tables by copying an existing base version and applies modifications
to production tables and model parameters as provided in a configuration file (see
the 'info.yml' examples in the simulation models repository).

Two main use cases are covered by this script:

1. full update: create a complete new set of production tables (e.g. for new major or minor
   versions of the simulation models). This will copy all production tables from the source
   directory and apply the modifications to the tables that are listed in the modifications file.
   If the full update is based on a previous patch update, the full history of changes is applied
   iteratively until the last base version is reached.

2. patch update: create a set of new production tables including the changes defined in the
   modifications file. No unmodified tables are copied. For new production tables with patch
   modifications, the key-value pair 'base_model_version: <base_model version>' is added.

Both use cases will also apply the modifications to the model parameters as defined in the
modifications file.

**Example**

The following example applies a patch update with changes defined in a YAML file.

.. code-block:: console

    simtools-maintain-simulation-model-add-production \\
        --simulation_models_path ../simulation-models-dev \\
        --model_version 6.0.2
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: maintain_simulation_model_add_production
   :no-heading:
```
