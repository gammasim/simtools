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

To initialize a production version that does not yet have a directory, provide
``--initialize_from`` with the existing base version. The command creates an initial
``info.yml`` for the new version and exits without generating production tables. For example:

.. code-block:: console

    simtools-maintain-simulation-model-add-production \\
        --simulation_models_path ../simulation-models \\
        --model_version 7.0.1 \\
        --initialize_from 7.0.0

Initialization only works when the target production directory does not exist. The
generated ``info.yml`` starts as a patch update with no changes and identifies the base
version in ``model_version_history``. Add the desired changes to ``info.yml``, then run the
command again without ``--initialize_from`` to generate the patch tables and model parameters.
Use ``--initialize_update_type full_update`` to initialize a full update instead.

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
