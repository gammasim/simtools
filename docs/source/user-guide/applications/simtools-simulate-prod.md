```{eval-rst}
.. _simulate_prod:
```

# simtools-simulate-prod

```{eval-rst}
.. automodule:: simulate_prod
   :members:
   :exclude-members: main
```

## Overview

The application produces multipipe scripts and runs array-layout simulations that include shower
and detector simulations. It can execute only the CORSIKA shower simulation or pipe CORSIKA output
directly to sim_telarray using the sim_telarray multipipe mechanism.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: simulate_prod
   :no-heading:
```

## Examples

To read the simulation model from files instead of MongoDB, pass the root directory containing
the nested `simulation-models` directory:

```console
simtools-simulate-prod \
    --simulation_models_path /path/to/model-files \
    --model_version 7.0.0 \
    --site North \
    --array_layout_name CTAO-North-Alpha \
    --simulation_software corsika_sim_telarray \
    --config production.yml
```

The explicitly configured path takes precedence over MongoDB environment settings. Filesystem
access is read-only.

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_proton_20_deg_north_check_output.yml
```
