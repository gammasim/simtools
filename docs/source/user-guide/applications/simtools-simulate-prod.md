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

The installed CORSIKA build determines which hadronic interaction-model combinations are
available. Use `--list_available_corsika_models` to inspect them. For simulations that run
CORSIKA, `--corsika_hadronic_transition_energy` controls the `HILOW` transition between the
low- and high-energy models. If omitted, the selected CORSIKA build default is retained.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :module: simtools.applications.simulate_prod
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_proton_20_deg_north_check_output.yml
```
