# simtools-simulate-prod

```{eval-rst}
.. _simulate_prod:

.. automodule:: simtools.applications.simulate_prod
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
   :application: simulate_prod
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_gamma_20_deg_south_multiple_model_versions.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_gamma_40_deg_south_corsika_only.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_gamma_40_deg_south_sim_telarray_only.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_gamma_62_deg_south_check_output.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_proton_20_deg_north_check_output.yml
```
