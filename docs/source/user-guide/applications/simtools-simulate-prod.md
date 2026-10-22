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

After a successful job, the application writes `simulate_prod_job_metadata.yml` to the job output
directory. The manifest records the resolved configuration and the generated production files,
including files in the standard `sim_telarray/runNNNNNN` and `corsika/runNNNNNN` subdirectories.
For sim_telarray jobs, it also records simulated and triggered event counts under `statistics`;
these counts are derived from reduced-event data or the sim_telarray log without reading the
simtel event output.
When `--grid_output_path` is used, the manifest is written alongside the files packed for grid
registration.

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

## Shower input file format

For telescope simulations supplied with a shower input file,
`simulation_file_format` selects its reader. The default `eventio` reads current
CORSIKA IACT files. The reader supplies physical pointing, shower counts, reuse,
and observation height without constructing a CORSIKA configuration.

See [configuration and file formats](../../developer-guide/simulation_formats.md)
for the interfaces used to add formats. Available simulation programs remain
CORSIKA7 and sim_telarray.
