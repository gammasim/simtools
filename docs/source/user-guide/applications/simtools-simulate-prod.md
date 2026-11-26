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
For sim_telarray jobs, it also records simulated and triggered event counts per output under `statistics`;
these counts are derived from reduced-event data or the sim_telarray log without reading the
simtel event output.
When `--grid_output_path` is used, the manifest is written alongside the files packed for grid
registration.
sim_telarray event files include CTAO DataProducts metadata. Use `--production_id` to set the
production name recorded in the file and job manifest; when omitted, `--label` supplies it.
Existing simtools metadata keys remain available for older readers.

The installed CORSIKA build determines which hadronic interaction-model combinations are
available. Use `--list_available_corsika_models` to inspect them. For simulations that run
CORSIKA, `--corsika_hadronic_transition_energy` controls the `HILOW` transition between the
low- and high-energy models. If omitted, the selected CORSIKA build default is retained.

## Simulation chain

`--simulation_software` lists the stages of the simulation in order, each as
`STAGE` or `STAGE:KEY=VALUE,...`. The default is `corsika sim_telarray`: showers with CORSIKA,
telescopes and cameras with sim_telarray. A stage must consume what the previous stage produces.
The obdeect stages (for example `obdeect_atmosphere` and `obdeect_optics`), their options and the
efficiency factors of a chain are described in [obdeect](../../components/obdeect.md#simulation-chain).
Examples:

```bash
--simulation_software sim_telarray        # sim_telarray only, e.g. from a CORSIKA input file
--simulation_software corsika             # CORSIKA only
--simulation_software corsika obdeect_atmosphere obdeect_optics sim_telarray:bypass_optics=camera_only
```

simtools inserts the converters between programs with different photon formats
(`corsika_to_stream`, `stream_to_simtel`); list a converter to give it options.

The same list is used in configuration files:

```yaml
simulation_software:
  - corsika
  - sim_telarray
```

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

Use `--disable_nsb_autoscaling` to disable the zenith-dependent NSB correction. A configured manual NSB scaling factor still applies.
