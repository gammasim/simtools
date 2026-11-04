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
sim_telarray event files include CTAO DataProducts metadata. Use `--production_id` to set the
production name recorded in the file and job manifest; when omitted, `--label` supplies it.
Existing simtools metadata keys remain available for older readers.

The installed CORSIKA build determines which hadronic interaction-model combinations are
available. Use `--list_available_corsika_models` to inspect them. For simulations that run
CORSIKA, `--corsika_hadronic_transition_energy` controls the `HILOW` transition between the
low- and high-energy models. If omitted, the selected CORSIKA build default is retained.

## Shower ray tracing

For CORSIKA and sim_telarray simulations, use `--ray_tracing_backend obdeect` to trace
shower photons before sim_telarray. `--shower_route_mode stream` pipes the photons through
the tracer; `disk` retains the CORSIKA IACT file and writes camera photon-bunch files for
each telescope type. Each model version has a telescope map, and each route produces a log
and a resources JSON file. A failing route stops the job and prints its log to the job output.

`--shower_route_threads` sets the threads per route. The default, 1, runs each route as one
single-threaded process per telescope type, like sim_telarray. The value 0 divides the usable
cores among the telescope-type routes across all model versions, with a minimum of one; with
`--sequential`, this automatic setting uses one thread per route.

## Air-shower photons through obdeect: physics approach

CORSIKA supplies photon bunches at the telescope sphere centre, including their ground-frame
directions, emission altitudes, arrival times and wavelengths. The route projects them along
their directions to an input plane above the incoming obscurers. It transforms them to the
telescope frame, where obdeect applies the telescope axes offset and traces the optics,
including mirror reflectivity and shadowing. For a telescope without a structure, its
telescope transmission is applied. Wavelengths sampled from the CORSIKA wavelength range
follow the Cherenkov `1/lambda^2` law at integer nanometres. Photon thinning uses roulette;
seeds are derived per event, core reuse and telescope.

The route writes camera-frame bunches for sim_telarray with `Bypass_Optics` enabled.
sim_telarray applies quantum efficiency, camera and electronics. The route sets the 1201
height of each traced telescope to zero, so that sim_telarray does not shift the camera-frame
bunches along their directions (sim_telarray.c:2740-2790). Under bypass, sim_telarray takes the
airmass from the bunch direction it reads, which is near the telescope axis. The route therefore
multiplies each bunch by the ratio of the atmospheric transmission at the ground-frame airmass to
the transmission at the airmass sim_telarray reads under bypass, so that the absorption is
applied with the native airmass.

Telescope height convention: the height of each telescope in the 1201 object is taken as its
height above the observation level, as in sim_telarray. The reference-point altitude of the site
(16 m South, 21 m North) is not added to it.

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
