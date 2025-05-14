# Simulation Productions

Simtools allows large-scale productions of simulated events required for the calculation of instrument response functions.
Simtools is designed to use the CTAO workload management system (WMS) and the HT Condor job submission system.

Pre-requisites for running productions:

- simtools is installed and configured on the job submission host.
- simulation software (CORSIKA and sim_telarray) are installed on the worker nodes or installed in a container (findable through `$SIMTOOLS_SIMTEL_PATH`) .
- simulation models are accessible through the simulation model database. This includes both the production tables and the model parameters.
- configuration parameters for the production (see e.g., [simulate_prod_gamma_20_deg_North.yml](../tests/integration_tests/config/simulate_prod_gamma_20_deg_North.yml))

Productions are configured and submitted using the

## Local productions

Configure and submit a local production using the `simtools-simulate-prod` command.
The submit engine is configured with the `--submit_engine local` command line option, see
[simulate_prod_gamma_20_deg_North.yml](../tests/integration_tests/config/simulate_prod_gamma_20_deg_North.yml) for an example configuration.

Running a local production is a simple way to test the production system, but in general too slow to simulate a large number of events.

## Running simtools on HTCondor using Apptainers

Simtools can be run using HTCondor and Apptainers. Jobs are configured with the `simtools-simulate-prod-htcondor-generator` command.
Configuration is similar to the `simulate-prod` command, with the following additions:

- `APPTAINER_IMAGE` is the path to the Apptainer image to be used for the simulation.
- `PRIORITY` is the priority of the job in HTCondor.
- `RUN_NUMBER` and `NUMBER_OF_RUNS` are the first run number and the number of runs to be simulated.

The `simtools-simulate-prod-htcondor-generator` tool generates two files:

- a condor submission file to specify the apptainer, the number of jobs (equals the number of runs), the priority, and the output files.
- a condor submission script with the `simtools-simulate-prod` command to be run in the apptainer.

An example for the configuration `simtools-simulate-prod-htcondor-generator` can be found in [simulate_prod_htcondor_generator_gamma_20_deg_North.yml](tests/integration_tests/config/simulate_prod_htcondor_generator_gamma_20_deg_North.yml).

Container images are available from the GitHub and CTAO container registries and can be converted to Apptainer images using the `apptainer build` command.
Example:

```bash
apptainer build simtools.sif docker://ghcr.io/gammasim/simtools-prod-250304-corsika-78000-bernlohr-1.69-prod6-baseline-qgs3-avx2:20250507-154410
```

## Running Grid Productions

:::{danger}
Missing Documentation.
:::
