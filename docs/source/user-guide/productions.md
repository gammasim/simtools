# Simulation Productions

Simtools allows large-scale productions of simulated events required for the calculation of instrument response functions.
Simtools is designed to use the CTAO workload management system, but it is flexible enough to work with other job engines such as Grid Engine and HTCondor.

Pre-requisites for running productions:

- the required simulation software (typically CORSIKA and sim_telarray) are installed on the system (findable through `$SIMTOOLS_SIMTEL_PATH`)
- a complete simulation model is accessible through the simulation model database
- production configuration exists (see e.g., [simulate_prod_gamma_20_deg_North.yml](../tests/integration_tests/config/simulate_prod_gamma_20_deg_North.yml))
- simtools is installed and configured on the job submission host

Productions are configured and submitted using the `simtools-simulate-prod` command.
The submit engine is configured with the `--submit_engine` command line option.

## Running local production

Choose with `SUBMIT_ENGINE: local` (default value).

Running a local production is a simple way to test the production system, but too slow for anything else.

## Running productions using HCCondor or Grid Engine

Choose with `SUBMIT_ENGINE: htcondor` or `SUBMIT_ENGINE: gridengine`.

HTCondor can be configured with additional options, e.g. `SUBMIT_OPTIONS: "priority = 50"`

Job submission using HTCondor or Grid Engine is the preferred way to run medium-scale productions (hundreds of runs). However, the current implementation in simtools is not using any features of those job engines for efficient job submissions (jobs are configured and submitted one-by-one) and therefore probably not suitable for large-scale productions.

## Running Grid Productions

:::{danger}
Missing Documentation.
:::
