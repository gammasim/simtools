# Simtools Long Integration Tests

Long integration tests (LITs) allow to compare simulations results obtained with different simtools version (includes CORSIKA, sim-telarray, simtools), production models, or hadronic interaction models. Goal is to

- detect unintended physics regressions
- quantify differences between model versions (expected and unexpected differences)
- provide release qualification metrics
- provide long-term monitoring of execution statistics

Two types of tools are required - this work is about the implementation of the first type.

1. configure and run simulation productions for long integration tests
2. tools to compare two/several simulation production results

## Simulation productions for LITs

Assumption (for now) is that these productions are medium sized and run on the local HTCondor cluster. The future extension to a Grid-style setup should be taken into account (separation of production definition from execution backend).

Start with fixed energy simulations for easier comparison (and quicker testing). This can easily be extended later to energy ranges.

Comparison grid:

- simtools versions
- production model versions
- interaction models

Production grid:

- zenith angles (e.g., 20, 40, 60 deg)
- azimuth angles (e.g., 0, 180 deg)
- energy (e.g., [0.03, 0.1, 0.3, 1., 3., 10., 30., 100. TeV])
- primary (e.g. [‘gamma’, ‘proton’])

The grid might be adjustable (testing with 1 grid point, release with many).

Additional input:

- event statistics per energy bin (number of runs x events per run) *fixed during prototyping*
- scatter area (E) *fixed during prototyping*
- adjust energy grid as function of zenith *fixed during prototyping*

Implementation:

- tool similar to `simtools-simulate-prod-htcondor-generator` (or adaption of this tool)
- write one condor file per simtools versions (as these are defined by Apptainer times)
- modify the configuration paramaeter `apptainer_image` such that it can digest a dict with (label: apptainer image path) entries, e.g.:
```
apptainer_image:
  7.0.0: <apptainer image path for simtools 7.0.0>
  6.3.0: <apptainer image path for simtools 6.3.0>
```
- write a config file with one line per production grid point (similar to VTSSimpipe) to be ingested to HTCondor (x number of runs)

This means that there will be at most len(simtools versions) job submission steps.

## Technical details

Suggestion is to modify `simtools-simulate-prod-htcondor-generator`.

Copy and adapt the example in tests/integration_tests/config/simulate_prod_htcondor_generator_gamma_20_deg_north.yml:

- value listed in the comparison and production grid should be allowed to be lists in this configuration
- one condor output file per apptainer_image (this is the `container_image` field in the condor file)

simtools-simulate-prod-htcondor-generator should be modified that it doesn't write a file with the values from the grid included (like the example in htcondor_submit/simulate_prod.submit.sh) but :

- a parameter file with one line per grid point (job) with the value listed:
 
 Example for params.txt:

 ```text
 0.1 20 0 gamma epos 7.0.0 <grid output path>
 0.1 20 0 gamma epos 6.3.0 <grid output path>
 ```

 - note that this is the combination of the comparison and production grid values (energy, zenith, azimuth, primary, interaction model), so this file might be quite long (depending on the grid size); the simtools version is handled by the container image fields.

Condor should submit then with `queue name,value,tag from params.txt` (instead of `queue <nruns>`).

<grid output path> is the parameter given with `--pack_for_grid_register` and should contain the apptainer label (see modification of `apptainer_image` field in the config file) to be used for the job execution. This label should also be used in condor files, parameter file, and the job execution script.

Remember to keep the following for job execution:

```text
# Process ID used to generate run number
process_id="$1"
# Load environment variables (for DB access)
set -a; source "$2"
```

For the implementation:

- try to minimize code changes to the existing tool
- even when running with the current functionality (no grid), the should should use a params.txt file
- Scalar config values are internally normalized to lists of length 1.
- all runs should start with the same random seed and run number (this might change later; let's start simple)
  

Do not implement:

- Grid backend execution
- adaptive statistics
- energy-dependent scatter area
- comparison tooling

Make a plan with detailed suggestions. Goal is to have a work plan in markdown format that can be used for implementation. This should include technical details and code snippets where appropriate.
