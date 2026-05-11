## Summary

Add grid-style HTCondor submission support to `simtools-simulate-prod-htcondor-generator`.

This change expands selected production and comparison parameters into a `params.txt` file, writes HTCondor submit files that queue from those params, and updates the submit script to consume per-job values from HTCondor arguments instead of hardcoded values.

## What Changed

- allow grid axes such as `primary`, `azimuth_angle`, `zenith_angle`, `model_version`, and interaction models to be provided as lists
- support `apptainer_image` as either a single image path or a label-to-image mapping
- generate one HTCondor params file row per expanded job and one condor file per apptainer label
- pass job-specific values, including resolved `array_layout_name`, through the params file into the submit script
- preserve energy units in the params file and forward them correctly to `simtools-simulate-prod`
- add unit and integration coverage for parser behavior and HTCondor grid generation

## Example Config

See the grid example in:

`tests/integration_tests/config/simulate_prod_htcondor_generator_grid_example.yml`

## Validation

- `pytest -q tests/unit_tests/configuration/test_configurator.py tests/unit_tests/applications/test_simulate_prod_htcondor_generator.py tests/unit_tests/job_execution/test_htcondor_script_generator.py`
