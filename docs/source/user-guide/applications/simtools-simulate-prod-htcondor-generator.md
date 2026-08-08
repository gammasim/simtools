# simtools-simulate-prod-htcondor-generator

```{eval-rst}
.. automodule:: simulate_prod_htcondor_generator
   :members:
   :exclude-members: main
```

## Overview

This application converts an executable production grid into HTCondor submission files for
`simtools-simulate-prod`. It supports one Apptainer image or a mapping of image labels to image
paths. With multiple images, each label receives its own Condor and parameter file pair.

The generated files are written below `output_path`. HTCondor logs are placed below
`htcondor_log_path`, which defaults to `output_path/htcondor_logs`. Simulation products are written
below `simulation_output` when the generated wrapper runs.

The input grid should be created with
[`simtools-production-generate-grid`](simtools-production-generate-grid.md). Keep
`run_number_offset` at zero when that option was already applied while generating the grid.

## Input and output

| Role | Argument or file | Description |
| --- | --- | --- |
| Input | `job_grid_file` | Executable ECSV production grid. |
| Input | `apptainer_image` | One image path or a label-to-image mapping. |
| Output | `output_path` | Condor files and the generated wrapper script. |
| Output | `htcondor_log_path` | Condor log, error, and output directories. |
| Runtime | `simulation_output` | Base directory passed to the simulation jobs. |

Each image label produces `simulate_prod.submit.<label>.condor` and
`simulate_prod.submit.<label>.params.txt`. A single default image retains the unsuffixed filenames.

## Submitting jobs

Change to the output directory and submit the generated file:

```console
condor_submit simulate_prod.submit.condor
```

For labelled images, submit the corresponding `simulate_prod.submit.<label>.condor` file.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: simulate_prod_htcondor_generator
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_htcondor_generator_gamma_20_deg_north.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: simulate_prod_htcondor_generator_grid_horizontal.yml
```
