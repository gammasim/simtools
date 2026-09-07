# simtools-write-production-metadata

```{eval-rst}
.. automodule:: simtools.applications.write_production_metadata
   :members:
   :exclude-members: main
```

## Overview

The application writes job-level metadata manifests for an existing simulation production. It
uses the authoritative ECSV job grid to reconstruct each job configuration and discovers output
files below each `job-*` directory, including the standard `sim_telarray/runNNNNNN` and
`corsika/runNNNNNN` subdirectories.

Use `--check` to validate manifests that already exist. In write mode, use `--overwrite` to
replace existing manifests.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `production_path` | Directory | Directory containing `job-*` output directories. |
| Input | `job_grid_file` | ECSV | Job grid used to resolve the production configuration. |
| Output | `simulate_prod_job_metadata.yml` | YAML | One manifest per job directory. |

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: write_production_metadata
   :no-heading:
```

## Example

```console
simtools-write-production-metadata \
    --production_path /data/production \
    --job_grid_file /data/production_grid.ecsv \
    --overwrite
```
