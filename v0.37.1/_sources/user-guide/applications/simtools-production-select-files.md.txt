# simtools-production-select-files

```{eval-rst}
.. automodule:: simtools.applications.production_select_files
   :members:
   :exclude-members: main
```

## Overview

The application selects production files from the
`simulate_prod_job_metadata.yml` manifests written for completed simulation jobs. Selection
expressions address manifest fields using dotted paths, for example
`configuration.zenith_angle=20 deg`.

The selected files are grouped by their simulation configuration. Use
`--require_complete_runs` to reject groups with missing run numbers.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `production_path` | Directory | Directory containing `job-*` output directories. |
| Input | `select` | `KEY=VALUE` | Optional selection expression; may be repeated. |
| Output | `output_file` | YAML | Optional file containing the selected groups. |

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: production_select_files
   :no-heading:
```

## Example

```console
simtools-production-select-files \
    --production_path /data/production \
    --select configuration.zenith_angle="20 deg" \
    --select configuration.primary=gamma \
    --file_type reduced_event_data \
    --output_file selected_files.yml
```
