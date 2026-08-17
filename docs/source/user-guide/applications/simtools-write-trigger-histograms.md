# simtools-write-trigger-histograms

```{eval-rst}
.. automodule:: write_trigger_histograms
   :members:
   :exclude-members: main
```

## Overview

Reads reduced event-data HDF5 files and writes trigger histograms. The products
contain simulated and triggered event distributions used for production checks, CORSIKA-limit
derivation, and Monte Carlo statistics estimates.

Use one or more input files or glob patterns. Files from each selected production are accumulated
into the output HDF5 file. `model_version`, `site`, and the array-layout selection describe the
simulation model associated with the input data.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `event_data_files` | HDF5 | Reduced event-data files or glob patterns. |
| Output | `output_file` | HDF5 | Trigger-histogram product. |

The output can be passed to the CORSIKA-limit and Monte Carlo-statistics applications.

Malformed input files stop processing by default. Use `skip_invalid_event_data_files` to skip
invalid files, and use `max_workers` to control parallel processing.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: write_trigger_histograms
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: write_trigger_histograms.yml
```
