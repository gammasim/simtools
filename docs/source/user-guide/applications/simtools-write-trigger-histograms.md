# simtools-write-trigger-histograms

```{eval-rst}
.. automodule:: simtools.applications.write_trigger_histograms
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

By default, only events triggering at least two telescopes contribute to the trigger histograms.
Set `minimum_triggered_telescopes: 1` (or use `--minimum_triggered_telescopes 1`) when single-
telescope triggers must be included, for example for a single-telescope layout.

For a directory of reduced event-data products, use `event_data_directory`. The application scans
direct-child `*.reduced_event_data.hdf5` files and groups files whose names differ only by a
`.part<digits>` suffix. It writes one `<group>.trigger_histograms.hdf5` product per group below
`output_path`. With the `htcondor` backend, one job is submitted per group and the command returns
after writing the submission manifest.

For a completed `simulate_prod` production, use `production_path` and one or more `select`
expressions. The application reads the job manifests, groups reduced event-data files by their
recorded configuration, and writes one trigger-histogram product per group. The site, model
version, and array layout are read from each selected manifest; explicit layout or model options
must agree with that metadata.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `event_data_files` | HDF5 | Reduced event-data files or glob patterns. |
| Input | `event_data_directory` | Directory | Direct-child reduced event-data files. |
| Input | `production_path` | Directory | Production job manifests and their reduced event-data files. |
| Input | `select` | `KEY=VALUE` | Optional repeated metadata selection. |
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

## Examples

### Explicit input file

```{eval-rst}
.. simtools-integration-example::
    :file: write_trigger_histograms_from_file.yml
```

### Input directory

```{eval-rst}
.. simtools-integration-example::
    :file: write_trigger_histograms_from_directory.yml
```

### Production metadata

```console
simtools-write-trigger-histograms \
    --production_path /data/production \
    --select configuration.zenith_angle="20 deg" \
    --select configuration.energy_min="1 TeV" \
    --output_path trigger_histograms
```
