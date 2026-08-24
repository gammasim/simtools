# simtools-production-derive-corsika-limits

```{eval-rst}
.. automodule:: simtools.applications.production_derive_corsika_limits
   :members:
   :exclude-members: main
```

## Overview

This application derives CORSIKA limits from broad-range trigger histograms. The limits
cover the lower energy bound (`ERANGE`), the maximum core distance (`CSCAT`), and the viewcone
radius (`VIEWCONE`).

`allowed_losses` defines the accepted loss for the core-distance and angular-distance axes. Both
axes must be provided, or `all` can be used as a shorthand. Its three comma-separated fields are
the axis, fractional loss, and minimum number of lost events. The fraction is in `[0, 1]`, and
the event count is a non-negative integer:

```console
--allowed_losses core_distance,0.001,10 angular_distance,0.001,10
```

Use integrated limits by leaving `differential_loss_bins_per_decade` at zero, or derive
energy-dependent limits with a positive number of bins per decade. The lower energy limit is
derived from the triggered-energy peak using `energy_threshold_fraction`, which defaults to
`0.01` and must be in `[0, 1]`.

When `array_layout_name` is omitted, all layouts in the input file are processed. The option
accepts multiple names, for example `--array_layout_name CTAO-North-Alpha CTAO-North-Beta`.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `trigger_histogram_file` | HDF5 | Product from `simtools-write-trigger-histograms`. |
| Output | `output_file` | ECSV | Derived limits, including the production index. |
| Output | `output_path` | Directory | Limits and optional diagnostic plots. |

The output table includes the selected particle, array layout, pointing, NSB level, derived limits,
the broad-range values used for the derivation, and standard simtools metadata in its ECSV header.
The loss settings are also recorded in the ECSV metadata. `output_file` may be an absolute path
or a path relative to `output_path`; if omitted, shared simtools startup generates a name from
the activity ID and optional label. The output table is always ECSV.

Use `--plot_histograms` to write diagnostic plots below `output_path`. With multiple production
indices, plots are grouped below `output_path/production_<index>/`; with one production they are
written directly below `output_path`. The current plotting path writes PNG files; the shared
`figure_format` option is not yet applied here.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: production_derive_corsika_limits
   :no-heading:
```

## Example

The example below is rendered automatically from the integration-test configuration, so the
documented command and YAML remain aligned with the tested workflow.

```{eval-rst}
.. simtools-integration-example::
    :file: production_derive_corsika_limits.yml
```
