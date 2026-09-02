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

`plot_histograms` controls diagnostic plotting independently of which layouts are processed. Set
it to `false` to disable plots, to `true` or `all` to plot every processed layout, or to a list of
array-layout names to plot only those layouts. A bare `--plot_histograms` command-line flag is
equivalent to `all`.

For example, this configuration derives limits for every layout but writes diagnostic plots only
for two layouts:

```yaml
plot_histograms:
- CTAO-North-Alpha
- CTAO-North-Beta
```

Set `plot_reduced_histograms: true` (or use `--plot_reduced_histograms`) to limit each selected
layout to `angular_distance_vs_energy_triggered` and `core_distance_vs_energy_triggered`. This
flag affects plotting only; all input histograms are still used for deriving limits.

For the distance-versus-energy plots, the top-right distance projection and bottom-right energy
projection include the derived maximum distance and minimum energy limits as red dashed lines.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `trigger_histogram_file` | HDF5 or glob | Product from `simtools-write-trigger-histograms`, or a glob matching several products. |
| Input | `trigger_histogram_directory` | Directory | Site-specific directory of existing trigger-histogram HDF5 products; processed one particle at a time. |
| Output | `output_file` | ECSV | Derived limits for single-file or merged-glob mode. |
| Output | `output_path` | Directory | Limits and optional diagnostic plots; directory mode creates one subdirectory per particle. |

The output table includes the selected particle, array layout, pointing, NSB level, derived limits,
the broad-range values used for the derivation, and standard simtools metadata in its ECSV header.
The loss settings are also recorded in the ECSV metadata. `output_file` may be an absolute path
or a path relative to `output_path`; if omitted, shared simtools startup generates a name from
the activity ID and optional label. The shared `--output_file_format` option is accepted for
configuration compatibility but does not change this application's output: the table is always
written as ECSV.

Rows with no positive entries in the energy histogram are skipped with a warning. This allows
layouts that are not available at every pointing or zenith angle to coexist in one trigger-histogram
product; valid layouts continue to produce limits.

Use `trigger_histogram_directory` when all existing trigger-histogram products for one site are
stored together. The application scans direct-child `*.trigger_histograms.hdf5` files and groups
them using their `TRIGGER_REFERENCE_METADATA` tables, then writes
`<output_path>/<particle>/corsika_limits.ecsv` for each particle. New primary particles are
discovered automatically. Gamma products with a zero-to-zero viewcone are labelled
`gamma-0.00deg`; other gamma products are labelled `gamma`. Files without valid metadata are
ignored with a warning. Site selection remains explicit: run a North and South configuration
separately. This mode does not read reduced event data or refill histograms.

For example:

```console
simtools-production-derive-corsika-limits \
    --trigger_histogram_directory /lustre/fs25/group/cta/prod6/north/trigger_histograms \
    --allowed_losses all,0.001,10 \
    --energy_threshold_fraction 0.001 \
    --output_path tmp_corsika_limits/north
```

Use `--plot_histograms` to write diagnostic plots below `output_path`. With multiple production
indices, plots are grouped below `output_path/production_<index>/`; with one production they are
written directly below the selected output directory. In directory mode, each particle receives
its own output directory first. Only selected array layouts are plotted. The current plotting path
writes PNG files; the shared `figure_format` option is not yet applied here.

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

The directory-mode workflow is also covered by the following integration-test
configuration:

```{eval-rst}
.. simtools-integration-example::
    :file: production_derive_corsika_limits_from_directory.yml
```
