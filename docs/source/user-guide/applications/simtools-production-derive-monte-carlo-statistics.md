# simtools-production-derive-monte-carlo-statistics

```{eval-rst}
.. automodule:: production_derive_monte_carlo_statistics
   :members:
   :exclude-members: main
```

## Overview

This application estimates the total number of thrown Monte Carlo events required by a production.
It evaluates expected triggered events from a trigger-histogram product after reweighting to an
optional target power-law spectrum.

Choose exactly one optimization target:

- `target_relative_uncertainty` requires every relevant energy bin to meet the requested relative
  statistical uncertainty.
- `target_triggered_events` requires the selected energy range to contain the requested number of
  triggered events.

Use `optimization_energy_min` and `optimization_energy_max` to limit the optimization range. The
optional reduced core and view-cone radii report statistics for a smaller analysis region.

## Input and output

| Role | Argument or file | Format | Description |
| --- | --- | --- | --- |
| Input | `trigger_histogram_file` | HDF5 | Product from `simtools-write-trigger-histograms`. |
| Configuration | `spectral_index` | Float | Target power-law index for reweighting. |
| Configuration | `target_relative_uncertainty` or `target_triggered_events` | Number | Exclusive target. |
| Output | `output_file` | ECSV | Estimated Monte Carlo statistics. |
| Output | `output_path` | Directory | Result and optional diagnostic plots. |

Set `plot_diagnostics` to write plots of expected events and relative uncertainty.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: production_derive_monte_carlo_statistics
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: production_derive_monte_carlo_statistics_target_uncertainty.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: production_derive_monte_carlo_statistics_target_triggered_events.yml
```
