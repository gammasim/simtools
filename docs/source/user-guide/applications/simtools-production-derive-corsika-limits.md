# simtools-production-derive-corsika-limits

```{eval-rst}
.. automodule:: production_derive_corsika_limits
   :members:
   :exclude-members: main
```

## Overview

This application derives CORSIKA limits from broad-range trigger histograms. The limits
cover the lower energy bound (`ERANGE`), the maximum core distance (`CSCAT`), and the viewcone
radius (`VIEWCONE`).

`allowed_losses` defines the accepted loss for the core-distance and angular-distance axes. Use
integrated limits by leaving `differential_loss_bins_per_decade` at zero, or derive energy-dependent
limits with differential bins. The lower energy limit is derived from the triggered-energy peak
using `energy_threshold_fraction`.

## Input and output

| Role | Argument | Format | Description |
| --- | --- | --- | --- |
| Input | `trigger_histogram_file` | HDF5 | Product from `simtools-write-trigger-histograms`. |
| Output | `output_file` | ECSV | Derived limits, including the production index. |
| Output | `output_path` | Directory | Limits and optional diagnostic plots. |

The output table includes the selected particle, array layout, pointing, NSB level, derived limits,
the broad-range values used for the derivation, and standard simtools metadata in its ECSV header.
With multiple productions, plots are grouped into production-specific subdirectories.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: production_derive_corsika_limits
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: production_derive_corsika_limits.yml
```
