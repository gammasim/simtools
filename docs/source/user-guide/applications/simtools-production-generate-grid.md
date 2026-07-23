# simtools-production-generate-grid

```{eval-rst}
.. automodule:: production_generate_grid
   :members:
   :exclude-members: main
```

## Overview

The application expands a production definition consisting of configuration axes, energy ranges,
and run statistics into a grid of executable simulation jobs. Possible axes include particle type,
simulation model version, interaction models, and pointing directions such as azimuth, zenith,
hour angle, and declination.

The generated grid can be used as input for local production execution or workload-management
submission tools. Different levels of night-sky background (NSB) can be configured through the
production model version.

## Input/Output

The application reads and writes the following user-visible files:

| Role | Argument | Format | Purpose | Schema |
| --- | --- | --- | --- | --- |
| Input | `--corsika_limits` | ECSV | Lookup table for CORSIKA simulation limits (optional). | Expected to follow `corsika_limits_table.schema.yml`. |
| Output | `--output_file` | ECSV | Executable production job grid. | Validated against `job_grid_density.schema.yml`. |

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: production_generate_grid
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: production_generate_grid_horizontal_explicit.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: production_generate_grid_horizontal.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: production_generate_grid_horizontal_density.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: production_generate_grid_ha_dec_density.yml
```
