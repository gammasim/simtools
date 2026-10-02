# simtools-production-generate-grid

```{eval-rst}
.. automodule:: simtools.applications.production_generate_grid
   :members:
   :exclude-members: main
```

## Overview

The application expands a grid of simulation configuration parameters to be used
for a simulation production.
Possible axes include particle type,
simulation model version, interaction models, and pointing directions such as azimuth, zenith,
hour angle, and declination.

The generated grid can be used as input for local production execution or workload-management
submission tools. Different levels of night-sky background (NSB) can be configured through the
production model version.

## Input/Output

The application reads and writes the following user-visible files:

| Role | Argument | Format | Purpose | Schema |
| --- | --- | --- | --- | --- |
| Input | `corsika_limits` | ECSV | Lookup table for CORSIKA simulation limits (optional). | [`corsika_limits_table.schema.yml`](https://github.com/gammasim/simtools/blob/main/src/simtools/schemas/corsika_limits_table.schema.yml). |
| Output | `output_file` | ECSV | Executable production job grid. | [`job_grid_density.schema.yml`](https://github.com/gammasim/simtools/blob/main/src/simtools/schemas/job_grid_density.schema.yml). |

## Defining the number of showers

`showers_per_run` is required and defines the baseline number of CORSIKA showers in one job.
For every combination of primary, model version, interaction model, pointing, and energy interval,
the application first determines the showers in each generated job and then determines how many jobs
to create. The resulting value is written to the `showers_per_run` column; the unscaled configured
value is retained in `configured_showers_per_run`.

Choose one of the following allocation modes:

| Mode | Configuration | Result |
| --- | --- | --- |
| Fixed number of jobs | Set `number_of_runs`. | Creates that many jobs for every grid point and energy interval. The total is `number_of_runs * showers_per_run` after any per-run scaling. If omitted, one job is created. |
| Target number of showers | Set `total_showers`. | Creates enough equal-sized jobs to meet the target at every grid point and energy interval. This cannot be used with `number_of_runs`. |

With `total_showers`, the target is rounded up to a whole number of jobs. For example,
`total_showers: 2500` and `showers_per_run: 1000` create three jobs of 1000 showers, for an actual
total of 3000. A warning reports each such adjustment; `max_total_showers_rounding_warnings` limits
how many of these warnings are emitted (20 by default).

The following options alter the shower allocation. Per-run scaling is applied before the number of
jobs is determined, so it affects the total when `number_of_runs` is used and the required number of
jobs when `total_showers` is used.

| Option | Choices or value | Effect |
| --- | --- | --- |
| `showers_per_run_power_law` | `INDEX REFERENCE_ENERGY` | Scales the baseline per-run value by `(E_mid / E_ref)^INDEX`, where `E_mid` is the logarithmic midpoint of the selected energy interval. The result is rounded up to an integer. Omit it, or use index `0`, to keep the per-run count independent of energy. |
| `showers_per_run_scaling` | `fixed` (default), `cosine_zenith` | `cosine_zenith` sets the per-run value to `ceil(N * cos(zenith))`; `fixed` leaves it unchanged. |
| `total_showers_scaling` | `fixed` (default), `zenith_scaled` | Applies only with `total_showers`. `zenith_scaled` changes the target to `ceil(N * exp(F * (cos(zenith) - 1)))`, where `F` is `zenith_angle_scaling_factor` (3.9781 by default). |

`total_showers_scaling` changes the requested total, whereas `showers_per_run_scaling` changes the
size of each job. They can be combined: the first sets a zenith-dependent target and the second sets
a zenith-dependent job size, after which the application rounds the number of jobs up as needed.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: production_generate_grid
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: production_generate_grid_ha_dec.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: production_generate_grid_ha_dec_density.yml
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
    :file: production_generate_grid_horizontal_explicit.yml
```
