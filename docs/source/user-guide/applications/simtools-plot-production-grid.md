# simtools-plot-production-grid

```{eval-rst}
.. automodule:: plot_production_grid
   :members:
   :exclude-members: main
```

## Overview

This application reads an ECSV production grid, produced by
`simtools-production-generate-grid`, and creates PNG plots of the grid pointings and production
limits.

Horizontal grids use `azimuth_angle` and `zenith_angle`. Equatorial grids use `ra`
and `dec`. If both coordinate systems are present, both panels are shown; equatorial-only points
are not converted to local Alt/Az coordinates.

## Input and output

| Role | Argument or filename | Format | Description |
| --- | --- | --- | --- |
| Input | `--grid_points_file` | ECSV | Production-grid table. |
| Output | `production_grid_sky_projection.png` | PNG | Sky coverage in local Alt/Az, with an RA/Dec panel when available. |
| Output | `production_grid_altaz_<value>.png` | PNG | Alt/Az points colored by each supported value column. |
| Output | `production_grid_zenith_profile_<value>.png` | PNG | Value versus zenith angle at azimuth 0 and 180 degrees. |

The input follows
the [job grid schema](https://github.com/gammasim/simtools/blob/main/src/simtools/schemas/job_grid_density.schema.yml).

## Sky projection

The main plot uses a polar projection with azimuth increasing clockwise, north at the top, and
zenith angle increasing from 0 degrees at the center to 90 degrees at the edge. Grid metadata such
as site, density, and observation time is shown when available.

The `--plot_ra_dec_tracks` and `--dec_values` options are retained for compatibility with older
configurations. They do not add guide tracks in the current file-driven implementation.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_production_grid
   :no-heading:
```

## Example

```console
simtools-plot-production-grid --grid_points_file path/to/production_grid.ecsv \
    --output_path simtools-output
```

```{eval-rst}
.. simtools-integration-example::
    :file: plot_production_grid.yml
```
