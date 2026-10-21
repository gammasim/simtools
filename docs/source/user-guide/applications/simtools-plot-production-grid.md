# simtools-plot-production-grid

```{eval-rst}
.. automodule:: simtools.applications.plot_production_grid
   :members:
   :exclude-members: main
```

## Overview

This application reads an ECSV production grid, produced by
`simtools-production-generate-grid`, and creates plots of the grid pointings and production
limits.

Horizontal grids use `azimuth_angle` and `zenith_angle`. HA/Dec grids use `ha`
and `dec`. If both coordinate systems are present, both panels are shown.

## Input and output

| Role | Argument or filename | Format | Description |
| --- | --- | --- | --- |
| Input | `--grid_points_file` | ECSV | Production-grid table. |
| Output | `production_grid_sky_projection.png` | PNG | Sky coverage in local Alt/Az, with an HA/Dec panel when available. |
| Output | `production_grid_altaz_<value>.png` | PNG | Alt/Az points colored by each supported value column. |
| Output | `showers_vs_zenith_*.png` | PNG | Total showers per pointing versus zenith, for a selected azimuth interval and all azimuths. |
| Output | `showers_per_run_vs_zenith_single_row.png` | PNG | Showers per run versus zenith, colored by azimuth. |

The shower-statistics plots are made when the grid contains Alt/Az coordinates and a
`showers_per_run` column. Use `azimuth_center` and `azimuth_width` to choose the interval for the
selected-azimuth plot; their defaults are 180 degrees and 1 degree.

The input follows
the [job grid schema](https://github.com/gammasim/simtools/blob/main/src/simtools/schemas/job_grid_density.schema.yml).

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: plot_production_grid
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: plot_production_grid.yml
```
