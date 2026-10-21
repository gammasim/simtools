# simtools-compare-atmospheres

```{eval-rst}
.. automodule:: simtools.applications.compare_atmospheres
   :members:
   :exclude-members: main
```

Compare one or more atmospheric profile and transmission model parameters from the
configured simulation-model repository. Give the site, profile parameter version,
and transmission parameter version for each atmosphere. The first atmosphere is
the reference. Repeating the same combination is not supported.

The application writes profile figures for density, atmospheric depth, refractivity,
temperature, pressure, and water-vapor partial pressure against altitude, with
absolute differences from the first profile on
the overlapping altitude range and relative differences in percent in a separate
row. It writes a separate figure showing direct transmission and differences from
the reference at altitudes above sea level: top of atmosphere, 20 km, 10 km, 5 km,
the highest observation height plus 1.5 km and 0.5 km. The two lowest panel
titles show their resulting absolute a.s.l. heights. The transmission difference plots
show relative differences in percent. It also writes `atmosphere_summary.json`
with model identities and numerical
comparisons. Transmission is calculated as `exp(-optical depth)` at an emission
altitude of 10 km above sea level. The requested transmission levels are
interpolated in optical depth, with linear extrapolation where a table does not
reach a requested level. Each transmission table uses its own recorded
observation level; comparisons can therefore include different path endpoints.
All relative-difference axes use a symmetric range of ±35% by default. Override
this with the `relative_difference_limit` command-line argument.
The summary reports unweighted differences over common tabulated wavelengths.
A value is required at 10 km in each transmission table and each profile must
cover that altitude. Missing wavelength cells are not interpolated. Direct
transmission does not account for light scattered into the camera or predict
instrument response.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: compare_atmospheres
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
   :file: compare_atmospheres_north_south.yml
```
