# Configure Productions

Production configuration turns a compact production definition into an executable job grid.
Each row of the grid represents one simulation run with a resolved primary particle, model
version, array layout, pointing direction, energy range, CORSIKA limits, shower count, and run
number.
The grid is created with
[simtools-production-generate-grid](../applications/simtools-production-generate-grid).

```{warning}
Production configuration is a policy-heavy step, with many parameter choices defined by the user.
Small changes to pointing coverage, event
statistics, CORSIKA limits, or energy scaling can multiply the number of generated jobs or change
the simulated phase space. A review of the generated grid is required before submission.
```

## Configuration Workflow

The production context includes:

- the site, model version, and array layout;
- the primary particles and CORSIKA interaction models;
- the simulation energy range, core scatter, and view cone;
- whether the pointing grid is horizontal (`azimuth`/`zenith`) or hour-angle (`ha`/`dec`);
- the source-offset grid points;
- the model version, which determines the NSB rate used by the production grid;
- whether the number of runs is fixed or derived from a total shower target;
- whether direction-dependent CORSIKA limits or zenith-dependent scaling should be applied.

The CORSIKA and sim_telarray executable versions are not selected directly in the grid. They are
determined by the production environment, usually the Apptainer image used for execution. The
exact software versions used by a production are identified from the image tag, image release
notes, and generated run logs.

Some parameters are normally production-policy settings rather than per-production tuning knobs.
Examples are `simulation_software` and `curved_atmosphere_min_zenith_angle`. These parameters
normally remain at their defaults unless the production definition explicitly requires otherwise.

## Pointing Strategy

Pointing directions and other grid dimensions are configured with `axis` entries. A production
grid always needs direction axes plus the non-direction axes used by the simulation, usually
`offset`.

NSB is not a grid axis. simtools resolves the integrated NSB rate from the site model for each
selected `model_version` and writes it to the job grid as `nsb_rate`. This means that NSB values
are directly dependent on the model version. When a production requests several model versions,
the same pointing and offset grid can result in different `nsb_rate` values for each version.

A horizontal grid is used when the production is defined directly in local coordinates. This is
the usual choice for simulations binned by zenith and azimuth:

```yaml
axis:
- azimuth 310 deg 20 deg 3 linear
- zenith 30 deg 40 deg 2 linear
- offset 0 deg 0 deg 1 linear
```

This example defines a horizontal grid with evenly spaced bins between the listed minimum and
maximum azimuth and zenith values, plus a single offset point.

Directed azimuth ranges may cross the 0 deg boundary. For example, `310 deg` to `20 deg` covers
the interval through North.

An hour-angle grid is used when the production is defined by local hour angle and declination.
simtools converts each HA/Dec point to local azimuth and zenith using the site latitude, because
CORSIKA and sim_telarray still need local horizontal coordinates:

```yaml
axis:
- ha 0 deg 360 deg 36 linear
- dec -90 deg 90 deg 18 linear
- offset 0 deg 10 deg 2 linear
```

For either coordinate system, `direction_grid_density` can be used instead of an explicit number
of direction bins. The axis ranges still define the accepted domain, while simtools derives the
number of direction points from the requested density. This is useful when the grid is intended to
sample solid angle more uniformly or follow the interpolation scale of CORSIKA limits:

```yaml
axis:
- azimuth 0 deg 360 deg 0 linear
- zenith 0 deg 70 deg 0 linear
- offset 0 deg 10 deg 2 linear
direction_grid_density: 0.25 1/deg^2
```

For density-based HA/Dec grids, local-sky filters restrict the generated points to the observable
region needed for the production:

```yaml
axis:
- ha 0 deg 360 deg 1 linear
- dec -40 deg 80 deg 1 linear
- offset 0 deg 10 deg 2 linear
direction_grid_density: 0.25 1/deg^2
local_zenith_range: 0 deg 70 deg
local_azimuth_range: 300 deg 60 deg
```

## Energy and CORSIKA Limits

`energy_range`, `core_scatter`, and `view_cone` define the requested simulation phase space. Fixed
energies can be simulated by setting the minimum and maximum energy to the same value. Multiple
fixed energies or energy bins can be provided as a list of energy-range pairs in a configuration
file.

If a CORSIKA limits table is passed through `corsika_limits`, simtools interpolates the table for
each direction point using zenith, azimuth, and the model-version-dependent `nsb_rate`. The
configured values remain absolute bounds: lookup thresholds can tighten the selected energy range,
core scatter, or view cone, but they do not expand beyond the configured limits.

This separation is important for production reviews:

- the configuration expresses the intended maximum phase space;
- the lookup table expresses what is valid for a given direction, NSB rate, and array layout;
- the generated grid shows the final values that will be executed.

The CORSIKA-limit workflow described in the
[pre-requisites documentation](pre-requisites.md) applies when the production depends on
lookup-table limits.

## Event Statistics

Production statistics can be configured in two common ways:

- `number_of_runs` and `showers_per_run` when the number of jobs per grid point is fixed;
- `total_showers` and `showers_per_run` when the number of jobs should be derived from a
  physics-motivated total shower target.

The second mode rounds up the number of runs when the requested total is not divisible by the
selected showers per run, so all runs at a grid point have the same shower count.

For fixed-energy or wide-zenith productions, the grid generator also supports energy- and
zenith-dependent scaling of showers per run, total showers, and maximum energy. The exact
formulae and parameter names are documented in the
[simtools-production-generate-grid](../applications/simtools-production-generate-grid) reference.

## Example Configuration

The following compact configuration generates a horizontal gamma-production grid with CORSIKA
limits and a fixed number of runs per grid point:

```yaml
axis:
- azimuth 310 deg 20 deg 3 linear
- zenith 30 deg 40 deg 2 linear
- offset 0 deg 0 deg 1 linear
primary: gamma
model_version: 7.0.0
array_layout_name: LSTN-01
corsika_le_interaction: urqmd
corsika_he_interaction: epos
energy_range: 30 GeV 300 GeV
core_scatter: 10 500 m
view_cone: 0 deg 10 deg
showers_per_run: 5
number_of_runs: 2
corsika_limits: tests/resources/corsika_simulation_limits/corsika_limits_north.ecsv
output_file: job_grid_horizontal.ecsv
```

Grid generation command:

```bash
simtools-production-generate-grid --config path/to/production_generate_grid.yml
```

Several integration-test configurations are useful starting points:

- `tests/integration_tests/config/production_generate_grid_horizontal.yml`
- `tests/integration_tests/config/production_generate_grid_ha_dec.yml`

## Production Grid Plotting

The output is an ECSV production grid. Each row describes one executable run, including primary,
pointing, energy range, core scatter settings, view cone, showers per run, model version,
model-version-dependent `nsb_rate`, array layout, interaction models, and run number.

The generated table is inspected before job submission. The review includes:

- the number of rows and run numbers;
- the final energy ranges after lookup limits and scaling;
- the final core-scatter and view-cone values;
- the `nsb_rate` values for each model version;
- the direction coverage and any skipped grid points.

[simtools-plot-production-grid](../applications/simtools-plot-production-grid) visualizes the sky
coverage:

```bash
simtools-plot-production-grid \
    --grid_points_file tests/resources/production_job_grid_horizontal.ecsv \
    --output_path simtools-output
```
