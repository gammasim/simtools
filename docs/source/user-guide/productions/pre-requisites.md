# Production Pre-Requisites

Production pre-requisites are inputs derived from broad-range simulations or other sources.

```{Note}
This step is not required for every new production.
The configuration inputs found in the
[production configuration repository][production-configuration]
are expected to include the necessary pre-requisites for most productions.
```

## Trigger histograms

The estimation of required Monte Carlo statistics and the derivation of CORSIKA limits require
triggered-event histograms built from broad-range reduced event-data files. The histograms are
binned in energy, angular distance to the camera center, and shower-core distance. The original
broad-range simulation geometry is kept in the metadata, including the energy range, view-cone
range, core-scatter range, and scatter area.

Example command:

```bash
simtools-write-trigger-histograms \
    --event_data_file "reduced_event_data/gamma_20deg_0deg_run*.hdf5" \
    --array_layout_name LSTN-01 \
    --site North \
    --model_version 7.0.0 \
    --energy_bins_per_decade 10 \
    --angular_distance_bin_width "0.5 deg" \
    --output_file trigger_histograms.hdf5
```

## Derivation of CORSIKA Limits

CORSIKA configuration limits are derived from broad-range simulations and include
the lower CORSIKA energy limit, the maximum core scatter radius, and the maximum viewcone radius.
All these limits depend on observation parameters like zenith angle and the integrated NSB rate.
Limits are derived for each array layout from distributions of triggered events.

Input for the derivation is a trigger-histogram HDF5 file generated with
[simtools-write-trigger-histograms][build-trigger-histograms] from reduced event-data files.

[simtools-production-derive-corsika-limits][derive-corsika-limits] reads the trigger-histogram
file and derives the limits used by production grids:

- the lower CORSIKA energy limit from the triggered-energy distribution
- the maximum core scatter radius from the accepted loss in triggered events
- the maximum viewcone radius from the accepted loss in triggered events

The result is an ECSV lookup table with one row per production point and array layout. The table
contains the derived values and the broad-range simulation settings from which they were derived.
An example table is located in the test resources
(`tests/resources/corsika_simulation_limits/corsika_limits_north.ecsv`).

Production lookup tables are maintained in the
CTAO [production configuration repository][production-configuration] repository.

Example command:

```bash
simtools-production-derive-corsika-limits \
    --trigger_histogram_file trigger_histograms.hdf5 \
    --array_names LSTN-01 \
    --allowed_losses core_distance,1e-3,10 \
    --allowed_losses angular_distance,1e-3,10 \
    --energy_threshold_fraction 0.01 \
    --output_file corsika_limits_north.ecsv
```

## Plot CORSIKA Limits

Use [simtools-plot-corsika-limits](../applications/simtools-plot-corsika-limits) to inspect a
CORSIKA limits table. The application plots the derived limits as a function of zenith angle.

Deterministic example using the test lookup table:

```bash
simtools-plot-corsika-limits \
    --input tests/resources/corsika_simulation_limits/corsika_limits_north.ecsv \
    --output_path simtools-output
```

The command writes one plot per array-layout and azimuth combination to `simtools-output`.

## Derivation of Monte Carlo Statistics

Required Monte Carlo event statistics are estimated from triggered-event histograms built from
broad-range reduced event-data files. This workflow optimizes trigger statistics only. It does
not estimate post-cut reconstruction or DL2 analysis statistics.

[simtools-production-derive-monte-carlo-statistics][estimate-monte-carlo-statistics] reads this
HDF5 product and solves for the total number of thrown events needed to reach a requested
relative Poisson uncertainty in every estimable bin of the optimization energy range.

The estimator supports an optional `reduced_core_radius`. The builder always writes histograms
for the original broad-range core-scatter radius. At estimation time, the estimator integrates
the core-distance-binned simulated and triggered counts up to the selected radius, recomputes the
trigger efficiency, and then derives the required event count. The reduced-radius calculation
therefore uses the core-radius-dependent trigger efficiency; it is not a pure geometric
`r^2 / R^2` rescaling. Radius overrides larger than the original broad-range core-scatter radius
are rejected.

Example command:

```bash
simtools-production-derive-monte-carlo-statistics \
    --trigger_histogram_file trigger_histograms.hdf5 \
    --target_relative_uncertainty 0.1 \
    --optimization_energy_min "0.1 TeV" \
    --optimization_energy_max "10 TeV" \
    --reduced_core_radius "1000 m" \
    --plot_diagnostics \
    --output_file trigger_histograms_estimate.ecsv
```

The ECSV output reports the estimated total number of events and the value describing the
parameter space. Diagnostic plots show the expected triggered
events and relative uncertainty in energy and angular distance after applying the selected core
radius.


[derive-corsika-limits]: ../applications/simtools-production-derive-corsika-limits
[build-trigger-histograms]: ../applications/simtools-write-trigger-histograms
[estimate-monte-carlo-statistics]: ../applications/simtools-production-derive-monte-carlo-statistics
[production-configuration]: https://gitlab.cta-observatory.org/cta-science/simulations/productions/production-configuration
