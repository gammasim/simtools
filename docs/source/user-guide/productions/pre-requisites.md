# Production Pre-Requisites

Production pre-requisites are inputs derived from broad-range simulations or other sources.

```{Note}
This step is not required for every new production.
The configuration inputs found in the
[production configuration repository][production-configuration]
are expected to include the necessary pre-requisites for most productions.
```

## Derivation of CORSIKA Limits

CORSIKA configuration limits are derived from broad-range simulations and include
the lower CORSIKA energy limit, the maximum core scatter radius, and the maximum viewcone radius.
All these limits depend on observation parameters like zenith angle and the integrated NSB rate.
Limits are derived for each array layout from distributions of triggered events.

Input for the derivation are reduced event-data files from broad-range simulations generated
typically during productions using the
[simtools-generate-simtel-event-data](../applications/simtools-generate-simtel-event-data)
application.

[simtools-production-derive-corsika-limits][derive-corsika-limits] reads the reduced event-data
files and derives the limits used by production grids:

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
    --event_data_file "reduced_event_data/gamma_20deg_0deg_run*.hdf5" \
    --array_layout_name LSTN-01 \
    --model_version 7.0.0 \
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

[derive-corsika-limits]: ../applications/simtools-production-derive-corsika-limits
[production-configuration]: https://gitlab.cta-observatory.org/cta-science/simulations/productions/production-configuration
