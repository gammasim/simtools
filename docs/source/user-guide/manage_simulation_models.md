# Manage simulation models

Simulation models define the physical and technical state of the observatory used by Monte Carlo
productions. simtools supports model work in four stages:

1. **Receive input data** from telescope teams and calibration pipelines.
2. **Derive parameters** from those inputs with dedicated workflows.
3. **Validate parameters** against schema constraints and physics or engineering expectations.
4. **Publish and review** model versions for production use.

This guide focuses on process and responsibilities. Command-line details are documented in the
[Applications](applications.md) reference.

## Typical workflow

1. Prepare or update model-parameter entries in the model repository.
2. Validate consistency of schema, production tables, and file references.
3. Compare proposed model changes against a baseline model version.
4. Promote an accepted model release for production usage.

Model changes should be traceable to calibration inputs, analyses, or review decisions. Each model
version should be reproducible and attributable.

## Maintenance

- [Import simulation model parameters](manage_simulation_models/model_import.md)

```{toctree}
:hidden:
:glob: true
:maxdepth: 1
manage_simulation_models/model_import.md
```

## Intensity-dependent readout pulse templates

`fadc_hg_pulse_shape` and `fadc_lg_pulse_shape` are optional, independent ECSV
model parameters for complete narrow-pulse responses. Keep `fadc_pulse_shape`
for the linear single-photoelectron response. The existing table exporter writes
three columns, ordered by intensity and time: `intensity` (fluctuated p.e. sum),
`time` (ns relative to the median photoelectron arrival including transit delay),
and `amplitude` (complete baseline-subtracted waveform relative to the nominal
single-photoelectron peak of that gain). Amplitudes include the intensity
response; the exporter preserves their scale. A complete shared time grid is
required and validated before serialization.

The generated detector keys are `FADC_HG_PULSE_SHAPE` and
`FADC_LG_PULSE_SHAPE`. sim_telarray retains linear pulses at or below the lowest
intensity, interpolates linearly between templates, and holds the last waveform
above the largest intensity. Only Cherenkov readout is replaced; NSB is added
separately and the trigger remains linear. This whole-pixel approximation is
intended for narrow illumination and cannot resolve separated components.
Dummy telescopes disable both responses.

The LST templates in development model 8.0.0 are synthetic examples for testing,
with provenance and validity information in their ECSV metadata. Replace them
with calibrated, versioned templates before scientific use.
