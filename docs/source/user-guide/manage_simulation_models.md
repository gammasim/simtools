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
model parameters describing the pixel response to a narrow light flash as a
function of intensity. Keep `fadc_pulse_shape` for the linear
single-photoelectron response. The existing table exporter writes three columns,
ordered by intensity and time: `intensity` (fluctuated p.e. sum at nominal gain,
at least 1 p.e.), `time` (ns relative to the photoelectron arrival including
transit delay, the same origin as `fadc_pulse_shape`), and `amplitude`
(complete baseline-subtracted waveform relative to the nominal
single-photoelectron peak of that gain). Amplitudes include the intensity
response; the exporter preserves their scale. A complete shared time grid is
required and validated before serialization.

Templates include the `transit_time_jitter`, which sim_telarray does not apply
again to pixels using them. The lowest template should therefore equal its
intensity times the peak-normalized `fadc_pulse_shape` of the same telescope,
smeared with `transit_time_jitter`; sim_telarray warns if peak, area, or peak
time differ, since that would produce a step where pixels switch to the table.
Telescopes with their own `fadc_pulse_shape` therefore need their own templates.

Each signal photoelectron contributes the interpolated template, divided by its
intensity and weighted with the photoelectron amplitude, at its own arrival
time, so the photon arrival-time spread is preserved. The template is selected
by the effective intensity: the peak of the summed jitter-smeared linear pulses
relative to the single-photoelectron peak, converted to nominal gain. For
narrow illumination this is the pixel intensity; spread-out signals saturate
less. Pixel gain variations shift the intensity at which a template is selected
rather than scaling the saturated amplitude.

The generated detector keys are `FADC_HG_PULSE_SHAPE` and
`FADC_LG_PULSE_SHAPE`. sim_telarray retains linear pulses at or below the lowest
effective intensity, interpolates linearly between templates (warning if this
produces double-peaked pulses, in which case a denser intensity grid is needed),
and holds the last waveform above the largest intensity (with a warning). It
refuses tables combined with pixel cell saturation (`camera_pixel_types` with
micro-cells). Only Cherenkov readout is replaced; NSB is added separately and
the trigger remains linear. One template is used per pixel, so temporally
separated components are saturated according to the brightest one. Dummy
telescopes disable both responses.

The LST templates in development model 8.0.0 are synthetic examples for testing,
with provenance and validity information in their ECSV metadata. Replace them
with calibrated, versioned templates before scientific use.
