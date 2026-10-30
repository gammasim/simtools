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

The `fadc_pulse_shape` parameter stores both the linear reference pulses and
optional intensity-dependent HG/LG responses. Existing versions with columns
`time`, `amplitude`, and optional `amplitude_low_gain` remain valid. Schema
0.4.0 uses long-form rows with `template_kind` (`reference` or `response`),
`gain` (`HG` or `LG`), `intensity`, `time` (ns), and `amplitude`. Reference rows
have intensity 1 and contain the original unsmeared single-photoelectron
pulse. Response rows contain complete baseline-subtracted waveforms relative
to the nominal single-photoelectron peak. Each gain can use its own intensity
grid; each gain's response templates share a time grid, and the two reference
pulses share a time grid. simtools exports the reference to the linear pulse
input and writes one response file per gain. It passes the response files with
the `fadc_hg_response_table` and `fadc_lg_response_table` simulator keys; these
keys describe simulator inputs, not model parameters.

Response templates include `transit_time_jitter`, which sim_telarray does not
apply again to pixels using them. The reference pulse remains unsmeared in the
model and is used for the linear response; the simulator applies transit-time
jitter to individual photoelectrons on that path. The lowest response template
should equal its intensity times the peak-normalized reference smeared with
`transit_time_jitter`. Telescopes with their own reference pulse therefore need
their own response templates.

Each signal photoelectron contributes the interpolated template, divided by its
intensity and weighted with the photoelectron amplitude, at its own arrival
time, so the photon arrival-time spread is preserved. The template is selected
by the effective intensity: the peak of the summed jitter-smeared linear pulses
relative to the single-photoelectron peak, converted to nominal gain. For
narrow illumination this is the pixel intensity; spread-out signals saturate
less. Pixel gain variations shift the intensity at which a template is selected
rather than scaling the saturated amplitude.

sim_telarray retains linear pulses at or below the lowest
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
