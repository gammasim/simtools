# Telescope simulations with obdeect

[obdeect](https://github.com/gammasim/obdeect) simulates light from its source to the camera
entrance. Install the `obdeect` package in the same environment as simtools. simtools uses it for
ray tracing and as stages in simulation chains:

- **Ray tracing in process.** The ray-tracing applications (`simtools-validate-optics`,
  `simtools-derive-psf-parameters`, `simtools-derive-mirror-rnda`,
  `simtools-derive-incident-angle`) select the backend with
  `--simulation_software obdeect_star` (default `sim_telarray_star`). obdeect compiles the
  telescope model into an optical model and traces star light, with the same star positions,
  parameter changes, `--instrument_seed` and `--simulation_seed` as sim_telarray. Images stay
  in memory. Results do not depend on the number of threads.
- **Stages of a simulation chain.** `simtools-simulate-prod`, `simtools-simulate-flasher` and
  `simtools-simulate-illuminator` take obdeect stages in `--simulation_software`.

## Simulation chain

A chain is an ordered list of stages, `--simulation_software` (default `corsika sim_telarray`).
Each stage is `STAGE` or `STAGE:KEY=VALUE,...`. obdeect traces photons in photon streams and hands
camera-frame photons to sim_telarray, which simulates the camera and electronics.

| Stage | Task |
| --- | --- |
| `corsika_to_stream` | CORSIKA photon bunches to a photon stream |
| `obdeect_atmosphere` | emission point to the ground: refraction, timing and atmospheric extinction |
| `obdeect_optics` | mirrors, reflectivity, shadowing and telescope transmission; one process traces all telescopes of the array |
| `stream_to_simtel` | camera-frame photons to the sim_telarray input |
| `obdeect_flasher`, `obdeect_illuminator` | photons of a calibration device, traced to the camera |
| `light_emission`, `corsika`, `sim_telarray` | sim_telarray LightEmission programs, CORSIKA, sim_telarray |

Stages, their options and the files they write are declared in
[`resources/simulation_software.yml`](https://github.com/gammasim/simtools/blob/main/src/simtools/resources/simulation_software.yml);
options are checked against it when the chain is read. Files are named
`<run base name>.<stage name><suffix>`.

simtools inserts the converters `corsika_to_stream` and `stream_to_simtel` between two stages
with different photon formats; list a converter to give it options. A chain in which two stages
apply the same efficiency factor (for example the atmospheric extinction in `obdeect_atmosphere`
and in sim_telarray) is rejected.

### Air showers

```bash
simtools-simulate-prod --simulation_software corsika obdeect_atmosphere obdeect_optics \
  sim_telarray:bypass_optics=camera_only ...
```

- `bypass_optics=3` makes sim_telarray simulate the camera only; `1` and `2` take photons at the
  camera entrance and still apply the atmosphere. Keep the photons' fate in one place: with the
  full obdeect chain use `3`.
- simtools checks that the installed sim_telarray supports the chosen level before any job is
  written. For a locally patched build, `SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX=3` states the
  highest supported level.
- `obdeect_atmosphere` thins the photons by default, so simtools runs sim_telarray with
  `min_photons=0` (a chain with thinned photons would otherwise lose whole events); the run log
  states this. `min_photoelectrons` still applies.
- The seed of the thinning follows the simulation seed and the run number, so each run draws its
  own photons. `obdeect_atmosphere:seed=N` uses `N` for every run.
- The whole chain runs as one pipe, also for several telescopes: one `obdeect_optics` process
  traces every telescope with its own optical model and writes camera photons in event order.

### Calibration light

The calibration applications use obdeect for one telescope. The chain starts with a light-source
stage (`obdeect_flasher`, `obdeect_illuminator` or `light_emission`) and has no `corsika` stage.

```bash
simtools-simulate-flasher --run_mode full_simulation \
  --simulation_software obdeect_flasher sim_telarray:bypass_optics=camera_only ...
simtools-simulate-illuminator \
  --simulation_software obdeect_illuminator obdeect_atmosphere obdeect_optics \
  sim_telarray:bypass_optics=camera_only ...
```

- The light source follows the calibration model as for LightEmission: angular distribution,
  pulse shape, wavelength and photon number per pulse.
- Flasher light goes straight from the flasher to the camera; the telescope structure, if
  described, shadows it. Illuminator light crosses the atmosphere along the real path and the
  optics.
- obdeect stages deliver photons at the camera entrance, so `bypass_optics` is required. The
  value `1` or `2` follows the mirror class of the telescope (single or dual mirror), `3` applies
  no optical depth at all. Flasher photons stay in the telescope frame; illuminator photons arrive
  through the optics.

### What obdeect applies, and what sim_telarray applies

Each module records in the photon stream which efficiency factors it has applied, so no factor is
applied twice or forgotten. `stream_to_simtel` checks the record against the frame of the photons
and refuses a chain that would double-count or drop a factor; the error names the factor.
`allow_pending` and `scaled_factors` are the stage options for the rare cases where sim_telarray
applies a factor itself or uses a scaled table.

- The opt-in quantum-efficiency envelope (`obdeect_atmosphere:efficiency_envelope=qe`) thins the
  photons by the telescope QE divided by a headroom; sim_telarray then reads the QE table divided
  by the envelope. It is on by default for a chain ending in `sim_telarray:bypass_optics=3` and
  off otherwise.
- If the telescope model describes the telescope structure (`telescope_structure`), obdeect traces
  its shadows and does not apply `telescope_transmission`. sim_telarray does not simulate the
  structure.
- obdeect stops at the camera entrance: camera windows, light guides and the camera simulation are
  sim_telarray's.

## Notes

- obdeect is a Python package with native programs for the stream stages. Install it in the same
  environment as simtools.
- The chain and the ray-tracing backends are exercised by the `*_obdeect` integration-test
  configurations in `tests/integration_tests/config`. Details of the optical model, the photon
  stream and the physics are in the obdeect [documentation](https://gammasim.github.io/obdeect/).
