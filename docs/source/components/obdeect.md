# Telescope ray tracing with obdeect

[obdeect](https://github.com/gammasim/obdeect) is an optical ray tracer for
imaging atmospheric Cherenkov telescopes. It is a backend for the
ray-tracing applications of simtools (PSF, effective mirror area, effective
focal length, incident angles) and for the calibration light of flashers and
illuminators. Select it with `--ray_tracing_backend obdeect` or set
`SIMTOOLS_RAY_TRACING_BACKEND=obdeect`; `sim_telarray` remains the default.

The ray-tracing applications describe what to trace (star position, source
distance, photons, seeds and parameter changes such as the single-mirror mode)
and hand it to the backend (`simtools.ray_tracing.backends.interface`); both backends
receive the same requests. obdeect runs inside the simtools process: it compiles
the model parameters of the telescope model, exactly as simtools reads them for
sim_telarray, into an optical model and traces star light to the camera
entrance; images stay in memory (`--save_photons` writes them as ECSV tables).
`--instrument_seed` fixes the mirror errors of the
`--random_instrument_instances` realisations and `--simulation_seed` the photons,
for either backend. obdeect results do not depend on the number of threads.

If the telescope model describes the telescope structure
(`telescope_structure`), obdeect traces its shadows and `telescope_transmission`
is not applied to the effective area. sim_telarray does not simulate the
structure.

## Simulation chain

A simulation is an ordered list of stages, `--simulation_software` (default
`corsika sim_telarray`). Each stage is written `STAGE` or `STAGE:KEY=VALUE,...`, with list values in
brackets (`obdeect_atmosphere:wavelength_range_nm=[300,700]`); obdeect provides:

| Stage | Task |
| --- | --- |
| `corsika_to_stream` | CORSIKA photon bunches to a photon stream |
| `obdeect_atmosphere` | extinction of the atmosphere along the light path |
| `obdeect_optics` | mirrors, shadowing and telescope transmission, one process per telescope |
| `stream_to_simtel` | photon stream to the sim_telarray photon-bunch input |
| `obdeect_flasher`, `obdeect_illuminator` | calibration light; the flasher light is traced through the telescope structure |
| `light_emission` | sim_telarray LightEmission program (`ff-1m`, `xyzls`) as first stage of a calibration chain |

Air showers through obdeect atmosphere and optics, with sim_telarray simulating only the camera:

```bash
simtools-simulate-prod --simulation_software corsika corsika_to_stream obdeect_atmosphere obdeect_optics stream_to_simtel \
  sim_telarray:bypass_optics=3 ...
```

`obdeect_atmosphere` derives the seed of its photon thinning from the simulation seed and the run
number, so each run draws its own random numbers; `obdeect_atmosphere:seed=N` uses `N` for every
run.

List conversion stages explicitly between programs with different photon formats. A chain in which two stages apply the same efficiency factor (for example the atmosphere
in obdeect and in sim_telarray) is rejected. `Bypass_Optics=3` makes sim_telarray simulate the
camera only. simtools checks that the installed sim_telarray release supports it before any job is
written; for a locally patched build, `SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX=3` states the
highest supported level.

For multiple telescopes, the photon stream before optics and each telescope's camera stream are
written to disk. The camera streams are merged by event before sim_telarray runs, so the output
directory needs space for these intermediate files.

The calibration applications select obdeect with `--ray_tracing_backend` (next section) or with
`--simulation_software` (default `light_emission sim_telarray`), which runs the stages for one
telescope: a calibration chain starts with `light_emission`, `obdeect_flasher` or
`obdeect_illuminator` and has no `corsika` stage. The option `bypass_optics` of `sim_telarray` is
required when the photons come from obdeect (camera-frame photons); after `light_emission` it
follows the light source type.

The quantum-efficiency envelope (`obdeect_atmosphere:efficiency_envelope=qe`) thins the photons
by the telescope QE divided by a headroom, and sim_telarray then reads the QE table divided by
the envelope. It is on by default for a chain ending in `sim_telarray:bypass_optics=3` and off
otherwise; `efficiency_envelope=none` or `qe` overrides the default.

When an upstream stage thins the photons (`obdeect_atmosphere`, also with
`efficiency_envelope=qe`), simtools runs sim_telarray with `min_photons=0`. sim_telarray skips
a telescope event with fewer photons than `min_photons`, and the count is that of the thinned
photons, so the default cut would drop whole events. `min_photoelectrons` still applies after
the quantum efficiency. The run log states this ("Upstream obdeect_atmosphere thins photons").

```bash
simtools-simulate-flasher --run_mode full_simulation --simulation_software obdeect_flasher stream_to_simtel \
  sim_telarray:bypass_optics=1 ...
simtools-simulate-illuminator --simulation_software obdeect_illuminator obdeect_atmosphere \
  obdeect_optics stream_to_simtel sim_telarray:bypass_optics=3 ...
```

## Flashers and illuminators

With `--ray_tracing_backend obdeect`, `simtools-simulate-flasher --run_mode
full_simulation` and `simtools-simulate-illuminator` split the simulation into
stages: obdeect makes the light of the calibration device and traces it to the
camera entrance, simtools writes the photons as photon bunches
(`simtel.photon_bunch_file`), and sim_telarray simulates the camera and
electronics with its optics bypassed (`Bypass_Optics=1` for single-mirror,
`Bypass_Optics=2` for dual-mirror telescopes). This replaces the sim_telarray
LightEmission programs (`ff-1m`, `xyzls`) and sim_telarray's optics.

- The light source follows the calibration model as for LightEmission: angular
  distribution, pulse shape, wavelength and photon number per pulse; the
  number of photons reaching the telescope is Poisson distributed.
- Flasher light goes straight from the flasher to the camera; the telescope
  structure, if described, shadows it.
- Illuminator light crosses the atmosphere (the `atmospheric_transmission`
  table of the site, along the real path) and the optics; telescope position,
  pointing and axes offsets are those of the sim_telarray backend.
- `--instrument_seed` and `--simulation_seed` seed obdeect emission and the sim_telarray
  camera simulation. The default LightEmission programs choose their own emission seeds.
