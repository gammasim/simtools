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

A simulation is an ordered list of stages, `--simulation_chain` (default
`corsika sim_telarray`). Each stage is written `STAGE` or `STAGE:KEY=VALUE,...`; the stages and
the software they need are declared in `resources/simulation_stages.yml`. obdeect provides:

| Stage | Task |
| --- | --- |
| `corsika_to_stream` | CORSIKA photon bunches to a photon stream |
| `obdeect_atmosphere` | extinction of the atmosphere along the light path |
| `obdeect_optics` | mirrors, shadowing and telescope transmission, one process per telescope |
| `stream_to_simtel` | photon stream to the sim_telarray photon-bunch input |
| `obdeect_flasher`, `obdeect_illuminator` | calibration light; the flasher light is traced through the telescope structure |

Air showers through obdeect atmosphere and optics, with sim_telarray simulating only the camera:

```bash
simtools-simulate-prod --simulation_chain corsika corsika_to_stream obdeect_atmosphere \
  obdeect_optics stream_to_simtel sim_telarray:bypass_optics=3 ...
```

Stages are connected by the data products they consume and produce, and a chain in which two
stages apply the same efficiency factor (for example the atmosphere in obdeect and in
sim_telarray) is rejected. `Bypass_Optics=3` makes sim_telarray simulate the camera only. simtools
checks that the installed sim_telarray release supports it before any job is written; for a
locally patched build, `SIMTOOLS_SIM_TELARRAY_BYPASS_OPTICS_MAX=3` states the highest supported
level. The calibration applications select obdeect with `--ray_tracing_backend` (next section).

The programs of a stage are located through `resources/simulation_programs.yml`; a further
software package adds its stages and programs there, and its command builders in a sibling of
`simulation_chain/obdeect/`.

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
