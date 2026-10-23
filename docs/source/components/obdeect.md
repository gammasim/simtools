# Telescope ray tracing with obdeect

[obdeect](https://github.com/GernotMaier/obdeect) is an optical ray tracer for
imaging atmospheric Cherenkov telescopes. It is an optional backend for the
ray-tracing applications of simtools (PSF, effective mirror area, effective
focal length, incident angles) and for the calibration light of flashers and
illuminators. Install it with

```sh
python -m pip install 'gammasimtools[obdeect]'
```

and select it with `--ray_tracing_backend obdeect` (or the environment variable
`SIMTOOLS_RAY_TRACING_BACKEND=obdeect`); `sim_telarray` remains the default.

The ray-tracing applications describe what to trace (star position, source
distance, photons, seeds and parameter changes such as the single-mirror mode)
and hand it to the backend (`simtools.ray_tracing.backends`); both backends
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
- Illuminator light crosses the atmosphere (sim_telarray's transmission table
  and formula, along the real path) and the optics; telescope position,
  pointing and axes offsets are those of the sim_telarray run.
- `--instrument_seed` and `--simulation_seed` seed obdeect and sim_telarray
  with either backend.
