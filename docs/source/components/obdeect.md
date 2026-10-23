# Telescope ray tracing with obdeect

[obdeect](https://github.com/GernotMaier/obdeect) is an optical ray tracer for
imaging atmospheric Cherenkov telescopes. It is an optional backend for the
ray-tracing applications of simtools (PSF, effective mirror area, effective
focal length, incident angles). Install it with

```sh
python -m pip install 'gammasimtools[obdeect]'
```

and select it with `--ray_tracing_backend obdeect` (or the environment variable
`SIMTOOLS_RAY_TRACING_BACKEND=obdeect`); `sim_telarray` remains the default.

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
