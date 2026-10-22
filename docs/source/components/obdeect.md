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

obdeect runs inside the simtools process. It compiles the model parameters of
the telescope model, exactly as simtools reads them for sim_telarray, into an
optical model and traces star light to the camera entrance. Light source,
telescope pointing, photon numbers, mirror-error realisations and the
single-mirror mode follow the sim_telarray ray-tracing runs. Both backends use
the same seeds: `--instrument_seed` fixes the mirror errors of the
`--random_instrument_instances` realisations, `--simulation_seed` the photons.
obdeect results do not depend on the number of threads.

If the telescope model describes the telescope structure
(`telescope_structure`),
obdeect traces its shadows and `telescope_transmission` is not applied to the
effective area. sim_telarray does not simulate the structure.
