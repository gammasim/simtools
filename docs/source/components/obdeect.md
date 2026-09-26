# Telescope ray tracing with obdeect

The optional obdeect backend is distributed as the `obdeect-dev` PyPI package.
Install it together with simtools using:

```sh
python -m pip install 'gammasimtools[obdeect]'
```

The wheel contains the compiled C++ ray-tracing engine and the
`obdeect-simtools-raytrace` command. simtools resolves that command from the
installed package; a separate obdeect checkout, installation directory, or
runtime container is not required.

The `sim_telarray` backend remains the default. The obdeect backend can be
selected with `ray_tracing_backend=obdeect` for the packaged reference optical
prescriptions. Production model-driven runs remain gated on validation of the
versioned scene and optical-arrival contract for the requested telescope model.

For an imported nominal LST/MST scene, pass the file written by
`obdeect.scene_compiler.write_native_scene` as `obdeect_scene_file` in the
simulation configuration. The native CLI preserves scene provenance and
finite panel/detector apertures; run-specific alignment, materials, support
geometry and dual-mirror scenes remain validation gates.
The package can also be used independently of simtools:

```sh
obdeect-simtools-raytrace --telescope MST --photons 10000 --output trace.csv
```
