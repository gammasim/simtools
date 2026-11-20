# Add a ray-tracing backend

Ray-tracing backends implement the `RayTracingBackend` contract in
`simtools.ray_tracing.backends.interface`. A backend traces the requests made by
ray-tracing applications and returns image and incidence-angle results.
Calibration light and air-shower photons are traced by the stages of a simulation chain
(see [simulation chains](../../components/obdeect.md#simulation-chain)), not by a backend.

## Implementation checklist

1. Add a backend module under `simtools.ray_tracing.backends`.
2. Implement `simulate`, `psf_image`, and `incidence_angles` with the parameter
   signatures in `RayTracingBackend`.
3. Declare the backend as a software entry of `src/simtools/resources/simulation_software.yml`
   with a `ray_tracing_backend` field set to its `module:Class` path. The entry name is the
   backend name used by `--ray_tracing_backend`. Imports are lazy, so optional dependencies are
   needed only when the selected backend is loaded.
4. Add a contract test that checks the backend methods and their signatures.
