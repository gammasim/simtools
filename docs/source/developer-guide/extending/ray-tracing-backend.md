# Add a ray-tracing backend

Ray-tracing backends implement the `RayTracingBackend` contract in
`simtools.ray_tracing.backends.interface`. A backend traces the requests made by
ray-tracing applications and returns image and incidence-angle results. A
backend that traces calibration light sources also implements
`LightSourceRayTracingBackend` and provides `camera_photons`.

## Implementation checklist

1. Add a backend module under `simtools.ray_tracing.backends`.
2. Implement `simulate`, `psf_image`, and `incidence_angles` with the parameter
   signatures in `RayTracingBackend`.
3. For calibration light tracing, implement `camera_photons` with the signature
   in `LightSourceRayTracingBackend` and set `traces_light_sources` to `True`.
4. Add the backend name and its `module:class` path to `RAY_TRACING_BACKEND_CLASSES` in
   `simtools.constants`. Imports are lazy, so optional dependencies are needed
   only when the selected backend is loaded.
5. Add a contract test that checks the backend methods and their signatures.
