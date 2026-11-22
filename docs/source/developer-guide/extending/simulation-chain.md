# Extend the simulation chain

Simulation chains combine stages that produce, transform, and consume simulation data.
To change part of a chain, first identify the stage responsible for the work and the
products it exchanges with its neighbors. See [simulation chains](../../components/obdeect.md#simulation-chain)
for how applications select and run stages.

## Add or change a stage

1. Inspect `src/simtools/resources/simulation_software.yml` and the existing stage
   implementations in `simtools.runners`. Define the input and output products, their
   formats and frames, and which physical efficiency factors the stage applies.
2. Implement the stage using the existing stage implementations as examples. Keep
   simulator-specific behavior in the relevant simulation software package.
3. Declare the stage and its implementation in the software catalog. Describe its
   products, options, files, and applicable roles and efficiency factors. Add product
   or software entries when needed. Converters are declared as stages as well.
4. Check that neighboring stages can exchange the declared products and that the
   chain applies the required efficiency factors without duplication. Update an
   existing default chain only when the intended default behavior should change.
5. Add focused tests for the stage and its catalog declaration, including incompatible
   products or options where relevant. Exercise representative chains through the
   application integration tests and document any user-facing choices.

The catalog is validated against `schemas/simulation_software.schema.yml`. Existing
entries provide examples of source stages, converters, and simulation stages.

## Ray-tracing applications

The in-process ray-tracing applications use a separate backend contract. To add a
backend, implement `simulate`, `psf_image`, and `incidence_angles` with the signatures
in `simtools.ray_tracing.backends.interface.RayTracingBackend`. Declare its
`module:Class` path in the `ray_tracing_backend` field of a software entry in the
catalog, and add a contract test for its methods and signatures. The entry name selects
the backend through `--simulation_software`. Imports are lazy, so optional dependencies
are needed only when the selected backend is loaded.
