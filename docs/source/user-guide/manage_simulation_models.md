# Manage simulation models

Simulation models define the physical and technical state of the observatory used by Monte Carlo
productions. simtools supports model work in four stages:

1. **Receive input data** from telescope teams and calibration pipelines.
2. **Derive parameters** from those inputs with dedicated workflows.
3. **Validate parameters** against schema constraints and physics or engineering expectations.
4. **Publish and review** model versions for production use.

This guide focuses on process and responsibilities. Command-line details are documented in the
[Applications](applications.md) reference.

## Typical workflow

1. Prepare or update model-parameter entries in the model repository.
2. Validate consistency of schema, production tables, and file references.
3. Compare proposed model changes against a baseline model version.
4. Promote an accepted model release for production usage.

Model changes should be traceable to calibration inputs, analyses, or review decisions. Each model
version should be reproducible and attributable.

## Maintenance

- [Simulation models database and repository operations](manage_simulation_models/simulation_models_database.md)
- [Import simulation model parameters](manage_simulation_models/model_import.md)

```{toctree}
:hidden:
:glob: true
:maxdepth: 1
manage_simulation_models/simulation_models_database.md
manage_simulation_models/model_import.md
```
