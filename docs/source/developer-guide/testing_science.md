# Science Tests

Science tests are longer-running validation workflows for physics performance,
production behavior, and release qualification. They are not the default PR
test layer.

## Scope

Science tests should cover:

- physics performance with sufficient event statistics and relevant phase space
- production-like execution environments
- CPU, memory, storage, and wall-time behavior at realistic scale
- release-to-release comparisons that detect unintended physics regressions

Typical science-test comparisons include:

- simtools version against simtools version
- production model version against production model version
- hadronic interaction model against hadronic interaction model

Typical observables include:

- trigger rates
- telescope trigger multiplicities
- triggered-event distributions versus energy or core distance
- pedestal, signal, and timing distributions
- CPU time, wall time, and storage use

Keep the comparison grid and event statistics explicit in the workflow or
campaign description so results remain reproducible.

These tests run on release milestones or another explicitly agreed campaign,
not on every pull request.

```{warning}
Science tests are not implemented in the simtools testing workflow yet.
```
