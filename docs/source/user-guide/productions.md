# Simulation Productions

The production of large sets of simulated events is the core functionality of the simtools
framework. A production is configured as a grid of simulation jobs, executed locally or on a
workload management system, and verified against expected outputs or a baseline production.

## Production Workflow

The simulation production workflow is split into four steps:

1. [Pre-requisites](productions/pre-requisites.md): prepare common inputs
   needed by new productions.
2. [Configure productions](productions/configure-productions.md) with all requested options.
3. [Run productions](productions/run-productions.md) (locally or on a workload management system).
4. [Verify productions](productions/verify-productions.md): check simulations and compare them
   with a baseline production.

```{toctree}
:hidden:
:maxdepth: 1

productions/pre-requisites.md
productions/configure-productions.md
productions/run-productions.md
productions/verify-productions.md
```
